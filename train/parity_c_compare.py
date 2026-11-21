"""C side + comparison for the JAX<->C single-step parity harness.

Part 2 of 2. Consumes the dump from `train/parity_jax_dump.py`, feeds the
equivalent inputs to `infer/liblcm.so`, and reports the gap.

TWO C ENTRY POINTS, and the difference between them is the point:

  * `lcm_infer_step`    — inverse-distance fusion, no routing mask, no learned
                          alpha, no LayerNorm. The deployed behaviour.
  * `lcm_infer_step_v2` — the canonical fusion from
                          `train/fusion.py::fuse_lattices_with_aux`:
                          `soft_mask * alpha` normalised, weighted sum, then
                          LayerNorm. `soft_mask` is passed in rather than
                          computed, so a mismatch stays attributable to the
                          fusion instead of the routing gate's Gumbel draw.

Run both; the gap between them is what the fusion alignment bought, and the
residual against JAX is what the six lattice forwards still owe.

Usage:
    cd infer && make LCM_D=32          # LCM_D must equal cfg.d_model
    JAX_PLATFORMS=cpu be/bin/python -m train.parity_c_compare --dump /tmp/lcm_parity_jax.npz \
        --lib /tmp/lcm_parity_build/liblcm.so

The per-lattice parity harness used by `train/test_parity_lattices.py` lives
below the CLI helpers: `run_lattice_parity`, `run_canonical_fusion_parity` and
`run_engine_fusion_parity`. It builds/loads a dedicated `liblcm_d32.so` so the
deployed d=256 engine is never involved.
"""
import argparse
import ctypes
import os
import subprocess

import numpy as np

import jax
import jax.numpy as jnp

from train.cog_loop import dag_fuse
from train.fusion import fuse_lattices_with_aux
from train.hyp import exp_map, poincare_similarity, safe_unit
from train.lattices import (
    binding_forward, contrast_forward, hrq_forward, init_binding_params,
    init_contrast_params, init_hrq_params, init_lowrank_params,
    init_manifold_params, init_route_params, init_sparse_params,
    lowrank_forward, manifold_forward, routing_gate, sparse_forward,
)

D_MODEL = 32
N_LATTICES = 6
LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast")

DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "infer", "liblcm.so")

# ─── Per-lattice parity harness (design §6.2, task 11) ──────────────────────
#
# This half of the module is imported by `train/test_parity_lattices.py`. It
# deliberately does not reuse the dump-driven CLI above: parity should compare
# the two implementations of *the same function*, not the whole deployed step.
# Fixed seed, fixed z, fixed params; the only variables are the lattice and
# the value-bias strength.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARITY_D = 32
PARITY_LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding",
                   "contrast", "routing")
# lcm.h's LCM_MAX_LATTICES; distance_weighted_fusion always iterates this many.
C_MAX_LATTICES = 7

_FP = ctypes.POINTER(ctypes.c_float)
_IP = ctypes.POINTER(ctypes.c_int)


class HRQMem(ctypes.Structure):
    _fields_ = [
        ("C_top", _FP), ("M_top", ctypes.c_int),
        ("C_fine", _FP), ("M_fine", ctypes.c_int), ("n_fine", ctypes.c_int),
        ("value_top", _FP), ("value_fine", _FP),
    ]


class SparseMem(ctypes.Structure):
    _fields_ = [
        ("C", _FP), ("M", ctypes.c_int), ("zero_vec", _FP), ("value", _FP),
        ("N", _FP), ("m", _FP), ("last_used", _IP),
    ]


class LowrankMem(ctypes.Structure):
    _fields_ = [
        ("U", _FP), ("V", _FP), ("ranks", _IP),
        ("n_layers", ctypes.c_int), ("M_lr", ctypes.c_int),
        ("r_max", ctypes.c_int), ("value", _FP),
    ]


class ManifoldMem(ctypes.Structure):
    _fields_ = [
        ("C", _FP), ("M", ctypes.c_int), ("T", _FP),
        ("t_dim", ctypes.c_int), ("value", _FP),
    ]


class BindingMem(ctypes.Structure):
    _fields_ = [
        ("key_cb", _FP), ("val_cb", _FP), ("bind_cb", _FP),
        ("M", ctypes.c_int), ("n_layers", ctypes.c_int),
        ("r_max", ctypes.c_int),
        ("A_k", _FP), ("A_v", _FP),
        ("value_key", _FP), ("value_val", _FP), ("value_bind", _FP),
    ]


class ContrastMem(ctypes.Structure):
    _fields_ = [
        ("C_a", _FP), ("C_b", _FP), ("M", ctypes.c_int),
        ("n_layers", ctypes.c_int), ("value_a", _FP), ("value_b", _FP),
    ]


class LatticeMemory(ctypes.Structure):
    _fields_ = [("C", _FP), ("M", ctypes.c_int), ("D", ctypes.c_int)]


_LIB_CACHE = {}


def _np32(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _jax_to_np(x):
    return np.asarray(x, dtype=np.float32)


def _ptr(a):
    return a.ctypes.data_as(_FP)


def _iptr(a):
    return a.ctypes.data_as(_IP)


def parity_lib_path(d=PARITY_D):
    return os.path.join(REPO_ROOT, "infer", f"liblcm_d{int(d)}.so")


def ensure_parity_lib(d=PARITY_D, build=True):
    """Return a parity .so built for ``LCM_D == d``.

    ``LCM_PARITY_LIB`` overrides the path (used by callers with a prebuilt
    library). The override is trusted as-is: only its compiled ``LCM_D`` is
    verified, not that its sources match this worktree. Use the default builder
    unless you know the override is current.

    Otherwise ``make -C infer PARITY_D=<d> liblcm_d<d>.so`` uses the dedicated
    object directory, so the deployed d=256 build is not touched. A missing
    library or failed build raises with the compiler output; parity must never
    silently skip.
    """
    override = os.environ.get("LCM_PARITY_LIB")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"LCM_PARITY_LIB={override!r} does not exist; build the parity "
                f"library or unset the variable")
        return override

    path = parity_lib_path(d)
    if build:
        cmd = ["make", "-C", os.path.join(REPO_ROOT, "infer"),
               f"PARITY_D={int(d)}", f"liblcm_d{int(d)}.so"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise RuntimeError(
                f"cannot run {' '.join(cmd)}: {exc}. The per-lattice parity "
                f"tests need a C99 compiler and `make`.") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"parity C build failed (LCM_D={d}):\n"
                f"$ {' '.join(cmd)}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with:\n"
            f"    make -C infer PARITY_D={int(d)} liblcm_d{int(d)}.so\n"
            f"The engine is dimensioned at compile time, so the deployed "
            f"d=256 liblcm.so cannot be reused at d={d}.")
    return path


def load_parity_lib(path=None, d=PARITY_D):
    path = os.path.abspath(path or ensure_parity_lib(d))
    st = os.stat(path)
    cache_key = (path, st.st_mtime_ns, st.st_size)
    if cache_key in _LIB_CACHE:
        return _LIB_CACHE[cache_key]

    try:
        ctypes.CDLL("libm.so.6", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    lib = ctypes.CDLL(path)

    lib.hrq_forward_v31.restype = ctypes.c_int
    lib.hrq_forward_v31.argtypes = [
        _FP, ctypes.POINTER(HRQMem), ctypes.c_float, ctypes.c_int,
        ctypes.c_float, _FP, _IP, _FP]
    lib.sparse_forward_c.restype = ctypes.c_int
    lib.sparse_forward_c.argtypes = [
        _FP, ctypes.POINTER(SparseMem), ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_int, _FP, _IP]
    lib.lowrank_forward_c.restype = None
    lib.lowrank_forward_c.argtypes = [
        _FP, ctypes.POINTER(LowrankMem), ctypes.c_float, _FP, _IP]
    lib.manifold_forward_c.restype = None
    lib.manifold_forward_c.argtypes = [
        _FP, ctypes.POINTER(ManifoldMem), ctypes.c_float, _FP, _IP]
    lib.contrast_forward_v31.restype = None
    lib.contrast_forward_v31.argtypes = [
        _FP, ctypes.POINTER(ContrastMem), ctypes.c_float, _FP, _IP, _IP]
    lib.binding_forward_c.restype = None
    lib.binding_forward_c.argtypes = [
        _FP, ctypes.POINTER(BindingMem), _FP, ctypes.c_float, _FP, _FP, _FP,
        _IP]
    lib.routing_gate_c.restype = None
    lib.routing_gate_c.argtypes = [
        _FP, _FP, _FP, ctypes.c_int, _FP]
    lib.lcm_fuse_only.restype = ctypes.c_int
    lib.lcm_fuse_only.argtypes = [
        _FP, ctypes.c_int, ctypes.c_int, _FP, _FP, ctypes.c_int, _FP, _FP,
        _FP]
    lib.distance_weighted_fusion.restype = None
    lib.distance_weighted_fusion.argtypes = [
        _FP, _FP, _FP, _FP, ctypes.c_void_p, ctypes.c_float]
    lib.retrieve_single.restype = None
    lib.retrieve_single.argtypes = [
        _FP, ctypes.POINTER(LatticeMemory), _FP, _FP, _IP]
    lib.lcm_dim.restype = ctypes.c_int

    compiled_d = int(lib.lcm_dim())
    if compiled_d != int(d):
        raise RuntimeError(
            f"{path} was built with LCM_D={compiled_d}, expected {d}")
    _LIB_CACHE[cache_key] = lib
    return lib


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def load_lib(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Build a matching one:\n"
            f"    cd infer && make LCM_D={D_MODEL}\n"
            f"Note the C engine is dimensioned at compile time, so a .so built "
            f"for the deployed d_model (256) cannot run this comparison.")
    lib = ctypes.CDLL(path)
    lib.lcm_dim.restype = ctypes.c_int
    compiled_d = int(lib.lcm_dim())
    if compiled_d != D_MODEL:
        raise SystemExit(
            f"{path} was built with LCM_D={compiled_d}, but this harness "
            f"expects LCM_D={D_MODEL}. Rebuild with "
            f"`make -C infer PARITY_D={D_MODEL} liblcm_d{D_MODEL}.so` and "
            f"pass --lib, or use a matching .so.")
    common = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # z, d
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # hrq_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # sparse_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # lr_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # man_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # man_T, t_dim
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # bind_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # contrast_C, M
    ]
    lib.lcm_infer_step.restype = ctypes.c_int
    lib.lcm_infer_step.argtypes = common + [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # gv_pos, n
        ctypes.POINTER(ctypes.c_float),                        # gv_neg
        ctypes.c_int,                                          # n_lattices
        ctypes.POINTER(ctypes.c_float),                        # z_out
    ]
    lib.lcm_infer_step_v2.restype = ctypes.c_int
    lib.lcm_infer_step_v2.argtypes = common + [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # hrq_top_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # hrq_fine_C, M
        ctypes.c_int,                                          # hrq_n_fine
        ctypes.c_float,                                        # tau_route_fallback
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # soft_mask, n
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # alpha, n_alpha
        ctypes.POINTER(ctypes.c_float),                        # ln_scale
        ctypes.POINTER(ctypes.c_float),                        # ln_bias
        ctypes.POINTER(ctypes.c_float),                        # z_out
    ]
    lib.lcm_fuse_only.restype = ctypes.c_int
    lib.lcm_fuse_only.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,  # outputs, n, d
        ctypes.POINTER(ctypes.c_float),                              # soft_mask
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,                # alpha, n_alpha
        ctypes.POINTER(ctypes.c_float),                              # ln_scale
        ctypes.POINTER(ctypes.c_float),                              # ln_bias
        ctypes.POINTER(ctypes.c_float),                              # z_out
    ]
    return lib


def primary_codebooks(d):
    """The one codebook per lattice the C API can accept.

    Each lattice in JAX is a *stack* — HRQ has top+fine per layer, binding has
    key/value/bind per layer, contrast has C_a and C_b. The C API takes a single
    flat matrix per lattice, so this picks the primary one. Approximate on the
    input side by construction; reported, not hidden.
    """
    def cb(name):
        return _f32(np.asarray(d[f"params.{name}"], dtype=np.float32))

    def simvq(prefix):
        return _f32(cb(f"{prefix}.A") @ cb(f"{prefix}.W"))

    t_keys = sorted(k for k in d.files if k.startswith("params.manifold.T"))
    man_t = _f32(np.concatenate([d[k].ravel() for k in t_keys]))
    return {
        "hrq": simvq("hrq.top"),
        "sparse": cb("sparse.C"),
        "lowrank": _f32(cb("lowrank.A_V") @ cb("lowrank.W_V")),
        "manifold": cb("manifold.C"),
        "manifold_T": man_t,
        "binding": simvq("binding.key_cb.0"),
        "contrast": simvq("contrast.C_a.0"),
    }


def cb2(d, name):
    return np.asarray(d[f"params.{name}"], dtype=np.float32)


def _feed(lib, fn, z, cbs, man_t_dim, extra_args):
    z_c = np.zeros(D_MODEL, dtype=np.float32)
    args = [
        _ptr(_f32(z)), D_MODEL,
        _ptr(cbs["hrq"]), cbs["hrq"].shape[0],
        _ptr(cbs["sparse"]), cbs["sparse"].shape[0],
        _ptr(cbs["lowrank"]), cbs["lowrank"].shape[0],
        _ptr(cbs["manifold"]), cbs["manifold"].shape[0],
        _ptr(cbs["manifold_T"]), man_t_dim,
        _ptr(cbs["binding"]), cbs["binding"].shape[0],
        _ptr(cbs["contrast"]), cbs["contrast"].shape[0],
    ] + extra_args + [_ptr(z_c)]
    rc = fn(*args)
    if rc != 0:
        raise SystemExit(f"{fn.__name__} returned {rc}")
    return z_c.astype(np.float64)


def _report(label, z_jax, z_c):
    diff = np.abs(z_c - z_jax)
    rel = diff.max() / (np.abs(z_jax).max() + 1e-12)
    print(f"  {label:10s} max|diff|={diff.max():9.6f}  rel={rel:7.2%}  "
          f"|z_c|={np.abs(z_c).max():.4f}  |z_jax|={np.abs(z_jax).max():.4f}")
    return diff.max()


# ─── Fixed parity case ──────────────────────────────────────────────────────

PARITY_CASE = {
    "M_top": 8, "M_fine": 4, "n_fine": 2, "value_topk": 4,
    "M_sparse": 8, "M_lr": 8, "ranks": (2, 4),
    "M_man": 8, "t_dim": 3,
    "M_bind": 6, "n_bind": 2, "r_max": 4,
    "M_contrast": 6, "n_contrast": 2,
    "n_lattices": 6,
}


def _build_case(seed=0, d=PARITY_D):
    """Fixed parameters, fixed z, fixed value scalars.

    Every random quantity is derived from ``seed`` alone. The point is that
    both sides run the *same* function: if the JAX and C implementations drift,
    the only difference left is the implementation, not the inputs.
    """
    cfg = PARITY_CASE
    rng = jax.random.PRNGKey(int(seed))
    keys = jax.random.split(rng, 12)

    params = {
        "hrq": init_hrq_params(keys[0], d, cfg["M_top"], cfg["M_fine"],
                               cfg["n_fine"]),
        "sparse": init_sparse_params(keys[1], d, cfg["M_sparse"]),
        "lowrank": init_lowrank_params(keys[2], d, cfg["M_lr"],
                                       cfg["ranks"]),
        "manifold": init_manifold_params(keys[3], d, cfg["M_man"],
                                         cfg["t_dim"]),
        "binding": init_binding_params(keys[4], d, cfg["M_bind"],
                                       cfg["n_bind"], cfg["r_max"]),
        "contrast": init_contrast_params(keys[5], d, cfg["M_contrast"],
                                         cfg["n_contrast"]),
        "route": init_route_params(keys[6], cfg["n_lattices"], d),
    }

    # Norm != 1 on purpose: the R6 safe_unit boundary is part of the contract.
    z = jax.random.normal(keys[7], (1, d)) * 0.7

    vkeys = jax.random.split(keys[8], 10)

    def rv(k, n):
        return jax.random.normal(k, (n,)) * 0.5

    values = {
        "hrq": {"top": rv(vkeys[0], cfg["M_top"]),
                "fine": rv(vkeys[1], cfg["M_fine"])},
        "sparse": rv(vkeys[2], cfg["M_sparse"]),
        "lowrank": rv(vkeys[3], cfg["M_lr"]),
        "manifold": rv(vkeys[4], cfg["M_man"]),
        "binding": {"key": rv(vkeys[5], cfg["M_bind"]),
                    "val": rv(vkeys[6], cfg["M_bind"]),
                    "bind": rv(vkeys[7], cfg["M_bind"])},
        "contrast": {"a": rv(vkeys[8], cfg["M_contrast"]),
                     "b": rv(vkeys[9], cfg["M_contrast"])},
    }
    return {"seed": int(seed), "d": int(d), "z": z, "params": params,
            "values": values, "cfg": cfg}


def _first_idx_mismatch(name, jax_idx, c_idx):
    j = np.asarray(jax_idx, dtype=np.int64).ravel()
    c = np.asarray(c_idx, dtype=np.int64).ravel()
    if j.shape != c.shape:
        return (name, "shape", j.shape, c.shape)
    bad = np.flatnonzero(j != c)
    if bad.size == 0:
        return None
    k = int(bad[0])
    return (name, k, int(j[k]), int(c[k]))


def _parity_result(lattice, case, alpha, fields, idx_pairs,
                   jax_values, c_values):
    idx_equal = True
    idx_mismatch = None
    for name, jax_idx, c_idx in idx_pairs:
        m = _first_idx_mismatch(name, jax_idx, c_idx)
        if m is not None:
            idx_equal = False
            idx_mismatch = m
            break

    max_err = 0.0
    max_mm = None
    for name in fields:
        j = np.asarray(jax_values[name], dtype=np.float64)
        c = np.asarray(c_values[name], dtype=np.float64)
        if j.shape != c.shape:
            return {
                "lattice": lattice, "seed": case["seed"], "d": case["d"],
                "alpha": float(alpha), "fields": tuple(fields),
                "idx_equal": False, "max_abs_err": float("inf"),
                "jax": jax_values, "c": c_values,
                "idx_mismatch": (name, "shape", j.shape, c.shape),
                "max_mismatch": None,
            }
        if j.size == 0:
            continue
        diff = np.abs(j - c)
        finite = np.isfinite(diff)
        if not bool(finite.all()):
            # NaN must never compare as "not greater than max_err". Treat any
            # non-finite element as +inf and pin its index for the failure
            # message.
            k = int(np.flatnonzero(~finite.ravel())[0])
            idx = np.unravel_index(k, j.shape) if j.ndim > 1 else (k,)
            max_err = float("inf")
            max_mm = {"field": name, "index": idx,
                      "jax": float(j.ravel()[k]), "c": float(c.ravel()[k]),
                      "abs_err": float("inf")}
            break
        k = int(np.argmax(diff))
        idx = np.unravel_index(k, j.shape) if j.ndim > 1 else (k,)
        err = float(diff.ravel()[k])
        if err > max_err:
            max_err = err
            max_mm = {"field": name, "index": idx,
                      "jax": float(j.ravel()[k]), "c": float(c.ravel()[k]),
                      "abs_err": err}

    return {
        "lattice": lattice, "seed": case["seed"], "d": case["d"],
        "alpha": float(alpha), "fields": tuple(fields),
        "idx_equal": bool(idx_equal), "max_abs_err": float(max_err),
        "jax": jax_values, "c": c_values,
        "idx_mismatch": idx_mismatch, "max_mismatch": max_mm,
    }


# ─── One lattice at a time ──────────────────────────────────────────────────

def _hrq_parity(case, lib, alpha):
    p = case["params"]["hrq"]
    vs = case["values"]["hrq"]
    d = case["d"]
    z = case["z"]

    top = _np32(_jax_to_np(p["top"]["A"] @ p["top"]["W"]))
    fine = _np32(np.stack([_jax_to_np(fb["A"] @ fb["W"])
                           for fb in p["fine"]]))
    v_top = _np32(vs["top"])
    v_fine = _np32(vs["fine"])

    jax_out, jax_idx, jax_sim, _ = hrq_forward(
        p, z, tau_fallback=0.1, value_scalars={"top": vs["top"],
                                               "fine": vs["fine"]},
        alpha_val=float(alpha), value_topk=case["cfg"]["value_topk"])

    m = HRQMem()
    m.C_top = _ptr(top)
    m.M_top = top.shape[0]
    m.C_fine = _ptr(fine)
    m.M_fine = fine.shape[1]
    m.n_fine = fine.shape[0]
    m.value_top = _ptr(v_top)
    m.value_fine = _ptr(v_fine)

    z_np = _np32(_jax_to_np(z))
    out = np.zeros(d, dtype=np.float32)
    idx = ctypes.c_int(-1)
    sim = ctypes.c_float(float("nan"))
    rc = lib.hrq_forward_v31(
        _ptr(z_np), ctypes.byref(m), ctypes.c_float(float(alpha)),
        ctypes.c_int(case["cfg"]["value_topk"]), ctypes.c_float(0.1),
        _ptr(out), ctypes.byref(idx), ctypes.byref(sim))
    if rc != 0:
        raise RuntimeError(f"hrq_forward_v31 returned {rc}")

    return _parity_result(
        "hrq", case, alpha, ("out", "top_sim"),
        [("top_idx", _jax_to_np(jax_idx), np.array([idx.value]))],
        {"out": _jax_to_np(jax_out)[0],
         "top_sim": _jax_to_np(jax_sim).reshape(1)},
        {"out": out, "top_sim": np.array([sim.value], dtype=np.float32)})


def _sparse_parity(case, lib, alpha, training=True, d_top=None):
    p = case["params"]["sparse"]
    vs = case["values"]["sparse"]
    d = case["d"]
    z = case["z"]

    C = _np32(_jax_to_np(p["C"]))
    zero = _np32(_jax_to_np(p["zero_vec"][0]))
    v = _np32(vs)

    if d_top is None:
        jax_d_top = None
        jax_c_top = None
        c_d_top = 0.0
    else:
        # c_top only needs the (B, d) shape for the dependency-edge check; the
        # LFQ comparison itself uses d_top and the non-zero codebook.
        jax_d_top = jnp.full((1,), float(d_top), dtype=jnp.float32)
        jax_c_top = jnp.zeros((1, d), dtype=jnp.float32)
        c_d_top = float(d_top)

    jax_out, jax_idx = sparse_forward(
        p, z, training=training, lambda_lfq=1.5, d_top=jax_d_top,
        c_top=jax_c_top, value_scalars=vs, alpha_val=float(alpha))

    m = SparseMem()
    m.C = _ptr(C)
    m.M = C.shape[0]
    m.zero_vec = _ptr(zero)
    m.value = _ptr(v)

    z_np = _np32(_jax_to_np(z))
    out = np.zeros(d, dtype=np.float32)
    idx = ctypes.c_int(-1)
    rc = lib.sparse_forward_c(
        _ptr(z_np), ctypes.byref(m), ctypes.c_float(1.5),
        ctypes.c_float(float(alpha)), ctypes.c_float(c_d_top),
        ctypes.c_int(int(training)), _ptr(out), ctypes.byref(idx))
    if rc != 0:
        raise RuntimeError(f"sparse_forward_c returned {rc}")

    return _parity_result(
        "sparse", case, alpha, ("out",),
        [("idx", _jax_to_np(jax_idx), np.array([idx.value]))],
        {"out": _jax_to_np(jax_out)[0],
         "idx": np.asarray(jax_idx, dtype=np.int32).reshape(1)},
        {"out": out, "idx": np.array([idx.value], dtype=np.int32)})


def _lowrank_parity(case, lib, alpha):
    p = case["params"]["lowrank"]
    vs = case["values"]["lowrank"]
    ranks = tuple(case["cfg"]["ranks"])
    d = case["d"]
    z = case["z"]

    V = _np32(_jax_to_np(p["A_V"] @ p["W_V"]))
    r_max = V.shape[1]
    U = np.zeros((len(ranks), case["cfg"]["M_lr"], r_max),
                 dtype=np.float32)
    for layer, u in enumerate(p["U"]):
        u_np = _jax_to_np(u)
        U[layer, :, :u_np.shape[1]] = u_np
    ranks_np = np.ascontiguousarray(np.asarray(ranks, dtype=np.int32))
    v = _np32(vs)

    jax_out, aux = lowrank_forward(
        p, z, ranks, value_scalars=vs, alpha_val=float(alpha),
        return_aux=True)
    jax_idx = np.stack([_jax_to_np(i)[0] for i in aux["indices"]])

    m = LowrankMem()
    m.U = _ptr(U)
    m.V = _ptr(V)
    m.ranks = _iptr(ranks_np)
    m.n_layers = len(ranks)
    m.M_lr = U.shape[1]
    m.r_max = r_max
    m.value = _ptr(v)

    z_np = _np32(_jax_to_np(z))
    out = np.zeros(d, dtype=np.float32)
    idx = np.zeros(len(ranks), dtype=np.int32)
    lib.lowrank_forward_c(_ptr(z_np), ctypes.byref(m),
                          ctypes.c_float(float(alpha)), _ptr(out), _iptr(idx))

    return _parity_result(
        "lowrank", case, alpha, ("out",),
        [("idx", jax_idx, idx)],
        {"out": _jax_to_np(jax_out)[0]}, {"out": out})


def _manifold_parity(case, lib, alpha):
    p = case["params"]["manifold"]
    vs = case["values"]["manifold"]
    d = case["d"]

    # R6 (Ruling 29): both sides now apply exp_map(safe_unit(z)) themselves.
    # z keeps its raw norm (~3.8 here), so a regression to exp_map(z) or to a
    # harness-side normalisation is visible in the comparison.
    z = case["z"]
    z_np = _np32(_jax_to_np(z))
    C = _np32(_jax_to_np(p["C"]))
    T = _np32(_jax_to_np(p["T"]))
    v = _np32(vs)

    jax_out, jax_idx = manifold_forward(
        p, z, value_scalars=vs, alpha_val=float(alpha))

    m = ManifoldMem()
    m.C = _ptr(C)
    m.M = C.shape[0]
    m.T = _ptr(T)
    m.t_dim = T.shape[2]
    m.value = _ptr(v)

    out = np.zeros(d, dtype=np.float32)
    idx = ctypes.c_int(-1)
    lib.manifold_forward_c(_ptr(z_np), ctypes.byref(m),
                           ctypes.c_float(float(alpha)), _ptr(out),
                           ctypes.byref(idx))

    return _parity_result(
        "manifold", case, alpha, ("out",),
        [("idx", _jax_to_np(jax_idx), np.array([idx.value]))],
        {"out": _jax_to_np(jax_out)[0]}, {"out": out})


def _binding_parity(case, lib, alpha):
    p = case["params"]["binding"]
    vs = case["values"]["binding"]
    d = case["d"]
    z = case["z"]

    V = _np32(_jax_to_np(case["params"]["lowrank"]["A_V"]
                         @ case["params"]["lowrank"]["W_V"]))
    key = _np32(np.stack([_jax_to_np(cb["A"] @ cb["W"])
                          for cb in p["key_cb"]]))
    val = _np32(np.stack([_jax_to_np(cb["A"] @ cb["W"])
                          for cb in p["val_cb"]]))
    bind = _np32(np.stack([_jax_to_np(cb["A"] @ cb["W"])
                           for cb in p["bind_cb"]]))
    A_k = _np32(_jax_to_np(p["A_k"]))
    A_v = _np32(_jax_to_np(p["A_v"]))
    v_key = _np32(vs["key"])
    v_val = _np32(vs["val"])
    v_bind = _np32(vs["bind"])

    (jax_out, _residuals, _indices, jax_bq, jax_kq,
     _val_idx) = binding_forward(
        p, z, jnp.asarray(V), value_scalars=vs, alpha_val=float(alpha))

    m = BindingMem()
    m.key_cb = _ptr(key)
    m.val_cb = _ptr(val)
    m.bind_cb = _ptr(bind)
    m.M = key.shape[1]
    m.n_layers = key.shape[0]
    m.r_max = V.shape[1]
    m.A_k = _ptr(A_k)
    m.A_v = _ptr(A_v)
    m.value_key = _ptr(v_key)
    m.value_val = _ptr(v_val)
    m.value_bind = _ptr(v_bind)

    z_np = _np32(_jax_to_np(z))
    out = np.zeros(d, dtype=np.float32)
    b_q = np.zeros(d, dtype=np.float32)
    k_q = np.zeros(d, dtype=np.float32)
    lib.binding_forward_c(_ptr(z_np), ctypes.byref(m), _ptr(V),
                          ctypes.c_float(float(alpha)), _ptr(out),
                          _ptr(b_q), _ptr(k_q), None)

    return _parity_result(
        "binding", case, alpha, ("v_out", "b_q", "k_q"), [],
        {"v_out": _jax_to_np(jax_out)[0],
         "b_q": _jax_to_np(jax_bq)[0],
         "k_q": _jax_to_np(jax_kq)[0]},
        {"v_out": out, "b_q": b_q, "k_q": k_q})


def _contrast_parity(case, lib, alpha):
    p = case["params"]["contrast"]
    vs = case["values"]["contrast"]
    d = case["d"]
    z = case["z"]

    C_a = _np32(np.stack([_jax_to_np(cb["A"] @ cb["W"])
                          for cb in p["C_a"]]))
    C_b = _np32(np.stack([_jax_to_np(cb["A"] @ cb["W"])
                          for cb in p["C_b"]]))
    v_a = _np32(vs["a"])
    v_b = _np32(vs["b"])

    jax_out, jax_idx, _ = contrast_forward(
        p, z, value_scalars=vs, alpha_val=float(alpha))
    jax_idx_a = np.stack([_jax_to_np(i)[0] for i in jax_idx["a"]])
    jax_idx_b = np.stack([_jax_to_np(i)[0] for i in jax_idx["b"]])

    m = ContrastMem()
    m.C_a = _ptr(C_a)
    m.C_b = _ptr(C_b)
    m.M = C_a.shape[1]
    m.n_layers = C_a.shape[0]
    m.value_a = _ptr(v_a)
    m.value_b = _ptr(v_b)

    z_np = _np32(_jax_to_np(z))
    out = np.zeros(d, dtype=np.float32)
    idx_a = np.zeros(C_a.shape[0], dtype=np.int32)
    idx_b = np.zeros(C_b.shape[0], dtype=np.int32)
    lib.contrast_forward_v31(_ptr(z_np), ctypes.byref(m),
                             ctypes.c_float(float(alpha)), _ptr(out),
                             _iptr(idx_a), _iptr(idx_b))

    return _parity_result(
        "contrast", case, alpha, ("out",),
        [("idx_a", jax_idx_a, idx_a), ("idx_b", jax_idx_b, idx_b)],
        {"out": _jax_to_np(jax_out)[0]}, {"out": out})


def _routing_parity(case, lib, alpha):
    p = case["params"]["route"]
    d = case["d"]
    z = case["z"]
    n = case["cfg"]["n_lattices"]

    jax_mask, _z_route, _idx = routing_gate(p, z, tau=0.5, hard=True)
    jax_mask = _jax_to_np(jax_mask)[0]

    C_route = _np32(_jax_to_np(p["C_route"]))
    W_route = _np32(_jax_to_np(p["W_route"]))
    z_np = _np32(_jax_to_np(z))
    mask = np.zeros(n, dtype=np.float32)
    lib.routing_gate_c(_ptr(z_np), _ptr(C_route), _ptr(W_route),
                       ctypes.c_int(n), _ptr(mask))

    return _parity_result(
        "routing", case, alpha, ("mask",),
        [("mask", (jax_mask > 0.5).astype(np.int8),
          (mask > 0.5).astype(np.int8))],
        {"mask": jax_mask}, {"mask": mask})


def _dir_vec(theta, radius, d=PARITY_D):
    """Unit-plane prototype ``radius * (cos θ, sin θ)`` with zeros elsewhere."""
    v = np.zeros(d, dtype=np.float32)
    v[0] = radius * np.cos(theta)
    v[1] = radius * np.sin(theta)
    return v


def _simvq_identity(codebook):
    """Wrap an effective codebook as SimVQ params with ``W = I``."""
    cb = _np32(codebook)
    d = cb.shape[1]
    return {"A": jnp.asarray(cb), "W": jnp.eye(d, dtype=jnp.float32)}


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64)
                               - np.asarray(b, np.float64))))


def _hrq_pair(lib, top, z, fine=None, alpha=0.0, tau=0.1,
              value_topk=None, value_top=None, value_fine=None):
    """JAX/C HRQ call on explicit effective codebooks (no random params)."""
    top = _np32(top)
    z = _np32(z)
    d = top.shape[1]
    if fine is None:
        fine = np.zeros((0, 0), dtype=np.float32)
    else:
        fine = _np32(fine)

    jax_fine = ([_simvq_identity(layer) for layer in fine]
                if fine.shape[0] else [])
    jax_params = {"top": _simvq_identity(top), "fine": jax_fine}

    if value_top is not None:
        v_top = _np32(value_top)
        v_fine = None if value_fine is None else _np32(value_fine)
        jax_vs = {"top": jnp.asarray(v_top)}
        if v_fine is not None:
            jax_vs["fine"] = jnp.asarray(v_fine)
    else:
        v_top = v_fine = None
        jax_vs = None

    k = int(top.shape[0] if value_topk is None else value_topk)
    jax_out, jax_idx, jax_sim, _ = hrq_forward(
        jax_params, jnp.asarray(z)[None, :], tau_fallback=tau,
        value_scalars=jax_vs, alpha_val=float(alpha), value_topk=k)

    m = HRQMem()
    m.C_top = _ptr(top)
    m.M_top = top.shape[0]
    if fine.shape[0]:
        m.C_fine = _ptr(fine)
        m.M_fine = fine.shape[1]
        m.n_fine = fine.shape[0]
    m.value_top = _ptr(v_top) if v_top is not None else None
    m.value_fine = _ptr(v_fine) if v_fine is not None else None

    out = np.zeros(d, dtype=np.float32)
    idx = ctypes.c_int(-1)
    sim = ctypes.c_float(float("nan"))
    rc = lib.hrq_forward_v31(
        _ptr(z), ctypes.byref(m), ctypes.c_float(float(alpha)),
        ctypes.c_int(k), ctypes.c_float(float(tau)), _ptr(out),
        ctypes.byref(idx), ctypes.byref(sim))
    if rc != 0:
        raise RuntimeError(f"hrq_forward_v31 returned {rc}")

    return _parity_result(
        "hrq_crafted", {"seed": 0, "d": d}, alpha, ("out", "top_sim"),
        [("top_idx", _jax_to_np(jax_idx), np.array([idx.value]))],
        {"out": _jax_to_np(jax_out)[0],
         "top_sim": _jax_to_np(jax_sim).reshape(1),
         "idx": np.asarray(jax_idx, dtype=np.int32).reshape(1)},
        {"out": out, "top_sim": np.array([sim.value], dtype=np.float32),
         "idx": np.array([idx.value], dtype=np.int32)})


def run_hrq_poincare_parity(d=PARITY_D):
    """Crafted fixture where Euclidean and Poincaré top-1 disagree."""
    lib = load_parity_lib(d=d)
    top = np.stack([_dir_vec(0.0, 0.4, d), _dir_vec(0.5, 0.95, d)])
    z = _dir_vec(0.0, 0.9, d)

    e_idx = int(np.argmin(np.sum((z[None, :] - top) ** 2, axis=-1)))
    z_P = exp_map(safe_unit(jnp.asarray(z)[None, :]))
    c_P = exp_map(jnp.asarray(top))
    p_sims = np.asarray(poincare_similarity(z_P, c_P)).reshape(-1)
    p_idx = int(np.argmin(p_sims))

    res = _hrq_pair(lib, top, z, tau=0.1)
    res.update({
        "euclidean_idx": e_idx,
        "poincare_idx": p_idx,
        "jax_idx": int(res["jax"]["idx"][0]),
        "c_idx": int(res["c"]["idx"][0]),
        "poincare_top_sim": float(p_sims[p_idx]),
    })
    return res


def _hrq_gap_case(lib, top, fine, z, name):
    z_P = exp_map(safe_unit(jnp.asarray(_np32(z))[None, :]))
    sims = np.asarray(poincare_similarity(
        z_P, exp_map(jnp.asarray(_np32(top))))).reshape(-1)
    s0, s1 = float(sims[0]), float(sims[1])
    abs_gap = s1 - s0
    rel_gap = abs_gap / s0

    default = _hrq_pair(lib, top, z, fine=fine, tau=0.1)
    hard = _hrq_pair(lib, top, z, fine=fine, tau=0.0)
    soft = _hrq_pair(lib, top, z, fine=fine, tau=1e9)

    return {
        "name": name,
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
        "default": default,
        "hard": hard,
        "soft": soft,
        "jax_default_hard": _maxdiff(default["jax"]["out"],
                                     hard["jax"]["out"]),
        "jax_default_soft": _maxdiff(default["jax"]["out"],
                                     soft["jax"]["out"]),
        "c_default_hard": _maxdiff(default["c"]["out"], hard["c"]["out"]),
        "c_default_soft": _maxdiff(default["c"]["out"], soft["c"]["out"]),
        "jax_hard_soft": _maxdiff(hard["jax"]["out"], soft["jax"]["out"]),
        "c_hard_soft": _maxdiff(hard["c"]["out"], soft["c"]["out"]),
    }


def run_hrq_gap_parity(d=PARITY_D):
    """Task 8's absolute-vs-relative gap cases, run through both sides.

    Case A: absolute gap > tau > relative gap -> fallback.
    Case B: relative gap > tau > absolute gap -> hard.
    """
    lib = load_parity_lib(d=d)
    fine = np.stack([np.stack([_dir_vec(-0.75, 1.0, d),
                               _dir_vec(0.0, 1.0, d)])])
    z = _dir_vec(0.0, 1.0, d)

    top_a = np.stack([_dir_vec(0.78, 1.0, d),
                      _dir_vec(-0.81, 1.0, d),
                      _dir_vec(np.pi, 1.0, d),
                      _dir_vec(np.pi, 1.0, d)])
    top_b = np.stack([_dir_vec(0.195, 1.0, d),
                      _dir_vec(-0.2138, 1.0, d),
                      _dir_vec(np.pi, 1.0, d),
                      _dir_vec(np.pi, 1.0, d)])

    return {
        "case_a": _hrq_gap_case(lib, top_a, fine, z, "A"),
        "case_b": _hrq_gap_case(lib, top_b, fine, z, "B"),
    }


def _binding_pair(lib, key, val, bind, A_k, A_v, V, z):
    """JAX/C binding call on explicit effective codebooks."""
    key = _np32(key)
    val = _np32(val)
    bind = _np32(bind)
    A_k = _np32(A_k)
    A_v = _np32(A_v)
    V = _np32(V)
    z = _np32(z)
    d = z.shape[0]

    params = {
        "key_cb": [_simvq_identity(layer) for layer in key],
        "val_cb": [_simvq_identity(layer) for layer in val],
        "bind_cb": [_simvq_identity(layer) for layer in bind],
        "A_k": jnp.asarray(A_k),
        "A_v": jnp.asarray(A_v),
    }
    (jax_out, _residuals, _indices, jax_bq, jax_kq,
     _val_idx) = binding_forward(params, jnp.asarray(z)[None, :],
                                 jnp.asarray(V))

    m = BindingMem()
    m.key_cb = _ptr(key)
    m.val_cb = _ptr(val)
    m.bind_cb = _ptr(bind)
    m.M = key.shape[1]
    m.n_layers = key.shape[0]
    m.r_max = V.shape[1]
    m.A_k = _ptr(A_k)
    m.A_v = _ptr(A_v)

    out = np.zeros(d, dtype=np.float32)
    b_q = np.zeros(d, dtype=np.float32)
    k_q = np.zeros(d, dtype=np.float32)
    lib.binding_forward_c(_ptr(z), ctypes.byref(m), _ptr(V),
                          ctypes.c_float(0.0), _ptr(out), _ptr(b_q),
                          _ptr(k_q), None)

    return _parity_result(
        "binding_crafted", {"seed": 0, "d": d}, 0.0,
        ("v_out", "b_q", "k_q"), [],
        {"v_out": _jax_to_np(jax_out)[0],
         "b_q": _jax_to_np(jax_bq)[0],
         "k_q": _jax_to_np(jax_kq)[0]},
        {"v_out": out, "b_q": b_q, "k_q": k_q})


def run_binding_projection_parity(d=PARITY_D):
    """Task 9's crafted projection fixture: ``V @ A_k @ z`` vs ``A_k^T V^T z``."""
    lib = load_parity_lib(d=d)
    L, M, R = 1, 2, 2
    V = np.zeros((d, R), dtype=np.float32)
    V[0, 0] = 1.0
    V[0, 1] = 0.25
    V[1, 0] = 0.5
    V[2, 0] = 0.75
    V[2, 1] = 1.0

    A_k = np.zeros((R, d), dtype=np.float32)
    A_k[0, 0] = 1.0
    A_k[0, 1] = 0.5
    A_k[1, 2] = 1.0
    A_v = A_k.copy()

    z = np.zeros(d, dtype=np.float32)
    z[0], z[1], z[2] = 1.0, 0.3, 0.7

    u = A_k.astype(np.float64) @ z.astype(np.float64)
    correct = (V.astype(np.float64) @ u).astype(np.float32)
    t = V.astype(np.float64).T @ z.astype(np.float64)
    transposed = (A_k.astype(np.float64).T @ t).astype(np.float32)

    key = np.zeros((L, M, d), dtype=np.float32)
    key[0, 0] = correct
    key[0, 1] = transposed
    val = np.zeros((L, M, d), dtype=np.float32)
    val[0, 0, 3] = 1.0
    val[0, 1, 4] = 1.0
    bind = np.zeros((L, M, d), dtype=np.float32)
    bind[0, 0, 5] = 1.0
    bind[0, 1, 6] = 1.0

    res = _binding_pair(lib, key, val, bind, A_k, A_v, V, z)
    res.update({
        "correct": correct,
        "transposed": transposed,
        "jax_correct_err": _maxdiff(res["jax"]["k_q"], correct),
        "jax_transposed_err": _maxdiff(res["jax"]["k_q"], transposed),
        "c_correct_err": _maxdiff(res["c"]["k_q"], correct),
        "c_transposed_err": _maxdiff(res["c"]["k_q"], transposed),
    })
    return res


def run_sparse_value_bias_parity(d=PARITY_D):
    """Crafted sparse fixture where the value term provably flips the row.

    z = 0.6·e0, C = [0.7·e0, 0.1·e0], v = [−1, +1]. Without the bias the
    nearest row is 0; at alpha=2 row 1 wins on ``+alpha·tanh(v)·mean_d²``.
    """
    lib = load_parity_lib(d=d)
    C = np.zeros((2, d), dtype=np.float32)
    C[0, 0] = 0.7
    C[1, 0] = 0.1
    z = np.zeros(d, dtype=np.float32)
    z[0] = 0.6
    vs = np.array([-1.0, 1.0], dtype=np.float32)

    params = {"C": jnp.asarray(C), "zero_vec": jnp.zeros((1, d))}
    zero = np.zeros(d, dtype=np.float32)
    out = {}
    for alpha in (0.0, 2.0):
        jax_out, jax_idx = sparse_forward(
            params, jnp.asarray(z)[None, :], training=True,
            value_scalars=jnp.asarray(vs), alpha_val=float(alpha))

        m = SparseMem()
        m.C = _ptr(C)
        m.M = C.shape[0]
        m.zero_vec = _ptr(zero)
        m.value = _ptr(vs)
        c_out = np.zeros(d, dtype=np.float32)
        c_idx = ctypes.c_int(-1)
        rc = lib.sparse_forward_c(
            _ptr(z), ctypes.byref(m), ctypes.c_float(1.5),
            ctypes.c_float(float(alpha)), ctypes.c_float(0.0),
            ctypes.c_int(1), _ptr(c_out), ctypes.byref(c_idx))
        if rc != 0:
            raise RuntimeError(f"sparse_forward_c returned {rc}")

        out[alpha] = _parity_result(
            "sparse_crafted", {"seed": 0, "d": d}, alpha, ("out",),
            [("idx", _jax_to_np(jax_idx), np.array([c_idx.value]))],
            {"out": _jax_to_np(jax_out)[0],
             "idx": np.asarray(jax_idx, dtype=np.int32).reshape(1)},
            {"out": c_out,
             "idx": np.array([c_idx.value], dtype=np.int32)})
    out["jax_idx_0"] = int(out[0.0]["jax"]["idx"][0])
    out["jax_idx_2"] = int(out[2.0]["jax"]["idx"][0])
    out["c_idx_0"] = int(out[0.0]["c"]["idx"][0])
    out["c_idx_2"] = int(out[2.0]["c"]["idx"][0])
    return out


def run_lattice_parity(lattice, seed=0, d=PARITY_D, alpha=0.0,
                       training=True, d_top=None):
    """Run one lattice through JAX and the C parity build; compare.

    Returns ``{'idx_equal', 'max_abs_err', 'jax', 'c', ...}``. Discrete
    outputs are compared exactly; continuous outputs by max absolute error.
    ``alpha`` is the R22 value-bias strength (0 disables it); ``training``
    selects the Sparse branch. ``d_top`` enables Sparse's LFQ decision (None
    leaves it disabled, matching "HRQ did not activate").
    """
    if lattice not in PARITY_LATTICES:
        raise ValueError(f"unknown lattice {lattice!r}; "
                         f"expected one of {PARITY_LATTICES}")
    case = _build_case(seed=seed, d=d)
    lib = load_parity_lib(d=d)

    if lattice == "hrq":
        return _hrq_parity(case, lib, alpha)
    if lattice == "sparse":
        return _sparse_parity(case, lib, alpha, training=training,
                              d_top=d_top)
    if lattice == "lowrank":
        return _lowrank_parity(case, lib, alpha)
    if lattice == "manifold":
        return _manifold_parity(case, lib, alpha)
    if lattice == "binding":
        return _binding_parity(case, lib, alpha)
    if lattice == "contrast":
        return _contrast_parity(case, lib, alpha)
    return _routing_parity(case, lib, alpha)


# ─── Fusion parity ──────────────────────────────────────────────────────────

def run_canonical_fusion_parity(seed=0, d=PARITY_D):
    """R9 canonical fusion: JAX ``fuse_lattices_with_aux`` vs C ``lcm_fuse_only``."""
    lib = load_parity_lib(d=d)
    cfg = PARITY_CASE
    n = cfg["n_lattices"]
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), 5)

    outputs = jax.random.normal(keys[0], (n, d))
    logits = jax.random.normal(keys[1], (1, n)) * 2.0
    soft_mask = jax.nn.softmax(logits, axis=-1)
    alpha = jax.random.uniform(keys[2], (n,), minval=0.5, maxval=1.5)
    ln_scale = 1.0 + 0.1 * jax.random.normal(keys[3], (d,))
    ln_bias = 0.05 * jax.random.normal(keys[4], (d,))

    jax_z, _, _ = fuse_lattices_with_aux(
        [outputs[i][None, :] for i in range(n)], soft_mask,
        {"alpha": alpha, "ln_scale": ln_scale, "ln_bias": ln_bias})

    out_flat = _np32(_jax_to_np(outputs).reshape(-1))
    mask = _np32(_jax_to_np(soft_mask)[0])
    alpha_np = _np32(_jax_to_np(alpha))
    scale_np = _np32(_jax_to_np(ln_scale))
    bias_np = _np32(_jax_to_np(ln_bias))
    z_c = np.zeros(d, dtype=np.float32)
    rc = lib.lcm_fuse_only(
        _ptr(out_flat), ctypes.c_int(n), ctypes.c_int(d), _ptr(mask),
        _ptr(alpha_np), ctypes.c_int(n), _ptr(scale_np), _ptr(bias_np),
        _ptr(z_c))
    if rc != 0:
        raise RuntimeError(f"lcm_fuse_only returned {rc}")

    jax_z = _jax_to_np(jax_z)[0]
    diff = np.abs(jax_z.astype(np.float64) - z_c.astype(np.float64))
    err = float(np.max(diff))
    if not np.isfinite(err):
        err = float("inf")
    k = int(np.argmax(diff))
    return {
        "lattice": "canonical_fusion", "seed": int(seed), "d": int(d),
        "alpha": 0.0, "fields": ("z",), "idx_equal": True,
        "max_abs_err": err,
        "jax": {"z": jax_z}, "c": {"z": z_c},
        "idx_mismatch": None,
        "max_mismatch": {"field": "z", "index": (k,),
                         "jax": float(jax_z[k]), "c": float(z_c[k]),
                         "abs_err": err},
    }


def run_engine_fusion_parity(seed=0, d=PARITY_D):
    """R9 engine fusion: JAX ``cog_loop.dag_fuse`` vs C engine fusion.

    The C side performs its own nearest-neighbour retrieval (``retrieve_single``)
    and reciprocal-distance fusion (``distance_weighted_fusion``), so the
    comparison covers the whole engine formula, not just the weighted sum.
    """
    lib = load_parity_lib(d=d)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), 8)
    z = jax.random.normal(keys[0], (d,)) * 0.7

    n = PARITY_CASE["n_lattices"]
    sizes = (9, 7, 8, 6, 7, 9)
    codebooks = [jax.random.normal(keys[1 + i], (sizes[i], d)) * 0.3
                 for i in range(n)]
    jax_next, _diff, _entropy = dag_fuse(
        z, codebooks, [1e9] * n, tau=0.1, eps=1e-6)

    z_np = _np32(_jax_to_np(z))
    n_buf = max(C_MAX_LATTICES, n)
    outputs = np.zeros((n_buf, d), dtype=np.float32)
    confidences = np.zeros(n_buf, dtype=np.float32)
    for i, cb in enumerate(codebooks):
        cb_np = _np32(_jax_to_np(cb))
        mem = LatticeMemory()
        mem.C = _ptr(cb_np)
        mem.M = cb_np.shape[0]
        mem.D = d
        out = np.zeros(d, dtype=np.float32)
        dist = ctypes.c_float(float("nan"))
        idx = ctypes.c_int(-1)
        lib.retrieve_single(_ptr(z_np), ctypes.byref(mem), _ptr(out),
                            ctypes.byref(dist), ctypes.byref(idx))
        outputs[i] = out
        confidences[i] = 1.0 / (float(np.sqrt(dist.value)) + 1e-6)

    z_c = np.zeros(d, dtype=np.float32)
    weights = np.zeros(C_MAX_LATTICES, dtype=np.float32)
    lib.distance_weighted_fusion(_ptr(outputs), _ptr(confidences), _ptr(z_c),
                                 _ptr(weights), None, ctypes.c_float(0.0))

    jax_next = _jax_to_np(jax_next)
    diff = np.abs(jax_next.astype(np.float64) - z_c.astype(np.float64))
    err = float(np.max(diff))
    if not np.isfinite(err):
        err = float("inf")
    k = int(np.argmax(diff))
    return {
        "lattice": "engine_fusion", "seed": int(seed), "d": int(d),
        "alpha": 0.0, "fields": ("z",), "idx_equal": True,
        "max_abs_err": err,
        "jax": {"z": jax_next}, "c": {"z": z_c},
        "idx_mismatch": None,
        "max_mismatch": {"field": "z", "index": (k,),
                         "jax": float(jax_next[k]), "c": float(z_c[k]),
                         "abs_err": err},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/lcm_parity_jax.npz")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    args = ap.parse_args()

    d = np.load(args.dump)
    z = _f32(d["in.z"][0])
    z_jax = np.asarray(d["out.z_next"], dtype=np.float64)[0]

    cbs = primary_codebooks(d)
    t_keys = sorted(k for k in d.files if k.startswith("params.manifold.T"))
    man_t_dim = int(d[t_keys[0]].shape[-1]) if t_keys else 4

    lib = load_lib(args.lib)

    print("=== inputs fed to C ===")
    for k in LATTICES:
        print(f"  {k:9s} {cbs[k].shape}")
    print(f"  manifold_T {cbs['manifold_T'].shape} (t_dim={man_t_dim})")
    print("  of the 36 parameter leaves the JAX step consumes, this supplies 7;")
    print("  route / fusion / value_scalars have no place in lcm_infer_step_v1.")
    print()

    gv = np.zeros((4, D_MODEL), dtype=np.float32)
    z_v1 = _feed(lib, lib.lcm_infer_step, z, cbs, man_t_dim,
                 [_ptr(gv), 4, _ptr(gv), N_LATTICES])

    soft_mask = _f32(d["out.soft_mask"][0])
    alpha = _f32(d["params.fusion.alpha"])
    ln_scale = _f32(d["params.fusion.ln_scale"])
    ln_bias = _f32(d["params.fusion.ln_bias"])
    # HRQ stack: top codebook plus the concatenated fine codebooks per layer.
    hrq_top = _f32(cb2(d, "hrq.top.A") @ cb2(d, "hrq.top.W"))
    fine_keys = sorted(k for k in d.files
                       if k.startswith("params.hrq.fine.") and k.endswith(".A"))
    # fine_keys are already full archive keys; cb2() would double the prefix.
    fine_layers = [_f32(np.asarray(d[k], dtype=np.float32)
                        @ np.asarray(d[k[:-2] + ".W"], dtype=np.float32))
                   for k in fine_keys]
    hrq_n_fine = len(fine_layers)
    hrq_fine = _f32(np.concatenate(fine_layers)) if fine_layers else _f32(np.zeros((0, D_MODEL)))
    hrq_fine_M = fine_layers[0].shape[0] if fine_layers else 0

    z_v2 = _feed(lib, lib.lcm_infer_step_v2, z, cbs, man_t_dim,
                 [_ptr(hrq_top), hrq_top.shape[0],
                  _ptr(hrq_fine), hrq_fine_M, hrq_n_fine, 0.1,
                  _ptr(soft_mask), N_LATTICES, _ptr(alpha), alpha.size,
                  _ptr(ln_scale), _ptr(ln_bias)])

    # Decisive separation: feed C the JAX lattice outputs and run ONLY the
    # fusion. Whatever gap survives here is fusion error; the rest is retrieval.
    lat = np.concatenate([_f32(d[f"out.lattice.{k}"][0]) for k in LATTICES])
    z_fuse = np.zeros(D_MODEL, dtype=np.float32)
    rc = lib.lcm_fuse_only(_ptr(_f32(lat)), N_LATTICES, D_MODEL,
                           _ptr(soft_mask), _ptr(alpha), int(alpha.size),
                           _ptr(ln_scale), _ptr(ln_bias), _ptr(z_fuse))
    if rc != 0:
        raise SystemExit(f"lcm_fuse_only returned {rc}")
    z_fuse = z_fuse.astype(np.float64)

    print("=== result ===")
    print(f"  soft_mask  = {soft_mask}")
    print(f"  alpha      = {alpha}")
    print()
    _report("v1 (old)", z_jax, z_v1)
    d2 = _report("v2 (canon)", z_jax, z_v2)
    d1 = np.abs(z_v1 - z_jax).max()
    print()
    print(f"  fusion alignment moved the gap {d1:.6f} -> {d2:.6f}"
          f"  ({'improved' if d2 < d1 else 'NOT improved'})")
    _report("fuse-only", z_jax, z_fuse)
    print()
    print("  fuse-only feeds C the JAX lattice outputs and runs ONLY the fusion.")
    print("  A ~0 gap there means the fusion is correct and the entire v2")
    print("  residual above is retrieval (the six lattice forwards).")
    print()
    print("v2 now uses the faithful HRQ port for lattice 0 and flat retrieval for")
    print("the rest. HRQ was 99.7% of the original gap: 2.9406 -> 0.2622 (fusion)")
    print("-> 0.0007 (HRQ). What is left is the other five lattices; re-run")
    print("train/parity_ablate.py to see which of them matters next -- not the")
    print("mask weights, which point the wrong way.")


if __name__ == "__main__":
    main()
