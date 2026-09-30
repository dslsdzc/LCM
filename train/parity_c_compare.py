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
"""
import argparse
import ctypes
import os

import numpy as np

D_MODEL = 32
N_LATTICES = 6
LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast")

_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "infer", "liblcm.so")


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_lib(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. Build a matching one:\n"
            f"    cd infer && make LCM_D={D_MODEL}\n"
            f"Note the C engine is dimensioned at compile time, so a .so built "
            f"for the deployed d_model (256) cannot run this comparison.")
    lib = ctypes.CDLL(path)
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
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # soft_mask, n
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # alpha, n_alpha
        ctypes.POINTER(ctypes.c_float),                        # ln_scale
        ctypes.POINTER(ctypes.c_float),                        # ln_bias
        ctypes.POINTER(ctypes.c_float),                        # z_out
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/lcm_parity_jax.npz")
    ap.add_argument("--lib", default=_DEFAULT_LIB)
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
    z_v2 = _feed(lib, lib.lcm_infer_step_v2, z, cbs, man_t_dim,
                 [_ptr(soft_mask), N_LATTICES, _ptr(alpha), alpha.size,
                  _ptr(ln_scale), _ptr(ln_bias)])

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
    print()
    print("Remaining gap after v2 is attributable to the six lattice forwards,")
    print("which C still computes as a flat nearest-neighbour retrieval while JAX")
    print("runs six distinct learned operators.")


if __name__ == "__main__":
    main()
