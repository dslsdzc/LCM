"""C side + comparison for the JAX<->C single-step parity harness.

Part 2 of 2. Consumes the dump from `train/parity_jax_dump.py`, feeds the
equivalent inputs to `infer/liblcm.so::lcm_infer_step`, and reports the gap.

WHAT THIS IS EXPECTED TO SHOW ON CURRENT CODE: a large mismatch. That is the
point. `six_lattice_step` does learned routing -> six distinct lattice forwards
-> `soft_mask * alpha * exp(beta*v)` weighted fusion -> LayerNorm, while
`lcm_infer_step` does a flat nearest-neighbour retrieval per codebook followed by
inverse-distance fusion. Running it now validates the harness itself: a harness
that cannot detect a known difference cannot certify an unknown one.

The C entry point also cannot express the JAX step — `lcm_infer_step` takes one
flat codebook per lattice and has no `route`, `fusion` or `value_scalars`
parameter at all. So this comparison is necessarily approximate on the input
side too, and the script prints exactly what it fed.

Usage:
    cd infer && make LCM_D=32          # LCM_D must equal cfg.d_model
    JAX_PLATFORMS=cpu be/bin/python -m train.parity_c_compare --dump /tmp/lcm_parity_jax.npz
"""
import argparse
import ctypes
import os

import numpy as np

from train.config import LCMConfig

# Must match the CFG in parity_jax_dump.py.
D_MODEL = 32
N_LATTICES = 6
LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "infer", "liblcm.so")


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_lib():
    if not os.path.exists(LIB_PATH):
        raise SystemExit(
            f"{LIB_PATH} not found. Build it first:\n"
            f"    cd infer && make LCM_D={D_MODEL}")
    lib = ctypes.CDLL(LIB_PATH)
    lib.lcm_infer_step.restype = ctypes.c_int
    lib.lcm_infer_step.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # z, d
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # hrq_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # sparse_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # lr_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # man_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # man_T, t_dim
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # bind_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # contrast_C, M
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,          # gv_pos, n
        ctypes.POINTER(ctypes.c_float),                        # gv_neg
        ctypes.c_int,                                          # n_lattices
        ctypes.POINTER(ctypes.c_float),                        # z_out
    ]
    return lib


def primary_codebooks(d):
    """The one codebook per lattice the C API can accept.

    Each lattice in JAX is a *stack* — HRQ has top+fine per layer, binding has
    key/value/bind per layer, contrast has C_a and C_b. `lcm_infer_step` takes a
    single flat matrix per lattice, so this picks the primary one and the
    comparison is approximate on the input side by construction. Reported, not
    hidden.
    """
    def cb(name):
        return _f32(np.asarray(d[f"params.{name}"], dtype=np.float32))

    def simvq(prefix):
        A = cb(f"{prefix}.A")
        W = cb(f"{prefix}.W")
        return _f32(A @ W)

    hrq_C = simvq("hrq.top")
    lr_mat = cb("lowrank.A_V") @ cb("lowrank.W_V")
    man_C = cb("manifold.C")
    # manifold T is stored as a flat leaf under manifold.T or manifold.T.0..n
    t_keys = sorted(k for k in d.files if k.startswith("params.manifold.T"))
    man_T = _f32(np.concatenate([d[k].ravel() for k in t_keys]))
    return {
        "hrq": hrq_C,
        "sparse": cb("sparse.C"),
        "lowrank": _f32(lr_mat),
        "manifold": man_C,
        "manifold_T": man_T,
        "binding": simvq("binding.key_cb.0"),
        "contrast": simvq("contrast.C_a.0"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/lcm_parity_jax.npz")
    ap.add_argument("--lib", default=None,
                    help="path to a liblcm.so built with LCM_D=<cfg.d_model>. "
                         "Defaults to infer/liblcm.so, which is normally built "
                         "for the deployed d_model (256) and will NOT match.")
    args = ap.parse_args()

    d = np.load(args.dump)
    z = _f32(d["in.z"])
    if z.shape != (1, D_MODEL):
        raise SystemExit(f"dump z has shape {z.shape}, expected (1, {D_MODEL})")
    z_jax = np.asarray(d["out.z_next"], dtype=np.float64)[0]

    cbs = primary_codebooks(d)
    man_t_dim = int(d["params.manifold.T"].shape[-1]) if "params.manifold.T" in d.files else 4

    global LIB_PATH
    if args.lib:
        LIB_PATH = args.lib
    lib = load_lib()
    z_c = np.zeros(D_MODEL, dtype=np.float32)
    gv_pos = np.zeros((4, D_MODEL), dtype=np.float32)
    gv_neg = np.zeros((4, D_MODEL), dtype=np.float32)

    rc = lib.lcm_infer_step(
        _ptr(_f32(z[0])), D_MODEL,
        _ptr(cbs["hrq"]), cbs["hrq"].shape[0],
        _ptr(cbs["sparse"]), cbs["sparse"].shape[0],
        _ptr(cbs["lowrank"]), cbs["lowrank"].shape[0],
        _ptr(cbs["manifold"]), cbs["manifold"].shape[0],
        _ptr(cbs["manifold_T"]), man_t_dim,
        _ptr(cbs["binding"]), cbs["binding"].shape[0],
        _ptr(cbs["contrast"]), cbs["contrast"].shape[0],
        _ptr(gv_pos), 4,
        _ptr(gv_neg),
        N_LATTICES,
        _ptr(z_c),
    )
    if rc != 0:
        raise SystemExit(f"lcm_infer_step returned {rc}")

    z_c = z_c.astype(np.float64)
    diff = np.abs(z_c - z_jax)

    print("=== inputs actually fed to C (approximate by construction) ===")
    for k in ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast"):
        print(f"  {k:9s} {cbs[k].shape}")
    print(f"  manifold_T {cbs['manifold_T'].shape} (t_dim={man_t_dim})")
    print("  NOT EXPRESSIBLE in lcm_infer_step: route, fusion, value_scalars")
    print()
    print("=== result ===")
    print(f"  z_jax[:6] = {z_jax[:6]}")
    print(f"  z_c  [:6] = {z_c[:6]}")
    print(f"  max|diff| = {diff.max():.6f}")
    print(f"  rel       = {diff.max() / (np.abs(z_jax).max() + 1e-12):.4%}")
    print()
    print("A mismatch here is EXPECTED and is the harness working. Certifying a")
    print("match requires giving C the same semantics, which its API cannot yet")
    print("express — that is the work item, not a bug in this script.")


if __name__ == "__main__":
    main()
