"""Ablate the JAX<->C gap per lattice, to decide what to port first.

The mask weights are NOT the answer. `soft_mask` reads 98.4% on `sparse`, which
looks like "port sparse first" — and that is wrong, because the fusion ends in a
LayerNorm, which normalises by the fused vector's own scale. The pre-LayerNorm
fused vector is small (~0.05), so an absolute error of 1.43 on a lattice carrying
only 0.43% of the weight dominates after normalisation. Measured: HRQ alone is
99.7% of the gap.

Method: model C's per-lattice retrieval as plain nearest-neighbour over the
primary codebook, then swap one lattice at a time to the JAX output and re-fuse.

The model is validated, not assumed: fused, it reproduces `lcm_infer_step_v2`'s
output to 4 decimals (0.262144 vs the C-reported 0.262156, float64 vs float32).
`infer/lattice.c::retrieve_single` is plain Euclidean argmin, which is exactly
`sparse_forward`'s training-mode value path when `value_scalars == 0` (their
init is `jnp.zeros`), so the model holds for every lattice that C retrieves that
way.

Usage:
    JAX_PLATFORMS=cpu be/bin/python -m train.parity_ablate --dump /tmp/lcm_parity_jax.npz
"""
import argparse

import numpy as np

LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast")


def _cb(d, name):
    return np.asarray(d[f"params.{name}"], dtype=np.float64)


def primary_codebooks(d):
    """What C's single flat matrix per lattice actually holds."""
    V = _cb(d, "lowrank.A_V") @ _cb(d, "lowrank.W_V")
    r0 = _cb(d, "lowrank.U.0").shape[-1]
    return {
        "hrq": _cb(d, "hrq.top.A") @ _cb(d, "hrq.top.W"),
        "sparse": _cb(d, "sparse.C"),
        "lowrank": _cb(d, "lowrank.U.0") @ V[:, :r0].T,
        "manifold": _cb(d, "manifold.C"),
        "binding": _cb(d, "binding.key_cb.0.A") @ _cb(d, "binding.key_cb.0.W"),
        "contrast": _cb(d, "contrast.C_a.0.A") @ _cb(d, "contrast.C_a.0.W"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/lcm_parity_jax.npz")
    args = ap.parse_args()

    d = np.load(args.dump)
    z = d["in.z"][0].astype(np.float64)
    mask = d["out.soft_mask"][0].astype(np.float64)
    alpha = d["params.fusion.alpha"].astype(np.float64)
    ln_s = d["params.fusion.ln_scale"].astype(np.float64)
    ln_b = d["params.fusion.ln_bias"].astype(np.float64)
    z_jax = d["out.z_next"][0].astype(np.float64)

    def fuse(outs):
        w = mask * alpha
        w = w / w.sum()
        y = sum(w[i] * outs[i] for i in range(len(outs)))
        return (y - y.mean()) / np.sqrt(y.var() + 1e-6) * ln_s + ln_b

    jax_outs = [d[f"out.lattice.{n}"][0].astype(np.float64) for n in LATTICES]
    cbs = primary_codebooks(d)
    nn_outs = []
    for n in LATTICES:
        C = cbs[n]
        dist2 = ((z[None, :] - C) ** 2).sum(-1)
        nn_outs.append(C[int(np.argmin(dist2))])

    print(f"fuse(JAX outputs)  max|.-z_next| = {np.abs(fuse(jax_outs)-z_jax).max():.8f}")
    base = np.abs(fuse(nn_outs) - z_jax).max()
    print(f"fuse(all-NN)       max|.-z_next| = {base:.8f}   "
          f"(lcm_infer_step_v2 reports ~0.262156)")
    print()
    print(f"{'swap to JAX':28s} {'gap':>12s} {'removed':>12s} {'share':>8s}")
    rows = []
    for i, n in enumerate(LATTICES):
        o = list(nn_outs)
        o[i] = jax_outs[i]
        g = np.abs(fuse(o) - z_jax).max()
        rows.append((base - g, n, g))
        print(f"  {n:26s} {g:12.6f} {base-g:12.6f} {(base-g)/base:8.2%}")
    rows.sort(reverse=True)
    print()
    print(f"=> port FIRST: {rows[0][1]}  "
          f"(removes {rows[0][0]/base:.1%} of the gap)")
    print("   Port order after that is whatever the next ablation says, not the"
          " mask weights.")


if __name__ == "__main__":
    main()
