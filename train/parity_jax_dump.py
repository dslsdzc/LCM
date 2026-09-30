"""JAX side of the JAX<->C single-step parity harness.

Part 1 of 2. This produces the reference values; the C side (part 2) must
reproduce them from the same inputs.

Why this exists: the cognitive step is currently computed two different ways.
`train/cognitive_step.py::six_lattice_step` does learned routing -> six distinct
lattice forwards -> `soft_mask * alpha * exp(beta*v)` weighted fusion -> LayerNorm,
while `infer/engine.c::distance_weighted_fusion` does inverse-distance weighting
over a flat nearest-neighbour retrieval, with no routing mask, no learned alpha
and no LayerNorm. Those are different computations, not different precisions.

Before any C change can be judged, there has to be a way to compare one step.
That is what this file and its C counterpart provide.

Usage:
    JAX_PLATFORMS=cpu be/bin/python -m train.parity_jax_dump --out /tmp/parity.npz
    JAX_PLATFORMS=cpu be/bin/python -m train.parity_jax_dump --list

Deliberately a script rather than a test: on the CURRENT code it is expected to
mismatch, and a red test in the suite would be noise. Once C is aligned, the
comparison belongs in a test.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from train.cognitive_step import six_lattice_step
from train.config import LCMConfig

LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast")

# Small and fully specified. Any value that affects the step must be pinned here
# or the C side has no way to reproduce it.
CFG = LCMConfig(
    d_model=32, d_ff=48, n_heads=4, d_head=8, vocab_size=64,
    max_seq_len=16, n_encoder_layers=1,
    M_top=16, M_fine=8, n_hrq_layers=1,
    M_sparse=16, M_lr=16, M_man=16, M_bind=16, M_contrast=16,
    n_bind_layers=1, n_contrast_layers=1,
    n_self_codes=8, use_bf16=False,
    # r_max must equal max(ranks): binding_forward projects the shared low-rank
    # base V (last dim max(ranks)) through params['A_k'], shaped (r_max, ...).
    r_max=8, ranks=(2, 4, 8),
)

SEED_PARAMS = 0
SEED_Z = 1234
SEED_STEP_RNG = 7


def build_case():
    """Fixed params and a fixed z. No encoder: the step's input is the contract."""
    from train.model import init_all_params

    params, _gvalue, _self_state = init_all_params(CFG, jax.random.PRNGKey(SEED_PARAMS))
    z = jax.random.normal(jax.random.PRNGKey(SEED_Z), (1, CFG.d_model))

    z_next, aux = six_lattice_step(
        z, params, CFG, training=True, rng=jax.random.PRNGKey(SEED_STEP_RNG))
    return params, z, z_next, aux


def flatten_params(params):
    """Name every leaf the C side would need, so the gap is visible and countable."""
    out = {}
    for name in ("route", "fusion"):
        for k, v in params[name].items():
            out[f"params.{name}.{k}"] = np.asarray(v, dtype=np.float32)
    for name in LATTICES:
        for k, v in jax.tree_util.tree_flatten_with_path(params[name])[0]:
            path = ".".join(str(p.key) if hasattr(p, "key") else str(p.idx)
                            for p in k)
            out[f"params.{name}.{path}"] = np.asarray(v, dtype=np.float32)
    if params.get("value_scalars"):
        for k, v in params["value_scalars"].items():
            out[f"params.value_scalars.{k}"] = np.asarray(v, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/lcm_parity_jax.npz",
                    help="where to write the reference values")
    ap.add_argument("--list", action="store_true",
                    help="print what the C side must be given, then exit")
    args = ap.parse_args()

    params, z, z_next, aux = build_case()
    p = flatten_params(params)

    if args.list:
        print(f"cfg.d_model   = {CFG.d_model}")
        print(f"cfg.n_lattices= {CFG.n_lattices}")
        print(f"cfg.tau_route = {CFG.tau_route}")
        print(f"cfg.tau_route_fallback = {CFG.tau_route_fallback}")
        print(f"cfg.alpha_val = {CFG.alpha_val}")
        print(f"cfg.beta_val  = {CFG.beta_val}")
        print(f"cfg.tau_val_signal = {CFG.tau_val_signal}")
        print(f"cfg.lambda_sparse  = {CFG.lambda_sparse}")
        print(f"cfg.alpha_self     = {CFG.alpha_self}")
        print(f"\n{len(p)} parameter leaves the C side must be given:")
        for k in sorted(p):
            print(f"  {k:44s} {p[k].shape}")
        return

    payload = dict(p)
    payload["in.z"] = np.asarray(z, dtype=np.float32)
    payload["out.z_next"] = np.asarray(z_next, dtype=np.float32)
    payload["out.soft_mask"] = np.asarray(aux["soft_mask"], dtype=np.float32)
    payload["out.weights"] = np.asarray(aux["weights"], dtype=np.float32)
    payload["out.entropy"] = np.asarray(aux["entropy"], dtype=np.float32)
    for i, name in enumerate(LATTICES):
        payload[f"out.lattice.{name}"] = np.asarray(
            aux["lattice_outputs"][i], dtype=np.float32)

    np.savez(args.out, **payload)
    print(f"wrote {args.out}: {len(payload)} arrays, "
          f"{len(p)} of them inputs the C side needs")
    print(f"z_next[0,:6] = {np.asarray(z_next)[0, :6]}")
    print(f"soft_mask    = {np.asarray(aux['soft_mask'])[0]}")


if __name__ == "__main__":
    main()
