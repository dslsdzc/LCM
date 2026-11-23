#!/usr/bin/env python3
"""End-to-end C DAG <-> JAX reconstruction check.

The C engine runs a dynamic DAG: a lattice may be inactive even when it is in
the routed top-1, in which case ``lcm_fuse_canonical`` falls back to the active
set (or identity when the active set is empty). A random single-step parity
does not cover those branches. This script:

1. runs the C v3 loop for several inputs and reads the v3 trace, including
   per-lattice ``active`` flags;
2. computes every lattice's JAX output on the same input;
3. reconstructs C's fusion using the trace's weights/active flags/self term,
   and compares against the C ``z_next``.

It is read-only and uses the real checkpoint. Exit code is nonzero if any
step exceeds the tolerance.

Usage:
    JAX_PLATFORMS=cpu python scripts/validate_c_dag_parity.py \
        --checkpoint /home/DslsDZC/LCM/checkpoints_calibrated_fixed5 \
        --inputs 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lcm import LCMInferEngine  # noqa: E402
from train.checkpoint import load_checkpoint  # noqa: E402
from train.cognitive_step import six_lattice_step  # noqa: E402
from train.config import LCMConfig  # noqa: E402
from train.fusion import layer_norm  # noqa: E402
from train.self_lattice import self_lattice_forward  # noqa: E402

TOL = 1e-3


def _load_cfg(checkpoint: Path) -> LCMConfig:
    import json

    with open(checkpoint / "config.json") as f:
        raw = json.load(f)
    fields = LCMConfig.__dataclass_fields__
    return LCMConfig(**{k: v for k, v in raw.items() if k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    eng = LCMInferEngine(str(ckpt), lt_enabled=False)
    cfg = _load_cfg(ckpt)
    params, *_ = load_checkpoint(
        str(ckpt), cfg=cfg, load_opt=False, allow_missing_aux=True)

    def jax_outputs(z):
        zz = jnp.asarray(z, jnp.float32)[None]
        z_self, self_bias, _ = self_lattice_forward(
            params["self"], zz, training=False, return_aux=True,
            alpha_self=cfg.alpha_self)
        _, aux = six_lattice_step(
            zz, params, cfg, training=False, rng=jax.random.PRNGKey(0),
            self_output=z_self, self_bias=self_bias)
        return ([np.asarray(o[0]) for o in aux["lattice_outputs"]],
                np.asarray(z_self[0]),
                float(np.asarray(self_bias).reshape(-1)[0]))

    def fuse(z_in, outs, z_self, self_bias, weights, active):
        w = weights[:6].astype(np.float64)
        wsum = w.sum()
        if wsum > 0:
            acc = sum((w[k] / wsum) * outs[k] for k in range(6))
            if active[6]:
                acc = acc + self_bias * z_self
            return np.asarray(layer_norm(
                jnp.asarray(np.asarray(acc, np.float32))[None],
                params["fusion"]["ln_scale"],
                params["fusion"]["ln_bias"])[0])
        routed = [k for k in range(6) if active[k]]
        if routed:
            # C's active-set uniform fallback when the routed soft mask is
            # zero but some routed lattice is still active.
            acc = sum(outs[k] for k in routed) / len(routed)
            if active[6]:
                acc = acc + self_bias * z_self
            return np.asarray(layer_norm(
                jnp.asarray(np.asarray(acc, np.float32))[None],
                params["fusion"]["ln_scale"],
                params["fusion"]["ln_bias"])[0])
        if active[6]:
            # Ruling 24: self-only activation still fuses and LayerNorms.
            acc = self_bias * z_self
            return np.asarray(layer_norm(
                jnp.asarray(np.asarray(acc, np.float32))[None],
                params["fusion"]["ln_scale"],
                params["fusion"]["ln_bias"])[0])
        # No lattice active: C copies z_cur without LayerNorm.
        return np.asarray(z_in, np.float32)

    rng = np.random.default_rng(args.seed)
    masks = {}
    max_err = 0.0
    failures = 0
    for i in range(args.inputs):
        z0 = rng.standard_normal(eng.d).astype(np.float32)
        z0 /= np.linalg.norm(z0)
        eng.cognitive_loop(z0, max_steps=4, use_safety=False)
        trace = eng.get_trace_v3()
        z_in = z0
        for t, step in enumerate(trace or []):
            outs, z_self, self_bias = jax_outputs(z_in)
            rebuilt = fuse(z_in, outs, z_self, self_bias,
                           step["weights"], step["active"])
            err = float(np.max(np.abs(step["z_next"] - rebuilt)))
            max_err = max(max_err, err)
            mask = tuple(int(x) for x in step["active"])
            masks[mask] = masks.get(mask, 0) + 1
            if err > TOL:
                failures += 1
                print(f"FAIL input={i} step={t} mask={mask} "
                      f"weights={np.round(step['weights'], 3)} "
                      f"max_abs={err:.3e}")
            z_in = step["z_next"]

    print(f"inputs={args.inputs} masks={len(masks)} "
          f"max_abs={max_err:.3e} failures={failures}")
    for mask, count in sorted(masks.items(), key=lambda kv: -kv[1]):
        print(f"  active={mask} steps={count}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
