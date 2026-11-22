#!/usr/bin/env python3
"""FPS-initialise Binding codebooks from real projected queries, then train.

Binding's projected query (``z_k = z @ (V @ A_k)^T``) has far lower effective
rank/diversity than the random 512-row codebook. This script chooses codebook
rows from the actual residual-query distribution with farthest-point sampling
(FPS), installs them as ``A`` with ``W=I``, and passes the tree to ``train_cog``
as the initial model.

Usage:
    JAX_PLATFORMS=cuda python scripts/train_binding_fps.py \
        --output /home/DslsDZC/LCM/checkpoints_binding_fps \
        --steps 100 --r-max 32
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train.cog_train import init_cog_params, train_cog  # noqa: E402
from train.config import LCMConfig  # noqa: E402
from train.data import WikiDataIter  # noqa: E402
from train.encoder import encoder_forward  # noqa: E402
from train.hyp import safe_unit  # noqa: E402
from train.lattices import normalize_fft  # noqa: E402


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _fps(x: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(x)))
    idx = [first]
    d = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, m):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.sum((x - x[j]) ** 2, axis=1))
    return x[idx]


def _residual_rows(q: np.ndarray, m: int, layers: int, seed: int):
    rows, r = [], q
    for layer in range(layers):
        layer_rows = _fps(r, m, seed + layer)
        rows.append(layer_rows)
        idx = np.argmin(
            np.sum((_unit(r)[:, None, :] - _unit(layer_rows)[None, :, :]) ** 2,
                   axis=-1), axis=1)
        r = r - layer_rows[idx]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="/home/DslsDZC/LCM/checkpoints_binding_fps")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--r-max", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat")
    p.add_argument("--shape", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json")
    p.add_argument("--spans", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy")
    args = p.parse_args()

    cfg = dataclasses.replace(LCMConfig(), r_max=args.r_max)
    params, _self_state = init_cog_params(
        cfg, jax.random.split(jax.random.PRNGKey(42))[1], qwen_ckpt=None)

    # Real encoder states from the training corpus.
    it = WikiDataIter(args.data, args.shape, args.spans, B=1, N=args.seq)
    zs = []
    for _ in range(args.samples):
        x, _y = next(it)
        z = safe_unit(encoder_forward(params["encoder"],
                                      jnp.asarray(x), cfg.n_heads))
        zs.append(np.asarray(z)[0])
    z = np.stack(zs).astype(np.float32)

    V = np.asarray(params["lowrank"]["A_V"] @ params["lowrank"]["W_V"])
    q_key = z @ (V @ np.asarray(params["binding"]["A_k"])).T
    q_val = z @ (V @ np.asarray(params["binding"]["A_v"])).T
    n_layers = len(params["binding"]["key_cb"])
    key_rows = _residual_rows(q_key, cfg.M_bind, n_layers, 11)
    val_rows = _residual_rows(q_val, cfg.M_bind, n_layers, 22)

    # b_raw samples from the same per-layer key/value rows.
    b_raw = []
    for zi in z:
        ks = [key_rows[l][np.argmin(
            np.sum((_unit(zi)[None, :] - _unit(key_rows[l])) ** 2, axis=1))]
            for l in range(n_layers)]
        vs = [val_rows[l][np.argmin(
            np.sum((_unit(zi)[None, :] - _unit(val_rows[l])) ** 2, axis=1))]
            for l in range(n_layers)]
        acc = np.zeros(cfg.d_model, dtype=np.float32)
        for a in ks:
            for b in vs:
                acc += np.fft.irfft(
                    _unit(np.fft.rfft(a)) * _unit(np.fft.rfft(b)),
                    n=cfg.d_model)
        b_raw.append(acc)
    bind_rows = _residual_rows(np.stack(b_raw).astype(np.float32),
                               cfg.M_bind, n_layers, 33)

    for family, rows in (("key", key_rows), ("val", val_rows),
                         ("bind", bind_rows)):
        for layer in range(n_layers):
            params["binding"][f"{family}_cb"][layer]["A"] = jnp.asarray(rows[layer])
            params["binding"][f"{family}_cb"][layer]["W"] = jnp.eye(
                cfg.d_model, dtype=jnp.float32)
    print(f"[fps] initialised {n_layers} binding layers from {args.samples} "
          f"samples, r_max={args.r_max}")

    train_cog(
        cfg, output_dir=args.output, steps=args.steps, lr=args.lr,
        batch_size=args.batch, seq_len=args.seq,
        log_every=max(1, args.steps // 4), save_every=1000,
        data_path=args.data, shape_path=args.shape, spans_path=args.spans,
        joint=True, self_awareness=False, overfit_one_batch=False,
        init_params=params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
