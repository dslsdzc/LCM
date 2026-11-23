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
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _fps(x: np.ndarray, m: int, seed: int) -> np.ndarray:
    # A random first point can be a near-zero residual. Under direction-based
    # selection that row becomes a universal attractor; start from the
    # largest-norm point instead.
    first = int(np.argmax(np.linalg.norm(x, axis=1)))
    idx = [first]
    d = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, m):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.sum((x - x[j]) ** 2, axis=1))
    return x[idx]


def _fill_repeat(rows: np.ndarray, m: int) -> np.ndarray:
    """Repeat the selected rows to the codebook size.

    Do not fill with extra query samples: a full FPS cover leaves a zero
    residual for the next layer, and extra query rows repeat the same effect
    under Euclidean selection.
    """
    return np.concatenate([rows] * int(np.ceil(m / len(rows))), axis=0)[:m]


def _residual_rows(q: np.ndarray, m: int, layers: int, seed: int,
                   distinct: int | None = None, cosine: bool = True):
    """Per-layer residual coverage with a bounded distinct-row budget.

    ``distinct`` caps how many *different* rows a layer may install. A layer
    that installs every query direction leaves the next layer with an
    all-zero residual (observed on calibrated binding.bind layers 1/2).
    """
    rows, r = [], q
    n_distinct = int(m if distinct is None else min(m, distinct))
    for layer in range(layers):
        layer_rows = _fps(r, n_distinct, seed + layer)
        if cosine:
            idx = np.argmin(
                np.sum((_unit(r)[:, None, :] - _unit(layer_rows)[None, :, :]) ** 2,
                       axis=-1), axis=1)
        else:
            idx = np.argmin(
                np.sum((r[:, None, :] - layer_rows[None, :, :]) ** 2,
                       axis=-1), axis=1)
        rows.append(_fill_repeat(layer_rows, m) if n_distinct < m else layer_rows)
        r = r - layer_rows[idx]
    return rows


def _residual_vq_chain_np(codebooks, r, cosine=True):
    """Numpy mirror of ``train.lattices._residual_vq_chain``.

    The quantised forward value is the raw selected row ``C[idx]``; the next
    residual is ``r - C[idx]``. This is what ``binding_forward`` binds.
    """
    quantised, inputs, indices = [], [], []
    for cb in codebooks:
        C = np.asarray(cb["A"] @ cb["W"])
        inputs.append(r)
        if cosine:
            r_sel, C_sel = _unit(r), _unit(C)
        else:
            r_sel, C_sel = r, C
        idx = np.argmin(
            np.sum((r_sel[:, None, :] - C_sel[None, :, :]) ** 2,
                   axis=-1), axis=1)
        z_q = C[idx]
        quantised.append(z_q)
        indices.append(idx)
        r = r - z_q
    return quantised, inputs, indices


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
    p.add_argument("--init-lowrank", action="store_true",
                   help="Also FPS-initialise the LowRank per-layer codebooks")
    p.add_argument("--init-contrast", action="store_true",
                   help="FPS-initialise the Contrast per-family/per-layer "
                        "codebooks with a bounded distinct-row budget")
    p.add_argument("--freeze-codebooks", action="store_true",
                   help="Keep binding/lowrank out of optimizer and EMA")
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
    # Budget half the rows per layer so the next layer still has a residual.
    bind_distinct = max(1, int(cfg.M_bind) // 2)
    key_rows = _residual_rows(q_key, cfg.M_bind, n_layers, 11,
                              distinct=bind_distinct)
    val_rows = _residual_rows(q_val, cfg.M_bind, n_layers, 22,
                              distinct=bind_distinct)

    # Install key/val first, then run the *real* projected residual chain.
    # Selecting rows from raw z (the old prototype) does not reproduce the
    # tensor binding_forward actually binds.
    for family, rows in (("key", key_rows), ("val", val_rows)):
        for layer in range(n_layers):
            params["binding"][f"{family}_cb"][layer]["A"] = jnp.asarray(
                rows[layer])
            params["binding"][f"{family}_cb"][layer]["W"] = jnp.eye(
                cfg.d_model, dtype=jnp.float32)

    k_q, _k_inputs, _k_idx = _residual_vq_chain_np(
        params["binding"]["key_cb"], q_key, cosine=True)
    v_q, _v_inputs, _v_idx = _residual_vq_chain_np(
        params["binding"]["val_cb"], q_val, cosine=True)
    b_raw = np.zeros((len(z), cfg.d_model), dtype=np.float32)
    for a in k_q:
        for b in v_q:
            for i in range(len(z)):
                b_raw[i] += np.fft.irfft(
                    _unit(np.fft.rfft(a[i])) * _unit(np.fft.rfft(b[i])),
                    n=cfg.d_model)
    bind_rows = _residual_rows(b_raw, cfg.M_bind, n_layers, 33,
                               distinct=bind_distinct)
    for layer in range(n_layers):
        params["binding"]["bind_cb"][layer]["A"] = jnp.asarray(
            bind_rows[layer])
        params["binding"]["bind_cb"][layer]["W"] = jnp.eye(
            cfg.d_model, dtype=jnp.float32)
    print(f"[fps] initialised {n_layers} binding layers from {args.samples} "
          f"samples, r_max={args.r_max}")

    if args.init_contrast:
        n_ct = len(params["contrast"]["C_a"])
        # contrast_forward selects by raw Euclidean distance (no direction
        # normalisation), so calibrate in that same domain.
        ct_distinct = max(1, (3 * int(cfg.M_contrast)) // 4)
        for family, seed in (("a", 55), ("b", 66)):
            rows = _residual_rows(
                z, cfg.M_contrast, n_ct, seed,
                distinct=ct_distinct, cosine=False)
            for layer in range(n_ct):
                entry = params["contrast"][f"C_{family}"][layer]
                entry["A"] = jnp.asarray(rows[layer])
                entry["W"] = jnp.eye(cfg.d_model, dtype=jnp.float32)
        print(f"[fps] initialised {n_ct} contrast layers x2 families from "
              f"{args.samples} samples")

    if args.init_lowrank:
        M_lr = int(params["lowrank"]["U"][0].shape[0])
        r = z
        for layer, rank in enumerate(cfg.ranks):
            rows = _fps(r, M_lr, 44 + layer)
            Vs = V[:, :rank]  # (d, rank)
            # Rows = U @ Vs.T; solve the small rank x rank system for U.
            U = rows @ Vs @ np.linalg.pinv(Vs.T @ Vs)
            params["lowrank"]["U"][layer] = jnp.asarray(U)
            idx = np.argmin(
                np.sum((r[:, None, :] - rows[None, :, :]) ** 2, axis=-1),
                axis=1)
            r = r - rows[idx]
        print(f"[fps] initialised {len(cfg.ranks)} lowrank layers from "
              f"{args.samples} samples")

    train_cog(
        cfg, output_dir=args.output, steps=args.steps, lr=args.lr,
        batch_size=args.batch, seq_len=args.seq,
        log_every=max(1, args.steps // 4), save_every=1000,
        data_path=args.data, shape_path=args.shape, spans_path=args.spans,
        joint=True, self_awareness=False, overfit_one_batch=False,
        init_params=params,
        freeze_lattices=("binding", "lowrank") if args.freeze_codebooks else ())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
