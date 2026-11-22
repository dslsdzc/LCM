#!/usr/bin/env python3
"""Calibrate lattice codebooks to the final encoder's query distribution.

Read-only diagnostic: loads a checkpoint, samples encoder z, replaces selected
codebook rows with farthest-point coverage rows, then runs the official
streaming utilisation probe. No training and no checkpoint write.

Usage:
    JAX_PLATFORMS=cpu python scripts/calibrate_coverage.py \
        --checkpoint /home/DslsDZC/LCM/checkpoints_fps_frozen/step_000100 \
        --samples 1000 --batches 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train.checkpoint import load_checkpoint  # noqa: E402
from train.config import LCMConfig  # noqa: E402
from train.data import WikiDataIter  # noqa: E402
from train.encoder import encoder_forward  # noqa: E402
from train.hyp import safe_unit  # noqa: E402
from train.smoke_conformance import _streaming_utilization_probe  # noqa: E402


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _fps(x, m, seed):
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(x)))
    idx = [first]
    d = np.sum((x - x[first]) ** 2, axis=1)
    for _ in range(1, m):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.sum((x - x[j]) ** 2, axis=1))
    return x[idx]


def _residual_rows(q, m, layers, seed, cosine=True):
    rows, r = [], q
    for layer in range(layers):
        rr = _fps(r, m, seed + layer)
        rows.append(rr)
        if cosine:
            idx = np.argmin(
                np.sum((_unit(r)[:, None, :] - _unit(rr)[None, :, :]) ** 2,
                       axis=-1), axis=1)
        else:
            idx = np.argmin(
                np.sum((r[:, None, :] - rr[None, :, :]) ** 2, axis=-1),
                axis=1)
        r = r - rr[idx]
    return rows


def _install_simvq(entry, rows, d):
    entry["A"] = jnp.asarray(rows)
    entry["W"] = jnp.eye(d, dtype=jnp.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat")
    p.add_argument("--shape", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json")
    p.add_argument("--spans", default="/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--seq", type=int, default=64)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    with open(ckpt / "config.json") as f:
        cfg = LCMConfig(**{k: v for k, v in json.load(f).items()
                           if k in LCMConfig.__dataclass_fields__})
    params, *_ = load_checkpoint(str(ckpt), cfg=cfg, load_opt=False,
                                 allow_missing_aux=True)

    it = WikiDataIter(args.data, args.shape, args.spans, B=1, N=args.seq)
    zs = []
    for _ in range(args.samples):
        x, _y = next(it)
        z = safe_unit(encoder_forward(params["encoder"],
                                      jnp.asarray(x), cfg.n_heads))
        zs.append(np.asarray(z)[0])
    z = np.stack(zs).astype(np.float32)
    d = cfg.d_model

    # HRQ top + fine layers. The probe tracks top; fine rows still get
    # direction coverage so the residual path is not left random.
    top_rows = _fps(z, int(cfg.M_top), 101)
    _install_simvq(params["hrq"]["top"], top_rows, d)
    for layer, fine in enumerate(params["hrq"]["fine"]):
        _install_simvq(fine, _fps(z, int(cfg.M_fine), 111 + layer), d)

    # Sparse: its LFQ gate depends on HRQ top_sim, so re-cover its code rows
    # after the HRQ recalibration.
    params["sparse"]["C"] = jnp.asarray(
        _fps(z, int(cfg.M_sparse), 501))

    # Contrast: per-family, per-layer residual coverage.
    for family in ("a", "b"):
        rows = _residual_rows(z, int(cfg.M_contrast),
                              len(params["contrast"][f"C_{family}"]),
                              201 if family == "a" else 221)
        for layer, entry in enumerate(params["contrast"][f"C_{family}"]):
            _install_simvq(entry, rows[layer], d)

    # LowRank: per-layer residual coverage projected back through V.
    V = np.asarray(params["lowrank"]["A_V"] @ params["lowrank"]["W_V"])
    r = z
    for layer, rank in enumerate(cfg.ranks):
        rows = _fps(r, int(cfg.M_lr), 301 + layer)
        Vs = V[:, :rank]
        U = rows @ Vs @ np.linalg.pinv(Vs.T @ Vs)
        params["lowrank"]["U"][layer] = jnp.asarray(U)
        idx = np.argmin(np.sum((r[:, None, :] - rows[None, :, :]) ** 2,
                               axis=-1), axis=1)
        r = r - rows[idx]

    # Binding: key/val residual coverage; bind from the resulting HRR samples.
    qk = z @ (V @ np.asarray(params["binding"]["A_k"])).T
    qv = z @ (V @ np.asarray(params["binding"]["A_v"])).T
    keys = _residual_rows(qk, int(cfg.M_bind), 3, 401)
    vals = _residual_rows(qv, int(cfg.M_bind), 3, 421)
    b_raw = []
    for zi in z:
        ks = [keys[l][np.argmin(
            np.sum((_unit(zi)[None, :] - _unit(keys[l])) ** 2, axis=1))]
            for l in range(3)]
        vs = [vals[l][np.argmin(
            np.sum((_unit(zi)[None, :] - _unit(vals[l])) ** 2, axis=1))]
            for l in range(3)]
        acc = np.zeros(d, dtype=np.float32)
        for a in ks:
            for b in vs:
                acc += np.fft.irfft(
                    _unit(np.fft.rfft(a)) * _unit(np.fft.rfft(b)), n=d)
        b_raw.append(acc)
    binds = _residual_rows(np.stack(b_raw).astype(np.float32),
                           int(cfg.M_bind), 3, 441)
    for family, rows in (("key", keys), ("val", vals), ("bind", binds)):
        for layer, entry in enumerate(params["binding"][f"{family}_cb"]):
            _install_simvq(entry, rows[layer], d)

    report = _streaming_utilization_probe(
        cfg, params, args.data, args.shape, args.spans,
        seq_len=args.seq, batches=args.batches)
    print(json.dumps(report["per_lattice"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
