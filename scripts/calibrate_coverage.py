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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from train.checkpoint import load_cog_state  # noqa: E402
from train.cog_train import (  # noqa: E402
    RunIdentity, resolve_dataset_identity, save_cog_checkpoint,
)
from train.config import LCMConfig  # noqa: E402
from train.data import WikiDataIter  # noqa: E402
from train.encoder import encoder_forward  # noqa: E402
from train.hyp import exp_map, log_map, safe_unit  # noqa: E402
from train.smoke_conformance import _streaming_utilization_probe  # noqa: E402


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _fps(x, m, seed):
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


def _fill_repeat(rows, m):
    """Repeat the selected rows to the codebook size.

    Filling with extra query samples re-covers every direction and leaves a
    zero residual for the next layer; repeats keep the budget visible.
    """
    return np.concatenate([rows] * int(np.ceil(m / len(rows))), axis=0)[:m]


def _residual_rows(q, m, layers, seed, cosine=True, distinct=0):
    """Per-layer residual coverage, optionally with a distinct-row budget.

    ``distinct > 0`` caps how many *different* rows a layer installs; the
    remaining rows are repeats. Without a budget, a layer can consume every
    query direction and make the next layer's residual identically zero.
    """
    rows, r = [], q
    n_distinct = int(m if distinct <= 0 else min(m, distinct))
    for layer in range(layers):
        rr = _fps(r, n_distinct, seed + layer)
        if cosine:
            idx = np.argmin(
                np.sum((_unit(r)[:, None, :] - _unit(rr)[None, :, :]) ** 2,
                       axis=-1), axis=1)
        else:
            idx = np.argmin(
                np.sum((r[:, None, :] - rr[None, :, :]) ** 2, axis=-1),
                axis=1)
        rows.append(_fill_repeat(rr, m) if n_distinct < m else rr)
        r = r - rr[idx]
    return rows


def _residual_vq_chain_np(codebooks, r, cosine=True):
    """Numpy mirror of ``train.lattices._residual_vq_chain``.

    Returns per-layer ``(quantised_rows, residual_inputs, indices)``. The
    forward quantised value is the raw selected row ``C[idx]``, not the
    residual query; ``binding_forward`` feeds these rows into HRR bind.
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
    p.add_argument("--bind-distinct", type=int, default=0,
                   help="Distinct rows per Binding layer (0 = full M)")
    p.add_argument("--contrast-distinct", type=int, default=0,
                   help="Distinct rows per Contrast layer (0 = full M)")
    p.add_argument("--save", default="",
                   help="Optional output directory for a calibrated v2 checkpoint")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    with open(ckpt / "config.json") as f:
        cfg = LCMConfig(**{k: v for k, v in json.load(f).items()
                           if k in LCMConfig.__dataclass_fields__})
    state = load_cog_state(str(ckpt), require_buffers=False)
    params = jax.tree_util.tree_map(jnp.asarray, state["params"])
    self_state = state.get("self_state")

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

    # Manifold: retrieval runs in the R6 domain z_P = exp_map(safe_unit(z)),
    # while the stored codebook C is tangent-space and is pushed to the ball
    # by exp_map(C). Calibrating on raw z (the old behaviour) put the rows in
    # the wrong domain. FPS on z_P, then store log_map(z_P) in C.
    z_P = np.asarray(exp_map(jnp.asarray(safe_unit(z))))
    man_rows = _fps(z_P, int(cfg.M_man), 601)
    params["manifold"]["C"] = jnp.asarray(log_map(jnp.asarray(man_rows)))

    # Contrast: per-family, per-layer residual coverage. contrast_forward
    # selects by raw Euclidean distance (no direction normalisation), so the
    # calibration must use the same domain.
    for family in ("a", "b"):
        rows = _residual_rows(z, int(cfg.M_contrast),
                              len(params["contrast"][f"C_{family}"]),
                              201 if family == "a" else 221,
                              cosine=False,
                              distinct=args.contrast_distinct)
        for layer, entry in enumerate(params["contrast"][f"C_{family}"]):
            _install_simvq(entry, rows[layer], d)

    # LowRank: per-layer residual coverage projected back through V. The
    # installed codebook is C = U @ V^T, not the FPS rows themselves; the
    # residual update must use the row the forward actually selects.
    V = np.asarray(params["lowrank"]["A_V"] @ params["lowrank"]["W_V"])
    r = z
    for layer, rank in enumerate(cfg.ranks):
        rows = _fps(r, int(cfg.M_lr), 301 + layer)
        Vs = V[:, :rank]
        U = rows @ Vs @ np.linalg.pinv(Vs.T @ Vs)
        C_rows = U @ Vs.T
        params["lowrank"]["U"][layer] = jnp.asarray(U)
        idx = np.argmin(np.sum((r[:, None, :] - C_rows[None, :, :]) ** 2,
                               axis=-1), axis=1)
        r = r - C_rows[idx]

    # Binding: key/val residual coverage; bind from the *real* forward
    # chain. The old prototype selected key/val rows from raw z and ignored
    # the residual update, so its b_raw was not the tensor binding_forward
    # actually binds.
    qk = z @ (V @ np.asarray(params["binding"]["A_k"])).T
    qv = z @ (V @ np.asarray(params["binding"]["A_v"])).T
    n_bind = len(params["binding"]["key_cb"])
    keys = _residual_rows(qk, int(cfg.M_bind), n_bind, 401,
                          distinct=args.bind_distinct)
    vals = _residual_rows(qv, int(cfg.M_bind), n_bind, 421,
                          distinct=args.bind_distinct)
    for layer in range(n_bind):
        _install_simvq(params["binding"]["key_cb"][layer], keys[layer], d)
        _install_simvq(params["binding"]["val_cb"][layer], vals[layer], d)

    # Run the installed key/val codebooks through the same residual chain as
    # binding_forward, then bind every (key layer, value layer) pair.
    k_q, _k_inputs, _k_idx = _residual_vq_chain_np(
        params["binding"]["key_cb"], qk, cosine=True)
    v_q, _v_inputs, _v_idx = _residual_vq_chain_np(
        params["binding"]["val_cb"], qv, cosine=True)
    b_raw = np.zeros((len(z), d), dtype=np.float32)
    for a in k_q:
        for b in v_q:
            for i in range(len(z)):
                b_raw[i] += np.fft.irfft(
                    _unit(np.fft.rfft(a[i])) * _unit(np.fft.rfft(b[i])),
                    n=d)
    binds = _residual_rows(b_raw, int(cfg.M_bind), n_bind, 441,
                           distinct=args.bind_distinct)
    for layer, entry in enumerate(params["binding"]["bind_cb"]):
        _install_simvq(entry, binds[layer], d)

    report = _streaming_utilization_probe(
        cfg, params, args.data, args.shape, args.spans,
        seq_len=args.seq, batches=args.batches)
    print(json.dumps(report["per_lattice"], indent=1))

    if args.save:
        spec, meta = resolve_dataset_identity(
            args.data, args.shape, args.spans, full_verify=True)
        save_cog_checkpoint(
            params, args.save, int(state["step"]),
            run_identity=RunIdentity(spec, meta), train_cfg=cfg,
            self_state=self_state, allow_missing_buffers=True)
        print(f"[calib] saved calibrated checkpoint -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
