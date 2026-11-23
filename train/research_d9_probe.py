#!/usr/bin/env python3
"""D-9 codebook-utilisation diagnostics (read-only research probe).

The conformance smoke reports only "unique rows / M". This probe answers the
next question: *why* does a lattice stay below 10%? It records, per codebook
layer, the selection histogram, its normalised entropy, the top-row share, and
whether the selection is input-independent (the same index on every batch).

Usage (CPU, one checkpoint, no training):

    JAX_PLATFORMS=cpu python -m train.research_d9_probe \
        --checkpoint checkpoints/conformance_smoke/step_000100 \
        --data .../conformance.dat --shape ..._shape.json \
        --spans ..._docs.npy --batches 100 \
        --output /tmp/d9_probe.json

The probe never changes parameters, thresholds, or checkpoints. It is a
diagnostic for the research plan in
``docs/d9-codebook-utilization-research-plan.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from train.checkpoint import load_checkpoint
from train.cognitive_step import six_lattice_step
from train.config import LCMConfig
from train.data import WikiDataIter
from train.encoder import encoder_forward
from train.hyp import safe_unit
from train.self_lattice import self_lattice_forward

LATTICE_GROUPS = ("hrq", "sparse", "lowrank", "manifold", "binding",
                  "contrast", "route", "self")


def _load_cfg(checkpoint: Path) -> LCMConfig:
    with open(checkpoint / "config.json") as f:
        raw = json.load(f)
    fields = LCMConfig.__dataclass_fields__
    return LCMConfig(**{k: v for k, v in raw.items() if k in fields})


class Histogram:
    """Accumulate selected row indices for one codebook layer."""

    def __init__(self, name: str, layer: int, M: int):
        self.name = name
        self.layer = layer
        self.M = int(M)
        self.counts = np.zeros(self.M, dtype=np.int64)
        self.n_selections = 0
        self.per_batch: list[tuple[int, ...]] = []

    def add(self, indices) -> None:
        idx = np.asarray(indices).reshape(-1)
        for value in idx:
            value = int(value)
            if 0 <= value < self.M:
                self.counts[value] += 1
                self.n_selections += 1
        self.per_batch.append(tuple(sorted(int(v) for v in idx if v >= 0)))

    def summary(self) -> dict:
        used = int(np.count_nonzero(self.counts))
        probs = self.counts[self.counts > 0].astype(np.float64)
        if probs.size > 1:
            probs /= probs.sum()
            entropy = float(-(probs * np.log(probs)).sum())
            entropy_norm = entropy / math.log(float(probs.size))
        else:
            entropy_norm = 0.0
        top_share = (float(self.counts.max()) / self.n_selections
                     if self.n_selections else 0.0)
        # "constant" means every batch selected exactly the same set of rows.
        constant = len(set(self.per_batch)) <= 1 if self.per_batch else False
        return {
            "M": self.M,
            "unique_rows": used,
            "fraction": used / max(1, self.M),
            "n_selections": self.n_selections,
            "entropy_norm_over_used": entropy_norm,
            "top_row_share": top_share,
            "constant_across_batches": bool(constant),
            "top_rows": [
                {"row": int(r), "count": int(c)}
                for r, c in sorted(enumerate(self.counts),
                                   key=lambda kv: kv[1], reverse=True)[:8]
                if c > 0
            ],
        }


def _codebook_geometry(params: dict) -> dict:
    """Cheap geometry signals for the codebooks that collapse hardest."""
    geometry = {}

    hrq = np.asarray(params["hrq"]["top"]["A"], dtype=np.float64)
    norms = np.linalg.norm(hrq.reshape(hrq.shape[0], -1), axis=1)
    geometry["hrq_top_norm"] = {
        "min": float(norms.min()), "mean": float(norms.mean()),
        "max": float(norms.max()),
    }

    route = np.asarray(params["route"]["C_route"], dtype=np.float64)
    route = route.reshape(route.shape[0], -1)
    if route.shape[0] > 1:
        gn = route / np.maximum(np.linalg.norm(route, axis=1, keepdims=True),
                                1e-12)
        sim = gn @ gn.T
        off = sim[~np.eye(sim.shape[0], dtype=bool)]
        geometry["route_pair_cosine"] = {
            "min": float(off.min()), "mean": float(off.mean()),
            "max": float(off.max()),
        }

    u0 = np.asarray(params["lowrank"]["U"][0], dtype=np.float64)
    u0 = u0.reshape(u0.shape[0], -1)
    sv = np.linalg.svd(u0, compute_uv=False)
    energy = sv ** 2
    eff_rank = (float(energy.sum() ** 2 / max(1e-24, (energy ** 2).sum()))
                if energy.size else 0.0)
    geometry["lowrank_u0"] = {
        "rows": int(u0.shape[0]),
        "rank": int(min(u0.shape)),
        "effective_rank": eff_rank,
        "singular_value_max": float(sv.max()) if sv.size else 0.0,
        "singular_value_min": float(sv.min()) if sv.size else 0.0,
    }
    return geometry


def run_probe(checkpoint: Path, data: str, shape: str, spans: str,
              batches: int, seq_len: int) -> dict:
    cfg = _load_cfg(checkpoint)
    params, _gvalue, _opt, _step = load_checkpoint(
        str(checkpoint), cfg=cfg, load_opt=False, allow_missing_aux=True)

    M_top = int(np.asarray(params["hrq"]["top"]["A"]).shape[0])
    M_fine = (int(np.asarray(params["hrq"]["fine"][0]["A"]).shape[0])
              if params["hrq"]["fine"] else None)
    M_sparse = int(np.asarray(params["sparse"]["C"]).shape[0])
    M_lr = int(np.asarray(params["lowrank"]["U"][0]).shape[0])
    M_man = int(np.asarray(params["manifold"]["C"]).shape[0])
    M_bind = int(np.asarray(params["binding"]["key_cb"][0]["A"]).shape[0])
    M_ct = int(np.asarray(params["contrast"]["C_a"][0]["A"]).shape[0])
    M_route = int(np.asarray(params["route"]["C_route"]).shape[0])
    M_self = int(np.asarray(params["self"]["C_self"]).shape[0])

    histograms: dict[tuple[str, int], Histogram] = {}

    def track(name: str, layer: int, indices, M: int) -> None:
        key = (name, int(layer))
        hist = histograms.get(key)
        if hist is None:
            hist = histograms[key] = Histogram(name, int(layer), int(M))
        hist.add(indices)

    it = WikiDataIter(data, shape, spans, B=1, N=seq_len)

    def forward(z):
        zself, bias, saux = self_lattice_forward(
            params["self"], z, training=False, return_aux=True,
            alpha_self=cfg.alpha_self)
        _, aux = six_lattice_step(
            z, params, cfg, training=False, rng=jax.random.PRNGKey(0),
            self_output=zself, self_bias=bias)
        aux = dict(aux)
        aux["self_idx"] = saux["idx"]
        return aux

    fn = jax.jit(forward)
    for _ in range(int(batches)):
        inputs, _targets = next(it)
        z = safe_unit(encoder_forward(params["encoder"],
                                      jnp.asarray(inputs), cfg.n_heads))
        aux = jax.tree_util.tree_map(
            lambda a: np.asarray(a), jax.device_get(fn(z)))

        track("hrq", 0, aux["hrq_idx"], M_top)
        for layer, idx in enumerate(aux.get("hrq_fine_idx", [])):
            track("hrq", 1 + layer, idx, M_fine)
        sidx = np.asarray(aux["sparse_idx"])
        track("sparse", 0, np.where(sidx > 0, sidx - 1, -1), M_sparse)
        for layer, idx in enumerate(aux["lowrank_aux"]["indices"]):
            track("lowrank", layer, idx, M_lr)
        track("manifold", 0, aux["man_idx"], M_man)
        for family in ("key", "val", "bind"):
            for layer, idx in enumerate(aux["binding_indices"][family]):
                track(f"binding.{family}", layer, idx, M_bind)
        for family in ("a", "b"):
            for layer, idx in enumerate(aux["contrast_indices"][family]):
                track(f"contrast.{family}", layer, idx, M_ct)
        track("route", 0, aux["route_idx"], M_route)
        track("self", 0, aux["self_idx"], M_self)

    by_lattice: dict[str, dict] = {}
    for (name, layer), hist in sorted(histograms.items()):
        by_lattice.setdefault(name, {})[str(layer)] = hist.summary()

    aggregate = {}
    for group in LATTICE_GROUPS:
        members = [v for k, v in by_lattice.items()
                   if k == group or k.startswith(group + ".")]
        fractions = [m["fraction"] for fam in members for m in fam.values()]
        aggregate[group] = {
            "mean_fraction": float(np.mean(fractions)) if fractions else 0.0,
            "min_fraction": float(np.min(fractions)) if fractions else 0.0,
            "max_fraction": float(np.max(fractions)) if fractions else 0.0,
            "passes_10pct": bool(fractions and min(fractions) >= 0.10),
        }

    return {
        "checkpoint": str(checkpoint),
        "batches": int(batches),
        "batch_size": 1,
        "seq_len": int(seq_len),
        "thresholds": {"utilization": 0.10},
        "aggregate": aggregate,
        "per_layer": by_lattice,
        "geometry": _codebook_geometry(params),
        "note": ("Read-only diagnostics. Failing rows are evidence, not a "
                 "verdict change: the 10% threshold is not lowered."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = run_probe(Path(args.checkpoint), args.data, args.shape,
                       args.spans, args.batches, args.seq_len)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"[D9] wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
