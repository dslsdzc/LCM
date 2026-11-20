"""≤100-step conformance smoke for the lattice stack (design §6.3).

The smoke is deliberately *not* a language-quality experiment. It answers one
question: after 100 real optimizer/EMA steps, did every lattice update, every
safety/conformance path fire, and is every persisted tensor finite?

What it does
------------
* Calls the production ``train_cog`` loop (passive-only, ``joint=True``,
  ``self_awareness=True``, no Qwen, no resume) with a diagnostic callback.
* Records loss per step, per-lattice parameter/EMA deltas, codebook
  utilisation/variance, route-vs-weight-decay baseline, sparse LFQ, HRQ
  value-routing invariants, the entropy gate, feature-bank resets, the C danger
  abort, gvalue margins, self-bias range, 100-step wall-clock overhead and a
  save→load→2-step consistency check.
* Writes ``smoke_report.json`` and ``smoke_report.md`` next to the checkpoint.

Run:
    python train/smoke_conformance.py --steps 100 --batch 1 --seq 64 \
        --data <corpus.dat> --shape <corpus_shape.json> \
        --spans <corpus_docs.npy> --output checkpoints/conformance_smoke \
        --passive-only
"""
import argparse
import ctypes
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

# Running ``python train/smoke_conformance.py`` puts ``train/`` first on
# sys.path, where the module ``train`` resolves to ``train/train.py`` and the
# package import below fails. Make the repo root win and drop the script dir;
# ``python -m train.smoke_conformance`` is unaffected.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path
            if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train.behavior_explorer import BehaviorExplorer
from train.checkpoint import load_cog_state
from train.cog_train import (
    RunIdentity, init_cog_params, make_train_step,
    resolve_dataset_identity, train_cog, _self_observe_step,
)
from train.config import LCMConfig
from train.data import WikiDataIter
from train.ema_state import init_ema_state
from train.encoder import encoder_forward
from train.feature_bank import FeatureBank
from train.fusion import fuse_lattices_with_aux
from train.hyp import exp_map, poincare_similarity, safe_unit
from train.lattices import (
    _hrq_sims, hrq_forward, hrq_top_prototype, sparse_forward,
)
from train.narrative_memory import NarrativeMemory
from train.observability import ObservabilityRecorder
from train.param_groups import partition_params
from train.reflection_loop import ReflectionLoop
from train.self_lattice import self_lattice_forward

REQUIRED_REPORT_KEYS = (
    "meta", "loss", "lattice_updates", "codebook_utilization", "ema",
    "route", "sparse_inference", "hrq_routing", "entropy_gate",
    "feature_bank", "safety", "self_lattice", "overhead", "persistence",
    "checks",
)

# Tiny preset for the unit test. It must keep every structural feature
# (layers, self, banks, safety) but compile in seconds on CPU.
TINY_OVERRIDES = dict(
    d_model=32,
    n_heads=4,
    d_head=8,
    d_ff=64,
    n_encoder_layers=1,
    max_seq_len=128,
    M_top=16,
    M_fine=8,
    n_hrq_layers=2,
    M_sparse=16,
    M_lr=16,
    n_lr_layers=2,
    r_max=4,
    ranks=(2, 4),
    M_man=16,
    t_dim=2,
    M_bind=16,
    n_bind_layers=1,
    M_contrast=16,
    n_contrast_layers=1,
    M_danger=8,
    n_self_codes=8,
    max_inference_steps=2,
    n_orth_samples=4,
    bank_capacity=16,
    bank_check_interval=2,
    bank_dead_threshold=2,
)


def smoke_config(preset="full", max_inference_steps=None):
    """Return the config used by the smoke.

    ``tiny`` is only for the unit test; the 100-step report must use ``full``
    (d_model=256 and the training lattice sizes), because a reduced model would
    not exercise the shipped shapes.
    """
    cfg = LCMConfig()
    if preset == "tiny":
        cfg = dataclasses.replace(cfg, **TINY_OVERRIDES)
    elif preset != "full":
        raise ValueError(f"unknown preset {preset!r}")
    if max_inference_steps is not None:
        cfg = dataclasses.replace(cfg, max_inference_steps=int(max_inference_steps))
    return cfg


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _np(x):
    return np.asarray(jax.device_get(x))


def _to_jax_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)


def _tree_delta(a, b, prefix=""):
    """Per-leaf max-abs and L2 differences between two pytrees.

    Returns ``{'max_abs', 'l2', 'n_leaves', 'n_changed'}``. Non-array leaves
    are ignored (None, strings); shape mismatches are recorded as an error.
    """
    max_abs = 0.0
    l2 = 0.0
    n_leaves = 0
    n_changed = 0
    errors = []

    def walk(x, y, path):
        nonlocal max_abs, l2, n_leaves, n_changed
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), f"{path}.{key}" if path else str(key))
            return
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                errors.append(f"{path}: length {len(x)} != {len(y)}")
                return
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{path}[{i}]")
            return
        if x is None or y is None:
            if (x is None) != (y is None):
                errors.append(f"{path}: None mismatch")
            return
        ax = np.asarray(x, dtype=np.float64)
        ay = np.asarray(y, dtype=np.float64)
        if ax.shape != ay.shape:
            errors.append(f"{path}: shape {ax.shape} != {ay.shape}")
            return
        if ax.size == 0:
            return
        diff = ax - ay
        n_leaves += 1
        if not np.all(np.isfinite(diff)):
            errors.append(f"{path}: non-finite difference")
            max_abs = float("inf")
            l2 = float("inf")
            return
        m = float(np.max(np.abs(diff)))
        max_abs = max(max_abs, m)
        l2 += float(np.sum(diff * diff))
        if m > 0:
            n_changed += 1

    walk(a, b, prefix)
    return {
        "max_abs": max_abs,
        "l2": float(math.sqrt(l2)),
        "n_leaves": n_leaves,
        "n_changed": n_changed,
        "errors": errors,
    }


def _codebook_variances(params):
    """Variance of each effective codebook, for the collapse check."""
    out = {}

    def add(name, array):
        arr = np.asarray(array, dtype=np.float64)
        out[name] = float(arr.var()) if arr.size else 0.0

    if "hrq" in params:
        add("hrq_top", _np(params["hrq"]["top"]["A"])
            @ _np(params["hrq"]["top"]["W"]))
        fine = [_np(fb["A"]) @ _np(fb["W"]) for fb in params["hrq"]["fine"]]
        if fine:
            out["hrq_fine"] = float(np.mean([f.var() for f in fine]))
    if "sparse" in params:
        add("sparse", params["sparse"]["C"])
    if "lowrank" in params:
        lr = params["lowrank"]
        V = _np(lr["A_V"]) @ _np(lr["W_V"])
        vals = []
        for u, r in zip(lr["U"], _ranks(params)):
            C = _np(u) @ V[:, :int(r)].T
            vals.append(C.var())
        out["lowrank"] = float(np.mean(vals)) if vals else 0.0
    if "manifold" in params:
        add("manifold", params["manifold"]["C"])
    if "binding" in params:
        vals = []
        for family in ("key_cb", "val_cb", "bind_cb"):
            for cb in params["binding"][family]:
                vals.append((_np(cb["A"]) @ _np(cb["W"])).var())
        out["binding"] = float(np.mean(vals)) if vals else 0.0
    if "contrast" in params:
        vals = []
        for family in ("C_a", "C_b"):
            for cb in params["contrast"][family]:
                vals.append((_np(cb["A"]) @ _np(cb["W"])).var())
        out["contrast"] = float(np.mean(vals)) if vals else 0.0
    if "route" in params:
        add("route", params["route"]["C_route"])
    if "self" in params:
        add("self", params["self"]["C_self"])
    return out


def _ranks(params):
    """Per-layer lowrank ranks, inferred from U shapes when cfg is absent."""
    return [int(np.asarray(u).shape[1]) for u in params["lowrank"]["U"]]


class _UtilTracker:
    """Count unique selected rows per lattice layer."""

    def __init__(self):
        self.layers = {}

    def add(self, name, layer, indices, M):
        indices = np.asarray(indices).reshape(-1)
        key = (str(name), int(layer))
        entry = self.layers.setdefault(
            key, {"seen": set(), "M": int(M), "n_selections": 0})
        entry["seen"].update(int(i) for i in indices if int(i) >= 0)
        entry["n_selections"] += int(indices.size)

    def summary(self):
        grouped = {}
        for (name, _layer), entry in self.layers.items():
            frac = len(entry["seen"]) / max(1, entry["M"])
            grouped.setdefault(name, []).append(frac)
        return {
            name: {
                "layers": len(vals),
                "mean_fraction": float(np.mean(vals)),
                "min_fraction": float(np.min(vals)),
                "max_fraction": float(np.max(vals)),
                "unique_total": int(sum(
                    len(self.layers[(name, l)]["seen"])
                    for (n, l) in self.layers if n == name)),
                "selections_total": int(sum(
                    self.layers[(name, l)]["n_selections"]
                    for (n, l) in self.layers if n == name)),
            }
            for name, vals in grouped.items()
        }


def _sample_batch(data, shape, spans, cfg, B=1, N=None):
    N = int(N or cfg.max_seq_len)
    it = WikiDataIter(data, shape, spans, B=B, N=N)
    inputs, targets = next(it)
    return inputs.astype(np.int32), targets.astype(np.int32)


def _encode_z(params, inputs, cfg):
    z = encoder_forward(params["encoder"], jnp.asarray(inputs), cfg.n_heads)
    return safe_unit(z)


def _ema_branch_summary(initial, final):
    """Aggregate one EMA branch's N/m/last_used movement."""
    N0 = _np(initial["N"]).astype(np.float64)
    N1 = _np(final["N"]).astype(np.float64)
    m0 = _np(initial["m"]).astype(np.float64)
    m1 = _np(final["m"]).astype(np.float64)
    l0 = _np(initial["last_used"])
    l1 = _np(final["last_used"])
    return {
        "N_sum_initial": float(N0.sum()),
        "N_sum_final": float(N1.sum()),
        "delta_N_abs": float(np.abs(N1 - N0).sum()),
        "m_norm_initial": float(np.linalg.norm(m0)),
        "m_norm_final": float(np.linalg.norm(m1)),
        "delta_m_norm": float(np.linalg.norm(m1 - m0)),
        "rows_used_final": int(np.count_nonzero(N1 > 0)),
        "rows_total": int(N1.size),
        "last_used_max": int(l1.max()) if l1.size else 0,
        "last_used_nonzero": int(np.count_nonzero(l1 > 0)),
        "changed": bool(np.any(N1 != N0) or np.any(m1 != m0)),
    }


def _ema_binding_summary(initial, final):
    """Aggregate binding's three families × layers."""
    N0 = N1 = 0.0
    m0 = m1 = 0.0
    rows_used = rows_total = 0
    last_max = 0
    last_nonzero = 0
    changed = False
    for family in ("key", "val", "bind"):
        for s0, s1 in zip(initial[family], final[family]):
            a = _np(s0["N"]).astype(np.float64)
            b = _np(s1["N"]).astype(np.float64)
            ma = _np(s0["m"]).astype(np.float64)
            mb = _np(s1["m"]).astype(np.float64)
            N0 += float(a.sum())
            N1 += float(b.sum())
            m0 += float(np.sum(ma * ma))
            m1 += float(np.sum(mb * mb))
            rows_used += int(np.count_nonzero(b > 0))
            rows_total += int(b.size)
            last = _np(s1["last_used"])
            last_max = max(last_max, int(last.max()) if last.size else 0)
            last_nonzero += int(np.count_nonzero(last > 0))
            changed = changed or bool(np.any(b != a) or np.any(mb != ma))
    return {
        "N_sum_initial": N0,
        "N_sum_final": N1,
        "delta_N_abs": abs(N1 - N0),
        "m_norm_initial": math.sqrt(m0),
        "m_norm_final": math.sqrt(m1),
        "delta_m_norm": abs(math.sqrt(m1) - math.sqrt(m0)),
        "rows_used_final": rows_used,
        "rows_total": rows_total,
        "last_used_max": last_max,
        "last_used_nonzero": last_nonzero,
        "changed": changed,
    }


def _make_optimizer(cfg):
    """The production optimizer construction (train_cog)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0, b1=cfg.adam_beta1,
                    b2=cfg.adam_beta2, eps=cfg.adam_eps,
                    weight_decay=cfg.weight_decay),
    )


def _route_baseline_delta(initial_params, cfg, steps, lr):
    """C_route movement from weight decay alone (zero gradients)."""
    optimizer = _make_optimizer(cfg)
    trainable, _, _ = partition_params(initial_params)
    opt_state = optimizer.init(trainable)
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=max(1, int(steps)), alpha=0.1)
    p = trainable
    for i in range(int(steps)):
        grads = jax.tree_util.tree_map(jnp.zeros_like, p)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        updates = jax.tree_util.tree_map(
            lambda u, lr_i=float(schedule(i)): lr_i * u, updates)
        p = optax.apply_updates(p, updates)
    if "route" not in p:
        return {"max_abs": 0.0, "l2": 0.0}
    return _tree_delta(initial_params["route"], p["route"])


def _normalised_entropy(weights):
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)
    n_active = int(np.count_nonzero(w > 1e-6))
    if n_active <= 1:
        return 0.0, n_active
    h = -float(np.sum(w * np.log(w + 1e-12)))
    return h / math.log(n_active), n_active


# ---------------------------------------------------------------------------
# Safety probes
# ---------------------------------------------------------------------------

class _DangerLattice(ctypes.Structure):
    _fields_ = [
        ("C_threats", ctypes.POINTER(ctypes.c_float)),
        ("C_normal", ctypes.POINTER(ctypes.c_float)),
        ("M_danger", ctypes.c_int),
        ("D", ctypes.c_int),
        ("safety_threshold", ctypes.c_float),
        ("integrity_hash", ctypes.c_char * 65),
    ]


class _Conflict(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_int),
        ("type", ctypes.c_int),
        ("detail", ctypes.c_char * 256),
        ("step", ctypes.c_int),
        ("timestamp", ctypes.c_double),
    ]


def _c_danger_probe(so_path):
    """Constructed threat/aligned/then normal vectors through the real C API."""
    D = 256  # LCM_D is a compile-time constant in infer/lcm.h
    try:
        ctypes.CDLL("libm.so.6", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    lib = ctypes.CDLL(str(so_path))
    lib.danger_init.argtypes = [
        ctypes.POINTER(_DangerLattice), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.danger_init.restype = None
    lib.detect_any_conflict.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.POINTER(_DangerLattice), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.POINTER(_Conflict)]
    lib.detect_any_conflict.restype = ctypes.c_bool

    threat = np.zeros((2, D), np.float32)
    normal = np.zeros((2, D), np.float32)
    threat[0, 0] = 1.0
    threat[1, 2] = 1.0
    normal[0, 1] = 1.0
    normal[1, 3] = 1.0

    def ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dl = _DangerLattice()
    lib.danger_init(ctypes.byref(dl), ptr(threat), ptr(normal), 2, D)
    hash_one = bytes(dl.integrity_hash)
    dl2 = _DangerLattice()
    lib.danger_init(ctypes.byref(dl2), ptr(threat), ptr(normal), 2, D)
    hash_two = bytes(dl2.integrity_hash)

    counts = (ctypes.c_int * 8)(*([0] * 8))
    z_threat = np.zeros(D, np.float32)
    z_threat[0] = 1.0
    conflict = _Conflict()
    blocked = bool(lib.detect_any_conflict(
        ptr(z_threat), ptr(z_threat), 0, ctypes.byref(dl), None, counts,
        1.0, 0.5, 16, ctypes.byref(conflict)))

    z_normal = np.zeros(D, np.float32)
    z_normal[1] = 1.0
    conflict_normal = _Conflict()
    blocked_normal = bool(lib.detect_any_conflict(
        ptr(z_normal), ptr(z_normal), 0, ctypes.byref(dl), None, counts,
        1.0, 0.5, 16, ctypes.byref(conflict_normal)))

    return {
        "available": True,
        "threat_blocked": blocked,
        "threat_source": int(conflict.source),
        "threat_detail": bytes(conflict.detail).decode(errors="replace").rstrip("\x00"),
        "neutral_blocked": blocked_normal,
        "hash_stable": hash_one == hash_two,
        "hash_prefix": hash_one.decode(errors="replace")[:16],
        "threshold": float(dl.safety_threshold),
        "constructed_codebooks_distinct": not np.array_equal(threat, normal),
    }


class _AbortTokenizer:
    def encode(self, prompt):
        return SimpleNamespace(ids=[2])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x41 + (int(i) % 26)) for i in ids)


def _python_no_token_probe():
    """lcm.generate must return no token when the loop returns -2."""
    from unittest import mock
    import lcm

    class _AbortEngine(lcm.LCMInferEngine):
        """Inherits the real generate() short-circuit, no C engine needed."""

        def __init__(self, d=256):
            self.d = d
            self.V = 8
            self.max_seq_len = 8
            self.n_heads = 1
            self.tokenizer = _AbortTokenizer()
            self.encoder = None
            self.decoder = {
                "format": "cog", "w_out": np.zeros((d, 8), np.float32)}
            self._pred_enabled = False
            self.pred = None
            self._last_agency_mod = {"tension": 0.0, "explore": 0.0}
            self._last_loop_ret = 0

        def cognitive_loop(self, z, **kwargs):
            self._last_loop_ret = -2
            return np.zeros(self.d, np.float32), False

    eng = _AbortEngine()
    z = np.zeros(eng.d, np.float32)
    z[0] = 1.0
    with mock.patch.object(lcm, "_encoder_full_with_state",
                           lambda *a, **k: (z.copy(), None)):
        tokens = list(eng.generate("hello", max_new=1, use_safety=False))
    return {
        "abort_code": int(getattr(lcm, "CONFLICT_ABORT_TOKEN", -2)),
        "loop_ret": int(eng._last_loop_ret),
        "tokens_emitted": len(tokens),
        "no_token": len(tokens) == 0 and int(eng._last_loop_ret) == -2,
    }


def _gvalue_probe():
    from train.gvalue import GValueCodebook, make_global_value_vectors

    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    pos = jnp.zeros(256).at[0].set(0.9)
    neg = jnp.zeros(256).at[0].set(-0.9)
    neutral = jnp.zeros(256)
    safe_pos_0, _ = gv.check_safety(pos, 0.0)
    safe_pos_05, _ = gv.check_safety(pos, 0.5)
    safe_neg_05, violated = gv.check_safety(neg, 0.5)
    safe_neutral_05, _ = gv.check_safety(neutral, 0.5)
    return {
        "pos_neg_distinct": not bool(jnp.array_equal(C_pos, C_neg)),
        "verify_integrity": bool(gv.verify_integrity()),
        "positive_margin_0_safe": bool(safe_pos_0),
        "positive_margin_0.5_safe": bool(safe_pos_05),
        "negative_margin_0.5_safe": bool(safe_neg_05),
        "negative_violated_law": int(violated),
        "neutral_margin_0.5_safe": bool(safe_neutral_05),
        "strict_neutral_violates": not bool(safe_neutral_05),
    }


def _exported_safety_probe(final_dir):
    """Read the exported gvalue/danger files; verify the halves differ."""
    import lcm

    out = {}
    try:
        gv, gv_n, d = lcm.load_codebook_for_c(
            final_dir, "gvalue_codebook.bin")
        gv = np.asarray(gv)
        out["gvalue_pairs"] = int(gv_n)
        out["gvalue_pos_neg_distinct"] = not np.array_equal(
            gv[: gv_n * d], gv[gv_n * d:])
    except Exception as exc:  # pragma: no cover - diagnostic
        out["gvalue_error"] = f"{type(exc).__name__}: {exc}"
    try:
        danger, dm, dd = lcm.load_codebook_for_c(
            final_dir, "danger_codebook.bin")
        danger = np.asarray(danger)
        out["danger_rows"] = int(dm)
        out["danger_threat_normal_distinct"] = not np.array_equal(
            danger[: dm * dd], danger[dm * dd:])
    except Exception as exc:  # pragma: no cover - diagnostic
        out["danger_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persistence_probe(final_dir, final_params, final_ema, final_banks,
                       cfg, data, shape, spans, steps, seq_len):
    loaded = load_cog_state(str(final_dir), require_buffers=True)
    loaded_params = _to_jax_tree(loaded["params"])
    loaded_ema = _to_jax_tree(loaded["ema_state"])

    params_err = _tree_delta(final_params, loaded_params)["max_abs"]
    ema_err = _tree_delta(final_ema, loaded_ema)["max_abs"]

    bank_err = 0.0
    for name in ("feature_bank", "contrast_bank"):
        a = final_banks[name]
        b = loaded[name]
        bank_err = max(
            bank_err,
            float(np.max(np.abs(np.asarray(a.data) - np.asarray(b.data))))
            if a.data.size else 0.0,
            float(abs(int(a.size) - int(b.size))),
            float(abs(int(a.pos) - int(b.pos))),
        )

    # Two steps from the in-memory state and from the loaded state, both with a
    # freshly initialised optimizer. Optimizer state is not part of the
    # checkpoint contract (R10 says params/EMA/buffers), so using the same fresh
    # optimizer on both sides is what isolates save/load losslessness.
    optimizer = _make_optimizer(cfg)
    train_step = make_train_step(cfg, optimizer, joint=True)
    try:
        b1 = _sample_batch(data, shape, spans, cfg, B=1, N=seq_len)
        b2 = _sample_batch(data, shape, spans, cfg, B=1, N=seq_len)
    except Exception:
        b1 = b2 = None

    def two_steps(p, ema):
        if b1 is None:
            return p, ema, float("inf")
        tr, _, _ = partition_params(p)
        opt = optimizer.init(tr)
        rng = jax.random.PRNGKey(1234)
        for i, batch in enumerate((b1, b2)):
            rng, sub = jax.random.split(rng)
            p, opt, ema, _, _ = train_step(
                p, opt, ema, batch, 3e-4, sub, self_state=None,
                step=steps + i, routing_bias=jnp.zeros(cfg.n_lattices))
        return p, ema, 0.0

    p_a, ema_a, _ = two_steps(final_params, final_ema)
    p_b, ema_b, _ = two_steps(loaded_params, loaded_ema)
    two_step_params = _tree_delta(p_a, p_b)["max_abs"]
    two_step_ema = _tree_delta(ema_a, ema_b)["max_abs"]

    # Old-format rejection: mutate only a copy of the pickle.
    old_rejected = False
    old_error = None
    with tempfile.TemporaryDirectory(prefix="old_ckpt_", dir=str(Path(final_dir).parent)) as tmp:
        src = Path(final_dir) / "cog_params.pkl"
        dst = Path(tmp) / "cog_params.pkl"
        with open(src, "rb") as f:
            payload = __import__("pickle").load(f)
        payload["format_version"] = 1
        with open(dst, "wb") as f:
            __import__("pickle").dump(payload, f)
        try:
            load_cog_state(tmp, require_buffers=True)
        except ValueError as exc:
            old_rejected = True
            old_error = str(exc)

    return {
        "params_roundtrip_max_abs_err": float(params_err),
        "ema_roundtrip_max_abs_err": float(ema_err),
        "bank_roundtrip_max_abs_err": float(bank_err),
        "two_step_params_max_abs_err": float(two_step_params),
        "two_step_ema_max_abs_err": float(two_step_ema),
        "old_checkpoint_rejected": bool(old_rejected),
        "old_checkpoint_error": old_error,
        "optimizer_state_note": (
            "Optimizer moments are deliberately not part of the v2 checkpoint "
            "(R10: params/EMA/buffers). Both sides use a fresh optimizer, so "
            "the comparison tests state round-trip, not AdamW continuity."),
        "pass": bool(
            params_err < 1e-6 and ema_err < 1e-6 and bank_err < 1e-6
            and two_step_params < 1e-6 and two_step_ema < 1e-6
            and old_rejected),
    }


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------

def run_smoke(*, steps, data, shape, spans, output,
              batch=1, seq_len=64, preset="full", passive_only=True,
              tokenizer_path=None, max_inference_steps=None, lr=3e-4,
              joint=True, self_awareness=True, qwen_ckpt=None,
              save_every=None, progress=True, overfit_one_batch=True):
    """Run the smoke and return the report dict. Writes JSON + Markdown."""
    if passive_only and qwen_ckpt:
        raise ValueError("--passive-only and a Qwen checkpoint are exclusive")
    if not passive_only and qwen_ckpt is None:
        raise ValueError("active channel requested without --qwen-ckpt")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = smoke_config(preset, max_inference_steps=max_inference_steps)

    t_start = time.time()
    spec, meta = resolve_dataset_identity(
        data, shape, spans, full_verify=True, tokenizer_path=tokenizer_path)
    cfg = dataclasses.replace(cfg, vocab_size=int(meta["model_vocab_size"]))
    run_identity = RunIdentity(spec, meta)

    # Recreate the exact initial params/EMA that train_cog will build. The init
    # seed is a documented constant (PRNGKey(42), split once) in train_cog.
    init_rng = jax.random.split(jax.random.PRNGKey(42))[1]
    init_params, _ = init_cog_params(
        cfg, init_rng, qwen_ckpt=qwen_ckpt, resume=None)
    init_ema = init_ema_state(init_params, cfg)

    tracker = _UtilTracker()
    losses = []
    step_times_ms = []
    callback_times = []
    is_safe_steps = []
    soft_masks = []
    self_bias_steps = []
    final = {"params": None, "ema": None, "feature_bank": None,
             "contrast_bank": None, "contrast_last_used": None, "aux": None,
             "loss": None}

    def _track(name, indices, M):
        arr = _np(indices)
        if arr.ndim == 0:
            tracker.add(name, 0, arr, M)
        elif arr.ndim == 1:
            tracker.add(name, 0, arr, M)
        else:
            # (B, layers) is not used by the current forwards; keep it honest.
            for layer in range(arr.shape[-1]):
                tracker.add(name, layer, arr[..., layer], M)

    def on_step(abs_step, loss, aux, params, ema_state, feature_bank,
                contrast_bank, contrast_last_used, step_time_ms):
        losses.append(float(loss))
        step_times_ms.append(float(step_time_ms))
        callback_times.append(time.perf_counter())
        is_safe_steps.append(bool(_np(aux.get("is_safe", True))))
        mask = _np(aux.get("soft_mask"))
        if mask.ndim >= 1 and mask.size:
            soft_masks.append(mask.reshape(-1) if mask.ndim == 1 else mask[0])
        if "self_bias" in aux:
            self_bias_steps.append(float(_np(aux["self_bias"])))
        ls = aux.get("loop_stats") or {}
        if ls:
            _track("hrq", ls["hrq_idx"],
                   int(np.asarray(params["hrq"]["top"]["A"]).shape[0]))
            _track("sparse", ls["sparse_idx"],
                   int(np.asarray(params["sparse"]["C"]).shape[0]))
            _track("manifold", ls["man_idx"],
                   int(np.asarray(params["manifold"]["C"]).shape[0]))
            if "route_idx" in ls:
                _track("route", ls["route_idx"],
                       int(np.asarray(params["route"]["C_route"]).shape[0]))
            M_lr = int(np.asarray(params["lowrank"]["U"][0]).shape[0])
            for layer, idx in enumerate(
                    ls.get("lowrank_aux", {}).get("indices", [])):
                _track("lowrank", idx, M_lr) if layer == 0 else \
                    tracker.add("lowrank", layer, _np(idx), M_lr)
            M_bind = int(np.asarray(
                params["binding"]["key_cb"][0]["A"]).shape[0])
            for family in ("key", "val", "bind"):
                for layer, idx in enumerate(
                        ls.get("binding_indices", {}).get(family, [])):
                    tracker.add(f"binding.{family}", layer, _np(idx), M_bind)
            M_ct = int(np.asarray(
                params["contrast"]["C_a"][0]["A"]).shape[0])
            for family in ("a", "b"):
                for layer, idx in enumerate(
                        ls.get("contrast_indices", {}).get(family, [])):
                    tracker.add(f"contrast.{family}", layer, _np(idx), M_ct)
            if "self_idx" in ls:
                _track("self", ls["self_idx"],
                       int(np.asarray(params["self"]["C_self"]).shape[0]))
        final.update(params=params, ema=ema_state,
                     feature_bank=feature_bank, contrast_bank=contrast_bank,
                     contrast_last_used=contrast_last_used, aux=aux,
                     loss=float(loss))

    train_cog(
        cfg, str(output), steps=int(steps), lr=lr, batch_size=int(batch),
        seq_len=int(seq_len), log_every=max(1, int(steps)),
        save_every=(int(save_every) if save_every else int(steps)),
        data_path=data, shape_path=shape, spans_path=spans,
        qwen_ckpt=qwen_ckpt, resume=None, joint=joint,
        auto_mode=False, full_verify=True, allow_missing_self=False,
        self_awareness=self_awareness, verifier=None, causal_subject=None,
        tokenizer_path=tokenizer_path, on_step=on_step,
        overfit_one_batch=overfit_one_batch)

    final_dir = output / f"step_{int(steps):06d}"
    if final["params"] is None:
        loaded = load_cog_state(str(final_dir), require_buffers=True)
        final["params"] = _to_jax_tree(loaded["params"])
        final["ema"] = _to_jax_tree(loaded["ema_state"])
        final["feature_bank"] = loaded["feature_bank"]
        final["contrast_bank"] = loaded["contrast_bank"]
        final["contrast_last_used"] = loaded["contrast_last_used"]
        final["aux"] = {}

    fp = final["params"]
    fe = final["ema"]
    elapsed = time.time() - t_start

    # ── Loss ────────────────────────────────────────────────────────────────
    n_eval = min(10, len(losses))
    first_mean = float(np.mean(losses[:n_eval])) if n_eval else None
    last_mean = float(np.mean(losses[-n_eval:])) if n_eval else None
    ratio = (last_mean / first_mean) if first_mean else None
    loss_report = {
        "per_step": [float(x) for x in losses],
        "first_10_mean": first_mean,
        "last_10_mean": last_mean,
        "ratio": ratio,
        "all_finite": bool(all(math.isfinite(x) for x in losses)),
        "decreased_below_0.8": (bool(ratio < 0.8)
                                if (ratio is not None and len(losses) >= 10)
                                else None),
    }

    # ── Lattice updates / utilisation / variance ────────────────────────────
    var0 = _codebook_variances(init_params)
    var1 = _codebook_variances(fp)
    util = tracker.summary()

    lattice_updates = {}
    for key in ("encoder", "hrq", "sparse", "lowrank", "manifold",
                "binding", "contrast", "route", "fusion", "self",
                "value_scalars"):
        a = init_params.get(key)
        b = fp.get(key)
        delta = _tree_delta(a, b) if (a is not None and b is not None) else {
            "max_abs": 0.0, "l2": 0.0, "n_leaves": 0, "n_changed": 0,
            "errors": ["missing"]}
        entry = {
            "param_delta_max_abs": float(delta["max_abs"]),
            "param_delta_l2": float(delta["l2"]),
            "leaves_changed": int(delta["n_changed"]),
            "leaves_total": int(delta["n_leaves"]),
            "updated_gt_1e-6": bool(delta["max_abs"] > 1e-6),
            "delta_errors": delta["errors"],
        }
        for cb_name in (key, f"{key}_top", f"{key}_fine"):
            if cb_name in var0:
                ratio_v = (var1.get(cb_name, 0.0)
                           / var0[cb_name]) if var0[cb_name] > 0 else None
                entry["codebook_variance_ratio"] = ratio_v
                entry["codebook_variance_ok"] = (
                    bool(ratio_v is not None and ratio_v >= 0.5))
                break
        lattice_updates[key] = entry

    # Attach per-family utilisation to the parent lattice and evaluate the
    # design's "at least 10% of codewords selected once in 100 steps" check.
    # At batch=1 a lattice can select at most 100 rows, so 10% of a 512-row
    # codebook is 51 distinct rows; this is a real requirement, not a
    # formality, and it can fail.
    utilization_per_lattice = {}
    for name in ("hrq", "sparse", "lowrank", "manifold", "binding",
                 "contrast", "route", "self"):
        lattice_updates.setdefault(name, {})
        if name in util:
            lattice_updates[name]["utilization"] = util[name]
            fraction = util[name]["mean_fraction"]
        else:
            families = [v for k, v in util.items() if k.startswith(name + ".")]
            if families:
                fraction = float(np.mean(
                    [f["mean_fraction"] for f in families]))
                lattice_updates[name]["utilization"] = {
                    "mean_fraction": fraction,
                    "families": {k: v for k, v in util.items()
                                 if k.startswith(name + ".")},
                }
            else:
                fraction = None
        utilization_per_lattice[name] = {
            "fraction": fraction,
            "threshold": 0.10,
            "pass": bool(fraction is not None and fraction >= 0.10),
        }
    codebook_utilization = {
        "threshold": 0.10,
        "per_lattice": utilization_per_lattice,
        "pass": bool(all(v["pass"] for v in utilization_per_lattice.values())),
        "note": ("Design §6.3 A: at least 10% of each codebook selected once "
                 "in 100 steps. Batch=1 gives at most 100 selections per "
                 "lattice, so this is the strictest possible reading."),
    }

    # ── EMA ─────────────────────────────────────────────────────────────────
    ema = {}
    for name in ("sparse", "manifold", "self"):
        if name in init_ema and name in fe:
            ema[name] = _ema_branch_summary(init_ema[name], fe[name])
    if "binding" in init_ema and "binding" in fe:
        ema["binding"] = _ema_binding_summary(
            init_ema["binding"], fe["binding"])
    ema["pass"] = bool(ema) and all(v["changed"] for v in ema.values())

    # ── Route baseline ──────────────────────────────────────────────────────
    route_actual = _tree_delta(init_params["route"], fp["route"])
    route_baseline = _route_baseline_delta(init_params, cfg, steps, lr)
    route = {
        "C_route_delta_max_abs": float(route_actual["max_abs"]),
        "C_route_delta_l2": float(route_actual["l2"]),
        "weight_decay_only_delta_max_abs": float(route_baseline["max_abs"]),
        "weight_decay_only_delta_l2": float(route_baseline["l2"]),
        "beats_weight_decay_baseline": bool(
            route_actual["max_abs"] > route_baseline["max_abs"]),
        "note": ("Baseline runs the same AdamW/cosine schedule with zero "
                 "gradients for the same number of steps."),
    }

    # ── Sparse inference / LFQ ──────────────────────────────────────────────
    sparse_report = {}
    try:
        z_batch = _encode_z(fp, _sample_batch(
            data, shape, spans, cfg, B=min(16, 8), N=seq_len)[0], cfg)
        hrq_out, hrq_idx, hrq_sim, _ = hrq_forward(
            fp["hrq"], z_batch, cfg.tau_route_fallback,
            value_scalars=None, alpha_val=0.0)
        c_top = hrq_top_prototype(fp["hrq"], hrq_idx)
        _, sparse_idx = sparse_forward(
            fp["sparse"], z_batch, training=False,
            lambda_lfq=cfg.lambda_lfq, d_top=hrq_sim, c_top=c_top,
            value_scalars=None, alpha_val=0.0)
        idx = _np(sparse_idx)
        sparse_report["inference_nonzero_fraction"] = float(
            np.mean(idx != 0)) if idx.size else None
        sparse_report["inference_nonzero_count"] = int(np.count_nonzero(idx != 0))
        sparse_report["inference_samples"] = int(idx.size)

        C_sparse = _np(fp["sparse"]["C"])
        near_row = int(np.argmax(np.linalg.norm(C_sparse, axis=-1)))
        z_near = jnp.asarray(C_sparse[near_row][None, :])
        _, near_idx, near_sim, _ = hrq_forward(
            fp["hrq"], z_near, cfg.tau_route_fallback,
            value_scalars=None, alpha_val=0.0)
        near_c = hrq_top_prototype(fp["hrq"], near_idx)
        _, near_sparse = sparse_forward(
            fp["sparse"], z_near, training=False,
            lambda_lfq=cfg.lambda_lfq, d_top=near_sim, c_top=near_c,
            value_scalars=None, alpha_val=0.0)
        sparse_report["near_lfq_nonzero"] = bool(_np(near_sparse)[0] != 0)

        # Far: an HRQ top prototype whose direction is farthest from all sparse
        # codes. d_top is then ~0 and any non-zero d_min must force zero_vec.
        top_cb = (_np(fp["hrq"]["top"]["A"]) @ _np(fp["hrq"]["top"]["W"]))
        sparse_cb = C_sparse
        best_far, best_dist = 0, -1.0
        for r in range(min(top_cb.shape[0], 64)):
            zr = top_cb[r:r + 1].astype(np.float32)
            zr_P = _np(exp_map(safe_unit(jnp.asarray(zr))))
            dmin = float(np.min(poincare_similarity(
                jnp.asarray(zr_P)[:, None, :],
                exp_map(jnp.asarray(sparse_cb))[None, :, :])))
            if dmin > best_dist:
                best_dist, best_far = dmin, r
        z_far = jnp.asarray(top_cb[best_far][None, :])
        _, far_idx, far_sim, _ = hrq_forward(
            fp["hrq"], z_far, cfg.tau_route_fallback,
            value_scalars=None, alpha_val=0.0)
        far_c = hrq_top_prototype(fp["hrq"], far_idx)
        _, far_sparse = sparse_forward(
            fp["sparse"], z_far, training=False,
            lambda_lfq=cfg.lambda_lfq, d_top=far_sim, c_top=far_c,
            value_scalars=None, alpha_val=0.0)
        sparse_report["far_lfq_is_zero"] = bool(_np(far_sparse)[0] == 0)
        sparse_report["far_d_top"] = float(_np(far_sim)[0])
        sparse_report["far_d_min"] = float(best_dist)

        _, no_c_idx = sparse_forward(
            fp["sparse"], z_near, training=False,
            lambda_lfq=cfg.lambda_lfq, d_top=None, c_top=None,
            value_scalars=None, alpha_val=0.0)
        sparse_report["lfq_disabled_without_c_top"] = bool(
            _np(no_c_idx)[0] != 0)
        sparse_report["lambda_lfq"] = float(cfg.lambda_lfq)
    except Exception as exc:  # pragma: no cover - diagnostic
        sparse_report["error"] = f"{type(exc).__name__}: {exc}"

    # d_top shape / per-sample variation.
    try:
        z2 = _encode_z(fp, _sample_batch(
            data, shape, spans, cfg, B=2, N=seq_len)[0], cfg)
        _, _, sims2, _ = hrq_forward(
            fp["hrq"], z2, cfg.tau_route_fallback,
            value_scalars=None, alpha_val=0.0)
        sparse_report["d_top_shape"] = list(np.shape(_np(sims2)))
        sparse_report["d_top_differs_across_samples"] = bool(
            abs(float(_np(sims2)[0]) - float(_np(sims2)[1])) > 1e-8)
    except Exception as exc:  # pragma: no cover
        sparse_report["d_top_error"] = f"{type(exc).__name__}: {exc}"

    sparse_report["pass"] = bool(
        sparse_report.get("inference_nonzero_fraction", 0) > 0
        and sparse_report.get("near_lfq_nonzero")
        and sparse_report.get("far_lfq_is_zero")
        and sparse_report.get("lfq_disabled_without_c_top")
        and sparse_report.get("d_top_shape", [0]) == [2]
        and sparse_report.get("d_top_differs_across_samples")
    )

    # ── HRQ routing invariants ──────────────────────────────────────────────
    hrq_route = {}
    try:
        z4 = _encode_z(fp, _sample_batch(
            data, shape, spans, cfg, B=min(4, 8), N=seq_len)[0], cfg)
        M_top = int(np.asarray(fp["hrq"]["top"]["A"]).shape[0])
        v0 = jnp.zeros((M_top,), dtype=jnp.float32)
        _, idx0, _, _ = hrq_forward(
            fp["hrq"], z4, cfg.tau_route_fallback, value_scalars=v0,
            alpha_val=cfg.alpha_val, value_topk=cfg.hrq_value_topk)
        sims = _np(_hrq_sims(fp["hrq"], z4)[2])
        pure = np.argmin(sims, axis=-1)
        k = min(int(cfg.hrq_value_topk), M_top)
        topk = np.argsort(sims, axis=-1)[:, :k]
        v_probe = np.zeros(M_top, np.float32)
        if k > 1:
            v_probe[topk[:, 1]] = 10.0
        else:
            v_probe[topk[:, 0]] = 10.0
        _, idx_val, _, _ = hrq_forward(
            fp["hrq"], z4, cfg.tau_route_fallback,
            value_scalars=jnp.asarray(v_probe), alpha_val=10.0,
            value_topk=k)
        idx_val_np = _np(idx_val)
        in_topk = all(int(idx_val_np[i]) in set(int(x) for x in topk[i])
                      for i in range(idx_val_np.size))
        hrq_route = {
            "v0_equals_pure_poincare": bool(np.array_equal(_np(idx0), pure)),
            "value_top1_within_topk": bool(in_topk),
            "value_changed_top1": bool(
                not np.array_equal(_np(idx0), idx_val_np)),
            "probe_alpha": 10.0,
            "value_topk": k,
        }
        hrq_route["pass"] = bool(
            hrq_route["v0_equals_pure_poincare"]
            and hrq_route["value_top1_within_topk"])
    except Exception as exc:  # pragma: no cover
        hrq_route["error"] = f"{type(exc).__name__}: {exc}"

    # ── Entropy gate ────────────────────────────────────────────────────────
    entropy_report = {}
    try:
        d = cfg.d_model
        rng_np = np.random.default_rng(0)
        outs = [jnp.asarray(rng_np.standard_normal((1, d)).astype(np.float32))
                for _ in range(cfg.n_lattices)]
        peaked = np.asarray(
            [[0.99] + [0.002] * (cfg.n_lattices - 1)], np.float32)
        uniform = np.full((1, cfg.n_lattices),
                          1.0 / cfg.n_lattices, np.float32)
        _, w_peaked, _ = fuse_lattices_with_aux(
            outs, jnp.asarray(peaked), fp["fusion"])
        _, w_uniform, _ = fuse_lattices_with_aux(
            outs, jnp.asarray(uniform), fp["fusion"])
        h_peaked, n_peaked = _normalised_entropy(_np(w_peaked))
        h_uniform, n_uniform = _normalised_entropy(_np(w_uniform))
        entropy_report = {
            "peaked_H_norm": h_peaked,
            "peaked_n_active": n_peaked,
            "uniform_H_norm": h_uniform,
            "uniform_n_active": n_uniform,
            "peaked_passes_0.5": bool(h_peaked < 0.5),
            "uniform_fails_0.5": bool(h_uniform >= 0.5),
            "threshold": 0.5,
            "training_convergence_uses_raw_nats_threshold": float(
                cfg.entropy_threshold),
        }
        if soft_masks:
            real_mask = jnp.asarray(soft_masks[-1][None, :])
            _, w_real, _ = fuse_lattices_with_aux(
                outs, real_mask, fp["fusion"])
            h_real, n_real = _normalised_entropy(_np(w_real))
            entropy_report.update({
                "real_mask_H_norm": h_real,
                "real_mask_n_active": n_real,
                "real_mask_passes_0.5": bool(h_real < 0.5),
            })
        entropy_report["pass"] = bool(
            entropy_report["peaked_passes_0.5"]
            and entropy_report["uniform_fails_0.5"])
    except Exception as exc:  # pragma: no cover
        entropy_report["error"] = f"{type(exc).__name__}: {exc}"

    # ── Feature bank ────────────────────────────────────────────────────────
    bank_report = {}
    try:
        bank = FeatureBank(16, 8, seed=3)
        bank_vectors = np.asarray([
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1.0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1.0, 0, 0, 0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ], np.float32)
        bank.add(bank_vectors)
        C_dead = np.tile(bank_vectors[-1] + 100.0, (5, 1)).astype(np.float32)
        last = np.zeros(5, np.int32)
        C_new, last_new = bank.replace_dead_contrast(C_dead, last, 10, 5)
        replaced = int(np.count_nonzero(np.any(C_new != C_dead, axis=1)))
        expected = bank_vectors[
            int(np.argmax(np.linalg.norm(bank_vectors - C_dead[0], axis=1)))]
        farthest_correct = bool(np.allclose(C_new[0], expected))
        contrast_ok = bool(replaced == 5 and farthest_correct
                           and np.all(last_new == 10))

        sparse_bank = FeatureBank(16, 8, seed=4)
        sparse_bank.add(bank_vectors)
        state = {
            "N": np.ones(5, np.float32),
            "m": np.ones((5, 8), np.float32),
            "last_used": np.zeros(5, np.int32),
        }
        C2, state2 = sparse_bank.replace_dead_sparse(
            C_dead, state, 10, 5)
        replaced_s = int(np.count_nonzero(np.any(C2 != C_dead, axis=1)))
        cold = bool(np.all(state2["N"] == 0)
                    and np.all(state2["m"] == 0)
                    and np.all(state2["last_used"] == 10)
                    and np.all(state2["N"] + state2["m"].sum(axis=1) == 0))
        sparse_ok = bool(replaced_s == 5 and cold)
        bank_report = {
            "contrast_dead_injected": 5,
            "contrast_replaced": replaced,
            "contrast_used_farthest_point": farthest_correct,
            "contrast_last_used_updated": bool(np.all(last_new == 10)),
            "sparse_dead_injected": 5,
            "sparse_replaced": replaced_s,
            "sparse_cold_start": cold,
            "pass": bool(contrast_ok and sparse_ok),
        }
    except Exception as exc:  # pragma: no cover
        bank_report["error"] = f"{type(exc).__name__}: {exc}"

    # ── Safety ──────────────────────────────────────────────────────────────
    so_path = Path(__file__).resolve().parent.parent / "infer" / "liblcm.so"
    safety = {
        "training_is_safe_all_steps": bool(
            is_safe_steps and all(is_safe_steps)),
        "training_is_safe_fraction": (float(np.mean(is_safe_steps))
                                      if is_safe_steps else None),
        "structural_placeholder_note": (
            "gvalue/danger are structural placeholders built from fixed "
            "anchors and the first batch's z. They exercise the trigger chain; "
            "they do not establish semantic safety."),
    }
    if so_path.exists():
        try:
            safety["c_danger_probe"] = _c_danger_probe(so_path)
        except Exception as exc:  # pragma: no cover
            safety["c_danger_probe"] = {
                "available": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        safety["c_danger_probe"] = {
            "available": False, "error": f"{so_path} not built"}
    try:
        safety["python_no_token_probe"] = _python_no_token_probe()
    except Exception as exc:  # pragma: no cover
        safety["python_no_token_probe"] = {
            "error": f"{type(exc).__name__}: {exc}"}
    try:
        safety["gvalue"] = _gvalue_probe()
    except Exception as exc:  # pragma: no cover
        safety["gvalue"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        safety["export"] = _exported_safety_probe(str(final_dir))
    except Exception as exc:  # pragma: no cover
        safety["export"] = {"error": f"{type(exc).__name__}: {exc}"}
    c_probe = safety.get("c_danger_probe", {})
    no_token = safety.get("python_no_token_probe", {})
    export = safety.get("export", {})
    gv = safety.get("gvalue", {})
    safety["pass"] = bool(
        safety["training_is_safe_all_steps"]
        and c_probe.get("threat_blocked")
        and c_probe.get("neutral_blocked") is False
        and c_probe.get("constructed_codebooks_distinct")
        and c_probe.get("hash_stable")
        and no_token.get("no_token")
        and gv.get("pos_neg_distinct")
        and gv.get("strict_neutral_violates")
        and gv.get("verify_integrity")
        and export.get("gvalue_pos_neg_distinct")
        and export.get("danger_threat_normal_distinct")
    )

    # ── Self bias ───────────────────────────────────────────────────────────
    self_report = {}
    try:
        C_self = np.asarray(fp["self"]["C_self"])
        familiar_row = 1 if C_self.shape[0] > 1 else 0
        z_pair = jnp.asarray(np.stack([
            C_self[familiar_row],
            -C_self[familiar_row],
        ]).astype(np.float32))
        _, bias, aux_self = self_lattice_forward(
            fp["self"], z_pair, training=False, return_aux=True,
            alpha_self=cfg.alpha_self)
        b = _np(bias)
        self_report = {
            "alpha_self": float(cfg.alpha_self),
            "bias_min": float(np.min(b)),
            "bias_max": float(np.max(b)),
            "bias_familiar": float(b[0]),
            "bias_unfamiliar": float(b[1]),
            "in_range": bool(np.all(b >= -1e-6)
                             and np.all(b <= cfg.alpha_self + 1e-6)),
            "varies_with_z": bool(abs(float(b[0]) - float(b[1])) > 1e-4),
            "train_step_self_bias_min": (min(self_bias_steps)
                                         if self_bias_steps else None),
            "train_step_self_bias_max": (max(self_bias_steps)
                                         if self_bias_steps else None),
        }
        self_report["pass"] = bool(
            self_report["in_range"] and self_report["varies_with_z"])
    except Exception as exc:  # pragma: no cover
        self_report["error"] = f"{type(exc).__name__}: {exc}"

    # ── Self-observation overhead ───────────────────────────────────────────
    overhead = {}
    try:
        # train_cog's step_time_ms measures jitted dispatch, not device
        # completion (JAX is asynchronous). Use the delta between consecutive
        # post-step callbacks instead: each callback starts only after the
        # previous callback's device syncs, so the delta is real wall-clock for
        # one complete step + observation + bank/host work.
        intervals_ms = (np.diff(np.asarray(callback_times)) * 1000.0
                        if len(callback_times) >= 2 else np.array([]))
        step_median_ms = (float(np.median(intervals_ms))
                          if intervals_ms.size else None)
        dispatch_median_ms = (float(np.median(
            [t for t in step_times_ms if t > 0]))
            if any(t > 0 for t in step_times_ms) else None)
        obs = ObservabilityRecorder(
            raw_capacity=1000, long_term_capacity=10000,
            record_every_n=1, long_term_every=100)
        narr = NarrativeMemory(max_records=1000)
        reflection = ReflectionLoop(
            interval=200, log_path=str(output / "reflection.log"))
        explorer = BehaviorExplorer()
        aux_obs = final.get("aux") or {}
        obs_times = []
        for i in range(20):
            t0 = time.perf_counter()
            _self_observe_step(
                obs, narr, reflection, explorer, None,
                int(steps) + i, 0.0, aux_obs, float(final["loss"] or 0.0),
                cfg)
            obs_times.append(time.perf_counter() - t0)
        obs_median_s = float(np.median(obs_times))
        frac = None
        if step_median_ms and step_median_ms > 0:
            frac = obs_median_s / (step_median_ms / 1000.0)
        overhead = {
            "runs": 20,
            "observation_median_ms": obs_median_s * 1000.0,
            "training_step_median_ms": step_median_ms,
            "dispatch_median_ms_reference": dispatch_median_ms,
            "overhead_fraction": frac,
            "under_5pct": (bool(frac is not None and frac < 0.05)),
            "method": ("20-run median wall-clock of the real "
                       "_self_observe_step path divided by the median "
                       "callback-to-callback training-step interval. The "
                       "latter is device-synced wall-clock; train_cog's own "
                       "step_time_ms is only dispatch time. The first JIT "
                       "step is excluded because its callback precedes any "
                       "steady-state interval."),
        }
        try:
            reflection.wait(timeout=10.0)
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        overhead["error"] = f"{type(exc).__name__}: {exc}"

    # ── Persistence ─────────────────────────────────────────────────────────
    try:
        persistence = _persistence_probe(
            final_dir, fp, fe, final, cfg, data, shape, spans, steps, seq_len)
    except Exception as exc:  # pragma: no cover
        persistence = {"pass": False,
                       "error": f"{type(exc).__name__}: {exc}"}

    # ── Checks summary ──────────────────────────────────────────────────────
    checks = {
        "loss_all_finite": loss_report["all_finite"],
        "loss_decreased": loss_report["decreased_below_0.8"],
        "all_lattices_updated": bool(all(
            v.get("updated_gt_1e-6", False)
            for k, v in lattice_updates.items()
            if k in ("encoder", "hrq", "sparse", "lowrank", "manifold",
                     "binding", "contrast", "route", "fusion", "self",
                     "value_scalars"))),
        "ema_changed": ema.get("pass", False),
        "codebook_utilization_10pct": codebook_utilization["pass"],
        "route_beats_weight_decay": route["beats_weight_decay_baseline"],
        "sparse_lfq": sparse_report.get("pass", False),
        "hrq_value_routing": hrq_route.get("pass", False),
        "entropy_gate": entropy_report.get("pass", False),
        "feature_bank": bank_report.get("pass", False),
        "safety_chain": safety.get("pass", False),
        "self_bias": self_report.get("pass", False),
        "overhead_under_5pct": overhead.get("under_5pct", False),
        "persistence": persistence.get("pass", False),
    }

    report = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "steps": int(steps),
            "batch": int(batch),
            "seq_len": int(seq_len),
            "preset": preset,
            "joint": bool(joint),
            "self_awareness": bool(self_awareness),
            "passive_only": bool(passive_only),
            "overfit_one_batch": bool(overfit_one_batch),
            "qwen": qwen_ckpt,
            "lr": float(lr),
            "elapsed_sec": float(elapsed),
            "config": dataclasses.asdict(cfg),
            "data": {
                "data_path": os.path.abspath(data),
                "shape_path": os.path.abspath(shape),
                "spans_path": os.path.abspath(spans),
                "tokenizer_path": spec.path,
                "tokenizer_id": spec.tokenizer_id,
                "tokenizer_sha256": spec.sha256,
                "token_data_sha256": meta["token_data_sha256"],
                "document_spans_sha256": meta["document_spans_sha256"],
                "n_tokens": int(meta["n_tokens"]),
                "n_docs": int(meta["n_docs"]),
                "separator_id": int(meta["separator_id"]),
                "model_vocab_size": int(meta["model_vocab_size"]),
            },
            "output_dir": str(output),
            "final_checkpoint": str(final_dir),
            "report_json": str(output / "smoke_report.json"),
            "report_md": str(output / "smoke_report.md"),
        },
        "loss": loss_report,
        "lattice_updates": lattice_updates,
        "codebook_utilization": codebook_utilization,
        "ema": ema,
        "route": route,
        "sparse_inference": sparse_report,
        "hrq_routing": hrq_route,
        "entropy_gate": entropy_report,
        "feature_bank": bank_report,
        "safety": safety,
        "self_lattice": self_report,
        "overhead": overhead,
        "persistence": persistence,
        "checks": checks,
    }

    json_path = output / "smoke_report.json"
    with open(json_path, "w") as f:
        json.dump(_jsonable(report), f, indent=2, sort_keys=False)
    md_path = output / "smoke_report.md"
    md_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "non-finite"
        return f"{x:.{nd}f}"
    return str(x)


def _markdown(report):
    m = report["meta"]
    loss = report["loss"]
    lines = []
    lines.append("# Lattice conformance smoke report\n")
    lines.append(
        "**This is a conformance smoke, not a language-quality experiment.** "
        "100 steps prove that the update/safety paths fire and stay finite; "
        "they do not prove language ability.\n")
    lines.append("## Environment\n")
    lines.append(f"- git: `{m['git_commit']}`")
    lines.append(f"- platform: {m['platform']}")
    lines.append(f"- python {m['python']}, JAX {m['jax']}")
    lines.append(
        f"- steps={m['steps']} batch={m['batch']} seq={m['seq_len']} "
        f"joint={m['joint']} self_awareness={m['self_awareness']} "
        f"passive_only={m['passive_only']} preset={m['preset']} "
        f"overfit_one_batch={m.get('overfit_one_batch')}")
    lines.append(f"- elapsed: {m['elapsed_sec']:.1f}s")
    d = m["data"]
    lines.append(
        f"- data: `{d['data_path']}` ({d['n_tokens']} tokens, "
        f"{d['n_docs']} docs), tokenizer `{d['tokenizer_id']}` "
        f"`{d['tokenizer_sha256'][:16]}…`")
    lines.append("")
    lines.append("## Loss\n")
    lines.append(
        f"first-10 mean {_fmt(loss['first_10_mean'])} → last-10 mean "
        f"{_fmt(loss['last_10_mean'])} (ratio {_fmt(loss['ratio'])}), "
        f"finite={_fmt(loss['all_finite'])}, "
        f"<0.8={_fmt(loss['decreased_below_0.8'])}")
    lines.append("")
    lines.append("## Per-lattice updates\n")
    lines.append("| lattice | Δmax abs | updated>1e-6 | utilisation | var ratio |")
    lines.append("|---|---|---|---|---|")
    for key in ("encoder", "hrq", "sparse", "lowrank", "manifold",
                "binding", "contrast", "route", "fusion", "self",
                "value_scalars"):
        e = report["lattice_updates"].get(key, {})
        u = e.get("utilization", {})
        uf = u.get("mean_fraction") if isinstance(u, dict) else None
        lines.append(
            f"| {key} | {_fmt(e.get('param_delta_max_abs'))} | "
            f"{_fmt(e.get('updated_gt_1e-6'))} | "
            f"{_fmt(uf, 3)} | {_fmt(e.get('codebook_variance_ratio'), 3)} |")
    lines.append("")
    lines.append("## Codebook utilisation (design §6.3 A: ≥10%)\n")
    lines.append("| lattice | unique/M | pass |")
    lines.append("|---|---|---|")
    for name, e in report["codebook_utilization"]["per_lattice"].items():
        lines.append(
            f"| {name} | {_fmt(e.get('fraction'), 4)} | {_fmt(e.get('pass'))} |")
    lines.append("")
    lines.append("## EMA\n")
    lines.append("| branch | ΔN abs | rows used | Δm norm | changed |")
    lines.append("|---|---|---|---|---|")
    for name, e in report["ema"].items():
        if name == "pass":
            continue
        lines.append(
            f"| {name} | {_fmt(e['delta_N_abs'])} | "
            f"{e['rows_used_final']}/{e['rows_total']} | "
            f"{_fmt(e['delta_m_norm'])} | {_fmt(e['changed'])} |")
    lines.append("")
    lines.append("## Route\n")
    r = report["route"]
    lines.append(
        f"C_route Δmax {_fmt(r['C_route_delta_max_abs'])} vs weight-decay-only "
        f"{_fmt(r['weight_decay_only_delta_max_abs'])} → "
        f"{_fmt(r['beats_weight_decay_baseline'])}")
    lines.append("")
    lines.append("## Sparse / LFQ\n")
    sp = report["sparse_inference"]
    for key in ("inference_nonzero_fraction", "near_lfq_nonzero",
                "far_lfq_is_zero", "far_d_top", "far_d_min",
                "lfq_disabled_without_c_top", "d_top_shape",
                "d_top_differs_across_samples", "lambda_lfq"):
        if key in sp:
            lines.append(f"- {key}: {_fmt(sp[key])}")
    lines.append("")
    lines.append("## HRQ value routing\n")
    for key, value in report["hrq_routing"].items():
        if key != "pass":
            lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Entropy gate\n")
    for key, value in report["entropy_gate"].items():
        if key != "pass":
            lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Feature bank\n")
    for key, value in report["feature_bank"].items():
        if key != "pass":
            lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Safety\n")
    s = report["safety"]
    lines.append(
        f"- training is_safe all steps: "
        f"{_fmt(s.get('training_is_safe_all_steps'))}")
    cp = s.get("c_danger_probe", {})
    lines.append(
        f"- C danger probe: blocked={_fmt(cp.get('threat_blocked'))}, "
        f"neutral_blocked={_fmt(cp.get('neutral_blocked'))}, "
        f"source={cp.get('threat_source')}, hash_stable="
        f"{_fmt(cp.get('hash_stable'))}")
    nt = s.get("python_no_token_probe", {})
    lines.append(
        f"- Python abort: ret={nt.get('loop_ret')}, "
        f"tokens={nt.get('tokens_emitted')}, no_token="
        f"{_fmt(nt.get('no_token'))}")
    gv = s.get("gvalue", {})
    lines.append(
        f"- gvalue: pos≠neg={_fmt(gv.get('pos_neg_distinct'))}, "
        f"strict neutral violates={_fmt(gv.get('strict_neutral_violates'))}, "
        f"integrity={_fmt(gv.get('verify_integrity'))}")
    ex = s.get("export", {})
    lines.append(
        f"- exported: gvalue distinct={_fmt(ex.get('gvalue_pos_neg_distinct'))}, "
        f"danger distinct={_fmt(ex.get('danger_threat_normal_distinct'))}")
    lines.append(f"- {s.get('structural_placeholder_note', '')}")
    lines.append("")
    lines.append("## Self lattice\n")
    sl = report["self_lattice"]
    lines.append(
        f"- self_bias [{_fmt(sl.get('bias_min'))}, "
        f"{_fmt(sl.get('bias_max'))}] in [0, {_fmt(sl.get('alpha_self'))}]: "
        f"{_fmt(sl.get('in_range'))}; varies={_fmt(sl.get('varies_with_z'))}")
    lines.append("")
    lines.append("## Self-observation overhead\n")
    oh = report["overhead"]
    lines.append(
        f"- observation median {_fmt(oh.get('observation_median_ms'), 3)} ms; "
        f"train_step median {_fmt(oh.get('training_step_median_ms'), 3)} ms; "
        f"overhead {_fmt((oh.get('overhead_fraction') or 0) * 100, 2)}% "
        f"(<5%: {_fmt(oh.get('under_5pct'))})")
    lines.append(f"- {oh.get('method', '')}")
    lines.append("")
    lines.append("## Persistence\n")
    p = report["persistence"]
    for key in ("params_roundtrip_max_abs_err", "ema_roundtrip_max_abs_err",
                "bank_roundtrip_max_abs_err", "two_step_params_max_abs_err",
                "two_step_ema_max_abs_err", "old_checkpoint_rejected"):
        if key in p:
            lines.append(f"- {key}: {_fmt(p[key])}")
    lines.append(f"- {p.get('optimizer_state_note', '')}")
    lines.append("")
    lines.append("## Checks\n")
    lines.append("| check | result |")
    lines.append("|---|---|")
    for key, value in report["checks"].items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.append("")
    lines.append("## Known limitations\n")
    lines.append(
        "- 100 steps do not prove language ability; only conformance.\n"
        "- gvalue/danger codebooks are structural placeholders, not semantic "
        "safety anchors.\n"
        "- The entropy gate is verified on crafted softmasks; the real training "
        "mask result is reported above without being forced to pass.\n"
        "- JAX training still uses a raw-nats entropy threshold; R9's "
        "normalised gate is implemented in the C engine and in this probe.\n")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="≤100-step lattice conformance smoke")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--data", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--spans", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--preset", choices=("full", "tiny"), default="full")
    p.add_argument("--passive-only", action="store_true",
                   help="run without the Qwen active bridge (the 100-step "
                        "conformance configuration)")
    p.add_argument("--qwen-ckpt", default=None)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--max-inference-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-joint", action="store_true")
    p.add_argument("--no-self-awareness", action="store_true")
    p.add_argument("--save-every", type=int, default=None)
    args = p.parse_args(argv)

    report = run_smoke(
        steps=args.steps, data=args.data, shape=args.shape, spans=args.spans,
        output=args.output, batch=args.batch, seq_len=args.seq,
        preset=args.preset, passive_only=args.passive_only,
        tokenizer_path=args.tokenizer_path,
        max_inference_steps=args.max_inference_steps, lr=args.lr,
        joint=not args.no_joint,
        self_awareness=not args.no_self_awareness,
        qwen_ckpt=args.qwen_ckpt, save_every=args.save_every)
    checks = sum(1 for v in report["checks"].values() if v is True)
    total = len(report["checks"])
    print(f"[SMOKE] {checks}/{total} checks PASS")
    print(f"[SMOKE] report → {report['meta']['report_json']}")
    print(f"[SMOKE] table  → {report['meta']['report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
