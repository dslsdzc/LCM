"""Dual-channel cognitive training: passive introspection + active expression.

Two output channels from the same conscious state z_q:
  - Passive: z_q @ E.T — the tied encoder embedding, transparent readout
  - Active:  Qwen2.5-0.5B (frozen bridge; the only current active channel)

The passive channel keeps the model honest (cognitive state is directly readable).
The active channel is a frozen pretrained model the cognitive state must learn to
drive. Language LCM is retired and is not an alternative backend.

Usage (Stage 2):
    python lcm.py --cog-train -d zhwiki_qwen.dat --qwen-ckpt <qwen_params.npz>
"""
import dataclasses
import json
import os
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train.config import LCMConfig
from train.dataset_meta import (
    check_data_size, load_dataset_meta, token_dtype, validate_spans,
)
from train.tokenizer_spec import (
    TOKENIZER_REGISTRY, resolve_tokenizer_spec, sha256_file, tokenizer_path_for,
)
from train.encoder import init_encoder_params, encoder_forward
from train.hyp import safe_unit
from train.lattices import (
    init_hrq_params, init_sparse_params, init_lowrank_params,
    init_manifold_params, init_binding_params, init_contrast_params,
    init_route_params, init_value_scalars, init_danger_params,
)
from train.fusion import init_fusion_params
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_reg_loss,
    update_self_state,
)
from train.behavior_explorer import compute_tension_from_aux
from train.external_verifier import LOCK as VERIFIER_LOCK
from train.cog_loop import cog_loop_scan
from train.ema_state import (
    ema_update_binding, ema_update_manifold, ema_update_self,
    ema_update_sparse, init_ema_state,
)
from train.feature_bank import FeatureBank, init_contrast_last_used
from train.param_groups import merge_params, partition_params
from train.lattices import contrast_info_nce_loss, manifold_orth_loss
from train.losses import cognitive_vq_losses
from train.qwen_lm import qwen_forward, load_qwen_params, QWEN_CONFIG
from train.checkpoint import (
    FORMAT_VERSION, HEADER_FMT_V2, HEADER_SIZE_V2, save_manifest,
    _pack_header, _compute_checksum, load_cog_state, save_cog_state,
    _save_route, _save_fusion, _save_value_scalars, _save_self,
    _save_binding_proj,
)

# Global Qwen params cache (load once, reuse across calls)
_QWEN_PARAMS = None


# ─── Dataset / model identity enforcement ───────────────────────────────────
#
# Every check below raises. There is no warning path: no active-channel metric
# from a run whose token space cannot be verified is interpretable. The failure
# this guards against is silent — a 30k-token corpus full of ids below 151936
# looks entirely legal against a Qwen vocabulary, and the only thing that
# distinguishes it is the tokenizer's own digest.

@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """What a training run was: its tokenizer and its corpus.

    Constructed once at startup and handed to every checkpoint save, so no save
    path can guess a tokenizer from a global default.
    """
    tokenizer_spec: object      # TokenizerSpec
    dataset_meta: dict


def resolve_dataset_identity(data_path, shape_path, spans_path,
                             full_verify=False, tokenizer_path=None):
    """Verify the tokenizer, corpus, spans and metadata all agree.

    Returns (TokenizerSpec, metadata). Deliberately does not require a Qwen
    artifact: metadata is the authority for the tensor extent. Qwen is the
    current training bridge, but tokenizer/corpus identity is not owned by the
    bridge implementation.
    """
    meta = load_dataset_meta(shape_path)

    tokenizer_id = meta["tokenizer_id"]
    if tokenizer_id not in TOKENIZER_REGISTRY:
        raise ValueError(f"unknown tokenizer_id {tokenizer_id!r} in metadata")
    entry = TOKENIZER_REGISTRY[tokenizer_id]

    spec = resolve_tokenizer_spec(
        path=tokenizer_path or tokenizer_path_for(tokenizer_id),
        document_separator=entry["separator"],
        model_vocab_size=meta["model_vocab_size"],
        tokenizer_id=tokenizer_id,
    )

    if spec.sha256 != meta["tokenizer_sha256"]:
        raise ValueError("tokenizer file changed since the corpus was built")
    if spec.document_separator_id != meta["separator_id"]:
        raise ValueError(
            f"separator is {spec.document_separator_id} but metadata says "
            f"{meta['separator_id']}")
    if spec.max_token_id != meta["token_id_max"]:
        raise ValueError(
            f"tokenizer max id is {spec.max_token_id} but metadata says "
            f"{meta['token_id_max']}")
    if meta["dtype"] not in ("uint16", "uint32"):
        raise ValueError(f"unsupported token dtype {meta['dtype']!r}")
    if np.dtype(meta["dtype"]) != np.dtype(spec.dtype):
        raise ValueError(
            f"dataset dtype is {meta['dtype']} but tokenizer identity requires "
            f"{spec.dtype}")

    # Always on, and cheap: catches a truncated or replaced .dat. Runs after the
    # dtype identity check so a width drift is reported as identity drift rather
    # than merely as a byte-size mismatch.
    check_data_size(meta, data_path)

    if full_verify and sha256_file(data_path) != meta["token_data_sha256"]:
        raise ValueError(
            f"{data_path} does not match token_data_sha256; the corpus content "
            f"changed since this metadata was written")

    spans = validate_spans(spans_path, meta)

    tokens = np.memmap(data_path, dtype=token_dtype(meta), mode="r",
                       shape=(int(meta["n_tokens"]),))

    # Structure does not imply semantics: a builder that wrote wrong-but-
    # contiguous offsets passes every partition check above.
    if np.any(tokens[spans[:, 1] - 1] != meta["separator_id"]):
        raise ValueError(
            "some document span does not end on the separator; the boundary "
            "set is not what the metadata claims")

    if full_verify:
        n_sep = int(np.count_nonzero(tokens == meta["separator_id"]))
        if n_sep != int(meta["n_docs"]):
            raise ValueError(
                f"found {n_sep} separators but metadata claims "
                f"{meta['n_docs']} documents; the separator is not a unique "
                f"boundary marker")

    return spec, meta


def verify_tied_table(params, meta, cfg):
    """Assert the tied table was built at the metadata's extent.

    An assertion, not a derivation: the resume path can carry a checkpoint
    built under a different config.
    """
    E = params["encoder"]["embed"]
    want = (int(meta["model_vocab_size"]), cfg.d_model)
    if tuple(E.shape) != want:
        raise ValueError(
            f"tied table E has shape {tuple(E.shape)}, expected {want}")


def verify_qwen_extent(params, meta):
    """If the Qwen bridge is loaded, its tensor vocabulary must match metadata.

    Qwen is the only current cognitive active bridge. If it is absent the run is
    passive-only, and there is no Qwen extent to verify.
    """
    qwen = params.get("qwen")
    if qwen is None:
        return

    embed_vocab = int(qwen["model.embed_tokens.weight"].shape[0])
    if embed_vocab != int(meta["model_vocab_size"]):
        raise ValueError(
            f"Qwen embed_tokens has {embed_vocab} rows but the corpus was built "
            f"for model_vocab_size={meta['model_vocab_size']}")

    # Mirror train/qwen_lm.py: the head is weight-tied to the embedding when
    # lm_head.weight is absent. An lm_head-less checkpoint is valid.
    lm_weight = qwen.get("lm_head.weight", qwen["model.embed_tokens.weight"])
    logit_vocab = int(lm_weight.shape[0])
    if logit_vocab != int(meta["model_vocab_size"]):
        raise ValueError(
            f"Qwen lm_head has {logit_vocab} rows but the corpus was built for "
            f"model_vocab_size={meta['model_vocab_size']}")


# The fields that make a resume identity-preserving. The rest of the metadata
# (n_tokens, n_docs) is implied by these.
RESUME_IDENTITY_KEYS = (
    "tokenizer_sha256",
    "token_data_sha256",
    "document_spans_sha256",
    "model_vocab_size",
)


def verify_resume_identity(resume_dir, run_identity):
    """A resume must continue the dataset the checkpoint was fitted to.

    Dataset identity only. Optimizer state, RNG and scheduler position are not
    saved in the checkpoint, so this makes a resume identity-preserving, not
    state-exact.

    Resuming onto a different corpus is not a warm start with extra steps: the
    parameters were never fitted to that token space. Warm-starting from
    mismatched data is a separate, explicit operation and is not supported here.
    """
    path = os.path.join(resume_dir, "run_identity.json")
    if not os.path.exists(path):
        raise ValueError(
            f"{resume_dir} has no run identity; legacy cognitive checkpoints "
            f"cannot be resumed after tokenizer unification")

    with open(path) as f:
        stored = json.load(f)
    current = run_identity.dataset_meta

    missing = [k for k in RESUME_IDENTITY_KEYS if k not in stored]
    if missing:
        raise ValueError(f"{path} is missing identity keys: {missing}")

    stored_version = stored.get("format_version")
    if stored_version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: old checkpoint format; re-export required "
            f"(format_version={stored_version!r}, expected {FORMAT_VERSION})")

    for key in RESUME_IDENTITY_KEYS:
        if stored[key] != current[key]:
            raise ValueError(
                f"resume identity mismatch on {key}: checkpoint has "
                f"{stored[key]!r}, this run has {current[key]!r}; a resume must "
                f"continue the same dataset")


# ─── Load Stage 2 memory checkpoint ─────────────────────────────────────────

def _ensure_r15_self(params, cfg, rng, allow_missing_self):
    """Reject or explicitly reset a pre-R15 ``core + modes`` self lattice.

    Ruling 11: old weights are discarded, not migrated. A fresh ``C_self`` is
    acceptable for a new run, but only when the caller opts in with
    ``allow_missing_self=True``; silently starting from a random codebook in a
    resume is exactly the kind of mismatch this rejects.
    """
    d = int(cfg.d_model)
    M_self = int(cfg.n_self_codes)
    sp = params.get('self')
    valid = isinstance(sp, dict) and 'C_self' in sp and 'bias_self' in sp
    if valid:
        C = jnp.asarray(sp['C_self'])
        bias = jnp.asarray(sp['bias_self'])
        valid = (C.ndim == 2 and tuple(C.shape) == (M_self, d)
                 and bias.size == 1)
    if valid:
        # Drop pre-R15 leftovers if a hand-written checkpoint carries both.
        # Ruling 34: alpha_self is the frozen config scalar, so the checkpoint
        # mirror is refreshed from the current config rather than trained.
        params['self'] = {
            'C_self': jnp.asarray(sp['C_self']),
            'bias_self': jnp.asarray(cfg.alpha_self, dtype=jnp.float32),
        }
        return params

    if not allow_missing_self:
        raise ValueError(
            "checkpoint self lattice has no R15 C_self (found "
            f"{sorted(sp) if isinstance(sp, dict) else type(sp).__name__}); "
            "old core+modes weights are discarded, not migrated. Start a new "
            "run or pass allow_missing_self=True to reset C_self explicitly")
    params['self'] = init_self_params(
        jax.random.fold_in(rng, 13), d, M_self, alpha_self=cfg.alpha_self)
    print("[COG] R15: checkpoint had no C_self; reset to a fresh random "
          "self codebook (allow_missing_self=True)")
    return params


def load_stage2_params(resume, cfg, rng, allow_missing_self=False):
    """Load params from Stage 2 checkpoint (full pickle or legacy .bin format).

    Two formats:
      1. cog_params.pkl (new) — full params dict with self_state.
      2. .bin files (old) — loaded via checkpoint.load_checkpoint.

    R15: a checkpoint whose ``params['self']`` predates ``C_self`` is rejected
    unless ``allow_missing_self`` is explicit (Ruling 11: old weights are
    discarded, not migrated).

    Returns:
        (params, self_state, step) — the saved absolute step is the time origin
        of ``ema_state['last_used']`` and must be threaded back into training.
    """
    # Try new pickle format first (preserves self_state)
    pkl_path = os.path.join(resume, "cog_params.pkl")
    if os.path.exists(pkl_path):
        ckpt = load_cog_state(resume)
        params = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
            ckpt['params'])
        step = ckpt.get('step', 0)
        # Check W_out first: it is an even older, more specific schema break,
        # and a caller loading such a checkpoint should see that diagnosis
        # rather than the generic R15 self-lattice rejection.
        if 'W_out' in params:
            raise ValueError(
                "checkpoint carries W_out, which predates the tied readout; "
                "old cognitive checkpoints are not migrated")
        params = _ensure_r15_self(params, cfg, rng, allow_missing_self)
        self_state = ckpt.get('self_state')
        if self_state is None:
            self_state = init_self_state(cfg.n_self_codes, cfg.d_model)
        print(f"[COG] Loaded from cog_params.pkl (step {step})")
        return params, self_state, int(step)

    # Fallback: legacy .bin format — caller must supply --qwen-ckpt separately
    from train.checkpoint import load_checkpoint as bin_load
    loaded, _, _, step = bin_load(
        resume, cfg=cfg, rng=rng, load_opt=False,
        allow_missing_self=allow_missing_self)

    wanted = ['encoder', 'hrq', 'sparse', 'lowrank', 'manifold',
              'binding', 'contrast', 'self', 'gen_head']
    params = {k: loaded[k] for k in wanted if k in loaded}
    params = _ensure_r15_self(params, cfg, rng, allow_missing_self)

    self_state = init_self_state(cfg.n_self_codes, cfg.d_model)

    print(f"[COG] Loaded Stage 2 checkpoint from {resume} (step {step})")
    print(f"      encoder + {len([k for k in params if k not in ('gen_head',)])} codebooks + "
          f"{'gen_head' if 'gen_head' in params else 'no'} gen_head")
    return params, self_state, int(step)


def load_cog_buffers(resume, params, cfg, allow_missing_self=False):
    """Load EMA state, feature bank and the saved step for a resume.

    These are buffers, not parameters: a resume that cannot restore them would
    silently restart the EMA statistics/feature banks from zero, so the v2
    checkpoint is required to carry all four
    (``load_cog_state(require_buffers=True)``). The saved step is returned
    because restored ``last_used`` values are absolute step indices from the
    old run; the loop must continue from that origin, not from 0.

    R15: the EMA state must also carry the ``self`` branch (``C_self`` modes).
    A pre-R15 checkpoint has no such branch and is rejected unless
    ``allow_missing_self`` is explicit; then a fresh zeroed branch is created.

    Returns:
        ``(ema_state, feature_bank, contrast_bank, contrast_last_used, step)``.
    """
    if resume:
        pkl_path = os.path.join(resume, "cog_params.pkl")
        if not os.path.exists(pkl_path):
            raise ValueError(
                f"{resume} has no cog_params.pkl; only v2 cognitive "
                f"checkpoints with EMA/feature-bank buffers are resumable")
        ckpt = load_cog_state(resume, require_buffers=True)
        ema_state = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x), ckpt['ema_state'])
        ema_state = _ensure_r15_self_ema(
            ema_state, params, cfg, allow_missing_self)
        return (ema_state, ckpt['feature_bank'], ckpt['contrast_bank'],
                ckpt['contrast_last_used'], int(ckpt.get('step', 0)))

    return (init_ema_state(params, cfg),
            FeatureBank(cfg.bank_capacity, cfg.d_model),
            FeatureBank(cfg.bank_capacity, cfg.d_model),
            init_contrast_last_used(cfg.n_contrast_layers, cfg.M_contrast), 0)


def _ensure_r15_self_ema(ema_state, params, cfg, allow_missing_self):
    """Check the R15 ``ema_state['self']`` branch on a resume."""
    M_self = int(cfg.n_self_codes)
    d = int(cfg.d_model)
    branch = ema_state.get('self')
    valid = (isinstance(branch, dict)
             and branch.get('N') is not None
             and branch.get('m') is not None
             and branch.get('last_used') is not None
             and tuple(jnp.asarray(branch['N']).shape) == (M_self,)
             and tuple(jnp.asarray(branch['m']).shape) == (M_self, d)
             and tuple(jnp.asarray(branch['last_used']).shape) == (M_self,))
    if valid:
        return ema_state
    if not allow_missing_self:
        raise ValueError(
            "checkpoint EMA state has no R15 self branch (C_self modes); "
            "old checkpoints are discarded, not migrated. Start a new run or "
            "pass allow_missing_self=True to reset it explicitly")
    ema_state = dict(ema_state)
    ema_state['self'] = init_ema_state(params, cfg)['self']
    print("[COG] R15: checkpoint had no self EMA state; reset to zero "
          "(allow_missing_self=True)")
    return ema_state


# ─── Init full params ───────────────────────────────────────────────────────

def _load_qwen_checkpoint(qwen_path, params, d=256):
    """Load frozen Qwen2.5-0.5B as active channel.

    Qwen weights stay on CPU (read-only). A trainable z_proj
    projection layer maps (B, d) cognitive state → (B, 896) Qwen input.
    """
    global _QWEN_PARAMS
    if _QWEN_PARAMS is None:
        _QWEN_PARAMS = load_qwen_params(qwen_path)
    params['qwen'] = _QWEN_PARAMS
    # Trainable projection: z_q (d=256) → Qwen hidden (896).
    # Only initialise when the checkpoint lacks it — a resumed run must keep
    # the z_proj it already trained.
    if 'z_proj' not in params:
        rng = jax.random.PRNGKey(42)
        qwen_d = QWEN_CONFIG['d_model']
        params['z_proj'] = jax.random.normal(rng, (qwen_d, d)) * (d ** -0.5)
    n_layers = QWEN_CONFIG['n_layers']
    print(f"[COG] Frozen Qwen2.5-0.5B ({n_layers}x{QWEN_CONFIG['d_model']}) loaded as active channel")
    print(f"[COG]  z_proj: {'kept from checkpoint' if 'z_proj' in params else 'initialised'}")


def init_cog_params(cfg, rng, qwen_ckpt=None, resume=None,
                    allow_missing_self=False):
    """Initialize all trainable params for dual-channel cognitive training.

    Params:
        encoder + codebooks: trained in Stage 2 (cognitive). The encoder's
            `embed` is also the passive readout — one token table, not two.
        qwen: the frozen bridge, loaded from its `.npz` (stop_gradient).

    Args:
        qwen_ckpt: Path to the Qwen `.npz` bridge, or None for passive-only.
        resume: Optional Stage 2 checkpoint dir.
        allow_missing_self: R15 escape hatch for a pre-``C_self`` checkpoint;
            resets the self codebook explicitly instead of rejecting.

    Returns:
        params: Dict of all parameters.
        self_state: Dict for self-lattice runtime state.
    """
    keys = jax.random.split(rng, 12)
    d = cfg.d_model

    if resume:
        params, self_state, _ = load_stage2_params(
            resume, cfg, rng, allow_missing_self=allow_missing_self)
        # A checkpoint carrying W_out predates the tied readout. It is not
        # migrated: W_out was a second token table trained against a different
        # token space, so resuming it would silently mix the two.
        if 'W_out' in params:
            raise ValueError(
                "checkpoint carries W_out, which predates the tied readout; "
                "old cognitive checkpoints are not migrated")
    else:
        # ── Init from scratch ──
        params = {}
        params['encoder'] = init_encoder_params(
            keys[0], d, cfg.d_ff, cfg.n_heads, cfg.n_encoder_layers,
            cfg.vocab_size, cfg.max_seq_len)

        params['hrq'] = init_hrq_params(keys[1], d, cfg.M_top, cfg.M_fine, cfg.n_hrq_layers)
        params['sparse'] = init_sparse_params(keys[2], d, cfg.M_sparse)
        params['lowrank'] = init_lowrank_params(
            keys[3], d, cfg.M_lr, cfg.ranks, r_max=cfg.r_max)
        params['manifold'] = init_manifold_params(keys[4], d, cfg.M_man, cfg.t_dim)
        params['binding'] = init_binding_params(keys[5], d, cfg.M_bind, cfg.n_bind_layers, cfg.r_max)
        params['contrast'] = init_contrast_params(keys[6], d, cfg.M_contrast, cfg.n_contrast_layers)

        # R15: seed C_self[1:] from a small batch of bottleneck vectors from
        # the freshly initialised encoder (early-sample path); the core row is
        # a separate frozen anchor.
        n_modes = max(1, int(cfg.n_self_codes) - 1)
        seq = max(2, min(int(cfg.max_seq_len), 16))
        sample_tokens = jax.random.randint(
            jax.random.fold_in(keys[10], 101), (n_modes, seq), 0,
            cfg.vocab_size, dtype=jnp.int32)
        self_samples = safe_unit(encoder_forward(
            params['encoder'], sample_tokens, cfg.n_heads))
        params['self'] = init_self_params(
            keys[10], d, cfg.n_self_codes, init_samples=self_samples,
            alpha_self=cfg.alpha_self)
        self_state = init_self_state(cfg.n_self_codes, d)

        # Routing + fusion + per-lattice value scalars. The cognitive loop runs
        # the canonical six-lattice step, which needs all three; without them
        # there is no routing mask to fuse by and no per-lattice scaling.
        params['route'] = init_route_params(keys[8], cfg.n_lattices, d)
        params['fusion'] = init_fusion_params(keys[9], cfg.n_lattices, d)
        params['value_scalars'] = init_value_scalars(cfg)
        # R13: frozen structural safety codebooks. Both halves are distinct by
        # construction; the training loop replaces the normal half with a bank
        # of safe encoder states so the normal path covers the input
        # distribution. They never enter the optimizer (FROZEN_PATHS).
        params['danger'] = init_danger_params(keys[11], cfg.M_danger, d)

    # R13: a v2 checkpoint written before the two-half danger layout, or the
    # minimal fixtures used by resume tests, may lack C_normal. Initialise a
    # fresh structural pair rather than silently exporting equal halves.
    if 'danger' not in params or 'C_normal' not in params.get('danger', {}):
        fresh = init_danger_params(jax.random.fold_in(rng, 17), cfg.M_danger, d)
        params['danger'] = fresh
        print("[COG] danger: initialised structural C_threats/C_normal "
              "(checkpoint lacked the two-half R13 layout)")

    # Load the frozen Qwen bridge for the active channel. Language LCM is
    # retired: it is not an alternative backend, and a path that is not a .npz
    # fails rather than silently reviving it.
    if qwen_ckpt:
        if not qwen_ckpt.endswith('.npz'):
            raise ValueError(
                "Language LCM is retired; cognitive training accepts only the "
                "Qwen .npz bridge (or no bridge for passive-only runs), got "
                f"{qwen_ckpt!r}")
        _load_qwen_checkpoint(qwen_ckpt, params, d)
    else:
        print("[COG] No active bridge provided; active channel disabled")
        params['qwen'] = None
        # Keep a trained z_proj from a resumed checkpoint: a later resume with
        # a Qwen npz must not discard the projection it already trained.
        if 'z_proj' not in params:
            params['z_proj'] = None

    return params, self_state


def _simvq_codebook(simvq):
    """Extract actual codebook matrix from SimVQ params: A @ W."""
    return simvq['A'] @ simvq['W']


def pack_codebooks_for_c(p):
    """Pack per-lattice codebook matrices for threshold computation.

    Routing is returned **separately**: ``C_route`` is a codebook but
    ``W_route`` is a gate matrix, and neither belongs in the flat
    ``(K_i, d)`` list handed to ``avg_codebook_distances``. The actual
    ``.bin`` export (including route) lives in ``train/checkpoint.py``.
    """
    packed = {}
    # HRQ: top + fine per layer
    packed['hrq'] = [_simvq_codebook(p['hrq']['top'])]
    packed['hrq'] += [_simvq_codebook(fb) for fb in p['hrq']['fine']]

    # Sparse
    packed['sparse'] = [p['sparse']['C']]

    # LowRank: one per rank
    V = p['lowrank']['A_V'] @ p['lowrank']['W_V']
    packed['lowrank'] = [
        u_k @ V[:, :u_k.shape[-1]].T for u_k in p['lowrank']['U']]

    # Manifold
    packed['manifold'] = [p['manifold']['C']]

    # Binding: key, value, bind per layer (family order kept for thresholds)
    packed['binding'] = []
    for i in range(len(p['binding']['key_cb'])):
        packed['binding'].append(_simvq_codebook(p['binding']['key_cb'][i]))
        packed['binding'].append(_simvq_codebook(p['binding']['val_cb'][i]))
        packed['binding'].append(_simvq_codebook(p['binding']['bind_cb'][i]))

    # Contrast: C_a, C_b per layer
    packed['contrast'] = []
    for i in range(len(p['contrast']['C_a'])):
        packed['contrast'].append(_simvq_codebook(p['contrast']['C_a'][i]))
        packed['contrast'].append(_simvq_codebook(p['contrast']['C_b'][i]))

    # Routing data (R19): kept out of the threshold codebook list on purpose.
    packed['route'] = {
        'C_route': p['route']['C_route'],
        'W_route': p['route']['W_route'],
    }
    return packed


def avg_codebook_distances(codebooks):
    """Compute avg pairwise distance per codebook — for threshold setting."""
    avg_dists = []
    for cb in codebooks:
        n = cb.shape[0]
        if n > 100:
            idx = np.random.choice(n, min(100, n), replace=False)
            sample = cb[idx]
        else:
            sample = cb
        dists = jnp.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
        avg_dists.append(float(jnp.mean(dists)))
    return avg_dists


# ─── Passive channel: transparent introspection ─────────────────────────────

def passive_loss(logits_1d, target_token):
    """Single-token CE loss for passive introspection channel.

    Args:
        logits_1d: (V,) predicted logits from z_q @ E.T.
        target_token: int scalar — the next token.
    """
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_1d[None, :], jnp.array([target_token])).mean()


# ─── Active channel: frozen Qwen bridge conditioned on z_q ──────────────────

def active_loss(logits, targets):
    """Cross-entropy for active channel (full sequence)."""
    B, N, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()


# ─── Training step ──────────────────────────────────────────────────────────

def _apply_ema_updates(ema, ema_state, stats, cfg, step):
    """EMA pass over the EMA-owned codebooks (R3/R15).

    ``stats`` is the last cognitive-loop step: its input state and selection
    indices are the observations the codebooks must move toward. The optimizer
    never sees these leaves (see ``partition_params``), so this function is the
    only writer of ``ema``. ``C_self`` goes through ``ema_update_self``, which
    keeps row 0 (the frozen core) bit-identical.
    """
    new_ema = dict(ema)
    new_state = dict(ema_state)

    if 'sparse' in ema:
        C, state = ema_update_sparse(
            ema['sparse']['C'], ema_state['sparse'], stats['z_in'],
            stats['sparse_idx'], cfg.gamma_sparse, cfg.lambda_shrink,
            step=step)
        new_ema['sparse'] = dict(ema['sparse'], C=C)
        new_state['sparse'] = state

    if 'manifold' in ema:
        C, state = ema_update_manifold(
            ema['manifold']['C'], ema_state['manifold'], stats['z_in'],
            stats['man_idx'], cfg.gamma_man, step=step)
        new_ema['manifold'] = dict(ema['manifold'], C=C)
        new_state['manifold'] = state

    if 'binding' in ema:
        binding, state = ema_update_binding(
            ema['binding'], ema_state['binding'], stats['binding_residuals'],
            stats['binding_indices'], cfg.gamma_bind, step=step)
        new_ema['binding'] = binding
        new_state['binding'] = state

    if 'self' in ema:
        # R15: C_self[0] stays frozen inside ema_update_self. The forward's
        # selected row is what the EMA must accumulate; when a caller did not
        # provide it, recompute the (value-free) Euclidean argmin from the same
        # z the forward saw.
        idx = stats.get('self_idx')
        if idx is None:
            z_in = jnp.asarray(stats['z_in'])
            C_self = ema['self']['C_self']
            idx = jnp.argmin(
                jnp.sum((z_in[:, None, :] - C_self[None, :, :]) ** 2,
                        axis=-1), axis=-1)
        C_self, state = ema_update_self(
            ema['self']['C_self'], ema_state['self'], stats['z_in'], idx,
            cfg.gamma_self, step=step)
        new_ema['self'] = dict(ema['self'], C_self=C_self)
        new_state['self'] = state

    return new_ema, new_state


def make_optimizer(cfg):
    """The production optimizer chain (shared by train_cog and diagnostics).

    ``learning_rate=1.0`` is deliberate: ``train_step`` applies the live
    schedule value to the updates itself, which is what makes the Supervisor's
    mid-run LR changes effective.
    """
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0, b1=cfg.adam_beta1,
                    b2=cfg.adam_beta2, eps=cfg.adam_eps,
                    weight_decay=cfg.weight_decay),
    )


def make_train_step(cfg, optimizer, joint=False, return_grads=False,
                    freeze_lattices=()):
    """Create jitted training step with dual-channel output + self-lattice.

    The sequence is split at ``cog_context_frac``. z is computed from the
    **context** half only; both channels are then supervised on what follows:

      - Passive (introspection): z_q @ E.T → the first token after the context
      - Active (expression):     active_channel(z_q, generation segment)

    The generation segment is fed to the active channel on its own, so z is the
    only channel carrying the context. Supervising the whole shifted sequence
    from a global z — the previous behaviour — leaks every target through z,
    because the bidirectional encoder has already read x[i+1] by the time
    position i is asked to predict it. The loss looked excellent and measured
    nothing. A hinge on ``logit(true | z) − logit(true | z=0)`` stops the active
    channel from ignoring z and degenerating into a plain causal LM.

    The Qwen bridge is frozen (stop_gradient) so the gradient
    forces the cognitive state z_q to adapt to it.

    Self-lattice (R15) runs per macro step: nearest C_self row plus a cosine
    familiarity scalar, fused as the 7th element before LayerNorm.

    When joint=True, additional Stage 3 losses are computed:
      - Contrastive NCE
      - Manifold orthogonality

    The R18 layered codebook/commitment losses run unconditionally (also when
    joint=False); they are main training losses, not Stage 3 extras.

    ``return_grads=True`` adds the clipped trainable-gradient pytree to
    ``aux_out['grads']``. It is a diagnostics switch (Task-15 smoke): it is not
    used by the training loop, and retaining the tree for every production step
    would cost memory for no training benefit.
    """

    @jax.jit
    def train_step(params, opt_state, ema_state, batch, lr, rng,
                   self_state=None, step=None, routing_bias=None):
        inputs, targets = batch
        B, N = inputs.shape
        # Context / generation split for the active channel (see the
        # make_train_step docstring). Derived from the *batch* length, not
        # cfg.max_seq_len — those differ whenever --cog-seq is set, and a split
        # past the end would leave an empty generation segment.
        ctx_len = max(1, min(int(N * cfg.cog_context_frac), N - 1))
        # Context = the tokens z is allowed to see. The generation segment is
        # structurally disjoint from it, so no target is ever inside z's input.
        ctx = inputs[:, :ctx_len]
        # The bridge injects z by OVERWRITING position 0's embedding, so the
        # token placed there is consumed by the injection slot and never
        # predicted. Feed one extra token — x[ctx_len-1], which the injection
        # discards anyway — and every remaining position then predicts its own
        # successor: logits[:, j] <-> targets[:, ctx_len-1+j]. Starting the
        # segment at ctx_len instead skips x[ctx_len] entirely and leaves every
        # later target off by one (verified: an off-by-one target change at the
        # last position produced bit-identical losses).
        gen_in = inputs[:, ctx_len - 1:]
        k = gen_in.shape[1]

        # R3: only the trainable group is differentiated / optimized. The EMA
        # and frozen groups are closed over as constants inside loss_fn, so no
        # gradient (or AdamW moment, or weight-decay step) reaches them.
        trainable_params, ema_params, frozen_params = partition_params(params)
        for name in freeze_lattices:
            # Coverage-preserving research switch: keep a whole lattice subtree
            # out of both the optimizer and the EMA writer.
            if name in trainable_params:
                del trainable_params[name]
            if name in ema_params:
                del ema_params[name]
            if name in params:
                frozen_params = dict(frozen_params)
                frozen_params[name] = params[name]

        def loss_fn(p_trainable):
            p = merge_params(p_trainable, ema_params, frozen_params)
            z = encoder_forward(p['encoder'], ctx, cfg.n_heads)  # (B, d)

            # ── Normalise encoder output to the unit sphere ───────────────
            # The canonical path (model.forward) does exactly this, and the six
            # lattice forwards were written against unit-scale inputs. The old
            # codebook-scale normalisation existed to put z at the distance
            # scale dag_fuse's reciprocal weighting assumed; with the generic
            # path gone it has no basis.
            z = safe_unit(z)

            # ── Cognitive loop: the canonical six-lattice step, repeated ──
            # Batched (B, d), not vmapped over single vectors: the step calls
            # routing_gate, which indexes z[:, None, :]. The scan yields
            # (max_steps, B, ...), transposed back to the (B, max_steps, ...)
            # every consumer below expects.
            z_qs, diffs, entropies, loop_stats = cog_loop_scan(
                z, p, cfg, max_steps=cfg.max_inference_steps,
                rng=jax.random.fold_in(rng, 7), return_stats=True,
                routing_bias=routing_bias)
            z_qs = jnp.transpose(z_qs, (1, 0, 2))     # (B, max_steps, d)
            diffs = jnp.transpose(diffs, (1, 0))      # (B, max_steps)
            entropies = jnp.transpose(entropies, (1, 0))
            # The EMA pass consumes the last step's input state and selection
            # indices; the scan stacks them along the step axis.
            loop_stats = jax.tree_util.tree_map(lambda x: x[-1], loop_stats)

            # ── Self lattice (R15) ──────────────────────────────────────
            # The loop runs the self lattice each macro step and fuses
            # self_bias * z_self as the 7th element (cog_loop_scan). Here we
            # only keep its regulariser and the thin diagnostic state; there
            # is no second self forward pass.
            self_state_out = self_state
            loss_self = jnp.array(0.0)
            if 'self' in p:
                loss_self = self_lattice_reg_loss(p['self'])
                if self_state is not None and 'self_idx' in loop_stats:
                    self_state_out = update_self_state(
                        self_state, loop_stats['self_idx'],
                        loop_stats.get('z_self', loop_stats['z_in']),
                        gamma_self=cfg.gamma_self)

            # ── Passive channel: z_q @ E.T (tied to the encoder embedding) ──
            # One trainable token table: E is both the encoder's input
            # embedding and the readout matrix. That is what puts the passive
            # channel in the same token space as the active channel, and it is
            # what removes a second optimizer leaf (and its Adam state).
            p_logits = jnp.einsum('bsd,vd->bsv', z_qs, p['encoder']['embed'])
            # The honest readout is the token that follows the context. It is
            # outside the encoder's input by construction, so the passive
            # channel cannot degenerate into copying. (Reading targets[:, -1]
            # instead asks the converged state to jump N-ctx_len tokens ahead
            # — legitimate but no longer the same "next step" the active
            # channel is trained on.)
            p_target = inputs[:, ctx_len]
            p_loss = optax.softmax_cross_entropy_with_integer_labels(
                p_logits.reshape(-1, p_logits.shape[-1]),
                p_target[:, None].repeat(cfg.max_inference_steps, axis=1).reshape(-1),
            ).mean()

            # ── Active channel: frozen Qwen bridge ──────────────────────
            #
            # z is the ONLY channel carrying the context: the generation
            # segment is fed on its own, so the active channel cannot read the
            # answer off its own attention context. This is the structure
            # causal_student_train.py already validated.
            #
            # Supervising the whole shifted sequence from a *global* z (what
            # this used to do) leaks every target through z: the bidirectional
            # encoder has already read x[i+1] by the time position i is asked to
            # predict it. The loss then looks excellent and means nothing.
            z_final = z_qs[:, -1, :]  # (B, d)
            use_qwen = 'qwen' in p and p['qwen'] is not None
            # Contiguous window matching the overwrite-at-position-0 injection
            # (see the gen_in construction above): k logits, k targets, and the
            # window covers the whole generation segment x[ctx_len..N-1] plus
            # the out-of-sequence token x[N] == targets[:, N-1].
            a_targets = targets[:, ctx_len - 1:ctx_len - 1 + k]
            if use_qwen:
                qwen_params = jax.lax.stop_gradient(p['qwen'])
                z_proj = p['z_proj']  # trainable projection
                a_logits = qwen_forward(qwen_params, gen_in,
                                         z_q=z_final, z_proj=z_proj,
                                         n_layers=4)
            else:
                a_logits = None

            if a_logits is not None:
                a_loss = active_loss(a_logits, a_targets)
                # Force the active channel to actually depend on z: with a
                # generation segment of its own to attend over, the cheapest
                # solution is to ignore z entirely and behave like a plain
                # causal LM. Penalise any step where zeroing z does not hurt.
                z0 = jax.lax.stop_gradient(jnp.zeros_like(z_final))
                b_logits = qwen_forward(qwen_params, gen_in, z_q=z0,
                                        z_proj=p['z_proj'], n_layers=4)
                tgt = a_targets[:, :, None]
                with_z = jnp.take_along_axis(a_logits, tgt, axis=-1).squeeze(-1)
                without_z = jnp.take_along_axis(b_logits, tgt, axis=-1).squeeze(-1)
                z_margin = jnp.mean(jnp.maximum(
                    0.0, cfg.active_z_margin - with_z + without_z))
                a_loss = a_loss + cfg.active_z_margin_weight * z_margin
            else:
                a_loss = jnp.array(0.0)
                z_margin = jnp.array(0.0)

            # Convergence: R9's normalised entropy gate.
            # H_norm = H / ln(n_active), n_active = routed lattices with a
            # non-zero softmask entry, clamped to >=2 (ln(1)=0 is undefined;
            # a single active node is handled as H_norm=0 by R9, which passes).
            n_active = jnp.sum(
                loop_stats['soft_mask'] > 1e-6, axis=-1).astype(jnp.float32)
            n_active = jnp.maximum(n_active, 2.0)
            h_norm = entropies[:, -1] / jnp.log(n_active)
            conv = ((diffs[:, -1] < cfg.convergence_tol)
                    & (h_norm < cfg.entropy_threshold))
            n_steps = jnp.argmax((diffs < cfg.convergence_tol).astype(jnp.float32), axis=-1) + 1

            # Convergence bonus — reward FAST convergence (n=1 → -0.0035,
            # n=max_steps → 0). loss is MINIMISED, so the term must go negative
            # for fast loops. Written the other way up,
            # +log(max_steps / n_steps), it reads like a bonus but is a penalty:
            # n=1 scored +0.0035 and n=32 scored 0, so converging slowly was
            # still the better outcome. Inverting the ratio is what makes the
            # sign actually match the intent.
            loss = p_loss + a_loss + loss_self + jnp.mean(
                jnp.where(conv, 0.001 * jnp.log(
                    (n_steps.astype(jnp.float32) + 1e-8) / cfg.max_inference_steps), 0.0))

            # ── R18: layered codebook/commitment losses ──────────────────
            # These are main training losses, not Stage-3 extras: they run
            # unconditionally, including joint=False. They consume the last
            # loop step's forward outputs (the same stats the EMA pass uses),
            # so no second full forward pass is made.
            vq_loss, vq_parts = cognitive_vq_losses(p, loop_stats, cfg)
            loss = loss + vq_loss
            stage3_extra = {f'vq_{key}': value
                            for key, value in vq_parts.items()}
            if joint:
                if 'contrast' in p:
                    c_loss = cfg.lambda_contrast * contrast_info_nce_loss(
                        p['contrast'], z, tau=cfg.tau_contrast)
                    stage3_extra['contrast'] = c_loss
                    loss = loss + c_loss

                if cfg.lambda_orth > 0 and 'manifold' in p:
                    # λ·‖Tᵀ T − I‖², NOT λ·mean(Σ T²). The latter is plain
                    # weight decay on the tangent basis: it drives T → 0 rather
                    # than towards an orthonormal frame, collapsing the tangent
                    # space the manifold lattice projects into. The active rows
                    # come from the loop's own forward, not a recomputed argmin.
                    o_loss = manifold_orth_loss(
                        p['manifold']['T'], loop_stats['man_idx'],
                        cfg.n_orth_samples, lambda_orth=cfg.lambda_orth,
                        rng=rng)
                    stage3_extra['orth'] = o_loss
                    loss = loss + o_loss

            aux_out = {'self_state': self_state_out, 'loss_self': loss_self,
                       'self_bias': (jnp.mean(loop_stats['self_bias'])
                                     if 'self_bias' in loop_stats
                                     else jnp.array(0.0)),
                       'z_margin': z_margin, 'stage3': stage3_extra,
                       'loop_stats': loop_stats,
                       # R16 self-observation fields, from this same forward
                       # (no second pass).
                       'z_cur': z,
                       'z_next': z_qs[:, -1, :],
                       'soft_mask': loop_stats.get(
                           'soft_mask', jnp.zeros((B, cfg.n_lattices))),
                       'route_idx': loop_stats.get(
                           'route_idx', jnp.zeros((B,), dtype=jnp.int32)),
                       'convergence_diff': jnp.mean(diffs[:, -1]),
                       # Stage 2 has no safety oracle (no danger snapshot or
                       # gvalue decision). This is hardcoded True by design;
                       # real safety state lives on the C inference path.
                       'is_safe': jnp.array(True)}
            return loss, aux_out

        (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            trainable_params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads)
        if return_grads:
            aux_out['grads'] = grads
        # The optimizer sees trainable only. EMA and frozen leaves are not in
        # this tree, so their absence is structural, not a filtered-out zero.
        updates, new_opt = optimizer.update(grads, opt_state, trainable_params)
        # The optimizer is built with learning_rate=1.0, so its own scale is
        # already -1 (optax adds updates). Multiplying by the live `lr` here is
        # what actually applies the schedule — and what lets the Supervisor
        # reduce the rate mid-run. See the optimizer construction in train_cog.
        updates = jax.tree_util.tree_map(lambda u: lr * u, updates)
        new_trainable = optax.apply_updates(trainable_params, updates)

        new_ema, new_ema_state = _apply_ema_updates(
            ema_params, ema_state, aux_out['loop_stats'], cfg, step)
        new_params = merge_params(new_trainable, new_ema, frozen_params)
        # Exposed for the wiring test: proves the bias reached the routed step
        # rather than being computed and discarded.
        aux_out['routing_bias_used'] = routing_bias
        return new_params, new_opt, new_ema_state, loss, aux_out

    return train_step


# ─── Training loop ──────────────────────────────────────────────────────────

def _host_sparse_bank_update(params, ema_state, feature_bank, step, cfg):
    """Replace dead sparse rows with random bank vectors (host numpy).

    Runs outside jit: the number of dead rows is dynamic and must not become a
    traced shape. The replaced rows are cold-started (N/m zeroed) by
    ``FeatureBank``.
    """
    if (len(feature_bank) == 0 or 'sparse' not in ema_state
            or 'sparse' not in params):
        return params, ema_state
    state = ema_state['sparse']
    dead = feature_bank.dead_indices(
        state['last_used'], step, cfg.bank_dead_threshold)
    if dead.size == 0:
        return params, ema_state

    C_new, state_new = feature_bank.replace_dead_sparse(
        params['sparse']['C'], state, step, cfg.bank_dead_threshold)
    new_params = dict(params)
    new_params['sparse'] = dict(params['sparse'], C=jnp.asarray(C_new))
    new_ema = dict(ema_state)
    new_ema['sparse'] = jax.tree_util.tree_map(jnp.asarray, state_new)
    print(f"[COG] feature bank: replaced {len(dead)} dead sparse rows "
          f"at step {step}")
    return new_params, new_ema


def _record_contrast_usage(contrast_last_used, loop_stats, step):
    """Update contrast's per-layer ``last_used`` every training step (R17).

    Only the last macro step's indices are available from ``loop_stats``; that
    is the documented approximation. The clock is host-side and uses the
    absolute step, so a code selected between two bank checks stays live.
    """
    if loop_stats is None or 'contrast_indices' not in loop_stats:
        return contrast_last_used
    indices = loop_stats['contrast_indices']
    new_clock = {family: list(contrast_last_used[family])
                 for family in ('a', 'b')}
    for family in ('a', 'b'):
        for layer in range(len(new_clock[family])):
            last = np.array(new_clock[family][layer], copy=True)
            selected = np.asarray(indices[family][layer]).reshape(-1)
            last[selected] = step
            new_clock[family][layer] = last
    return new_clock


def _replace_dead_contrast(params, contrast_bank, contrast_last_used, step, cfg):
    """Replace only dead contrast rows, preserving every live row (R18).

    ``C = A @ W``. For each dead row we solve ``a @ W = c_new`` (``W`` is
    square, ``a = pinv(W.T) @ c_new``) and write only that row of ``A``; ``W``
    and all non-dead rows are left untouched, so their effective vectors are
    bit-identical. Adam moments for the replaced rows are *not* reset; the next
    gradient steps adapt them.
    """
    if len(contrast_bank) == 0 or 'contrast' not in params:
        return params, contrast_last_used

    new_params = params
    new_clock = {family: list(contrast_last_used[family])
                 for family in ('a', 'b')}
    n_replaced = 0
    for family in ('a', 'b'):
        for layer in range(len(params['contrast'][f'C_{family}'])):
            last = np.array(new_clock[family][layer], copy=True)
            dead = contrast_bank.dead_indices(
                last, step, cfg.bank_dead_threshold)
            if dead.size:
                cb = params['contrast'][f'C_{family}'][layer]
                A = np.asarray(cb['A'])
                W = np.asarray(cb['W'])
                C_new, last = contrast_bank.replace_dead_contrast(
                    A @ W, last, step, cfg.bank_dead_threshold)
                # Only the dead rows change: a = W^+ c_new. pinv handles a
                # singular W (the old np.linalg.solve raised LinAlgError and
                # aborted the training step); a non-finite replacement is a
                # hard error because it would poison the codebook silently.
                replacements = (np.linalg.pinv(W.T) @ C_new[dead].T).T
                if not np.all(np.isfinite(replacements)):
                    raise ValueError(
                        f"contrast dead-row replacement produced non-finite "
                        f"values (family={family}, layer={layer}, "
                        f"step={step}); refusing to poison the codebook")
                A_new = np.array(A, copy=True)
                A_new[dead] = replacements
                new_cb = {'A': jnp.asarray(A_new), 'W': cb['W']}
                rows = list(new_params['contrast'][f'C_{family}'])
                rows[layer] = new_cb
                new_params = dict(new_params)
                new_params['contrast'] = dict(
                    new_params['contrast'], **{f'C_{family}': rows})
                n_replaced += len(dead)
            new_clock[family][layer] = last
    if n_replaced:
        print(f"[COG] feature bank: replaced {n_replaced} dead contrast rows "
              f"at step {step}")
    return new_params, new_clock


def _contrast_bank_step(params, contrast_bank, contrast_last_used, loop_stats,
                        step, do_replace, cfg):
    """The contrast path ``train_cog`` runs every step (R17/R18).

    The usage clock is updated on every step; dead detection and replacement
    run only when ``do_replace`` (the ``bank_check_interval`` guard).
    """
    clock = _record_contrast_usage(contrast_last_used, loop_stats, step)
    if do_replace:
        params, clock = _replace_dead_contrast(
            params, contrast_bank, clock, step, cfg)
    return params, clock


def _self_observe_step(obs, narr, reflection, explorer, verifier_poller,
                       abs_step, step_time_ms, aux_out, loss_f, cfg,
                       causal_subject=None):
    """Record one training step into the R16 self-observation stack.

    Every sub-call is fault-isolated: a broken narrative/reflection/verifier
    must never change the training trajectory (e.md §1 core constraint).

    ``prediction_error`` is the Ruling-39 one-step residual
    ``||z_next - z_cur||²`` (or the model's prediction loss when present).
    ``T_conf`` has no value oracle in Stage 2, so behavior tension is the
    documented entropy + residual proxy (``T_conf == 0``), not full U.

    The training ``dag_nodes`` entry is a one-step summary, not the C-engine
    per-macro-step DAG trace. ``soft_mask`` is the routed six-lattice mask;
    the C inference trace's seventh self element is sliced off by ``lcm.py``.

    Returns the safety flag used by the next step's danger priority.
    """
    def _fail(category, exc):
        note = getattr(obs, 'note_error', None)
        if note is not None:
            note(category, exc)

    try:
        soft_mask = np.asarray(jax.device_get(aux_out['soft_mask']))[0]
        route_idx = int(np.asarray(
            jax.device_get(aux_out['route_idx'])).ravel()[0])
        z_cur = np.asarray(jax.device_get(aux_out['z_cur']))[0]
        z_next = np.asarray(jax.device_get(aux_out['z_next']))[0]
        conv = float(np.asarray(
            jax.device_get(aux_out['convergence_diff'])))
        is_safe = bool(np.asarray(jax.device_get(aux_out['is_safe'])))
        value_signals = aux_out.get('value_signals')
        if value_signals is not None:
            value_signals = np.asarray(jax.device_get(value_signals)).ravel()
    except Exception as exc:
        _fail('extract', exc)
        soft_mask, route_idx, z_cur, z_next = None, None, None, None
        conv, is_safe, value_signals = None, True, None
    is_safe = bool(is_safe and np.isfinite(loss_f))

    # Ruling 39: one-step residual. Prefer an explicit prediction loss if a
    # future training path exposes one; otherwise the residual is the error.
    pred_error = aux_out.get('prediction_loss')
    if pred_error is not None:
        try:
            pred_error = float(np.asarray(jax.device_get(pred_error)))
        except Exception:
            pred_error = None
    if pred_error is None and z_cur is not None and z_next is not None:
        pred_error = float(np.sum((z_next - z_cur) ** 2))

    dag_summary = None
    if soft_mask is not None:
        dag_summary = [{
            'step': abs_step,
            'route_idx': route_idx,
            'soft_mask': np.asarray(soft_mask).tolist(),
            'summary': True,
        }]
    record = obs.record_step(
        step=abs_step, source=0, step_time_ms=step_time_ms,
        soft_mask=soft_mask, route_idx=route_idx, is_safe=is_safe,
        convergence_diff=conv, prediction_error=pred_error,
        dag_nodes=dag_summary,
        z_cur=z_cur, z_next=z_next)

    if record is not None:
        try:
            narr.feed(record)
        except Exception as exc:
            _fail('narrative', exc)
        try:
            # Stage-2 training has no PredictiveSystem; the cache-hit trend
            # analysis is inference-only (tested against real cache counters).
            reflection.observe(abs_step, record, cache_hit=None,
                               is_safe=is_safe)
        except Exception as exc:
            _fail('reflection_observe', exc)
        if causal_subject is not None:
            try:
                causal_subject.step(
                    record, pred_cache=None,
                    action_desc=f"train_step_{abs_step}")
            except Exception as exc:
                _fail('causal', exc)

    try:
        tension = compute_tension_from_aux(
            {'soft_mask': soft_mask, 'value_signals': value_signals},
            z_cur, z_next, pred_error=pred_error, d_model=cfg.d_model)
        explorer.observe_tension(abs_step, tension,
                                 safety_breach=not is_safe)
    except Exception as exc:
        _fail('behavior_tension', exc)

    try:
        reflection.maybe_reflect(abs_step, narrative=narr)
        # Cap application: one highest-severity suggestion per step, with
        # the loop's own per-target cooldown. Low-severity noise never
        # directly moves Level 1.
        suggestions = sorted(reflection.drain_suggestions(),
                             key=lambda s: s.severity, reverse=True)
        applied = 0
        for suggestion in suggestions:
            if applied >= 1:
                break
            if suggestion.severity < 0.5:
                continue
            if reflection.apply_suggestion(
                    suggestion, explorer, danger=not is_safe):
                applied += 1
    except Exception as exc:
        _fail('reflection_apply', exc)

    if verifier_poller is not None:
        try:
            state = verifier_poller.poll(abs_step)
            if state == VERIFIER_LOCK:
                _fail('verifier_lock',
                      RuntimeError("external verifier LOCK while training"))
        except Exception as exc:
            _fail('verifier_poll', exc)

    return is_safe


def _lr_decay_steps(start_step, steps):
    """Policy A: a resumed run continues the original cosine curve.

    ``train_cog(steps=N)`` means N *new* steps; the decay horizon is the
    absolute end of the extended run, ``start_step + N``. A resumed run at
    ``start_step >= N`` therefore does not pin at the ``alpha=0.1`` floor.
    """
    return start_step + steps


def _resume_loop_plan(start_step, steps, save_every, log_every,
                      bank_check_interval):
    """Per-step clocks and guards for a (possibly resumed) training loop.

    Two clocks, deliberately separate:

    * ``offset`` (loop-relative, ``0..steps-1``) drives the LR schedule, the
      log/save/bank-interval guards and loss averaging.
    * ``abs_step = start_step + offset`` is the *state* clock: EMA
      ``last_used``/dead-code detection and checkpoint directory names.

    A saved step is not a multiple of every guard by construction, so mixing the
    clocks made the first resumed step overwrite ``step_{start_step}`` and
    diluted the first log average. Kept pure so this can be tested without
    running the jitted step.
    """
    plan = []
    for offset in range(steps):
        plan.append({
            'offset': offset,
            'abs_step': start_step + offset,
            # `offset > 0` keeps the first new step from re-saving the
            # checkpoint it was resumed from (and matches the old step-0 skip).
            'save': (save_every > 0 and offset > 0
                     and offset % save_every == 0),
            'log': (log_every > 0 and offset > 0
                    and offset % log_every == 0),
            'bank': (bank_check_interval > 0 and offset > 0
                     and offset % bank_check_interval == 0),
        })
    return plan


def train_cog(cfg, output_dir, steps=50000, lr=3e-4, batch_size=1,
              seq_len=256, log_every=100, save_every=1000,
              data_path=None, shape_path=None, spans_path=None,
              qwen_ckpt=None, resume=None, joint=False, auto_mode=False,
              full_verify=True, allow_missing_self=False,
              self_awareness=True, verifier=None,
              causal_subject=None, tokenizer_path=None, on_step=None,
              overfit_one_batch=False, init_params=None, freeze_lattices=()):
    """Run dual-channel cognitive training (Stage 2).

    Dual channels from cognitive state z_q:
      - Passive: z_q @ E.T (the encoder embedding, tied; honest readout)
      - Active:  Qwen2.5-0.5B (frozen bridge; the only current active channel)

    Args:
        data_path / shape_path / spans_path: the three files that constitute a
            corpus. Identity is verified before any parameter is allocated.
        qwen_ckpt: Path to the Qwen `.npz` bridge, or None for passive-only.
        resume: Optional checkpoint dir carrying a matching run identity.
        joint: When True, adds Stage 3 losses.
        auto_mode: When True, enables Supervisor.
        full_verify: Re-read the corpus and check its content hash. On by
            default: a production run must verify the corpus, not just its size.
        allow_missing_self: R15 escape hatch for resuming a pre-``C_self``
            checkpoint. Off by default; when on, the self codebook and its EMA
            branch are reset explicitly (old weights are not migrated).
        self_awareness: R16 wiring switch. When on, every step records the
            black box, feeds narrative memory, advances the behavior explorer
            (bias enters ``train_step``), and schedules the reflection loop
            every 200 steps. Off is the no-observation baseline.
        verifier: optional external verifier instance. It is polled with
            :class:`train.observability.VerifierPoller` each step. ``cog_train``
            does not create one by default: training has no gvalue/danger
            snapshot to submit, and a verifier process per training run is
            heavier than the 5% observation budget. The inference engine
            (``lcm.py``) owns the production wiring; this hook exists for
            callers that already have a verifier. Training only *observes*
            LOCK/OBSERVE callbacks; it does not halt on LOCK because Stage 2
            has no safety oracle. The C inference path halts.
        causal_subject: optional :class:`train.causal_subject.CausalSubject`.
            Per Ruling 43 it is wired only when supplied; the default omission
            is the accepted causal deviation (``causal_subject.py`` itself is
            not modified).
        tokenizer_path: optional override for dataset identity resolution. The
            default is the registry path for the metadata's ``tokenizer_id``;
            tests and the conformance smoke use this to point at a temporary
            tokenizer without modifying the global registry.
        on_step: optional callback invoked after a clean step has been applied,
            bank-updated and saved. Signature:
            ``on_step(abs_step, loss, aux_out, params, ema_state,
            feature_bank, contrast_bank, contrast_last_used, step_time_ms)``.
            It exists for the Task-15 smoke diagnostics; it is outside jit,
            receives the post-step state, and must not mutate it. Exceptions
            propagate (a diagnostic failure is not a training failure).
        overfit_one_batch: smoke-only switch. When True, every step reuses the
            first ``(inputs, targets)`` window instead of sampling a new one.
            Design §6.3's "single-batch overfit" check (last-10 mean < 0.8 ×
            first-10 mean) is only meaningful when the target does not change
            every step. Default False is the normal streaming path.
        init_params: optional research hook. A full parameter tree with the
            same shapes as ``init_cog_params``; used for coverage-based
            codebook initialisation. It is not a checkpoint migration: the
            passed tree is the initial model, and EMA buffers are rebuilt.
    """
    from train.data import WikiDataIter
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify corpus/tokenizer BEFORE allocating model parameters.
    spec, meta = resolve_dataset_identity(
        data_path, shape_path, spans_path, full_verify=full_verify,
        tokenizer_path=tokenizer_path,
    )

    # 2. The dataset owns the tensor vocabulary extent. Deliberately does not
    #    need a Qwen artifact: metadata is authoritative.
    cfg = dataclasses.replace(cfg, vocab_size=int(meta["model_vocab_size"]))

    # 3. Identity for this run.
    run_identity = RunIdentity(spec, meta)

    # 3b. A resume must continue the same dataset. Checked before allocation so
    #     a mismatched resume fails before anything is loaded.
    if resume:
        verify_resume_identity(resume, run_identity)

    # 4. Only now is model allocation allowed.
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    params, self_state = init_cog_params(
        cfg, init_rng, qwen_ckpt=qwen_ckpt, resume=resume,
        allow_missing_self=allow_missing_self)
    if init_params is not None:
        # Research hook: pre-initialised codebooks (e.g. FPS coverage) with
        # the same shapes as init_cog_params. Buffers are re-derived from this
        # tree below; no old checkpoint is migrated.
        params = jax.tree_util.tree_map(jnp.asarray, init_params)
    (ema_state, feature_bank, contrast_bank, contrast_last_used,
     start_step) = load_cog_buffers(
        resume, params, cfg, allow_missing_self=allow_missing_self)
    # Restored last_used values are absolute step indices from the old run, so
    # the loop continues from the saved step rather than restarting its clock.
    step = start_step

    # 5. Assert the actual tensors, do not assume init did the right thing.
    verify_tied_table(params, meta, cfg)
    verify_qwen_extent(params, meta)

    # Policy A: continue the original cosine curve across a resume. The horizon
    # is the absolute end of the extended run; `schedule(abs_step)` is below the
    # alpha floor only at that end, never merely because start_step >= steps.
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=_lr_decay_steps(start_step, steps),
        alpha=0.1)
    # learning_rate=1.0, deliberately not `schedule`. A schedule bound here is
    # stored inside opt_state and cannot be varied per step, which made
    # train_step's `lr` argument dead code: the Supervisor's "LR reduced" line
    # printed a new number while training carried on down the original curve.
    # train_step now applies the live schedule value to the updates itself.
    optimizer = make_optimizer(cfg)
    # R3: optimizer state exists only for the trainable group. EMA codebooks
    # are written by the EMA pass; frozen tensors (danger, the Qwen bridge)
    # must not even allocate Adam moments. Filtering `qwen` alone was not
    # enough: AdamW's decoupled weight decay moves zero-gradient leaves too.
    trainable_params, _, _ = partition_params(params)
    for name in freeze_lattices:
        trainable_params.pop(name, None)
    opt_state = optimizer.init(trainable_params)

    # 6. Sampling must consume the same dataset identity.
    data_iter = WikiDataIter(data_path, shape_path, spans_path,
                             B=batch_size, N=seq_len)
    fixed_batch = next(data_iter) if overfit_one_batch else None
    if overfit_one_batch:
        print("[COG] smoke: reusing one fixed batch for every step "
              "(single-batch overfit check)")
    train_step = make_train_step(cfg, optimizer, joint=joint,
                                 freeze_lattices=freeze_lattices)

    # ── Self-observation stack (R16 / e.md §1) ──────────────────────────
    # The external VerifierProcess is deliberately not created here: training
    # has no danger/gvalue snapshot to submit, and one process per training
    # run would be heavier than the 5% observation budget. lcm.py wires the
    # poller on the inference path; callers that already own a verifier can
    # pass ``verifier=`` above.
    obs = narr = reflection = explorer = verifier_poller = None
    routing_bias = np.zeros(cfg.n_lattices, dtype=np.float32)
    if self_awareness:
        from train.observability import ObservabilityRecorder, VerifierPoller
        from train.narrative_memory import NarrativeMemory
        from train.reflection_loop import ReflectionLoop, REFLECT_INTERVAL
        from train.behavior_explorer import BehaviorExplorer

        obs = ObservabilityRecorder(
            raw_capacity=1000, long_term_capacity=10000,
            record_every_n=1, long_term_every=100,
            anomaly_oversample=3)
        narr = NarrativeMemory(max_records=10000)
        reflection = ReflectionLoop(
            interval=REFLECT_INTERVAL,
            log_path=os.path.join(output_dir, 'reflection.log'))
        explorer = BehaviorExplorer()
        routing_bias = explorer.get_best_bias()
        if verifier is not None:
            verifier_poller = VerifierPoller(verifier)
        print("[SELF] Black box + narrative + reflection (200) + "
              "behavior (10) wired")

    packed_codebooks = pack_codebooks_for_c(params)
    codebooks_flat = [
        cb for name in ('hrq', 'sparse', 'lowrank', 'manifold',
                        'binding', 'contrast')
        for cb in packed_codebooks[name]]
    avg_dists = avg_codebook_distances(codebooks_flat)
    thresholds = [d * 0.15 for d in avg_dists]
    print(f"[COG] Codebook thresholds: {[f'{t:.3f}' for t in thresholds[:6]]}")

    d = cfg.d_model
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                       if hasattr(p, 'size'))
    has_qwen = 'qwen' in params and params['qwen'] is not None
    if has_qwen:
        active_name = f"Qwen2.5-0.5B (frozen, {len(params['qwen'])//12} layers)"
    else:
        active_name = "DISABLED"
    print(f"[COG] Dual-channel: passive (z_q @ E.T) + active ({active_name})")

    # Vocabulary agreement used to be checked here and merely printed. It is now
    # enforced before allocation by resolve_dataset_identity (tokenizer digest +
    # metadata + tensors) and verify_qwen_extent, which raise instead of warn:
    # a run whose token space cannot be verified has no interpretable metric.
    print(f"[COG] Self-lattice: {cfg.n_self_codes - 1} EMA modes + 1 frozen "
          f"core (C_self rows={cfg.n_self_codes})")
    print(f"[COG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")
    print("[COG] Layered codebook/commitment losses: always on (R18)")
    if joint:
        print(f"[COG] Joint mode: + Stage 3 losses (contrastive + orth)")
    print()

    import numpy as _np_np
    V = cfg.vocab_size
    _LN_V = float(_np_np.log(V))  # passive random baseline ≈ 10.31
    _LOSS_FLOOR = 0.0  # active channel floor = 0 (language LCM can reach low loss)
    print()

    running_loss = 0.0
    logged_steps = 0
    start_time = time.time()
    pbar = tqdm(total=steps, desc="cog training", unit="step", file=sys.stderr, mininterval=0.5)
    plan = _resume_loop_plan(start_step, steps, save_every, log_every,
                             cfg.bank_check_interval)

    import signal as _signal

    def _handler(sig, frame):
        print(f"\n[COG] Interrupt at step {step}, saving checkpoint...")
        save_cog_checkpoint(params, output_dir, step, self_state=self_state,
                            ema_state=ema_state, feature_bank=feature_bank,
                            contrast_bank=contrast_bank,
                            contrast_last_used=contrast_last_used,
                            run_identity=run_identity, train_cfg=cfg)
        print(f"[COG] Saved → {output_dir}/cog_params.pkl")
        sys.exit(0)

    _signal.signal(_signal.SIGINT, _handler)

    # ── Auto supervisor ──
    sup = None
    if auto_mode:
        from train.train_supervisor import Supervisor
        sup = Supervisor(output_dir, cfg, enable_auto=True,
                         val_data_path=data_path, val_shape_path=shape_path)

    last_is_safe = True  # danger priority for the next behavior-plan step
    obs_error_logged = False

    # R13: C_normal is generated from a bank of safe encoder states over many
    # batches, not one batch tiled. A single-batch normal set cannot cover the
    # input distribution and makes random threat rows win by chance. Frozen
    # write; never enters the optimizer.
    if 'danger' in params and 'C_normal' in params.get('danger', {}):
        try:
            M_d = int(params['danger']['C'].shape[0])
            normal_iter = WikiDataIter(data_path, shape_path, spans_path,
                                       B=1, N=seq_len)
            rows = []
            for _ in range(M_d):
                xn, _ = next(normal_iter)
                zn = safe_unit(encoder_forward(
                    params['encoder'], xn, cfg.n_heads))
                rows.append(np.asarray(jax.device_get(zn))[0])
            params['danger'] = dict(
                params['danger'],
                C_normal=jnp.asarray(np.stack(rows), dtype=jnp.float32))
            print(f"[COG] danger: C_normal seeded from {M_d} safe encoder "
                  f"states (structural placeholder, frozen)")
        except Exception as exc:
            print(f"[COG] danger: normal-bank seeding failed ({exc}); "
                  f"keeping structural C_normal", file=sys.stderr)

    for item in plan:
        # item['offset'] is loop-relative (schedule/log/save/bank guards);
        # abs_step is the state clock (EMA last_used, dead-code detection,
        # checkpoint names). Never use one for the other.
        abs_step = item['abs_step']
        step = abs_step
        batch = fixed_batch if fixed_batch is not None else next(data_iter)
        current_lr = schedule(abs_step)
        rng, step_rng = jax.random.split(rng)

        # Shallow backups: train_step returns fresh trees, never mutates these.
        params_backup, opt_state_backup = params, opt_state
        ema_state_backup, self_state_backup = ema_state, self_state

        # Level-1 behavior modulation: every 10 steps, danger first. The bias
        # is a real train_step input (R16), not a diagnostic that gets dropped.
        if explorer is not None:
            try:
                routing_bias = explorer.plan_step(
                    abs_step, danger=not last_is_safe)
            except Exception:
                routing_bias = np.zeros(cfg.n_lattices, dtype=np.float32)
        step_bias = jnp.asarray(routing_bias, dtype=jnp.float32)
        step_t0 = time.perf_counter()

        if sup:
            params, opt_state, loss_val, aux_out = sup.step(
                partial(train_step, routing_bias=step_bias),
                params, opt_state, batch, current_lr, step_rng,
                step=abs_step, self_state=self_state, ema_state=ema_state)
            # Supervisor keeps its historical 4-tuple and carries the EMA state
            # through aux, so its rollback path stays compatible with callers
            # that only know params/opt_state.
            ema_state = aux_out.pop('ema_state', ema_state)
        else:
            params, opt_state, ema_state, loss_val, aux_out = train_step(
                params, opt_state, ema_state, batch, current_lr, step_rng,
                self_state=self_state, step=abs_step,
                routing_bias=step_bias)
        step_time_ms = (time.perf_counter() - step_t0) * 1000.0
        if item['offset'] == 0:
            # First executed step includes JIT tracing/compilation. Exclude it
            # from black-box timing; it is not inference time.
            step_time_ms = 0.0

        # Update self state from forward pass
        if self_state is not None and aux_out.get('self_state') is not None:
            self_state = aux_out['self_state']

        loss_f = float(loss_val)

        # ── R16: self-observation stack ────────────────────────────────
        # Host-side, outside jit, and fault-isolated: a self-observation
        # failure must not change the training trajectory.
        if obs is not None:
            try:
                last_is_safe = _self_observe_step(
                    obs, narr, reflection, explorer, verifier_poller,
                    abs_step, step_time_ms, aux_out, loss_f, cfg,
                    causal_subject=causal_subject)
            except Exception as exc:
                if not obs_error_logged:
                    obs_error_logged = True
                    print(f"[SELF] observation error (training continues): "
                          f"{exc}", file=sys.stderr)

        # NaN/inf self-heal: without a Supervisor, one bad step would poison
        # params forever (every later checkpoint NaN). Roll back and skip.
        if np.isnan(loss_f) or np.isinf(loss_f):
            params, opt_state, ema_state, self_state = (
                params_backup, opt_state_backup, ema_state_backup,
                self_state_backup)
            pbar.update(1)
            continue

        # Feature bank: contrast's usage clock is updated EVERY step (R17);
        # dead detection/replacement and bank collection only at the interval.
        loop_stats = aux_out.get('loop_stats')
        if item['bank'] and loop_stats is not None and 'z_in' in loop_stats:
            # Host conversion happens only at the interval, not every step
            # (a per-step np.asarray of z forces a device sync).
            z_host = np.asarray(loop_stats['z_in'])
            feature_bank.add(z_host)
            contrast_bank.add(z_host)
        params, contrast_last_used = _contrast_bank_step(
            params, contrast_bank, contrast_last_used, loop_stats, abs_step,
            item['bank'], cfg)
        if item['bank']:
            params, ema_state = _host_sparse_bank_update(
                params, ema_state, feature_bank, abs_step, cfg)

        # Save checkpoint (only with a clean step — never on poisoned params).
        # The guard is loop-relative so resuming at step 5 with save_every=2
        # does not immediately overwrite step_000005; the name is absolute.
        if item['save']:
            ckpt_dir = os.path.join(output_dir, f"step_{abs_step:06d}")
            save_cog_checkpoint(params, ckpt_dir, abs_step,
                                self_state=self_state,
                                ema_state=ema_state, feature_bank=feature_bank,
                                contrast_bank=contrast_bank,
                                contrast_last_used=contrast_last_used,
                                run_identity=run_identity, train_cfg=cfg)

        # Task-15 diagnostics: post-bank, post-save state. Placed after the
        # checkpoint write so the callback sees exactly what was persisted.
        if on_step is not None:
            on_step(abs_step, loss_f, aux_out, params, ema_state,
                    feature_bank, contrast_bank, contrast_last_used,
                    step_time_ms)

        running_loss += loss_f
        logged_steps += 1

        if item['log']:
            # Average over the steps actually accumulated since the last log,
            # not blindly over log_every (the first resumed log starts from
            # offset 0 and would otherwise be divided by too large a number).
            avg_loss = running_loss / logged_steps
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * logged_steps / elapsed
            loss_self = float(aux_out.get('loss_self', 0.0))
            gap = avg_loss - _LOSS_FLOOR
            parts = [f"  step {abs_step:>6d} | loss={avg_loss:.4f}  gap={gap:.2f}"]
            if loss_self > 0:
                parts.append(f"self={loss_self:.6f}")
            parts.append(f"lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            tqdm.write(" | ".join(parts))
            if sup:
                sup.report(abs_step)
            running_loss = 0.0
            logged_steps = 0
            start_time = time.time()

        pbar.update(1)

    pbar.close()
    pbar.refresh()
    if reflection is not None:
        reflection.wait(timeout=30.0)
    # Task 15 (100-step smoke) reads the recorded trajectory after training.
    train_cog.last_observability = obs
    train_cog.last_narrative = narr
    train_cog.last_reflection = reflection
    train_cog.last_explorer = explorer
    final_step = start_step + steps
    final_dir = os.path.join(output_dir, f"step_{final_step:06d}")
    save_cog_checkpoint(params, final_dir, final_step, self_state=self_state,
                        ema_state=ema_state, feature_bank=feature_bank,
                        contrast_bank=contrast_bank,
                        contrast_last_used=contrast_last_used,
                        run_identity=run_identity, train_cfg=cfg)
    if sup and sup.best_params is not None:
        sup.save_best(sup.best_params, sup.best_opt_state, sup.best_step,
                      self_state=self_state, run_identity=run_identity,
                      ema_state=(sup.best_ema_state
                                 if sup.best_ema_state is not None
                                 else ema_state),
                      feature_bank=feature_bank,
                      contrast_bank=contrast_bank,
                      contrast_last_used=contrast_last_used)
    print(f"[COG] Training complete → {output_dir}/")


# ─── Checkpoint ──────────────────────────────────────────────────────────────

# Schema v2 header (32 bytes) — see train/checkpoint.py. The old 24-byte
# header is rejected on load; there is no migration path (R10).
_CB_HEADER_FMT = HEADER_FMT_V2


def _pack_cb_header(M, d, n_layers, cb_type=1, curvature=1.0):
    import struct
    return struct.pack(_CB_HEADER_FMT, int(M), int(d), int(n_layers),
                       int(cb_type), float(curvature), FORMAT_VERSION, 0, 0)


def _write_cb_bin(dir_path, filename, mat, cb_type):
    """Write single codebook matrix with 32-byte schema-v2 header."""
    import struct, zlib, os
    import numpy as _np
    mat = _np.asarray(mat, dtype=_np.float32)
    M, d = mat.shape
    data_bytes = mat.tobytes()
    crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
    hdr = struct.pack(_CB_HEADER_FMT, int(M), int(d), 1, int(cb_type), 1.0,
                      FORMAT_VERSION, 0, crc)
    path = os.path.join(dir_path, filename)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(data_bytes)
    return mat.ravel()


def _write_flat_cb(dir_path, filename, arrays, cb_type=1):
    """Write header + multiple arrays as concatenated data bytes."""
    import struct, zlib, os
    import numpy as _np
    arrays = [_np.asarray(a, dtype=_np.float32) for a in arrays]
    M = arrays[0].shape[0]
    d = arrays[0].shape[1]
    flat = _np.concatenate([a.ravel() for a in arrays]).astype(_np.float32)
    data_bytes = flat.tobytes()
    crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
    hdr = struct.pack(_CB_HEADER_FMT, int(M), int(d), len(arrays), int(cb_type),
                      1.0, FORMAT_VERSION, 0, crc)
    path = os.path.join(dir_path, filename)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(data_bytes)
    return flat


def _to_np(x):
    """Convert jax array → numpy, no-op if already numpy."""
    import numpy as _np
    return _np.asarray(x)


def _flatten_params_for_manifest(params):
    """Flatten params leaves to ``params.<path>`` arrays for manifest.json."""
    out = {}

    def _walk(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                _walk(f"{prefix}[{i}]", child)
        elif value is not None:
            arr = _to_np(value)
            if hasattr(arr, 'dtype'):
                out[f"params.{prefix}"] = arr

    _walk("", params)
    return out


def save_cog_checkpoint(params, output_dir, step, *, run_identity, train_cfg,
                        self_state=None, ema_state=None, feature_bank=None,
                        contrast_bank=None, contrast_last_used=None,
                        allow_missing_buffers=False):
    """Save full checkpoint + export codebooks + tied readout for C engine.

    run_identity is required: the checkpoint must record the tokenizer and
    corpus it was actually trained with, never a probed global default.
    train_cfg is required because the exported config feeds Python inference,
    which reshapes attention by n_heads.

    ``ema_state``, ``feature_bank``, ``contrast_bank`` and
    ``contrast_last_used`` are buffers: they live next to ``self_state`` in
    cog_params.pkl. Training must pass them. Export/test
    callers that do not train must opt in explicitly with
    ``allow_missing_buffers=True``; that records ``buffers_present=False`` and
    the checkpoint can never be resumed (``load_cog_state`` rejects it). This
    is deliberate: fabricating zeroed buffers would look resumable while
    silently restarting the EMA statistics.
    """
    import json, os, pickle, struct, hashlib, zlib
    import numpy as _np
    from train.gvalue import make_global_value_vectors

    os.makedirs(output_dir, exist_ok=True)

    # Refuse to write a checkpoint whose identity disagrees with its own tied
    # table. Cheap, and it catches a caller that threaded the wrong identity.
    _meta = run_identity.dataset_meta
    _E = _to_np(params['encoder']['embed'])
    if int(_meta["model_vocab_size"]) != _E.shape[0]:
        raise ValueError(
            f"tied table E has {_E.shape[0]} rows but run_identity says "
            f"model_vocab_size={_meta['model_vocab_size']}")
    os.makedirs(output_dir, exist_ok=True)

    buffers_present = (
        ema_state is not None and feature_bank is not None
        and contrast_bank is not None and contrast_last_used is not None)
    if not buffers_present and not allow_missing_buffers:
        raise ValueError(
            "save_cog_checkpoint requires ema_state, feature_bank, "
            "contrast_bank and contrast_last_used for a resumable checkpoint; "
            "pass allow_missing_buffers=True only for export/test checkpoints "
            "that must not be resumed")

    # Exclude the frozen Qwen weights from the pickle (≈2GB): they are
    # reloaded from the .npz on demand (see _load_qwen_checkpoint). The
    # trainable z_proj IS saved — resume must keep it.
    save_params = {k: v for k, v in params.items() if k != 'qwen'}
    save_cog_state(output_dir, params=save_params, step=step,
                   self_state=self_state, ema_state=ema_state,
                   feature_bank=feature_bank, contrast_bank=contrast_bank,
                   contrast_last_used=contrast_last_used,
                   buffers_present=buffers_present)

    # ── C推理引擎输出格式 ──────────────────────────────────────────────

    def _simvq_cb(simvq):
        return _to_np(simvq['A']) @ _to_np(simvq['W'])

    # No W_out: the readout is the encoder embedding, tied.
    E = _to_np(params['encoder']['embed'])
    d = E.shape[1]
    V = E.shape[0]

    # config.json
    enc = params.get('encoder', {})
    # Renamed from the local `cfg` it used to be: a parameter of that name
    # would have been shadowed, which is how n_heads stayed hardcoded at 4
    # against a training default of 8. Inference reshapes attention with
    # n_heads, so that was a wrong inference artifact, not a metadata quirk.
    # Nothing here is a literal any more: every entry is either a real tensor
    # shape or a training choice taken from train_cfg.
    export_cfg = {
        # Schema v2 (R10): read back by lcm.py before it loads any tensor.
        'format_version': FORMAT_VERSION,
        # Shape facts, from the tensors actually being saved.
        'd_model': d,
        'vocab_size': V,
        'n_encoder_layers': len(enc.get('layers', [])) if enc else train_cfg.n_encoder_layers,
        'M_top': _to_np(params['hrq']['top']['A']).shape[0],
        'M_fine': _to_np(params['hrq']['fine'][0]['A']).shape[0],
        'n_hrq_layers': len(params['hrq']['fine']),
        'M_sparse': _to_np(params['sparse']['C']).shape[0],
        # R10: M_lr is the low-rank codebook size (U[0].shape[0]), not the
        # shared base's row count (A_V.shape[0] == d, which mis-parsed every
        # exported lowrank file).
        'M_lr': _to_np(params['lowrank']['U'][0]).shape[0],
        'M_man': _to_np(params['manifold']['C']).shape[0],
        'M_bind': _to_np(params['binding']['key_cb'][0]['A']).shape[0],
        'M_contrast': _to_np(params['contrast']['C_a'][0]['A']).shape[0],
        'n_bind_layers': len(params['binding']['key_cb']),
        'n_contrast_layers': len(params['contrast']['C_a']),
        # R15: n_self_codes is the training-config row count (C_self rows).
        # The C loader reads n_self_modes (EMA rows, excluding the core) and
        # validates n_self_rows = 1 + n_self_modes.
        'n_self_codes': _to_np(params['self']['C_self']).shape[0],
        'n_self_rows': _to_np(params['self']['C_self']).shape[0],
        'n_self_modes': _to_np(params['self']['C_self']).shape[0] - 1,

        # Training choices, from train_cfg.
        'max_seq_len': train_cfg.max_seq_len,
        'n_heads': train_cfg.n_heads,
        'd_ff': train_cfg.d_ff,
        'n_lattices': train_cfg.n_lattices,
        'n_lr_layers': len(params['lowrank']['U']),
        'r_max': train_cfg.r_max,
        # Per-layer ranks: the lowrank .bin stores ragged U_k blocks, so the
        # loader cannot infer the split from r_max alone.
        'ranks': list(train_cfg.ranks),
        't_dim': train_cfg.t_dim,
        'n_value_pairs': train_cfg.n_value_pairs,
        'M_danger': train_cfg.M_danger,
        'alpha_self': max(float(train_cfg.alpha_self), 0.0),
        'tau_route': train_cfg.tau_route,
        'tau_route_fallback': train_cfg.tau_route_fallback,
        'lambda_lfq': train_cfg.lambda_lfq,
        'alpha_val': float(train_cfg.alpha_val),
        'max_inference_steps': train_cfg.max_inference_steps,
        'convergence_tol': train_cfg.convergence_tol,
        'entropy_threshold': train_cfg.entropy_threshold,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(export_cfg, f)

    # encoder.bin
    if enc:
        parts = [_to_np(enc['embed']).ravel(), _to_np(enc['rel_bias']).ravel()]
        for layer in enc['layers']:
            for k in ['ln1_scale','ln1_bias','w_q','w_k','w_v','w_o',
                       'ln2_scale','ln2_bias','w_1','w_2','w_3']:
                parts.append(_to_np(layer[k]).ravel())
        parts.append(_to_np(enc['q_pool']).ravel())
        parts.append(_to_np(enc['w_proj']).ravel())
        enc_flat = _np.concatenate(parts).astype(_np.float32)
        enc_flat.tofile(os.path.join(output_dir, "encoder.bin"))
    else:
        # dummy encoder (random) — exists for compatibility
        _np.random.seed(0)
        enc_flat = _np.random.randn(1).astype(_np.float32)
        enc_flat.tofile(os.path.join(output_dir, "encoder.bin"))

    # decoder.bin (new format: gen_head)
    gh = params.get('gen_head', {})
    if gh:
        parts = [_to_np(gh['w_embed']).ravel()]
        for k in ['w_q','w_k','w_v','w_o']:
            parts.append(_to_np(gh[k]).ravel())
        parts.append(_to_np(gh['w_1']).ravel())
        parts.append(_to_np(gh['w_2']).ravel())
        parts.append(_to_np(gh['w_3']).ravel())
        dec = _np.concatenate(parts).astype(_np.float32)
    else:
        # Phase 1: write E.T so the existing inference loader keeps working.
        # This duplicates bytes already in encoder.bin; it is not a second
        # training parameter and not a second Adam state. Dropping it requires
        # changing the loader, which is deliberately out of scope.
        dec = E.T.copy()
    dec.tofile(os.path.join(output_dir, "decoder.bin"))

    # Manifest payloads: the exact float32 stream written to each .bin file.
    # Named by logical tensor rather than path (R10: no paths in the manifest).
    manifest_tensors = {'encoder': enc_flat, 'decoder': dec}

    # 导出所有 codebook .bin 文件（带 LCM_CB 头部）
    codebooks_dir = output_dir

    # HRQ: all layers stacked in one file (header + all layer data)
    hrq_layers = [_simvq_cb(params['hrq']['top'])]
    for fb in params['hrq'].get('fine', []):
        hrq_layers.append(_simvq_cb(fb))
    manifest_tensors['hrq_codebook'] = _write_flat_cb(
        codebooks_dir, "hrq_codebook.bin", hrq_layers, 1)

    # Sparse: C + zero_vec in one file. The header's M counts both rows
    # (M_sparse + 1), matching train.checkpoint._save_codebook_from_params's
    # zero_vec=True layout. The C loader separates the last row as zero_vec.
    sparse_C = _to_np(params['sparse']['C'])
    sparse_zero = _to_np(params['sparse'].get(
        'zero_vec', _np.zeros((1, d), dtype=_np.float32)))
    sparse_all = _np.concatenate([sparse_C, sparse_zero], axis=0).astype(_np.float32)
    manifest_tensors['sparse_codebook'] = _write_cb_bin(
        codebooks_dir, "sparse_codebook.bin", sparse_all, 11)

    # LowRank: U_0..U_k + V (raw U matrices, not reconstructed)
    lr = params['lowrank']
    V_lr = _to_np(lr['A_V']) @ _to_np(lr['W_V'])
    parts_lr = []
    for u_k in lr['U']:
        parts_lr.append(_to_np(u_k).ravel())
    parts_lr.append(V_lr.ravel())
    lr_flat = _np.concatenate(parts_lr).astype(_np.float32)
    lr_flat.tofile(os.path.join(codebooks_dir, "lowrank_codebook.bin"))
    manifest_tensors['lowrank_codebook'] = lr_flat

    # Manifold: header + C + T
    C_m = _to_np(params['manifold']['C'])
    T_m = _to_np(params['manifold']['T'].reshape(C_m.shape[0], -1))
    manifest_tensors['manifold_codebook'] = _write_flat_cb(
        codebooks_dir, "manifold_codebook.bin", [C_m, T_m], 2)

    # Binding: single flat file (key_0, val_0, bind_0, key_1, ...)
    bind = params['binding']
    parts_bind = []
    for l in range(len(bind.get('key_cb', []))):
        for k in ['key_cb', 'val_cb', 'bind_cb']:
            parts_bind.append(_simvq_cb(bind[k][l]).ravel())
    bind_flat = _np.concatenate(parts_bind).astype(_np.float32)
    bind_flat.tofile(os.path.join(codebooks_dir, "bind_codebook.bin"))
    manifest_tensors['bind_codebook'] = bind_flat

    # Contrast: single flat file (C_a_0..C_a_n, C_b_0..C_b_n)
    contrast = params['contrast']
    parts_ct = []
    for ca in contrast.get('C_a', []):
        parts_ct.append(_simvq_cb(ca).ravel())
    for cb in contrast.get('C_b', []):
        parts_ct.append(_simvq_cb(cb).ravel())
    ct_flat = _np.concatenate(parts_ct).astype(_np.float32)
    ct_flat.tofile(os.path.join(codebooks_dir, "contrast_codebook.bin"))
    manifest_tensors['contrast_codebook'] = ct_flat

    # Routing / fusion / value scalars / self / binding projections. Before
    # this export these lived only in cog_params.pkl, so the C engine could
    # not receive them at all and inference fell back to first-layer data.
    # The file writers are shared with train.checkpoint._save_* so the two
    # export paths cannot drift; each returns the flat payload for the
    # manifest.
    manifest_tensors['route_codebook'] = _save_route(
        params, os.path.join(output_dir, "route_codebook.bin"))
    manifest_tensors['fusion'] = _save_fusion(
        params, os.path.join(output_dir, "fusion.bin"))
    manifest_tensors['value_scalars'] = _save_value_scalars(
        params, os.path.join(output_dir, "value_scalars.bin"), train_cfg)
    manifest_tensors['self_lattice'] = _save_self(
        params, os.path.join(output_dir, "self_lattice.bin"),
        alpha_self=train_cfg.alpha_self)
    manifest_tensors['binding_proj'] = _save_binding_proj(
        params, os.path.join(output_dir, "binding_proj.bin"))

    # The tokenizer this run actually trained with, not a probed default.
    # The old code copied data/tokenizer.json — the legacy 30k BPE — no matter
    # which tokenizer built the corpus.
    import shutil
    _spec = run_identity.tokenizer_spec
    shutil.copy2(_spec.path, os.path.join(output_dir, "tokenizer.json"))

    with open(os.path.join(output_dir, "run_identity.json"), "w") as f:
        json.dump({
            "format_version": FORMAT_VERSION,
            "tokenizer_id": _spec.tokenizer_id,
            "tokenizer_sha256": _spec.sha256,
            "token_data_sha256": _meta["token_data_sha256"],
            "document_spans_sha256": _meta["document_spans_sha256"],
            "model_vocab_size": int(_meta["model_vocab_size"]),
            "n_tokens": int(_meta["n_tokens"]),
            "n_docs": int(_meta["n_docs"]),
            "step": int(step),
        }, f, indent=2)

    # gvalue codebooks — schema-v2 32-byte header. The old 36-byte custom
    # header made lcm.py parse M as garbage (gv_n wrong → C engine got
    # misaligned pointers).
    gvalue_written = None
    try:
        C_pos, C_neg = make_global_value_vectors(d)
        C_p = _to_np(C_pos)
        C_n = _to_np(C_neg)
        if np.array_equal(C_p, C_n):
            raise ValueError(
                "gvalue anchors are identical; refusing to export a "
                "placeholder that lcm.py would reject")
        # STRUCTURAL, NOT SEMANTIC. These are the hand-designed R6 anchors, not
        # trained value examples. The strict margin=0.5 path will abort most
        # real inputs by design; the smoke uses use_safety=False for the
        # cognitive path and reports the strict result separately.
        _hdr = bytearray(_pack_header(C_p.shape[0], d, 1, 2, 1.0))
        _data = C_p.tobytes() + C_n.tobytes()
        struct.pack_into("<I", _hdr, HEADER_SIZE_V2 - 4,
                         _compute_checksum(_data))
        with open(os.path.join(output_dir, "gvalue_codebook.bin"), "wb") as _f:
            _f.write(_hdr)
            _f.write(_data)
            _f.write(hashlib.sha256(_data).digest())
        gvalue_written = (C_p, C_n)
    except Exception as e:
        print(f"[CKPT] gvalue write skipped: {e}")
    if gvalue_written is not None:
        C_p, C_n = gvalue_written
        manifest_tensors['gvalue_codebook'] = _np.concatenate(
            [C_p.ravel(), C_n.ravel()]).astype(_np.float32)
    # R13: danger two-half export. Both halves are frozen structural
    # placeholders, but they MUST be different data — lcm.py refuses an equal
    # pair, and an equal pair makes the danger lattice a no-op. The threat half
    # is the fixed random basis from init_cog_params; the normal half is a bank
    # of safe encoder states sampled across batches. This is not semantic
    # safety.
    danger_written = None
    try:
        danger = params.get('danger')
        if danger is None or 'C_normal' not in danger:
            raise ValueError(
                "params['danger'] has no C_normal; R13 requires a distinct "
                "threat/normal pair, not an equal placeholder")
        danger_t = _to_np(danger['C']).astype(np.float32)
        danger_n = _to_np(danger['C_normal']).astype(np.float32)
        if danger_t.shape != danger_n.shape:
            raise ValueError(
                f"danger halves have shapes {danger_t.shape} and "
                f"{danger_n.shape}")
        if np.array_equal(danger_t, danger_n):
            raise ValueError(
                "danger threat and normal halves are identical; lcm.py would "
                "reject this checkpoint")
        M_d = danger_t.shape[0]
        _data_d = danger_t.tobytes() + danger_n.tobytes()
        _sha_d = hashlib.sha256(_data_d).digest()
        _crc_d = zlib.crc32(_data_d) & 0xFFFFFFFF
        _hdr_d = struct.pack(_CB_HEADER_FMT, int(M_d), int(d), 1, 2, 1.0,
                             FORMAT_VERSION, 0, _crc_d)
        with open(os.path.join(output_dir, "danger_codebook.bin"), "wb") as _f:
            _f.write(_hdr_d)
            _f.write(_data_d)
            _f.write(_sha_d)
        danger_written = (danger_t, danger_n)
    except Exception as e:
        print(f"[CKPT] danger write skipped: {e}")
    if danger_written is not None:
        danger_t, danger_n = danger_written
        manifest_tensors['danger_codebook'] = _np.concatenate(
            [danger_t.ravel(), danger_n.ravel()]).astype(_np.float32)

    # Per-tensor hashes for the exported files, plus a summary of the pickled
    # training params so a resume can detect a truncated/mixed checkpoint.
    manifest_tensors.update(_flatten_params_for_manifest(save_params))

    # Ruling 9: logical tensor → file + payload byte offset. Header-bearing
    # v2 codebooks start after the 32-byte header; raw flat files at 0; the
    # pickled training params use storage="pickle" with offset=None.
    file_map = {
        'encoder': ('encoder.bin', 0),
        'decoder': ('decoder.bin', 0),
        'hrq_codebook': ('hrq_codebook.bin', HEADER_SIZE_V2),
        'sparse_codebook': ('sparse_codebook.bin', HEADER_SIZE_V2),
        'lowrank_codebook': ('lowrank_codebook.bin', 0),
        'manifold_codebook': ('manifold_codebook.bin', HEADER_SIZE_V2),
        'bind_codebook': ('bind_codebook.bin', 0),
        'contrast_codebook': ('contrast_codebook.bin', 0),
        'gvalue_codebook': ('gvalue_codebook.bin', HEADER_SIZE_V2),
        'danger_codebook': ('danger_codebook.bin', HEADER_SIZE_V2),
        'route_codebook': ('route_codebook.bin', 0),
        'fusion': ('fusion.bin', 0),
        'value_scalars': ('value_scalars.bin', HEADER_SIZE_V2),
        'self_lattice': ('self_lattice.bin', 0),
        'binding_proj': ('binding_proj.bin', 0),
    }
    for _name in manifest_tensors:
        if _name.startswith('params.'):
            # Pickle tensors have no byte offset; do not claim 0 as one.
            file_map[_name] = {'file': 'cog_params.pkl', 'storage': 'pickle'}
    save_manifest(output_dir, manifest_tensors, file_map=file_map)

    # 统计大小
    total_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.bin') or f.endswith('.json') or f.endswith('.pkl'):
                total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"[CKPT] Step {step}: inference format → {output_dir}/ ({total_bytes/1e6:.0f} MB)")


def load_cog_checkpoint(path, d_model=None, n_self_codes=64,
                        allow_missing_self=False, alpha_self=None):
    """Load full cognitive training checkpoint.

    R15: a checkpoint without ``params['self']['C_self']`` is rejected unless
    ``allow_missing_self`` is explicit, in which case a fresh codebook is
    created (Ruling 11: old core+modes weights are discarded, not migrated).

    Args:
        path: Path to checkpoint .pkl file.
        d_model: Model dimension (required when resetting a missing C_self).
        n_self_codes: Number of self codebook rows, including the frozen core.
        allow_missing_self: Explicit opt-in to reset a pre-R15 self codebook.
        alpha_self: Frozen config familiarity strength. When provided it
            overrides the checkpoint mirror (Ruling 34); otherwise the saved
            non-negative value is kept. Required in spirit when resetting.

    Returns:
        params, step, self_state
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        ckpt['params'])
    sp = params.get('self')
    if not (isinstance(sp, dict) and 'C_self' in sp and 'bias_self' in sp):
        if not allow_missing_self or d_model is None:
            raise ValueError(
                "checkpoint self lattice has no R15 C_self; old core+modes "
                "weights are discarded, not migrated"
                + ("" if allow_missing_self else
                   "; pass allow_missing_self=True to reset explicitly"))
        params['self'] = init_self_params(
            jax.random.PRNGKey(13), int(d_model), int(n_self_codes),
            alpha_self=0.05 if alpha_self is None else float(alpha_self))
    elif alpha_self is not None:
        params['self'] = dict(
            params['self'],
            bias_self=jnp.asarray(alpha_self, dtype=jnp.float32))
    self_state = ckpt.get('self_state')
    if self_state is None and d_model is not None:
        self_state = init_self_state(n_self_codes, d_model)
    print(f"[COG] Loaded checkpoint step {ckpt.get('step', '?')}")
    return params, ckpt.get('step', 0), self_state
