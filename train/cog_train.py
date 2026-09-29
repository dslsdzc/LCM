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
    init_route_params, init_value_scalars,
)
from train.fusion import init_fusion_params
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_forward,
    self_lattice_reg_loss,
)
from train.cog_loop import cog_loop_scan
from train.lattices import contrast_info_nce_loss, manifold_orth_loss
from train.qwen_lm import qwen_forward, load_qwen_params, QWEN_CONFIG

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

    for key in RESUME_IDENTITY_KEYS:
        if stored[key] != current[key]:
            raise ValueError(
                f"resume identity mismatch on {key}: checkpoint has "
                f"{stored[key]!r}, this run has {current[key]!r}; a resume must "
                f"continue the same dataset")


# ─── Load Stage 2 memory checkpoint ─────────────────────────────────────────

def load_stage2_params(resume, cfg, rng):
    """Load params from Stage 2 checkpoint (full pickle or legacy .bin format).

    Two formats:
      1. cog_params.pkl (new) — full params dict with self_state.
      2. .bin files (old) — loaded via checkpoint.load_checkpoint.

    Returns:
        (params, self_state)
    """
    # Try new pickle format first (preserves self_state)
    pkl_path = os.path.join(resume, "cog_params.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            ckpt = pickle.load(f)
        params = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
            ckpt['params'])
        step = ckpt.get('step', 0)
        self_state = ckpt.get('self_state')
        if self_state is None:
            self_state = init_self_state(cfg.n_self_codes, cfg.d_model)
        print(f"[COG] Loaded from cog_params.pkl (step {step})")
        return params, self_state

    # Fallback: legacy .bin format — caller must supply --qwen-ckpt separately
    from train.checkpoint import load_checkpoint as bin_load
    loaded, _, _, step = bin_load(resume, cfg=cfg, rng=rng, load_opt=False)

    wanted = ['encoder', 'hrq', 'sparse', 'lowrank', 'manifold',
              'binding', 'contrast', 'self', 'gen_head']
    params = {k: loaded[k] for k in wanted if k in loaded}

    self_state = init_self_state(cfg.n_self_codes, cfg.d_model)

    print(f"[COG] Loaded Stage 2 checkpoint from {resume} (step {step})")
    print(f"      encoder + {len([k for k in params if k not in ('gen_head',)])} codebooks + "
          f"{'gen_head' if 'gen_head' in params else 'no'} gen_head")
    return params, self_state


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


def init_cog_params(cfg, rng, qwen_ckpt=None, resume=None):
    """Initialize all trainable params for dual-channel cognitive training.

    Params:
        encoder + codebooks: trained in Stage 2 (cognitive). The encoder's
            `embed` is also the passive readout — one token table, not two.
        qwen: the frozen bridge, loaded from its `.npz` (stop_gradient).

    Args:
        qwen_ckpt: Path to the Qwen `.npz` bridge, or None for passive-only.
        resume: Optional Stage 2 checkpoint dir.

    Returns:
        params: Dict of all parameters.
        self_state: Dict for self-lattice runtime state.
    """
    keys = jax.random.split(rng, 12)
    d = cfg.d_model

    if resume:
        params, self_state = load_stage2_params(resume, cfg, rng)
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
        params['lowrank'] = init_lowrank_params(keys[3], d, cfg.M_lr, cfg.ranks)
        params['manifold'] = init_manifold_params(keys[4], d, cfg.M_man, cfg.t_dim)
        params['binding'] = init_binding_params(keys[5], d, cfg.M_bind, cfg.n_bind_layers, cfg.r_max)
        params['contrast'] = init_contrast_params(keys[6], d, cfg.M_contrast, cfg.n_contrast_layers)

        params['self'] = init_self_params(keys[10], d, cfg.n_self_codes)
        self_state = init_self_state(cfg.n_self_codes, d)

        # Routing + fusion + per-lattice value scalars. The cognitive loop runs
        # the canonical six-lattice step, which needs all three; without them
        # there is no routing mask to fuse by and no per-lattice scaling.
        params['route'] = init_route_params(keys[8], cfg.n_lattices, d)
        params['fusion'] = init_fusion_params(keys[9], cfg.n_lattices, d)
        params['value_scalars'] = init_value_scalars(keys[11], [
            ('hrq', cfg.M_top), ('sparse', cfg.M_sparse),
            ('lowrank', cfg.M_lr), ('manifold', cfg.M_man),
            ('binding', cfg.M_bind), ('contrast', cfg.M_contrast),
        ])

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
    """Extract all codebook (K_i, d) matrices into flat list for cognitive loop."""
    flat = []

    # HRQ: top + fine per layer
    flat.append(_simvq_codebook(p['hrq']['top']))
    for fb in p['hrq']['fine']:
        flat.append(_simvq_codebook(fb))

    # Sparse
    flat.append(p['sparse']['C'])

    # LowRank: one per rank
    V = p['lowrank']['A_V'] @ p['lowrank']['W_V']
    for l, u_k in enumerate(p['lowrank']['U']):
        r_k = p['lowrank']['U'][l].shape[-1]
        flat.append(u_k @ V[:, :r_k].T)

    # Manifold
    flat.append(p['manifold']['C'])

    # Binding: key, value, bind per layer
    for i in range(len(p['binding']['key_cb'])):
        flat.append(_simvq_codebook(p['binding']['key_cb'][i]))
        flat.append(_simvq_codebook(p['binding']['val_cb'][i]))
        flat.append(_simvq_codebook(p['binding']['bind_cb'][i]))

    # Contrast: C_a, C_b per layer
    for i in range(len(p['contrast']['C_a'])):
        flat.append(_simvq_codebook(p['contrast']['C_a'][i]))
        flat.append(_simvq_codebook(p['contrast']['C_b'][i]))

    return flat


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

def make_train_step(cfg, optimizer, joint=False):
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

    Self-lattice provides internal state machine (mode selection, self output).

    When joint=True, additional Stage 3 losses are computed:
      - VQ commitment (all codebooks)
      - Contrastive NCE
      - Manifold orthogonality
    """

    @jax.jit
    def train_step(params, opt_state, batch, lr, rng, self_state=None):
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

        def loss_fn(p):
            z = encoder_forward(p['encoder'], ctx, cfg.n_heads)  # (B, d)
            codebooks = pack_codebooks_for_c(p)

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
            z_qs, diffs, entropies = cog_loop_scan(
                z, p, cfg, max_steps=cfg.max_inference_steps,
                rng=jax.random.fold_in(rng, 7))
            z_qs = jnp.transpose(z_qs, (1, 0, 2))     # (B, max_steps, d)
            diffs = jnp.transpose(diffs, (1, 0))      # (B, max_steps)
            entropies = jnp.transpose(entropies, (1, 0))

            # ── Self lattice ────────────────────────────────────────────
            z_final_mean = z_qs[:, -1, :].mean(axis=0)
            rng_self = rng
            self_state_out = None
            loss_self = jnp.array(0.0)
            if self_state is not None and 'self' in p:
                o_self, self_state_out, world_dev = self_lattice_forward(
                    p['self'], self_state, z=z_final_mean[None, :],
                    rng=rng_self, training=True)
                loss_self = self_lattice_reg_loss(p['self'], self_state_out)

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

            # Convergence bonus
            conv = (diffs[:, -1] < cfg.convergence_tol) & (entropies[:, -1] < cfg.entropy_threshold)
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

            # ── Stage 3 joint losses ─────────────────────────────────────
            stage3_extra = {}
            if joint:
                vq_total = jnp.array(0.0)
                for cb in codebooks:
                    dists = jnp.sum((z[:, None, :] - cb[None, :, :]) ** 2, axis=-1)
                    vq_total = vq_total + jnp.mean(dists.min(axis=-1))
                stage3_extra['vq'] = vq_total
                loss = loss + cfg.beta_vq * vq_total

                if 'contrast' in p:
                    c_loss = cfg.lambda_contrast * contrast_info_nce_loss(
                        p['contrast'], z, tau=0.5)
                    stage3_extra['contrast'] = c_loss
                    loss = loss + c_loss

                if cfg.lambda_orth > 0 and 'manifold' in p:
                    # λ·‖Tᵀ T − I‖², NOT λ·mean(Σ T²). The latter is plain
                    # weight decay on the tangent basis: it drives T → 0 rather
                    # than towards an orthonormal frame, collapsing the tangent
                    # space the manifold lattice projects into.
                    # No forward pass here, so the active manifold codes are
                    # recomputed with the same Euclidean criterion this loop
                    # already uses for its VQ term.
                    C_man = p['manifold']['C']
                    man_idx = jnp.argmin(
                        jnp.sum((z[:, None, :] - C_man[None, :, :]) ** 2, axis=-1),
                        axis=-1)
                    o_loss = manifold_orth_loss(
                        p['manifold']['T'], man_idx,
                        cfg.n_orth_samples, lambda_orth=cfg.lambda_orth,
                        rng=rng)
                    stage3_extra['orth'] = o_loss
                    loss = loss + o_loss

            aux_out = {'self_state': self_state_out, 'loss_self': loss_self,
                       'z_margin': z_margin, 'stage3': stage3_extra}
            return loss, aux_out

        (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads)
        # Frozen Qwen is kept out of the optimizer tree (init side in train_cog):
        # it must not accumulate state or weight decay — it only serves loss_fn.
        trainable = {k: v for k, v in params.items() if k != 'qwen'}
        updates, new_opt = optimizer.update(
            {k: v for k, v in grads.items() if k != 'qwen'},
            opt_state, trainable)
        # The optimizer is built with learning_rate=1.0, so its own scale is
        # already -1 (optax adds updates). Multiplying by the live `lr` here is
        # what actually applies the schedule — and what lets the Supervisor
        # reduce the rate mid-run. See the optimizer construction in train_cog.
        updates = jax.tree_util.tree_map(lambda u: lr * u, updates)
        new_params = {**optax.apply_updates(trainable, updates),
                      'qwen': params['qwen']}
        return new_params, new_opt, loss, aux_out

    return train_step


# ─── Training loop ──────────────────────────────────────────────────────────

def train_cog(cfg, output_dir, steps=50000, lr=3e-4, batch_size=1,
              seq_len=256, log_every=100, save_every=1000,
              data_path=None, shape_path=None, spans_path=None,
              qwen_ckpt=None, resume=None, joint=False, auto_mode=False,
              full_verify=True):
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
    """
    from train.data import WikiDataIter
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify corpus/tokenizer BEFORE allocating model parameters.
    spec, meta = resolve_dataset_identity(
        data_path, shape_path, spans_path, full_verify=full_verify,
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
    params, self_state = init_cog_params(cfg, init_rng, qwen_ckpt=qwen_ckpt,
                                          resume=resume)

    # 5. Assert the actual tensors, do not assume init did the right thing.
    verify_tied_table(params, meta, cfg)
    verify_qwen_extent(params, meta)

    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    # learning_rate=1.0, deliberately not `schedule`. A schedule bound here is
    # stored inside opt_state and cannot be varied per step, which made
    # train_step's `lr` argument dead code: the Supervisor's "LR reduced" line
    # printed a new number while training carried on down the original curve.
    # train_step now applies the live schedule value to the updates itself.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    # Qwen stays frozen: keep it out of the optimizer tree entirely (no state,
    # no weight-decay drift — adamw's decoupled decay would otherwise slowly
    # erode the frozen bridge). train_step() filters it out of the update tree.
    opt_state = optimizer.init({k: v for k, v in params.items() if k != 'qwen'})

    # 6. Sampling must consume the same dataset identity.
    data_iter = WikiDataIter(data_path, shape_path, spans_path,
                             B=batch_size, N=seq_len)
    train_step = make_train_step(cfg, optimizer, joint=joint)

    codebooks_flat = pack_codebooks_for_c(params)
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
    print(f"[COG] Self-lattice: {cfg.n_self_codes} modes")
    print(f"[COG] Steps: {steps}, B={batch_size}, N={seq_len}, lr={lr}")
    if joint:
        print(f"[COG] Joint mode: + Stage 3 losses (VQ + contrastive + orth)")
    print()

    import numpy as _np_np
    V = cfg.vocab_size
    _LN_V = float(_np_np.log(V))  # passive random baseline ≈ 10.31
    _LOSS_FLOOR = 0.0  # active channel floor = 0 (language LCM can reach low loss)
    print()

    running_loss = 0.0
    start_time = time.time()
    pbar = tqdm(total=steps, desc="cog training", unit="step", file=sys.stderr, mininterval=0.5)

    import signal as _signal

    def _handler(sig, frame):
        print(f"\n[COG] Interrupt at step {step}, saving checkpoint...")
        save_cog_checkpoint(params, output_dir, step, self_state=self_state,
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

    for step in range(steps):
        batch = next(data_iter)
        current_lr = schedule(step)
        rng, step_rng = jax.random.split(rng)

        # Shallow backups: train_step returns fresh trees, never mutates these.
        params_backup, opt_state_backup = params, opt_state
        self_state_backup = self_state

        if sup:
            params, opt_state, loss_val, aux_out = sup.step(
                train_step, params, opt_state, batch, current_lr, step_rng,
                step=step, self_state=self_state)
        else:
            params, opt_state, loss_val, aux_out = train_step(
                params, opt_state, batch, current_lr, step_rng, self_state=self_state)

        # Update self state from forward pass
        if self_state is not None and aux_out.get('self_state') is not None:
            self_state = aux_out['self_state']

        loss_f = float(loss_val)

        # NaN/inf self-heal: without a Supervisor, one bad step would poison
        # params forever (every later checkpoint NaN). Roll back and skip.
        if np.isnan(loss_f) or np.isinf(loss_f):
            params, opt_state, self_state = (
                params_backup, opt_state_backup, self_state_backup)
            pbar.update(1)
            continue

        # Save checkpoint (only with a clean step — never on poisoned params)
        if save_every > 0 and step % save_every == 0 and step > 0:
            ckpt_dir = os.path.join(output_dir, f"step_{step:06d}")
            save_cog_checkpoint(params, ckpt_dir, step, self_state=self_state,
                                run_identity=run_identity, train_cfg=cfg)

        running_loss += loss_f

        if step % log_every == 0 and step > 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - start_time
            tok_s = batch_size * seq_len * log_every / elapsed
            loss_self = float(aux_out.get('loss_self', 0.0))
            gap = avg_loss - _LOSS_FLOOR
            parts = [f"  step {step:>6d} | loss={avg_loss:.4f}  gap={gap:.2f}"]
            if loss_self > 0:
                parts.append(f"self={loss_self:.6f}")
            parts.append(f"lr={current_lr:.2e} | {tok_s:.0f} tok/s")
            tqdm.write(" | ".join(parts))
            if sup:
                sup.report(step)
            running_loss = 0.0
            start_time = time.time()

        pbar.update(1)

    pbar.close()
    pbar.refresh()
    final_dir = os.path.join(output_dir, f"step_{steps:06d}")
    save_cog_checkpoint(params, final_dir, steps, self_state=self_state,
                        run_identity=run_identity, train_cfg=cfg)
    if sup and sup.best_params is not None:
        sup.save_best(sup.best_params, sup.best_opt_state, sup.best_step,
                      self_state=self_state, run_identity=run_identity)
    print(f"[COG] Training complete → {output_dir}/")


# ─── Checkpoint ──────────────────────────────────────────────────────────────

# 24-byte header format (matches checkpoint.py + lcm.py inference engine)
_CB_HEADER_FMT = '<iiiifI'


def _pack_cb_header(M, d, n_layers, cb_type=1, curvature=1.0):
    import struct
    return struct.pack(_CB_HEADER_FMT, int(M), int(d), int(n_layers), int(cb_type), float(curvature), 0)


def _write_cb_bin(dir_path, filename, mat, cb_type):
    """Write single codebook matrix with 24-byte header."""
    import struct, zlib, os
    import numpy as _np
    mat = _np.asarray(mat, dtype=_np.float32)
    M, d = mat.shape
    data_bytes = mat.tobytes()
    crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
    hdr = struct.pack(_CB_HEADER_FMT, int(M), int(d), 1, int(cb_type), 1.0, crc)
    path = os.path.join(dir_path, filename)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(data_bytes)


def _write_flat_cb(dir_path, filename, arrays, cb_type=1):
    """Write header + multiple arrays as concatenated data bytes."""
    import struct, zlib, os
    import numpy as _np
    arrays = [_np.asarray(a, dtype=_np.float32) for a in arrays]
    M = arrays[0].shape[0]
    d = arrays[0].shape[1]
    data_bytes = b''.join(a.tobytes() for a in arrays)
    crc = zlib.crc32(data_bytes) & 0xFFFFFFFF
    hdr = struct.pack(_CB_HEADER_FMT, int(M), int(d), len(arrays), int(cb_type), 1.0, crc)
    path = os.path.join(dir_path, filename)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(data_bytes)


def _to_np(x):
    """Convert jax array → numpy, no-op if already numpy."""
    import numpy as _np
    return _np.asarray(x)


def save_cog_checkpoint(params, output_dir, step, *, run_identity, train_cfg,
                        self_state=None):
    """Save full checkpoint + export codebooks + tied readout for C engine.

    run_identity is required: the checkpoint must record the tokenizer and
    corpus it was actually trained with, never a probed global default.
    train_cfg is required because the exported config feeds Python inference,
    which reshapes attention by n_heads.
    """
    import json, os, pickle, struct
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

    # Exclude the frozen Qwen weights from the pickle (≈2GB): they are
    # reloaded from the .npz on demand (see _load_qwen_checkpoint). The
    # trainable z_proj IS saved — resume must keep it.
    save_params = {k: v for k, v in params.items() if k != 'qwen'}
    ckpt = jax.tree_util.tree_map(_to_np, save_params)
    with open(os.path.join(output_dir, "cog_params.pkl"), "wb") as f:
        pickle.dump({'params': ckpt, 'step': step, 'self_state': self_state}, f)

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
        # Shape facts, from the tensors actually being saved.
        'd_model': d,
        'vocab_size': V,
        'n_encoder_layers': len(enc.get('layers', [])) if enc else train_cfg.n_encoder_layers,
        'M_top': _to_np(params['hrq']['top']['A']).shape[0],
        'M_fine': _to_np(params['hrq']['fine'][0]['A']).shape[0],
        'n_hrq_layers': len(params['hrq']['fine']),
        'M_sparse': _to_np(params['sparse']['C']).shape[0],
        'M_lr': _to_np(params['lowrank']['A_V']).shape[0],
        'M_man': _to_np(params['manifold']['C']).shape[0],
        'M_bind': _to_np(params['binding']['key_cb'][0]['A']).shape[0],
        'M_contrast': _to_np(params['contrast']['C_a'][0]['A']).shape[0],
        'n_bind_layers': len(params['binding']['key_cb']),
        'n_contrast_layers': len(params['contrast']['C_a']),
        'n_self_codes': _to_np(params['self']['modes']).shape[0],

        # Training choices, from train_cfg.
        'max_seq_len': train_cfg.max_seq_len,
        'n_heads': train_cfg.n_heads,
        'd_ff': train_cfg.d_ff,
        'n_lattices': train_cfg.n_lattices,
        'n_lr_layers': train_cfg.n_lr_layers,
        'r_max': train_cfg.r_max,
        't_dim': train_cfg.t_dim,
        'n_value_pairs': train_cfg.n_value_pairs,
        'M_danger': train_cfg.M_danger,
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
        _np.concatenate(parts).astype(_np.float32).tofile(
            os.path.join(output_dir, "encoder.bin"))
    else:
        # dummy encoder (random) — exists for compatibility
        _np.random.seed(0)
        dummy = _np.random.randn(1).astype(_np.float32)
        dummy.tofile(os.path.join(output_dir, "encoder.bin"))

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

    # 导出所有 codebook .bin 文件（带 LCM_CB 头部）
    codebooks_dir = output_dir
    cb_entries = []

    # HRQ: all layers stacked in one file (header + all layer data)
    hrq_layers = [_simvq_cb(params['hrq']['top'])]
    for fb in params['hrq'].get('fine', []):
        hrq_layers.append(_simvq_cb(fb))
    _write_flat_cb(codebooks_dir, "hrq_codebook.bin", hrq_layers, 1)

    # Sparse
    _write_cb_bin(codebooks_dir, "sparse_codebook.bin", _to_np(params['sparse']['C']), 11)

    # LowRank: U_0..U_k + V (raw U matrices, not reconstructed)
    lr = params['lowrank']
    V_lr = _to_np(lr['A_V']) @ _to_np(lr['W_V'])
    parts_lr = []
    for u_k in lr['U']:
        parts_lr.append(_to_np(u_k).ravel())
    parts_lr.append(V_lr.ravel())
    _np.concatenate(parts_lr).astype(_np.float32).tofile(
        os.path.join(codebooks_dir, "lowrank_codebook.bin"))

    # Manifold: header + C + T
    C_m = _to_np(params['manifold']['C'])
    T_m = _to_np(params['manifold']['T'].reshape(C_m.shape[0], -1))
    _write_flat_cb(codebooks_dir, "manifold_codebook.bin", [C_m, T_m], 2)

    # Binding: single flat file (key_0, val_0, bind_0, key_1, ...)
    bind = params['binding']
    parts_bind = []
    for l in range(len(bind.get('key_cb', []))):
        for k in ['key_cb', 'val_cb', 'bind_cb']:
            parts_bind.append(_simvq_cb(bind[k][l]).ravel())
    _np.concatenate(parts_bind).astype(_np.float32).tofile(
        os.path.join(codebooks_dir, "bind_codebook.bin"))

    # Contrast: single flat file (C_a_0..C_a_n, C_b_0..C_b_n)
    contrast = params['contrast']
    parts_ct = []
    for ca in contrast.get('C_a', []):
        parts_ct.append(_simvq_cb(ca).ravel())
    for cb in contrast.get('C_b', []):
        parts_ct.append(_simvq_cb(cb).ravel())
    _np.concatenate(parts_ct).astype(_np.float32).tofile(
        os.path.join(codebooks_dir, "contrast_codebook.bin"))

    # The tokenizer this run actually trained with, not a probed default.
    # The old code copied data/tokenizer.json — the legacy 30k BPE — no matter
    # which tokenizer built the corpus.
    import shutil
    _spec = run_identity.tokenizer_spec
    shutil.copy2(_spec.path, os.path.join(output_dir, "tokenizer.json"))

    with open(os.path.join(output_dir, "run_identity.json"), "w") as f:
        json.dump({
            "tokenizer_id": _spec.tokenizer_id,
            "tokenizer_sha256": _spec.sha256,
            "token_data_sha256": _meta["token_data_sha256"],
            "document_spans_sha256": _meta["document_spans_sha256"],
            "model_vocab_size": int(_meta["model_vocab_size"]),
            "n_tokens": int(_meta["n_tokens"]),
            "n_docs": int(_meta["n_docs"]),
            "step": int(step),
        }, f, indent=2)

    # gvalue codebooks — standard 24-byte header. The old 36-byte custom
    # header made lcm.py parse M as garbage (gv_n wrong → C engine got
    # misaligned pointers).
    try:
        import hashlib as _hl
        from train.checkpoint import _pack_header, _compute_checksum
        C_pos, C_neg = make_global_value_vectors(d)
        C_p = _to_np(C_pos)
        # PLACEHOLDER — NOT A WORKING SAFETY LAYER.
        # Identical halves → pos_d_min == neg_d_min → the C engine's margin
        # check (pos > neg - margin ⇒ unsafe) never fires, and lcm.py detects
        # the equality and disables gvalue entirely. Real anchors are not
        # trained yet; distinct halves would abort the cognitive loop on
        # (almost) every input instead, which is worse. See README §Safety.
        print("[CKPT] WARNING: exporting PLACEHOLDER gvalue (pos == neg) — "
              "the global value safety check is DISABLED in the C engine")
        C_n = C_p.copy()
        _hdr = bytearray(_pack_header(C_p.shape[0], d, 1, 2, 1.0))
        _data = C_p.tobytes() + C_n.tobytes()
        struct.pack_into("<I", _hdr, 20, _compute_checksum(_data))
        with open(os.path.join(output_dir, "gvalue_codebook.bin"), "wb") as _f:
            _f.write(_hdr)
            _f.write(_data)
            _f.write(_hl.sha256(_data).digest())
    except Exception as e:
        print(f"[CKPT] gvalue write skipped: {e}")
    # danger codebook — PLACEHOLDER, same story as gvalue above: identical
    # halves with a fixed seed → danger_score ≡ 0 → the danger lattice never
    # fires. The danger lattice is not part of cognitive training yet.
    # Matches checkpoint._save_danger.
    try:
        print("[CKPT] WARNING: exporting PLACEHOLDER danger codebook "
              "(threat == normal) — the danger lattice is INACTIVE")
        import zlib as _zl
        M_d = export_cfg["M_danger"]
        _np.random.seed(0)
        danger_t = _np.random.randn(M_d, d).astype(_np.float32) * 0.02
        danger_n = danger_t.copy()
        _data_d = danger_t.tobytes() + danger_n.tobytes()
        _sha_d = _hl.sha256(_data_d).digest()
        _crc_d = _zl.crc32(_data_d) & 0xFFFFFFFF
        _hdr_d = struct.pack("<iiiifI", int(M_d), int(d), 1, 2, 1.0, _crc_d)
        with open(os.path.join(output_dir, "danger_codebook.bin"), "wb") as _f:
            _f.write(_hdr_d)
            _f.write(_data_d)
            _f.write(_sha_d)
    except Exception as e:
        print(f"[CKPT] danger write skipped: {e}")

    # 统计大小
    total_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.bin') or f.endswith('.json') or f.endswith('.pkl'):
                total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"[CKPT] Step {step}: inference format → {output_dir}/ ({total_bytes/1e6:.0f} MB)")


def load_cog_checkpoint(path, d_model=None, n_self_codes=64):
    """Load full cognitive training checkpoint.

    Args:
        path: Path to checkpoint .pkl file.
        d_model: Model dimension (for re-init self_state if not saved).
        n_self_codes: Number of self modes.

    Returns:
        params, step, self_state
    """
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        ckpt['params'])
    self_state = ckpt.get('self_state')
    if self_state is None and d_model is not None:
        self_state = init_self_state(n_self_codes, d_model)
    print(f"[COG] Loaded checkpoint step {ckpt.get('step', '?')}")
    return params, ckpt.get('step', 0), self_state
