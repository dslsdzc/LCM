"""Dual-channel cognitive training: passive introspection + active expression.

Two output channels from the same conscious state z_q:
  - Passive: z_q @ W_out — transparent, always readable, no deception gap
  - Active:  Language LCM — fluent language model conditioned on cognitive state

The passive channel keeps the model honest (cognitive state is directly readable).
The active channel uses the Stage 1 Language LCM as a frozen decoder that
generates fluent text from cognitive state z_q (injected as start token).

Usage (Stage 2):
    python lcm.py --cog-train -d zhwiki_tokens.dat \\
      --from-lang-ckpt checkpoints/lang_lm/lang_final.pkl
"""
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train.config import LCMConfig
from train.encoder import init_encoder_params, encoder_forward
from train.lattices import (
    init_hrq_params, init_sparse_params, init_lowrank_params,
    init_manifold_params, init_binding_params, init_contrast_params,
)
from train.self_lattice import (
    init_self_params, init_self_state, self_lattice_forward,
    self_lattice_reg_loss,
)
from train.cog_loop import cog_loop_scan
from train.lattices import contrast_info_nce_loss
from train.lang_lcm import lang_lcm_forward
from train.qwen_lm import qwen_forward, load_qwen_params, QWEN_CONFIG

# Global Qwen params cache (load once, reuse across calls)
_QWEN_PARAMS = None


# ─── Load Stage 2 memory checkpoint ─────────────────────────────────────────

def load_stage2_params(resume, cfg, rng):
    """Load params from Stage 2 checkpoint (full pickle or legacy .bin format).

    Two formats:
      1. cog_params.pkl (new) — full params dict with lang_lcm, self_state.
      2. .bin files (old) — loaded via checkpoint.load_checkpoint, no lang_lcm.

    Returns:
        (params, self_state)
    """
    # Try new pickle format first (preserves lang_lcm + self_state)
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
        has_lang = 'lang_lcm' in params and params['lang_lcm'] is not None
        print(f"[COG] Loaded from cog_params.pkl (step {step}, "
              f"{'incl. lang_lcm' if has_lang else 'no lang_lcm'})")
        return params, self_state

    # Fallback: legacy .bin format — no lang_lcm, caller must provide --from-lang-ckpt
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

def _load_lang_lm_checkpoint(lang_ckpt, params):
    """Load Language LCM params from Stage 1 .pkl checkpoint.

    The Language LCM replaces the old gen_head as the active channel.
    Its parameters will be frozen (stop_gradient) during cognitive training
    so the cognitive state z_q must learn to drive the frozen language model.
    """
    print(f"[COG] Loading Language LCM from Stage 1: {lang_ckpt}")
    with open(lang_ckpt, 'rb') as f:
        ckpt = pickle.load(f)
    lang_params = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if hasattr(x, 'numpy') else x,
        ckpt['lang_params'])
    # Strip codebook entries (not used in pure-transformer Language LCM)
    lang_params.pop('codebook_entries', None)
    for layer in lang_params.get('decoder', []):
        layer.pop('cb_read', None)
        layer.pop('ln3_scale', None)
        layer.pop('ln3_bias', None)
    # Make sure pos_embed exists
    if 'pos_embed' not in lang_params:
        msg = (
            f"Checkpoint {lang_ckpt} has no pos_embed!\n"
            f"  Old checkpoints (before the pos_embed fix) were trained without\n"
            f"  position encoding and CANNOT be used for cognitive training.\n"
            f"  Train a new Language LCM first:\n"
            f"    python lcm.py --lang-train --lang-steps 1000 ..."
        )
        raise ValueError(msg)
    params['lang_lcm'] = lang_params
    print(f"[COG]  Language LCM loaded: {sum(p.size for p in jax.tree_util.tree_leaves(lang_params) if hasattr(p, 'size')):,} params")


def _load_qwen_checkpoint(qwen_path, params, d=256):
    """Load frozen Qwen2.5-0.5B as active channel.

    Qwen weights stay on CPU (read-only). A trainable z_proj
    projection layer maps (B, d) cognitive state → (B, 896) Qwen input.
    """
    global _QWEN_PARAMS
    if _QWEN_PARAMS is None:
        _QWEN_PARAMS = load_qwen_params(qwen_path)
    params['qwen'] = _QWEN_PARAMS
    # Trainable projection: z_q (d=256) → Qwen hidden (896)
    rng = jax.random.PRNGKey(42)
    qwen_d = QWEN_CONFIG['d_model']
    params['z_proj'] = jax.random.normal(rng, (qwen_d, d)) * (d ** -0.5)
    n_layers = QWEN_CONFIG['n_layers']
    print(f"[COG] Frozen Qwen2.5-0.5B ({n_layers}x{qwen_d}) loaded as active channel")
    print(f"[COG]  z_proj: ({qwen_d}, {d}) trainable")


def init_cog_params(cfg, rng, lang_ckpt=None, resume=None):
    """Initialize all trainable params for dual-channel cognitive training.

    Params:
        encoder + codebooks + W_out: trained in Stage 2 (cognitive).
        lang_lcm: loaded from Stage 1, frozen (stop_gradient).

    Args:
        lang_ckpt: Path to Language LCM .pkl checkpoint (Stage 1).
        resume: Optional Stage 2 checkpoint dir.

    Returns:
        params: Dict of all parameters.
        self_state: Dict for self-lattice runtime state.
    """
    keys = jax.random.split(rng, 12)
    d = cfg.d_model

    if resume:
        params, self_state = load_stage2_params(resume, cfg, rng)
        params['W_out'] = jax.random.normal(keys[7], (d, cfg.vocab_size)) * (d ** -0.5)
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

        params['W_out'] = jax.random.normal(keys[7], (d, cfg.vocab_size)) * (d ** -0.5)

    # Load frozen Language LCM for active channel
    use_qwen = getattr(cfg, 'use_qwen', True)
    if lang_ckpt:
        if use_qwen and lang_ckpt.endswith('.npz'):
            _load_qwen_checkpoint(lang_ckpt, params, d)
        else:
            _load_lang_lm_checkpoint(lang_ckpt, params)
    else:
        print("[COG] Warning: no Language LCM checkpoint provided; active channel disabled")
        params['lang_lcm'] = None
        params['qwen'] = None
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
        logits_1d: (V,) predicted logits from z_q @ W_out.
        target_token: int scalar — the next token.
    """
    return optax.softmax_cross_entropy_with_integer_labels(
        logits_1d[None, :], jnp.array([target_token])).mean()


# ─── Active channel: Language LCM conditioned on cognitive state ────────────

def active_channel_forward(lang_params, z_q, x, cfg):
    """Active channel: Language LM from cognitive state z_q.

    The Language LCM (4-layer transformer) is conditioned on z_q by
    replacing the first position's token embedding with z_q.
    Language LCM parameters are frozen (caller should use stop_gradient).

    Args:
        lang_params: Frozen Language LCM parameters.
        z_q: (B, d) cognitive state.
        x: (B, N) input token IDs.
        cfg: LCMConfig.

    Returns:
        logits: (B, N, V) next-token predictions.
    """
    # lang_lcm_forward returns (logits, h, aux)
    logits, _, _ = lang_lcm_forward(
        lang_params, x, cfg, rng=None, training=False,
        dropout_rate=0.0, z_q=z_q)
    return logits  # (B, N, V) — aligns with targets directly


def active_loss(logits, targets):
    """Cross-entropy for active channel (full sequence)."""
    B, N, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, V), targets.reshape(-1)).mean()


# ─── Training step ──────────────────────────────────────────────────────────

def make_train_step(cfg, optimizer, joint=False):
    """Create jitted training step with dual-channel output + self-lattice.

    Every macro step's z_q feeds both channels:
      - Passive (introspection):  z_q @ W_out → single-token CE
      - Active (expression):      Language LCM(z_q) → full-sequence CE

    The Language LCM is frozen (stop_gradient) so the gradient forces the
    cognitive state z_q to adapt to the frozen language model.

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

        def loss_fn(p):
            z = encoder_forward(p['encoder'], inputs, cfg.n_heads)  # (B, d)
            codebooks = pack_codebooks_for_c(p)

            # ── Normalise encoder output to codebook scale ────────────────
            cb_mean_norm = jnp.sqrt(sum(
                jnp.mean(jnp.sum(cb ** 2, axis=-1)) for cb in codebooks
            ) / len(codebooks))
            z_scale = jnp.sqrt(jnp.mean(jnp.sum(z ** 2, axis=-1)))
            z = z * (cb_mean_norm / (z_scale + 1e-8))

            # ── Adaptive tau ──────────────────────────────────────────────
            sample_cb = codebooks[0]
            z_sq = jnp.mean(jnp.sum(z ** 2, axis=-1))
            cb_sq = jnp.mean(jnp.sum(sample_cb ** 2, axis=-1))
            median_dist2_est = z_sq + cb_sq
            tau_adaptive = jnp.clip(median_dist2_est / 20.0, 0.1, 10.0)

            # ── Cognitive loop (vmap over batch) ──────────────────────────
            _cog = lambda zi: cog_loop_scan(
                zi, codebooks,
                max_steps=cfg.max_inference_steps,
                thresholds=None, tau=tau_adaptive)
            z_qs, diffs, entropies = jax.vmap(_cog, in_axes=0)(z)
            # z_qs: (B, max_steps, d)

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

            # ── Passive channel: z_q @ W_out ────────────────────────────
            p_logits = jnp.einsum('bsd,dv->bsv', z_qs, p['W_out'])
            p_target = targets[:, 0]
            p_loss = optax.softmax_cross_entropy_with_integer_labels(
                p_logits.reshape(-1, p_logits.shape[-1]),
                p_target[:, None].repeat(cfg.max_inference_steps, axis=1).reshape(-1),
            ).mean()

            # ── Active channel: Qwen or Language LCM (frozen) ───────────
            z_final = z_qs[:, -1, :]  # (B, d)
            use_qwen = 'qwen' in p and p['qwen'] is not None
            if use_qwen:
                qwen_params = jax.lax.stop_gradient(p['qwen'])
                z_proj = p['z_proj']  # trainable projection
                a_logits = qwen_forward(qwen_params, inputs,
                                         z_q=z_final, z_proj=z_proj,
                                         n_layers=4)
                a_loss = active_loss(a_logits, targets)
            elif p.get('lang_lcm') is not None:
                lang_params = jax.lax.stop_gradient(p['lang_lcm'])
                a_logits = active_channel_forward(lang_params, z_final, inputs, cfg)
                a_loss = active_loss(a_logits, targets)
            else:
                a_loss = jnp.array(0.0)

            # Convergence bonus
            conv = (diffs[:, -1] < cfg.convergence_tol) & (entropies[:, -1] < cfg.entropy_threshold)
            n_steps = jnp.argmax((diffs < cfg.convergence_tol).astype(jnp.float32), axis=-1) + 1

            loss = p_loss + a_loss + loss_self + jnp.mean(
                jnp.where(conv, -0.001 * jnp.log(n_steps.astype(jnp.float32) + 1e-8), 0.0))

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
                    o_loss = cfg.lambda_orth * jnp.mean(
                        jnp.sum(p['manifold']['T'] ** 2, axis=(-2, -1)))
                    stage3_extra['orth'] = o_loss
                    loss = loss + o_loss

            aux_out = {'self_state': self_state_out, 'loss_self': loss_self,
                       'stage3': stage3_extra}
            return loss, aux_out

        (loss, aux_out), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, aux_out

    return train_step


# ─── Training loop ──────────────────────────────────────────────────────────

def train_cog(cfg, output_dir, steps=50000, lr=3e-4, batch_size=1,
              seq_len=256, log_every=100, save_every=1000,
              data_path=None, shape_path=None, lang_ckpt=None,
              resume=None, joint=False, auto_mode=False):
    """Run dual-channel cognitive training (Stage 2).

    Dual channels from cognitive state z_q:
      - Passive: z_q @ W_out (transparent introspection)
      - Active:  Language LCM(z_q) (fluent language from frozen Stage 1 model)

    Args:
        lang_ckpt: Path to Stage 1 Language LCM .pkl checkpoint.
        resume: Optional Stage 2 checkpoint dir.
        joint: When True, adds Stage 3 losses.
        auto_mode: When True, enables Supervisor.
    """
    from train.data import WikiDataIter
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    rng = jax.random.PRNGKey(42)

    rng, init_rng = jax.random.split(rng)
    params, self_state = init_cog_params(cfg, init_rng, lang_ckpt=lang_ckpt,
                                          resume=resume)

    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=cfg.adam_beta1,
                     b2=cfg.adam_beta2, eps=cfg.adam_eps,
                     weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(params)

    data_iter = WikiDataIter(data_path, shape_path, B=batch_size, N=seq_len)
    train_step = make_train_step(cfg, optimizer, joint=joint)

    codebooks_flat = pack_codebooks_for_c(params)
    avg_dists = avg_codebook_distances(codebooks_flat)
    thresholds = [d * 0.15 for d in avg_dists]
    print(f"[COG] Codebook thresholds: {[f'{t:.3f}' for t in thresholds[:6]]}")

    d = cfg.d_model
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params)
                       if hasattr(p, 'size'))
    has_lang = 'lang_lcm' in params and params['lang_lcm'] is not None
    has_qwen = 'qwen' in params and params['qwen'] is not None
    if has_qwen:
        active_name = f"Qwen2.5-0.5B (frozen, {len(params['qwen'])//12} layers)"
    elif has_lang:
        active_name = "Language LCM (frozen)"
    else:
        active_name = "DISABLED"
    print(f"[COG] Dual-channel: passive (z_q @ W_out) + active ({active_name})")
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
    pbar = tqdm(total=steps, desc="cog training", unit="step")

    import signal as _signal

    def _handler(sig, frame):
        print(f"\n[COG] Interrupt at step {step}, saving checkpoint...")
        save_cog_checkpoint(params, output_dir, step, self_state=self_state)
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

        if sup:
            params, opt_state, loss_val, aux_out = sup.step(
                train_step, params, opt_state, batch, step_rng,
                step=step, self_state=self_state, lr=current_lr)
        else:
            params, opt_state, loss_val, aux_out = train_step(
                params, opt_state, batch, current_lr, step_rng, self_state=self_state)

        # Update self state from forward pass
        if self_state is not None and aux_out.get('self_state') is not None:
            self_state = aux_out['self_state']

        loss_f = float(loss_val)
        if np.isnan(loss_f) or np.isinf(loss_f):
            continue

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

        if save_every > 0 and step % save_every == 0 and step > 0:
            ckpt_dir = os.path.join(output_dir, f"step_{step:06d}")
            save_cog_checkpoint(params, ckpt_dir, step, self_state=self_state)

        pbar.update(1)

    pbar.close()
    final_dir = os.path.join(output_dir, f"step_{steps:06d}")
    save_cog_checkpoint(params, final_dir, steps, self_state=self_state)
    if sup and sup.best_params is not None:
        sup.save_best(sup.best_params, sup.best_opt_state, sup.best_step)
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


def save_cog_checkpoint(params, output_dir, step, self_state=None):
    """Save full checkpoint + export codebooks + W_out for C engine."""
    import json, os, pickle, struct
    import numpy as _np
    from train.gvalue import make_global_value_vectors

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    ckpt = jax.tree_util.tree_map(_to_np, params)
    with open(os.path.join(output_dir, "cog_params.pkl"), "wb") as f:
        pickle.dump({'params': ckpt, 'step': step, 'self_state': self_state}, f)

    # ── C推理引擎输出格式 ──────────────────────────────────────────────

    def _simvq_cb(simvq):
        return _to_np(simvq['A']) @ _to_np(simvq['W'])

    d = _to_np(params['W_out']).shape[0]
    V = _to_np(params['W_out']).shape[1]

    # config.json
    enc = params.get('encoder', {})
    cfg = {
        'd_model': d, 'vocab_size': V, 'max_seq_len': 512, 'n_heads': 4,
        'n_encoder_layers': len(enc.get('layers', [])) if enc else 2,
        'n_lattices': 6,
        'd_ff': int(1.5 * d),
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
        'n_lr_layers': 3, 'r_max': 8, 't_dim': 4,
        'n_value_pairs': 4, 'M_danger': 256,
        'n_self_codes': _to_np(params['self']['modes']).shape[0],
        'max_inference_steps': 32, 'convergence_tol': 1e-3,
        'entropy_threshold': 0.5,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

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
        dec = _to_np(params['W_out']).copy()  # fallback
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

    # tokenizer.json — 从 data/ 复制
    import shutil
    for cand in ['data/tokenizer.json', '../data/tokenizer.json']:
        if os.path.exists(cand):
            shutil.copy2(cand, os.path.join(output_dir, "tokenizer.json"))
            break

    # gvalue codebooks
    try:
        import hashlib as _hl
        C_pos, C_neg = make_global_value_vectors(d)
        C_p = _to_np(C_pos)
        C_n = _to_np(C_neg)
        _hdr = bytearray(36)
        _hdr[0:6] = b"LCM_CB"
        struct.pack_into("<I", _hdr, 6, 2)
        struct.pack_into("<I", _hdr, 10, C_p.shape[0])
        struct.pack_into("<I", _hdr, 14, d)
        struct.pack_into("<I", _hdr, 18, 1)
        _hdr[22] = 20
        struct.pack_into("<I", _hdr, 24, 0)
        struct.pack_into("<I", _hdr, 28, sum(_hdr[:28]) & 0xFFFFFFFF)
        _data = C_p.tobytes() + C_n.tobytes()
        with open(os.path.join(output_dir, "gvalue_codebook.bin"), "wb") as _f:
            _f.write(_hdr)
            _f.write(_data)
            _f.write(_hl.sha256(_data).digest())
    except Exception as e:
        print(f"[CKPT] gvalue write skipped: {e}")
    # danger codebook (dummy)
    try:
        M_d = cfg.get("M_danger", 256)
        danger_t = _np.random.randn(M_d, d).astype(_np.float32) * 0.02
        danger_n = _np.random.randn(M_d, d).astype(_np.float32) * 0.02
        _data_d = danger_t.tobytes() + danger_n.tobytes()
        _sha_d = _hl.sha256(_data_d).digest()
        _crc_d = zlib.crc32(_data_d) & 0xFFFFFFFF
        _hdr_d = struct.pack("<iiiifI", int(M_d), int(d), 1, 2, 1.0, _crc_d)
        with open(os.path.join(output_dir, "danger_codebook.bin"), "wb") as _f:
            _f.write(_hdr_d)
            _f.write(_data_d)
            _f.write(_sha_d)
    except Exception:
        pass

    # 统计大小
    total_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith('.bin') or f.endswith('.json') or f.endswith('.pkl'):
                total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"[CKPT] Step {step}: inference format → {output_dir}/ ({total_bytes/1e6:.0f} MB)")


def _write_cb_bin(dir_path, filename, mat, cb_type):
    """Write numpy matrix as LCM binary codebook file with header."""
    import numpy as _np
    import struct
    buf = bytearray(36)
    M, d = mat.shape
    buf[0:6] = b"LCM_CB"
    struct.pack_into("<I", buf, 6, 2)    # version
    struct.pack_into("<I", buf, 10, M)    # n_codes
    struct.pack_into("<I", buf, 14, d)    # dim
    struct.pack_into("<I", buf, 18, 1)    # n_layers
    buf[22] = cb_type
    buf[23] = 0
    struct.pack_into("<I", buf, 24, 0)    # c
    crc = sum(buf[:28]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 28, crc)
    struct.pack_into("<I", buf, 32, 0)    # reserved
    path = os.path.join(dir_path, filename)
    with open(path, "wb") as f:
        f.write(buf)
        mat.astype(_np.float32).tofile(f)


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
