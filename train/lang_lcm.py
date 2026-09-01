"""Language LCM — continuous language model with V4 innovations.

Architecture (active channel):
  tokens → embed → [transformer decoder × N] → LN → W_out → logits

V4 innovations integrated:
  - mHC: Manifold-Constrained Hyper-Connections (parallel residual streams + Sinkhorn mixing)
  - MTP: Multi-Token Prediction (predict D future tokens per position)

Key design:
  - Shared token embedding and W_out with the Cognitive LCM
  - Gradient flows freely (no STE, no VQ in the LM path)
"""
import functools
import jax
import jax.numpy as jnp
from jax import lax

from train.encoder import layer_norm


# ─── Dropout wrapper ──────────────────────────────────────────────────────────

def _dropout(x, rate, rng):
    if rate <= 0.0 or rng is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


# ─── Attention ────────────────────────────────────────────────────────────────

def _softmax_attention(q, k, v, mask=None):
    d_h = q.shape[-1]
    logits = jnp.einsum('bhnd,bhmd->bhnm', q, k) / jnp.sqrt(d_h)
    if mask is not None:
        logits = logits + mask
    attn = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhnm,bhmd->bhnd', attn, v)


def _causal_mask(N):
    # Causal: block j > i (future), allow j <= i (past + self).
    # triu(..., k=1) keeps the fill strictly above the diagonal; tril would
    # invert the mask and let earlier tokens attend to later ones.
    return jnp.triu(jnp.full((N, N), -1e9), k=1)


# ─── GLU ──────────────────────────────────────────────────────────────────────

def _glu(x, w_gate, w_up, w_down):
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    hidden = gate * up
    return hidden @ w_down


# ─── Codebook Soft Read ────────────────────────────────────────────────────────

def _codebook_soft_read(h, entries, w_proj, tau=0.5):
    """Differentiable soft read from a codebook.

    Continuous attention over codebook entries (no STE, no VQ).
    Hidden states attend to codebook entries via softmax, then the weighted
    sum of entries is projected back to hidden space as a residual.

    Args:
        h: (B, N, d) hidden states.
        entries: (M, d) codebook entries (language primitives).
        w_proj: (d, d) output projection.
        tau: Softmax temperature.

    Returns:
        (B, N, d) readout signal (to be added as residual).
    """
    # Attention weights: softmax over entries
    attn = jax.nn.softmax(h @ entries.T / tau, axis=-1)  # (B, N, M)
    # Weighted sum: retrieve primitives
    read = attn @ entries  # (B, N, d)
    # Project back to hidden space
    return read @ w_proj


# ─── Sinkhorn-Knopp (mHC) ─────────────────────────────────────────────────────

def _sinkhorn(logits, iters=5):
    """Sinkhorn-Knopp normalization in log-space → doubly-stochastic matrix.

    Args:
        logits: (n, n) unnormalized mixing logits.
        iters: Number of Sinkhorn iterations.

    Returns:
        (n, n) doubly-stochastic matrix (rows & columns sum to 1).
    """
    for _ in range(iters):
        logits = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits - jax.nn.logsumexp(logits, axis=-2, keepdims=True)
    return jnp.exp(logits)


# ─── mHC helpers ──────────────────────────────────────────────────────────────

def _mhc_pre_mix(h, pre_weights):
    """Combine n_hc residual streams into 1 for sublayer input.

    Args:
        h: (B, N, n_hc, d) residual streams.
        pre_weights: (n_hc, d, d) learned projections per stream.

    Returns:
        (B, N, d) combined.
    """
    n_hc = h.shape[2]
    combined = sum(h[:, :, k, :] @ pre_weights[k] for k in range(n_hc))
    return combined


def _mhc_post_mix(h_residual, h_processed, post_B, post_C_logits,
                   sinkhorn_iters=5):
    """Merge sublayer output back into n_hc residual streams.

    For each output stream r:
      h_new_r = sum_c B[r,c] * h_residual_c + C_sinkhorn[r,0] * h_processed

    C_sinkhorn is the first column of the (n_hc × n_hc) doubly-stochastic
    matrix produced by Sinkhorn-Knopp on post_C_logits.  Using the first
    column is equivalent because in a doubly-stochastic matrix all rows
    sum to 1 and all columns sum to 1, so for n_hc=2:
      C = [[a, 1-a], [1-a, a]] → column 0 = [a, 1-a]ᵀ
    which gives each stream a different processed-signal weight.

    Args:
        h_residual: (B, N, n_hc, d) input residual streams.
        h_processed: (B, N, d) sublayer output.
        post_B: (n_hc, n_hc) scalar residual mixing coefficients.
        post_C_logits: (n_hc, n_hc) mixing logits → Sinkhorn.
        sinkhorn_iters: Sinkhorn iterations.

    Returns:
        (B, N, n_hc, d) updated residual streams.
    """
    n_hc = h_residual.shape[2]
    B, N, d = h_processed.shape

    # Sinkhorn: doubly-stochastic matrix
    C = _sinkhorn(post_C_logits, sinkhorn_iters)  # (n_hc, n_hc)
    # Take first column as per-stream processed weight
    c_weights = C[:, 0]  # (n_hc,)

    # Build output streams
    parts = []
    for r in range(n_hc):
        # Residual mixing: sum_c post_B[r,c] * h_residual_c
        res = sum(post_B[r, c] * h_residual[:, :, c, :] for c in range(n_hc))
        # Processed signal
        proc = c_weights[r] * h_processed
        parts.append(res + proc)

    return jnp.stack(parts, axis=2)  # (B, N, n_hc, d)


# ─── Decoder layer ────────────────────────────────────────────────────────────

def decoder_layer_forward(h, params, N, n_heads=4, training=False,
                           dropout_rng=None, dropout_rate=0.0,
                           n_hc=1, hc_params=None, sinkhorn_iters=5,
                           cb_entries=None, tau_cb=0.1):
    """Transformer decoder layer with optional mHC + codebook soft read.

    Order: self_attn → codebook_soft_read → FFN

    Codebook read is BEFORE FFN so retrieved syntactic primitives inform
    the FFN's computation, not just decorate its output.

    Without mHC (n_hc=1 or hc_params=None): standard Pre-LN residual.
      h: (B, N, d) → (B, N, d)

    With mHC (n_hc>1 and hc_params is not None): mHC residual streams.
      h: (B, N, n_hc, d) → (B, N, n_hc, d)
    """
    d = h.shape[-1]
    causal_mask = _causal_mask(N)[None, None, :, :]
    has_mhc = n_hc > 1 and hc_params is not None
    has_cb = cb_entries is not None and 'cb_read' in params

    # ── Pre-mix (mHC) or identity ─────────────────────────────────────────
    if has_mhc:
        h_in = _mhc_pre_mix(h, hc_params['attn_pre'])  # (B,N,d)
    else:
        h_in = h

    # ── Multi-head self-attention (pre-LN) ────────────────────────────────
    h_norm = layer_norm(h_in, params['ln1_scale'], params['ln1_bias'])
    H = n_heads
    d_h = d // H

    def _split_heads(x):
        return x.reshape(-1, N, H, d_h).transpose(0, 2, 1, 3)

    q = _split_heads(h_norm @ params['w_q'])
    k = _split_heads(h_norm @ params['w_k'])
    v = _split_heads(h_norm @ params['w_v'])
    attn_out = _softmax_attention(q, k, v, causal_mask)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(-1, N, d)
    attn_out = _dropout(attn_out @ params['w_o'], dropout_rate, dropout_rng)

    # ── Post-mix (mHC) or residual add ────────────────────────────────────
    if has_mhc:
        h = _mhc_post_mix(h, attn_out, hc_params['attn_B'],
                           hc_params['attn_C_logits'], sinkhorn_iters)
    else:
        h = h + attn_out

    # ── Codebook soft read (BEFORE FFN, so FFN can use retrieved primitives) ──
    if has_cb:
        if has_mhc:
            h_in_cb = h.mean(axis=2)
        else:
            h_in_cb = h
        cb_out = jnp.zeros_like(h_in_cb)
        for i in range(len(cb_entries)):
            cb_out = cb_out + _codebook_soft_read(
                h_in_cb, cb_entries[i], params['cb_read'][i], tau=tau_cb)
        if has_mhc:
            h = h + cb_out[:, :, None, :]
        else:
            h = h + cb_out

    # ── Pre-mix FFN (mHC) or identity ─────────────────────────────────────
    if has_mhc:
        h_in_ffn = _mhc_pre_mix(h, hc_params['ffn_pre'])
    else:
        h_in_ffn = h

    # ── FFN (pre-LN) ──────────────────────────────────────────────────────
    h_norm = layer_norm(h_in_ffn, params['ln2_scale'], params['ln2_bias'])
    ffn_out = _glu(h_norm, params['w_gate'], params['w_up'], params['w_down'])
    ffn_out = _dropout(ffn_out, dropout_rate, dropout_rng)

    # ── Post-mix FFN (mHC) or residual add ────────────────────────────────
    if has_mhc:
        h = _mhc_post_mix(h, ffn_out, hc_params['ffn_B'],
                           hc_params['ffn_C_logits'], sinkhorn_iters)
    else:
        h = h + ffn_out

    return h


# ─── Init ──────────────────────────────────────────────────────────────────────

def _n_heads(cfg):
    return max(1, min(cfg.n_heads, cfg.d_model // 32))


def init_hc_params(rng, d, n_hc, n_layers):
    """Initialize mHC parameters for all decoder layers.

    Returns a list of dicts, one per decoder layer:
      attn_pre: (n_hc, d, d) — pre-mixing attention
      attn_B:   (n_hc, n_hc) — residual mixing attention
      attn_C_logits: (n_hc, n_hc) — Sinkhorn logits attention
      ffn_pre:  (n_hc, d, d) — pre-mixing FFN
      ffn_B:    (n_hc, n_hc) — residual mixing FFN
      ffn_C_logits: (n_hc, n_hc) — Sinkhorn logits FFN
    """
    if n_hc <= 1:
        return None

    keys = jax.random.split(rng, 6)
    # Pre-mix: identity initialization (stream k = input -> mostly itself)
    scale = (d ** -0.5)
    pre_all = jax.random.normal(keys[0], (n_layers, n_hc, d, d)) * scale

    # B and C: near-identity 2x2
    B_all = jnp.eye(n_hc)[None, :, :].repeat(n_layers, axis=0) * 1.0
    B_all = B_all + jax.random.normal(keys[1], (n_layers, n_hc, n_hc)) * 0.01
    # C logits: biased toward equal mixing
    C_all = jnp.zeros((n_layers, n_hc, n_hc))
    # Set diagonal slightly higher so after Sinkhorn each stream
    # gets roughly equal processed signal
    for l in range(n_layers):
        for k in range(n_hc):
            C_all = C_all.at[l, k, :].set(1.0 / n_hc)  # uniform init

    # Same for FFN
    pre_ffn = jax.random.normal(keys[3], (n_layers, n_hc, d, d)) * scale
    B_ffn = jnp.eye(n_hc)[None, :, :].repeat(n_layers, axis=0) * 1.0
    B_ffn = B_ffn + jax.random.normal(keys[4], (n_layers, n_hc, n_hc)) * 0.01
    C_ffn = jnp.zeros((n_layers, n_hc, n_hc))
    for l in range(n_layers):
        for k in range(n_hc):
            C_ffn = C_ffn.at[l, k, :].set(1.0 / n_hc)

    hc_list = []
    for l in range(n_layers):
        hc_list.append({
            'attn_pre': pre_all[l],
            'attn_B': B_all[l],
            'attn_C_logits': C_all[l],
            'ffn_pre': pre_ffn[l],
            'ffn_B': B_ffn[l],
            'ffn_C_logits': C_ffn[l],
        })
    return hc_list


def init_lang_lcm_params(rng, cfg):
    """Initialize Language LCM parameters with mHC + MTP support.

    Architecture: cfg.n_lang_layers transformer decoder with GLU.
    When cfg.n_hc > 1: adds mHC parallel residual streams.
    When cfg.n_mtp_depth > 1: adds MTP output heads.

    Structure:
      embed: (V, d) token embedding
      pos_embed: (max_seq_len, d) positional embedding
      decoder: list of transformer decoder layer params
      hc: optional list of mHC params per layer
      ln_final_scale, ln_final_bias: final LayerNorm
      W_out: (d, V) output projection
    """
    keys = jax.random.split(rng, 12)
    d = cfg.d_model
    n_layers = getattr(cfg, 'n_lang_layers', 4)
    n_hc = getattr(cfg, 'n_hc', 1)
    n_mtp = getattr(cfg, 'n_mtp_depth', 1)
    H = _n_heads(cfg)

    params = {}

    # Token embedding
    params['embed'] = jax.random.normal(keys[0], (cfg.vocab_size, d)) * (d ** -0.5)

    # Positional embedding
    max_len = getattr(cfg, 'max_seq_len', 512)
    params['pos_embed'] = jax.random.normal(keys[1], (max_len, d)) * (d ** -0.5)

    # Decoder layers + codebook read projections
    params['decoder'] = []
    for l in range(n_layers):
        kl = jax.random.split(keys[7], 10)
        layer = {
            'w_q': jax.random.normal(kl[0], (d, d)) * (d ** -0.5),
            'w_k': jax.random.normal(kl[1], (d, d)) * (d ** -0.5),
            'w_v': jax.random.normal(kl[2], (d, d)) * (d ** -0.5),
            'w_o': jax.random.normal(kl[3], (d, d)) * (d ** -0.5),
            'ln1_scale': jnp.ones(d), 'ln1_bias': jnp.zeros(d),
            'w_gate': jax.random.normal(kl[4], (d, d * 4)) * (d ** -0.5),
            'w_up': jax.random.normal(kl[5], (d, d * 4)) * (d ** -0.5),
            'w_down': jax.random.normal(kl[6], (d * 4, d)) * ((d * 4) ** -0.5),
            'ln2_scale': jnp.ones(d), 'ln2_bias': jnp.zeros(d),
            # Codebook read projections: one (d, d) per codebook type
            'cb_read': [jax.random.normal(kl[9 + i // 2], (d, d)) * (d ** -0.5)
                        for i in range(6)],
        }
        params['decoder'].append(layer)

    # Shared codebook entries (language primitives) — 6 types × 512 entries
    n_cb_types = 6
    M_cb = getattr(cfg, 'M_cb', 512)
    params['cb_entries'] = []
    for i in range(n_cb_types):
        cb = jax.random.normal(keys[2 + i % 3], (M_cb, d)) * (d ** -0.5)
        params['cb_entries'].append(cb)

    # mHC params
    if n_hc > 1:
        rng, hc_rng = jax.random.split(keys[9])
        params['hc'] = init_hc_params(hc_rng, d, n_hc, n_layers)
    else:
        params['hc'] = None

    # Final LayerNorm
    params['ln_final_scale'] = jnp.ones(d)
    params['ln_final_bias'] = jnp.zeros(d)

    # Output projection (shared for all MTP depths — weight-tied)
    params['W_out'] = jax.random.normal(keys[8], (d, cfg.vocab_size)) * (d ** -0.5)

    return params


# ─── Forward pass ─────────────────────────────────────────────────────────────

def lang_lcm_forward(params, x, cfg, rng=None, training=True,
                      dropout_rate=0.2, z_q=None, targets=None):
    """Language LCM forward pass with mHC + MTP.

    Teacher-forced training with optional dropout regularization.
    When z_q is provided, injects cognitive state at position 0.
    When targets is provided (and n_mtp_depth > 1), computes MTP aux logits.

    Args:
        params: Language LCM parameters.
        x: Input token IDs (B, N).
        cfg: LCMConfig.
        rng: JAX PRNG key for dropout.
        training: Enable dropout.
        dropout_rate: Dropout probability.
        z_q: Optional (B, d) cognitive state.
        targets: Optional (B, N) target IDs (needed for MTP embedding).

    Returns:
        logits: (B, N, V) next-token predictions.
        h: (B, N, d) final hidden state.
        aux: Dict with optional 'mtp_logits'.
    """
    B, N = x.shape
    d = cfg.d_model
    n_hc = getattr(cfg, 'n_hc', 1)
    n_mtp = getattr(cfg, 'n_mtp_depth', 1)

    # Embed
    h = params['embed'][x]  # (B, N, d)

    if z_q is not None:
        h = h.at[:, 0, :].set(z_q)

    # Positional embedding
    pos_indices = jnp.arange(N, dtype=jnp.int32)
    h = h + params['pos_embed'][pos_indices]

    # Expand to n_hc streams if mHC
    if n_hc > 1:
        h = jnp.broadcast_to(h[:, :, None, :], (B, N, n_hc, d))

    n_heads = _n_heads(cfg)
    hc_params_list = params.get('hc', None)
    sinkhorn_iters = getattr(cfg, 'hc_sinkhorn_iters', 5)

    for i, layer_params in enumerate(params['decoder']):
        if training and dropout_rate > 0.0 and rng is not None:
            rng, do_rng = jax.random.split(rng)
        else:
            do_rng = None
        l_hc = hc_params_list[i] if hc_params_list is not None else None
        h = decoder_layer_forward(
            h, layer_params, N, n_heads=n_heads,
            training=training, dropout_rng=do_rng,
            dropout_rate=dropout_rate,
            n_hc=n_hc, hc_params=l_hc,
            sinkhorn_iters=sinkhorn_iters)

    # Collapse mHC streams → single sequence
    if n_hc > 1:
        h = h.mean(axis=2)  # (B, N, d)

    h = layer_norm(h, params['ln_final_scale'], params['ln_final_bias'])
    logits = h @ params['W_out']  # (B, N, V)

    # ── MTP: Multi-Token Prediction ──────────────────────────────────────
    aux = {}
    if n_mtp > 1 and targets is not None and training:
        mtp_logits_list = []
        mtp_depths = []
        for k in range(1, n_mtp):  # k=1 → predict token_{t+2}
            N_k = N - k - 1
            if N_k <= 0:
                break
            # h[t] + embed[target_{t+k}] → predict token_{t+k+1}
            h_slice = h[:, :N_k, :]  # (B, N_k, d), positions 0..N_k-1
            e = params['embed'][targets[:, k:N - 1]]  # (B, N_k, d)
            mtp_input = h_slice + e
            mtp_l = mtp_input @ params['W_out']
            mtp_logits_list.append(mtp_l)
            mtp_depths.append(k + 1)

        if mtp_logits_list:
            aux['mtp_logits'] = mtp_logits_list
            aux['mtp_depths'] = mtp_depths

    return logits, h, aux


# ─── Autoregressive generation ────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=('cfg',))
def _gen_forward(params, x, cfg, rng):
    """JIT-compiled forward for generation (no dropout, no MTP)."""
    return lang_lcm_forward(params, x, cfg, rng=rng, training=False, dropout_rate=0.0)


def lang_lcm_generate(params, prompt, max_len, bos_id, eos_id, rng, cfg):
    """Autoregressive generation with Language LCM.

    Uses fixed-size input to avoid JIT recompilation per step.
    During generation only the main head is used (no MTP speculative decoding).
    """
    prompt_len = len(prompt)
    total = prompt_len + max_len

    tokens_list = list(prompt)
    x = jnp.zeros((1, total), dtype=jnp.int32)
    x = x.at[0, :prompt_len].set(jnp.array(prompt))

    _ = _gen_forward(params, x, cfg, rng)

    pos = prompt_len
    for _ in range(max_len):
        logits, _, _ = _gen_forward(params, x, cfg, rng)
        next_logits = logits[0, pos - 1, :]

        rng, sample_rng = jax.random.split(rng)
        next_id = int(jax.random.categorical(sample_rng, next_logits))
        tokens_list.append(next_id)

        if next_id == eos_id or next_id == 0:
            break

        x = x.at[0, pos].set(next_id)
        pos += 1

    return tokens_list
