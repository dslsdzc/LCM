"""Qwen2.5-0.5B JAX forward pass — frozen decoder for cognitive training.

Loads pretrained weights and provides teacher-forced forward pass.
All parameters frozen — only the z_q projection layer is trainable.

Architecture (Qwen2):
  embed → [decoder × 24] → RMSNorm → lm_head (weight-tied)

Each decoder layer:
  RoPE self-attention (GQA: 14 heads, 2 KV heads) → residual → RMSNorm
  SwiGLU MLP → residual → RMSNorm

Usage:
    params = load_qwen_params("checkpoints/qwen_model/qwen_params.npz")
    logits = qwen_forward(params, input_ids, z_q, z_proj)
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


# ─── RMSNorm ──────────────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    """RMS LayerNorm."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    var = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x = x / jnp.sqrt(var + eps)
    return (x * weight).astype(dtype)


# ─── RoPE ──────────────────────────────────────────────────────────────────────

def precompute_freqs(dim, max_pos, theta=1000000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)  # (max_pos, dim/2)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply RoPE to x (B, N, H, d_head). Standard rotate-half rotation.

    Qwen2 uses the HF rotate_half convention: the head dimension is split
    into front/back halves and each frequency rotates one front element with
    the corresponding back element (NOT interleaved even/odd pairing).
    """
    B, N, H, d = x.shape
    if d % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {d}")

    n_freqs = d // 2
    # cos/sin shape: (N_full, freqs_full). Slice to (N, n_freqs)
    cos_ = cos[:N, :n_freqs]  # (N, n_freqs)
    sin_ = sin[:N, :n_freqs]

    # (N, n_freqs) → (1, N, 1, n_freqs) for broadcasting with (B, N, H, n_freqs)
    cos_ = cos_[None, :, None, :]
    sin_ = sin_[None, :, None, :]

    # rotate_half: x' = x * cos + cat(-x_back, x_front) * sin
    x1 = x[..., :n_freqs]  # front half
    x2 = x[..., n_freqs:]  # back half

    rotated = jnp.concatenate(
        [x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], axis=-1)
    return rotated


# ─── Attention (GQA) ─────────────────────────────────────────────────────────

def qwen_attn(x, params, cos, sin, mask=None):
    """Grouped-query attention with RoPE.

    Args:
        x: (B, N, d) hidden states.
        params: Dict with q_proj, k_proj, v_proj, o_proj weights.
        cos, sin: Precomputed RoPE frequencies.
        mask: Optional (B, 1, N, N) causal mask.

    Returns:
        (B, N, d) attention output.
    """
    B, N, d = x.shape
    n_heads = 14
    n_kv_heads = 2
    head_dim = d // n_heads  # 896/14 = 64

    # Project Q, K, V
    q = (x @ params['q_proj'].T).reshape(B, N, n_heads, head_dim)
    k = (x @ params['k_proj'].T).reshape(B, N, n_kv_heads, head_dim)
    v = (x @ params['v_proj'].T).reshape(B, N, n_kv_heads, head_dim)

    # Apply RoPE
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    # Expand KV heads to match Q heads (GQA)
    k = jnp.repeat(k, n_heads // n_kv_heads, axis=2)
    v = jnp.repeat(v, n_heads // n_kv_heads, axis=2)

    # Transpose for attention
    q = q.transpose(0, 2, 1, 3)  # (B, H, N, d_h)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn_weights = jnp.einsum('bhnd,bhmd->bhnm', q, k) * scale

    if mask is not None:
        attn_weights = attn_weights + mask

    attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(x.dtype)
    attn_out = jnp.einsum('bhnm,bhmd->bhnd', attn_weights, v)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, d)

    # Output projection
    return attn_out @ params['o_proj'].T


# ─── SwiGLU MLP ──────────────────────────────────────────────────────────────

def qwen_mlp(x, params):
    """SwiGLU MLP: gate_proj, up_proj, down_proj."""
    gate = jax.nn.silu(x @ params['gate_proj'].T)
    up = x @ params['up_proj'].T
    return (gate * up) @ params['down_proj'].T


# ─── Decoder Layer ────────────────────────────────────────────────────────────

def qwen_decoder_layer(h, params, cos, sin, mask=None):
    """Single Qwen2 decoder layer.

    Args:
        params: Dict with keys like 'input_layernorm', 'self_attn', etc.
                (suffix .weight already stripped during loading).
    """
    # Self-attention with pre-RMSNorm
    h_norm = rms_norm(h, params['input_layernorm'])
    attn_out = qwen_attn(h_norm, params['self_attn'], cos, sin, mask)
    h = h + attn_out

    # MLP with pre-RMSNorm
    h_norm = rms_norm(h, params['post_attention_layernorm'])
    mlp_out = qwen_mlp(h_norm, params['mlp'])
    h = h + mlp_out

    return h


# ─── Full forward ─────────────────────────────────────────────────────────────

def qwen_forward(qwen_params, input_ids, z_q=None, z_proj=None, n_layers=24):
    """Teacher-forced forward pass through Qwen2.5-0.5B.

    Args:
        qwen_params: Dict of pretrained Qwen weights (frozen).
        input_ids: (B, N) integer token IDs.
        z_q: Optional (B, d_z) cognitive state.
        z_proj: Optional (d_z, d_model) projection weight.
        n_layers: Number of decoder layers to use (default 24).

    Returns:
        logits: (B, N, V) next-token predictions.
    """
    B, N = input_ids.shape
    d = qwen_params['model.embed_tokens.weight'].shape[1]  # 896

    # Token embedding
    h = qwen_params['model.embed_tokens.weight'][input_ids]  # (B, N, 896)

    # Inject z_q as first position
    if z_q is not None and z_proj is not None:
        z_projected = z_q @ z_proj.T  # (B, d_model)
        h = h.at[:, 0, :].set(z_projected)

    # Precompute RoPE
    cos, sin = precompute_freqs(d // 14, 2048, theta=1000000.0)
    cos_n = cos[:N]
    sin_n = sin[:N]

    # Causal mask: block j > i (future), allow j <= i (past + self).
    # triu(..., k=1) keeps -inf strictly above the diagonal; tril would
    # invert the mask and let earlier tokens attend to later ones.
    mask = jnp.triu(jnp.full((N, N), -float('inf'), dtype=jnp.float32), k=1)
    mask = mask[None, None, :, :]  # (1, 1, N, N)

    # Decoder layers
    for i in range(n_layers):
        prefix = f'model.layers.{i}.'
        layer_params = {}
        for k, v in qwen_params.items():
            if k.startswith(prefix):
                local_name = k[len(prefix):]
                # Strip .weight suffix
                if local_name.endswith('.weight'):
                    local_name = local_name[:-7]
                # Parse submodule
                if 'self_attn.' in local_name:
                    sub_key = local_name.split('.', 1)[1]
                    layer_params.setdefault('self_attn', {})[sub_key] = v
                elif 'mlp.' in local_name:
                    sub_key = local_name.split('.', 1)[1]
                    layer_params.setdefault('mlp', {})[sub_key] = v
                else:
                    layer_params[local_name] = v

        h = qwen_decoder_layer(h, layer_params, cos_n, sin_n, mask)

    # Final RMSNorm
    h = rms_norm(h, qwen_params['model.norm.weight'])

    # LM head (weight-tied with embedding in Qwen2)
    lm_weight = qwen_params.get('lm_head.weight', qwen_params['model.embed_tokens.weight'])
    logits = h @ lm_weight.T

    return logits


# ─── Load weights ─────────────────────────────────────────────────────────────

def load_qwen_params(path="checkpoints/qwen_model/qwen_params.npz"):
    """Load Qwen2.5-0.5B weights from npz file and convert to JAX arrays.

    Returns:
        Dict of JAX arrays (float32) on CPU.
    """
    data = np.load(path)
    params = {}
    for k in data.files:
        params[k] = jnp.array(data[k])
    print(f"[QWEN] Loaded {len(params)} tensors from {path}")
    return params


# ─── Config ───────────────────────────────────────────────────────────────────

QWEN_CONFIG = {
    'd_model': 896,
    'n_heads': 14,
    'n_kv_heads': 2,
    'd_head': 64,
    'n_layers': 24,
    'vocab_size': 151936,
    'intermediate_size': 4864,
    'rope_theta': 1000000.0,
    'max_seq_len': 32768,
}


# ─── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check():
    """Verify shapes with random input."""
    params = {
        'model.embed_tokens.weight': jnp.ones((151936, 896)),
        'model.norm.weight': jnp.ones(896),
        'lm_head.weight': jnp.ones((151936, 896)),
    }
    # Add dummy layer params
    for i in range(24):
        # Q/K/V: q has full head count, k/v have 2 heads
        params[f'model.layers.{i}.self_attn.q_proj'] = jnp.ones((896, 896))
        params[f'model.layers.{i}.self_attn.k_proj'] = jnp.ones((128, 896))
        params[f'model.layers.{i}.self_attn.v_proj'] = jnp.ones((128, 896))
        params[f'model.layers.{i}.self_attn.o_proj'] = jnp.ones((896, 896))
        for w in ['gate_proj', 'up_proj']:
            params[f'model.layers.{i}.mlp.{w}'] = jnp.ones((4864, 896))
        params[f'model.layers.{i}.mlp.down_proj'] = jnp.ones((896, 4864))
        params[f'model.layers.{i}.input_layernorm'] = jnp.ones(896)
        params[f'model.layers.{i}.post_attention_layernorm'] = jnp.ones(896)

    x = jnp.zeros((2, 4), dtype=jnp.int32)

    # jit compile for speed
    @jax.jit
    def fwd(params, x):
        return qwen_forward(params, x, n_layers=4)

    logits = fwd(params, x)
    print(f"Forward OK: {logits.shape}")
    print("Sanity check PASSED!")
    return True


if __name__ == "__main__":
    sanity_check()
