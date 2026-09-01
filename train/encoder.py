"""Linear attention encoder with GLU.

Transforms variable-length context into a fixed-dimension query vector z.
No learnable position embeddings — uses relative position bias.
"""
import jax
import jax.numpy as jnp
from jax import lax
import jax.nn as jnn


def init_encoder_params(rng, d_model, d_ff, n_heads, n_layers, vocab_size, max_seq_len):
    """Initialize all encoder parameters."""
    keys = jax.random.split(rng, 3 + n_layers * 4)

    # Token embedding
    embed = jax.random.normal(keys[0], (vocab_size, d_model)) * 0.02

    # Relative position bias
    rel_bias = jax.random.normal(keys[1], (2 * max_seq_len - 1,)) * 0.01

    params = {'embed': embed, 'rel_bias': rel_bias, 'layers': []}

    for l in range(n_layers):
        k = keys[2 + l * 4: 2 + (l + 1) * 4]
        layer = {
            'ln1_scale': jnp.ones(d_model),
            'ln1_bias': jnp.zeros(d_model),
            'w_q': jax.random.normal(k[0], (d_model, d_model)) * (d_model ** -0.5),
            'w_k': jax.random.normal(k[1], (d_model, d_model)) * (d_model ** -0.5),
            'w_v': jax.random.normal(k[2], (d_model, d_model)) * (d_model ** -0.5),
            'w_o': jax.random.normal(k[3], (d_model, d_model)) * (d_model ** -0.5),
            # GLU params
            'ln2_scale': jnp.ones(d_model),
            'ln2_bias': jnp.zeros(d_model),
            'w_1': jax.random.normal(k[0], (d_model, d_ff)) * (d_model ** -0.5),
            'w_2': jax.random.normal(k[1], (d_model, d_ff)) * (d_model ** -0.5),
            'w_3': jax.random.normal(k[2], (d_ff, d_model)) * (d_ff ** -0.5),
        }
        params['layers'].append(layer)

    # Pooling
    params['q_pool'] = jax.random.normal(keys[-2], (d_model,)) * 0.01
    params['w_proj'] = jax.random.normal(keys[-1], (d_model, d_model)) * (d_model ** -0.5)

    return params


def layer_norm(x, scale, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def linear_attn(x, w_q, w_k, w_v, w_o, n_heads):
    """Linear Transformer with standard normalization (ELU + 1)."""
    B, N, D = x.shape
    d_h = D // n_heads

    Q = x @ w_q
    K = x @ w_k
    V = x @ w_v

    def reshape_for_heads(t):
        return t.reshape(B, N, n_heads, d_h).transpose(0, 2, 1, 3)

    Q = reshape_for_heads(Q)
    K = reshape_for_heads(K)
    V = reshape_for_heads(V)

    # Kernel: ELU(x) + 1 (ensures non-negative)
    Q = jnn.elu(Q) + 1
    K = jnn.elu(K) + 1

    kv = jnp.einsum('b h n d, b h n e -> b h d e', K, V)
    Z = jnp.einsum('b h n d, b h d e -> b h n e', Q, kv)

    K_sum = K.sum(axis=2, keepdims=True)
    norm = jnp.einsum('b h n d, b h j d -> b h n j', Q, K_sum).squeeze(-1)
    Z = Z / (jnp.expand_dims(norm, -1) + 1e-6)

    Z = Z.transpose(0, 2, 1, 3).reshape(B, N, D)
    return Z @ w_o


def glu(x, w_1, w_2, w_3):
    """Gated Linear Unit with SiLU."""
    hidden = jnn.silu(x @ w_1) * (x @ w_2)
    return hidden @ w_3


def encoder_forward(params, x, n_heads):
    """Full encoder forward pass.

    Args:
        params: Encoder parameter tree.
        x: Input tokens (B, N) int32.
        n_heads: Number of attention heads.

    Returns:
        z: Bottleneck vector (B, d_model).
    """
    B, N = x.shape
    d_model = params['embed'].shape[1]

    # Token embedding
    h = params['embed'][x]  # (B, N, d)

    for layer in params['layers']:
        # Pre-LN + linear attention
        h_norm = layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])
        h_attn = linear_attn(h_norm, layer['w_q'], layer['w_k'],
                             layer['w_v'], layer['w_o'], n_heads)
        h = h + h_attn

        # Pre-LN + GLU
        h_norm = layer_norm(h, layer['ln2_scale'], layer['ln2_bias'])
        h_glu = glu(h_norm, layer['w_1'], layer['w_2'], layer['w_3'])
        h = h + h_glu

    # Global attention pooling
    d_model = h.shape[-1]
    q_pool = params['q_pool']
    q = jnn.elu(q_pool[None, None, :]) + 1  # (1, 1, d)

    k = jnn.elu(h.reshape(-1, d_model)) + 1
    k = k.reshape(B, N, d_model)
    v = h

    kv = jnp.einsum('b n d, b n e -> b d e', k, v)
    z_bn = jnp.einsum('b d, b d e -> b e', q.squeeze(1), kv)

    K_sum = k.sum(axis=1, keepdims=True)
    norm = jnp.einsum('b d, b d -> b', q.squeeze(1), K_sum.squeeze(1))
    z_bn = z_bn / (norm[:, None] + 1e-6)

    z = z_bn @ params['w_proj']
    return z


# ─── Incremental (recurrent) encoder for fast generation ──────────────────

def init_encoder_state(params, x, n_heads):
    """Full forward + return recurrent state for incremental updates.

    Returns:
        z: (B, d) bottleneck vector.
        state: dict of per-layer cumsum caches (KV, K_sum, plus pooled cumsum).
    """
    B, N = x.shape
    d = params['embed'].shape[1]
    d_h = d // n_heads

    h = params['embed'][x]  # (B, N, d)
    layers_state = []

    for layer in params['layers']:
        h_norm = layer_norm(h, layer['ln1_scale'], layer['ln1_bias'])

        # Transpose to (B, H, N, d_h) — matches the einsum layout below and
        # the numpy reference (lcm.py _encoder_full_with_state).
        Q = (jnn.elu(h_norm @ layer['w_q']) + 1).reshape(B, N, n_heads, d_h).transpose(0, 2, 1, 3)
        K = (jnn.elu(h_norm @ layer['w_k']) + 1).reshape(B, N, n_heads, d_h).transpose(0, 2, 1, 3)
        V = (h_norm @ layer['w_v']).reshape(B, N, n_heads, d_h).transpose(0, 2, 1, 3)

        # KV cumsum over ALL positions (for recurrent update later)
        kv = jnp.einsum('b h n d, b h n e -> b h d e', K, V)
        k_sum = K.sum(axis=2)  # (B, n_heads, d_h)

        # Full attention output (bidirectional)
        Z = jnp.einsum('b h n d, b h d e -> b h n e', Q, kv)
        norm = jnp.einsum('b h n d, b h d -> b h n', Q, k_sum)[..., None]  # (B,H,N,1)
        Z = Z / (norm + 1e-6)
        Z = Z.transpose(0, 2, 1, 3).reshape(B, N, d) @ layer['w_o']

        h = h + Z

        # GLU
        h_norm = layer_norm(h, layer['ln2_scale'], layer['ln2_bias'])
        h = h + glu(h_norm, layer['w_1'], layer['w_2'], layer['w_3'])

        # Keep prev layer h (used as query for next step when incremental),
        # and the cumsums so new tokens can append
        layers_state.append({
            'kv': kv,              # (B, H, d_h, d_h)
            'k': k_sum,            # (B, H, d_h)
            'h_prev': h[:, -1:, :],  # (B, 1, d) — last token's output
        })

    # Global attention pooling
    q_pool = jnn.elu(params['q_pool'][None, None, :]) + 1  # (1, 1, d)
    k_pool = jnn.elu(h) + 1
    pool_kv = jnp.einsum('b n d, b n e -> b d e', k_pool, h)
    pool_k = k_pool.sum(axis=1)

    z_num = jnp.einsum('b d, b d e -> b e', q_pool.squeeze(1), pool_kv)
    z_den = jnp.einsum('b d, b d -> b', q_pool.squeeze(1), pool_k)
    z = (z_num / (z_den[:, None] + 1e-6)) @ params['w_proj']

    state = {
        'layers': layers_state,
        'pool_kv': pool_kv,
        'pool_k': pool_k,
        'q_pool': params['q_pool'],
        'w_proj': params['w_proj'],
    }
    return z, state


def encoder_recurrent_step(state, new_embed, layer_params, n_heads):
    """Incremental encoder step: one new token, O(d²) per layer.

    Args:
        state: Recurrent state from init_encoder_state or previous step.
        new_embed: (B, d) embedding of the new token.
        layer_params: list of layer param dicts, one per encoder layer.
        n_heads: Number of attention heads.

    Returns:
        z_new: (B, d) updated bottleneck vector.
        state: Updated recurrent state.
    """
    B = new_embed.shape[0]
    d = new_embed.shape[-1]
    d_h = d // n_heads

    h = new_embed  # (B, d) — current token's hidden state (no sequence dim)

    for l, layer in enumerate(layer_params):
        ls = state['layers'][l]
        h_2d = h  # (B, d)

        # Pre-LN
        h_norm = layer_norm(h_2d, layer['ln1_scale'], layer['ln1_bias'])

        # Single-token QKV (no N dim, just (B, d))
        q = (jnn.elu(h_norm @ layer['w_q']) + 1).reshape(B, n_heads, d_h)   # (B, H, d_h)
        k = (jnn.elu(h_norm @ layer['w_k']) + 1).reshape(B, n_heads, d_h)   # (B, H, d_h)
        v = (h_norm @ layer['w_v']).reshape(B, n_heads, d_h)                # (B, H, d_h)

        # Append to cumsum: k ⊗ v  (broadcast outer product over heads)
        kv_new = jnp.einsum('b h d, b h e -> b h d e', k, v)  # (B, H, d_h, d_h)
        ls['kv'] = ls['kv'] + kv_new
        ls['k'] = ls['k'] + k

        # Attention output for this token: φ(q) @ kv / (φ(q) @ k_sum)
        num = jnp.einsum('b h d, b h d e -> b h e', q, ls['kv'])  # (B, H, d_h)
        den = jnp.einsum('b h d, b h d -> b h', q, ls['k'])       # (B, H)
        attn_out = (num / (den[:, :, None] + 1e-6)).reshape(B, d)  # (B, d)
        attn_out = attn_out @ layer['w_o']

        # Residual
        h_2d = h_2d + attn_out

        # GLU
        h_norm = layer_norm(h_2d, layer['ln2_scale'], layer['ln2_bias'])
        h_2d = h_2d + glu(h_norm, layer['w_1'], layer['w_2'], layer['w_3'])

        # Store this layer's output as prev h for next layer
        h = h_2d
        ls['h_prev'] = h[:, None, :]  # (B, 1, d)

    # Global attention pooling — incremental
    k_pool = jnn.elu(h) + 1  # (B, d)
    v_pool = h               # (B, d)
    state['pool_kv'] = state['pool_kv'] + jnp.einsum('b d, b e -> b d e', k_pool, v_pool)
    state['pool_k'] = state['pool_k'] + k_pool

    q_pool = jnn.elu(state['q_pool'][None, :]) + 1  # (1, d)
    z_num = jnp.einsum('b d, b d e -> b e', q_pool, state['pool_kv'])
    z_den = jnp.einsum('b d, b d -> b', q_pool, state['pool_k'])
    z = (z_num / (z_den[:, None] + 1e-6)) @ state['w_proj']

    return z, state
