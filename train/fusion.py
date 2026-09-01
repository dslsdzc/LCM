"""Multi-lattice fusion with global value integration.

Combines outputs from all six specialized lattices using:
- Routing soft masks (Gumbel-Softmax)
- Learnable scaling factors α_i
- Global value signals (proximity to positive/negative value anchors)
- No β_penalty (safety handled by hard interrupt upstream)
"""
import jax
import jax.numpy as jnp
from jax import lax


def init_fusion_params(rng, n_lattices, d):
    """Initialize fusion parameters."""
    return {
        'alpha': jnp.ones(n_lattices),        # Learnable scaling per lattice
        'ln_scale': jnp.ones(d),
        'ln_bias': jnp.zeros(d),
    }


def layer_norm(x, scale, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def compute_value_signal(gvalue, lattice_outputs, tau=0.1):
    """Compute value signal for each lattice output.

    Each lattice output is evaluated against the global value codebook:
    signal = softmax(-d_pos/τ) - softmax(-d_neg/τ)

    Args:
        gvalue: GValueCodebook instance.
        lattice_outputs: List of (B, d) arrays.
        tau: Temperature for softmax.

    Returns:
        value_signals: (B, n_lattices) signals in [-1, +1].
    """
    return gvalue.compute_value_signal_batch(lattice_outputs, tau)


def fuse_lattices(lattice_outputs, soft_mask, params, gvalue=None, beta_val=0.5,
                  tau_val=0.1, self_bias_weight=None):
    """Fuse all lattice outputs into a single representation.

    Fusion incorporates global value signals:
      w_i = soft_mask_i · α_i · exp(β_val · value_signal_i)

    When a self lattice output is appended (7th element), it is fused
    with a separate learned bias rather than the routing soft mask.

    Args:
        lattice_outputs: List of (B, d) arrays, one per lattice.
        soft_mask: (B, n_lattices) routing weights.
        params: Fusion parameters with 'alpha' scaling.
        gvalue: Optional GValueCodebook for value-biased fusion.
        beta_val: Value signal strength.
        tau_val: Temperature for value signal.
        self_bias_weight: Optional scalar weight for self lattice output.

    Returns:
        z_q: (B, d) fused representation.
    """
    n_lattices = len(lattice_outputs)

    # Separate self lattice if present (last element)
    if self_bias_weight is not None and n_lattices > soft_mask.shape[-1]:
        routed_outputs = lattice_outputs[:-1]
        self_output = lattice_outputs[-1]
        n_routed = len(routed_outputs)
    else:
        routed_outputs = lattice_outputs
        self_output = None
        n_routed = n_lattices

    stacked = jnp.stack(routed_outputs, axis=-1)  # (B, d, n_routed)

    # Base weights from routing × scaling
    weights = soft_mask * params['alpha'][None, :]  # (B, n_routed)

    # Value-biased modulation
    if gvalue is not None and beta_val > 0:
        value_signals = compute_value_signal(gvalue, routed_outputs, tau_val)
        weights = weights * jnp.exp(beta_val * value_signals)

    weights_norm = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

    z_q = jnp.einsum('bdn,bn->bd', stacked, weights_norm)  # (B, d)

    # Fuse self lattice output as additive bias
    if self_output is not None:
        z_q = z_q + self_bias_weight * self_output

    z_q = layer_norm(z_q, params['ln_scale'], params['ln_bias'])
    return z_q


def gen_head_forward(params_head, z_q, x, training=True):
    """Generation head with causal linear attention + GLU (teacher-forced).

    Internal: embeds all N tokens + prepends z_q (N+1 total), applies causal
    linear attention and GLU, returns logits for positions 1..N (matching
    the data iterator's shifted targets: targets[i] = x[i+1]).

    Args:
        params_head: Generation head parameters.
        z_q: Fused representation (B, d).
        x: Input token IDs (B, N). During training, used for teacher forcing.

    Returns:
        logits: (B, N, V) — logits[:, i, :] predicts targets[:, i] = x[:, i+1].
    """
    B, N = x.shape

    # Embed all input tokens, prepend z_q → (B, N+1, d)
    target_emb = params_head['w_embed'][x]  # (B, N, d)
    inputs = jnp.concatenate([z_q[:, None, :], target_emb], axis=1)  # (B, N+1, d)

    # ---- Causal linear attention: φ(x) = ELU(x) + 1 ----
    Q = jax.nn.elu(inputs @ params_head['w_q']) + 1.0  # (B, N+1, d)
    K = jax.nn.elu(inputs @ params_head['w_k']) + 1.0  # (B, N+1, d)
    V = inputs @ params_head['w_v']                     # (B, N+1, d)

    # Causal cumulative sum: KV_i = sum_{j <= i} φ(K_j) ⊗ V_j
    kv = K[:, :, :, None] @ V[:, :, None, :]              # (B, N+1, d, d)
    kv_cs = jnp.cumsum(kv, axis=1)                         # (B, N+1, d, d)
    k_cs = jnp.cumsum(K, axis=1)                           # (B, N+1, d)

    # O_i = φ(Q_i) @ KV_i / (φ(Q_i) @ sum_{j <= i} φ(K_j) + eps)
    attn = jnp.einsum('bnd,bnde->bne', Q, kv_cs) / (
        jnp.einsum('bnd,bnd->bn', Q, k_cs)[:, :, None] + 1e-8)
    attn_out = attn @ params_head['w_o']                   # (B, N+1, d)

    # ---- GLU ----
    gate = jax.nn.sigmoid(attn_out @ params_head['w_1'])
    up = attn_out @ params_head['w_2']
    glu_out = gate * up                                     # (B, N+1, d*4)

    full_logits = glu_out @ params_head['w_3']              # (B, N+1, V)
    # full_logits[:, 0, :] from z_q predicts x_0
    # full_logits[:, N, :] from [z_q, x_0..x_{N-1}] predicts x_N
    # Return positions 1..N matching targets[:, :] = x[:, 1:]
    return full_logits[:, 1:, :]  # (B, N, V)


def gen_head_generate(params_head, z_q, max_len, bos_token, eos_token, rng):
    """Autoregressive generation with causal linear attention + GLU (step-by-step).

    Uses recurrent KV-sum cache (O(d²) per step) for linear attention,
    avoiding N×N attention matrix.
    """
    d = z_q.shape[-1]

    # Initialise KV cumulative sum with z_q as the start token
    K_start = (jax.nn.elu(z_q @ params_head['w_k']) + 1.0)[None, :]  # (1, d)
    V_start = (z_q @ params_head['w_v'])[None, :]                     # (1, d)
    kv_sum = jnp.einsum('bd,be->de', K_start, V_start)               # (d, d)
    k_sum = K_start[0]                                                # (d,)

    tokens = [bos_token]
    rng_key = rng
    current_embed = z_q

    for step in range(max_len):
        # Compute query from current input
        Q = (jax.nn.elu(current_embed @ params_head['w_q']) + 1.0)  # (d,) or (B, d)

        # Receptive KV for this step: Q @ kv_sum / (Q @ k_sum + eps)
        numerator = jnp.einsum('d,de->e', Q if Q.ndim == 1 else Q[0], kv_sum)
        denominator = jnp.einsum('d,d->', Q if Q.ndim == 1 else Q[0], k_sum)[None] + 1e-8
        attn_out = (numerator / denominator) @ params_head['w_o']  # (d,)

        # GLU
        gate = jax.nn.sigmoid(attn_out @ params_head['w_1'])
        up = attn_out @ params_head['w_2']
        glu_out = gate * up  # (d*4,)

        # Vocab projection
        logits = glu_out @ params_head['w_3']  # (V,)

        # Sample
        rng_key, subkey = jax.random.split(rng_key)
        next_token = jax.random.categorical(subkey, logits)
        next_id = int(jnp.squeeze(jax.lax.stop_gradient(next_token)))

        tokens.append(next_id)

        if next_id == eos_token:
            break

        # Embed the sampled token for the next step
        current_embed = params_head['w_embed'][next_id]  # (d,)

        # Accumulate into KV cache
        K_new = (jax.nn.elu(current_embed @ params_head['w_k']) + 1.0)  # (d,)
        V_new = current_embed @ params_head['w_v']                       # (d,)
        kv_sum = kv_sum + jnp.einsum('d,e->de', K_new, V_new)
        k_sum = k_sum + K_new

    return jnp.array(tokens)


def init_gen_head_params(rng, d, vocab_size):
    """Initialize generation head parameters with causal linear attention + GLU."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(rng, 8)
    return {
        # Token embedding (teacher forcing)
        'w_embed': jax.random.normal(k1, (vocab_size, d)) * (d ** -0.5),
        # Causal linear attention: Q, K, V, O
        'w_q': jax.random.normal(k2, (d, d)) * (d ** -0.5),
        'w_k': jax.random.normal(k3, (d, d)) * (d ** -0.5),
        'w_v': jax.random.normal(k4, (d, d)) * (d ** -0.5),
        'w_o': jax.random.normal(k5, (d, d)) * (d ** -0.5),
        # GLU: gate, up, out
        'w_1': jax.random.normal(k6, (d, d * 4)) * (d ** -0.5),
        'w_2': jax.random.normal(k7, (d, d * 4)) * (d ** -0.5),
        'w_3': jax.random.normal(k8, (d * 4, vocab_size)) * ((d * 4) ** -0.5),
    }
