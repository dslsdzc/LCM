"""Canonical multi-lattice fusion.

R4/R5: combine the lattice outputs with routing soft masks and learnable
scaling factors only:

    z_q = layer_norm(Σ_i softmask_i · α_i · o_i + self_bias · o_self)

There is no value-signal soft penalty. Safety violations are handled by the
inference engine's hard interrupt, not by down-weighting a conflicting lattice
and continuing to reason (``b.md §4.8``, ``c.md §3``, design R5).
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


def fuse_lattices_with_aux(lattice_outputs, soft_mask, params, gvalue=None,
                           beta_val=0.5, tau_val=0.1, self_bias_weight=None):
    """Fuse all lattice outputs, returning the fusion weights alongside z_q.

    This is the canonical implementation. `fuse_lattices` is a thin wrapper for
    callers that only want z_q; anything needing the weights (or their entropy,
    as the cognitive loop's convergence criterion does) must come through here
    rather than recomputing them.

    Fusion weights are exactly ``soft_mask_i · α_i`` (normalised). The self
    lattice output, when present, is added *before* LayerNorm:

        z_q = layer_norm(Σ w_i o_i + self_bias · o_self)

    Args:
        lattice_outputs: List of (B, d) arrays, one per lattice.
        soft_mask: (B, n_lattices) routing weights.
        params: Fusion parameters with 'alpha' scaling.
        gvalue: DEPRECATED / ignored. Kept for API compatibility; R5 removed
            the value-signal soft penalty from fusion.
        beta_val: DEPRECATED / ignored, reserved for future value-conditioned
            routing. It must not change the weights (R5).
        tau_val: DEPRECATED / ignored temperature for the removed value
            signal.
        self_bias_weight: The self lattice's familiarity scalar. Despite the
            historical keyword name (kept for API compatibility, Ruling 2),
            this is no longer a constant weight: it is the per-sample
            ``self_bias`` returned by ``self_lattice_forward``. A scalar is
            broadcast over the batch; a ``(B,)`` or ``(B, 1)`` array applies
            one familiarity value per sample. May be None when no self output
            is present.

    Returns:
        z_q: (B, d) fused representation.
        weights_norm: (B, n_routed) normalised fusion weights.
        entropy: (B,) Shannon entropy of weights_norm, in nats.
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

    # Base weights from routing × scaling. R5: no value-signal modulation —
    # `gvalue`/`beta_val`/`tau_val` are deliberately ignored.
    weights = soft_mask * params['alpha'][None, :]  # (B, n_routed)

    weights_norm = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)

    z_q = jnp.einsum('bdn,bn->bd', stacked, weights_norm)  # (B, d)

    # Fuse self lattice output as additive bias. R15: self_bias is a
    # per-sample familiarity scalar, so it needs an explicit broadcast axis.
    if self_output is not None:
        bias = jnp.asarray(self_bias_weight)
        if bias.ndim == 0:
            z_q = z_q + bias * self_output
        else:
            if bias.ndim == 2 and bias.shape[-1] == 1:
                bias = bias[:, 0]
            if bias.ndim != 1 or bias.shape[0] != z_q.shape[0]:
                raise ValueError(
                    f"self_bias must be scalar or ({z_q.shape[0]},), got "
                    f"{tuple(bias.shape)}")
            z_q = z_q + bias[:, None] * self_output

    z_q = layer_norm(z_q, params['ln_scale'], params['ln_bias'])
    entropy = -jnp.sum(weights_norm * jnp.log(weights_norm + 1e-8), axis=-1)
    return z_q, weights_norm, entropy


def fuse_lattices(lattice_outputs, soft_mask, params, gvalue=None, beta_val=0.5,
                  tau_val=0.1, self_bias_weight=None):
    """Fuse all lattice outputs into a single representation.

    Thin wrapper over `fuse_lattices_with_aux` for callers that need only z_q.
    Kept so existing call sites and checkpoints are unaffected.
    """
    z_q, _, _ = fuse_lattices_with_aux(
        lattice_outputs, soft_mask, params, gvalue=gvalue, beta_val=beta_val,
        tau_val=tau_val, self_bias_weight=self_bias_weight)
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
    # Keep the cognitive state visible at every decoding position.  Injecting
    # it only as a prefix lets a shallow recurrent/linear decoder forget the
    # prompt after the first few tokens.
    target_emb = target_emb + 0.5 * z_q[:, None, :]
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
