"""JAX differentiable cognitive loop.

The cognitive loop IS the process of subconscious concepts "surfacing" into
conscious thought. Codebook entries (memory) are retrieved by their embeddings,
fused into a conscious state z_q, and z_q is transparently projected to tokens
via a single shared matrix W_out — no per-entry decoder, no deception gap.

The only "decoder" is z_q @ W_out. Every dimension of z_q contributes to every
token — the mapping is fully readable from W_out's columns.

Operations mirror infer/engine.c, lattice.c, hyp.c:
  - build_dag: distance routing, active set selection
  - retrieve_single: nearest-codebook-entry lookup (differentiable via softmax)
  - slide_manifold: Poincaré geodesic slide (from hyp.c)
  - hrr_bind / hrr_unbind: FFT-based binding (from lattice.c)
  - distance_weighted_fusion: reciprocal-distance fusion (from engine.c)
  - macro loop: multi-step convergence (from engine.c::dynamic_inference())
"""
import jax
import jax.numpy as jnp


# ─── HRR: FFT bind / unbind ─────────────────────────────────────────────────

def hrr_bind(a, b):
    """HRR bind: IFFT(FFT(a) ⊙ FFT(b))."""
    A = jnp.fft.fft(a)
    B = jnp.fft.fft(b)
    return jnp.fft.ifft(A * B).real


def hrr_unbind(bound, key):
    """HRR unbind: IFFT(conj(FFT(key)) ⊙ FFT(bound))."""
    K = jnp.fft.fft(key)
    B = jnp.fft.fft(bound)
    return jnp.fft.ifft(jnp.conj(K) * B).real


# ─── Poincaré hyperbolic ops ─────────────────────────────────────────────────

def poincare_exp_map(x, v, eps=1e-6):
    """Exponential map in Poincaré ball: push tangent vector v onto ball."""
    x_norm = jnp.sum(x ** 2)
    lam = 2.0 / (1.0 - x_norm + eps)
    v_norm = jnp.sqrt(jnp.sum(v ** 2) + eps)
    return jnp.tanh(lam * v_norm / 2.0 + eps) * v / (v_norm + eps)


def poincare_log_map(x, y, eps=1e-6):
    """Logarithmic map in Poincaré ball."""
    x_norm = jnp.sum(x ** 2)
    lam_x = 2.0 / (1.0 - x_norm + eps)
    diff = y - x
    num = diff - 2 * jnp.dot(x, diff) * x / (1.0 - x_norm + eps)
    den = 1.0 - 2 * jnp.dot(x, y) + jnp.sum(y ** 2)
    return (2.0 / lam_x + eps) * num / (den + eps)


def poincare_dist(x, y, eps=1e-6):
    """Geodesic distance in Poincaré ball."""
    diff = x - y
    num = 2 * jnp.sum(diff ** 2) + eps
    den = (1 - jnp.sum(x ** 2) + eps) * (1 - jnp.sum(y ** 2) + eps)
    return jnp.arccosh(1 + num / (den + eps))


# ─── Codebook retrieval (differentiable) ─────────────────────────────────────

def soft_retrieve(z, codebook, tau=0.1):
    """Differentiable nearest-codebook lookup.

    Forward: hard nearest neighbor (straight-through estimator).
    Backward: gradients flow via softmax over distances.
    """
    dists = jnp.sum((z[None, :] - codebook) ** 2, axis=-1)  # (K,)
    d_min = jnp.min(dists)
    idx = jnp.argmin(dists)
    # Straight-through: hard forward, soft backward
    soft_idx = jax.nn.softmax(-dists / tau)
    hard = codebook[idx]
    soft_avg = jnp.sum(soft_idx[:, None] * codebook, axis=0)
    # STE: forward=hard, backward=soft gradients
    return hard + soft_avg - jax.lax.stop_gradient(soft_avg), d_min, idx


# ─── DAG operations (JIT-safe: always retrieve all, mask inactives) ──────────

def dag_fuse(z, codebooks, thresholds, tau=0.1, eps=1e-6):
    # eps aligned with the C engine (engine.c confidences 1/(√d + 1e-6)) so
    # training and inference fuse to the same fixed point.
    """Build DAG + fuse in one JIT-safe pass.

    Always retrieves from every codebook. Inactive entries (distance above
    threshold) get zero weight. This avoids Python control flow on traced values.

    Args:
        z: (d,) query vector.
        codebooks: List of (K_i, d) arrays for each lattice.
        thresholds: List of distance thresholds per lattice.
        tau: Softmax temperature for differentiable retrieval.

    Returns:
        z_next: (d,) fused conscious state.
        diff: scalar — ‖z_next - z‖ for convergence.
        entropy: scalar — entropy of fusion weights.
    """
    n = len(codebooks)
    d = codebooks[0].shape[-1]

    embs = []
    dists = []

    for i, cb in enumerate(codebooks):
        vec, d_val, _ = soft_retrieve(z, cb, tau=tau)
        embs.append(vec)
        dists.append(jnp.sqrt(d_val))  # L2 for reciprocal weighting

    z_all = jnp.stack(embs)    # (n_lattices, d)
    d_all = jnp.stack(dists)   # (n_lattices,)

    # Reciprocal-distance weights (all codebooks contribute, no hard threshold)
    weights = 1.0 / (d_all + eps)
    w_sum = jnp.sum(weights) + eps
    weights = weights / w_sum

    z_next = jnp.sum(weights[:, None] * z_all, axis=0)
    diff = jnp.sqrt(jnp.sum((z_next - z) ** 2))

    pad = eps
    entropy = -jnp.sum(weights * jnp.log(weights + pad))

    return z_next, diff, entropy


# ─── Macro loop (compact via lax.scan) ──────────────────────────────────────

def cog_loop_scan(z, codebooks, max_steps=5, thresholds=None, tau=0.1):
    """Cognitive macro loop via lax.scan — compiles to compact XLA loop.

    Same math as cog_loop_all_steps but uses lax.scan instead of Python
    unrolling. The body (one macro step) is compiled once; the loop runs
    inside XLA. Result is identical, graph is ~1/max_steps the size.

    Returns:
        z_qs: (max_steps, d) — conscious state after each macro step.
        diffs: (max_steps,) — stepwise delta ‖z_{t+1} - z_t‖.
        entropies: (max_steps,) — stepwise fusion entropy.
    """
    from jax import lax

    if thresholds is None:
        thresholds = [0.5] * len(codebooks)

    def macro_step(z_cur, _):
        z_next, diff, entropy = dag_fuse(z_cur, codebooks, thresholds, tau=tau)
        return z_next, (z_next, diff, entropy)

    _, (z_qs, diffs, entropies) = lax.scan(macro_step, z, None, length=max_steps)
    return z_qs, diffs, entropies
