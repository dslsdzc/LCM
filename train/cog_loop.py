"""JAX differentiable cognitive loop.

The cognitive loop IS the process of subconscious concepts "surfacing" into
conscious thought. Codebook entries (memory) are retrieved by their embeddings,
fused into a conscious state z_q, and z_q is transparently projected to tokens
through a single shared matrix — no per-entry decoder, no deception gap.

That matrix is the encoder's token embedding E, used transposed: the readout is
z_q @ E.T. It is tied rather than separate so the passive channel reads out in
the same token space the encoder reads in, and so there is exactly one trainable
token table (and one set of Adam moments) rather than two. Every dimension of
z_q contributes to every token — the mapping is fully readable from E's rows.

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
    """Exponential map in Poincaré ball: push tangent vector v onto ball.

    exp_x(v) = tanh(λ_x·‖v‖/2) · v / (λ_x·‖v‖), λ_x = 2/(1-‖x‖²).
    (The old code dropped the 1/λ_x factor — outputs were λ_x× too long.)
    """
    x_norm = jnp.sum(x ** 2)
    lam = 2.0 / (1.0 - x_norm + eps)
    v_norm = jnp.sqrt(jnp.sum(v ** 2) + eps)
    return jnp.tanh(lam * v_norm / 2.0) * v / (lam * v_norm + eps)


def poincare_log_map(x, y, eps=1e-6):
    """Logarithmic map in Poincaré ball: log_x(y) = atanh(‖u‖)·u/‖u‖ with
    u = -x ⊕ y (Möbius subtraction). At x=0: atanh(‖y‖)·y/‖y‖."""
    x_norm2 = jnp.sum(x ** 2)
    y_norm2 = jnp.sum(y ** 2)
    xy = jnp.dot(x, y)
    denom = 1 - 2 * xy + x_norm2 * y_norm2 + eps
    u = ((1 - 2 * xy + y_norm2) * x + (1 - x_norm2) * y) / denom
    u_norm = jnp.sqrt(jnp.sum(u ** 2) + eps)
    return jnp.arctanh(jnp.clip(u_norm, 0.0, 0.999)) * u / (u_norm + eps)


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

def cog_loop_scan(z, params, cfg, max_steps=None, rng=None,
                  return_stats=False, routing_bias=None):
    """Cognitive macro loop: the canonical six-lattice step, repeated.

    The step itself lives in train/cognitive_step.py and is shared with
    train/model.py::forward. This function only supplies the repetition and the
    loop's own bookkeeping (stepwise delta, and the fusion entropy that the
    convergence bonus reads).

    It used to run dag_fuse — a generic nearest-neighbour over flattened
    codebooks, fused by reciprocal distance. That is not what model.py
    computes, so the loop was optimising a different model from the one the
    architecture describes. dag_fuse and soft_retrieve remain as primitives
    (the C-engine parity tests exercise them) but are no longer the cognitive
    loop.

    Args:
        z: (B, d) initial cognitive state.
        params: carries 'route', 'fusion' and the six lattice param dicts.
        cfg: LCMConfig.
        max_steps: loop length; defaults to cfg.max_inference_steps.
        rng: PRNG key, split into one subkey per step so the routing gate's
            Gumbel noise is reproducible rather than re-seeded identically.
        routing_bias: optional (6,) host/device bias added to the routing gate
            at every macro step (BehaviorExplorer's Level-1 modulation).
        return_stats: when True, append a per-step stats pytree carrying the
            selection indices and binding residual inputs the EMA pass needs.
            Default False keeps the historical 3-tuple return for the existing
            callers; the returned arrays are bit-identical in both modes.

    Returns:
        z_qs: (max_steps, B, d) — conscious state after each macro step.
        diffs: (max_steps, B) — stepwise delta ‖z_{t+1} - z_t‖.
        entropies: (max_steps, B) — stepwise fusion entropy, in nats.
        stats: only when return_stats=True — dict with per-step ``z_in``,
            ``route_idx``, ``hrq_idx``, ``sparse_idx``, ``man_idx``, ``hrq_vq``
            (R18 continuous/quantised pairs), ``lowrank_aux``,
            ``contrast_indices``, ``contrast_residuals``,
            ``binding_residuals``, ``binding_indices``, ``binding_b_q``,
            ``binding_k_q`` and ``binding_val_out_idx`` (each stacked along
            axis 0). R15 adds ``self_idx`` and ``self_bias`` when the self
            lattice is present. These are exactly the
            inputs the R18 losses and the EMA pass consume, so neither needs a
            second forward pass.
    """
    from jax import lax

    from train.cognitive_step import six_lattice_step
    from train.self_lattice import self_lattice_forward

    if max_steps is None:
        max_steps = cfg.max_inference_steps
    if rng is None:
        rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, max_steps)

    def _step(z_cur, key):
        # R15: self is a real seventh fused element and is recomputed from the
        # current state each macro step (the same contract as infer/loops_v3.c).
        # Its nearest-row index is carried in aux so the EMA pass writes the
        # row this forward actually selected.
        self_output = None
        self_bias = None
        self_extras = {}
        if 'self' in params:
            self_output, self_bias, self_aux = self_lattice_forward(
                params['self'], z_cur, training=True, return_aux=True,
                alpha_self=cfg.alpha_self)
            self_extras = {
                'self_idx': self_aux['idx'],
                'self_bias': self_bias,
                'z_self': self_output,
            }
        z_next, aux = six_lattice_step(
            z_cur, params, cfg, training=True, rng=key,
            routing_bias=routing_bias, self_output=self_output,
            self_bias=self_bias)
        if self_extras:
            aux = {**aux, **self_extras}
        diff = jnp.sqrt(jnp.sum((z_next - z_cur) ** 2, axis=-1))
        return z_next, diff, aux

    if return_stats:
        def macro_step(z_cur, key):
            z_next, diff, aux = _step(z_cur, key)
            stats = {
                # The state the lattices received, not the fused output: EMA
                # accumulates the vector each codebook actually quantised.
                'z_in': z_cur,
                'soft_mask': aux['soft_mask'],
                'entropy': aux['entropy'],
                'route_idx': aux['route_idx'],
                'hrq_idx': aux['hrq_idx'],
                'sparse_idx': aux['sparse_idx'],
                'man_idx': aux['man_idx'],
                # R18 loss targets: per-layer continuous/quantised pairs and
                # residual/selected-row inputs from the same forward pass.
                'hrq_vq': aux['hrq_vq'],
                'lowrank_aux': aux['lowrank_aux'],
                'contrast_indices': aux['contrast_indices'],
                'contrast_residuals': aux['contrast_residuals'],
                'binding_residuals': aux['binding_residuals'],
                'binding_indices': aux['binding_indices'],
                # R20: the quantised bound representation, summed key and
                # observable unbind row.
                'binding_b_q': aux['binding_b_q'],
                'binding_k_q': aux['binding_k_q'],
                'binding_val_out_idx': aux['binding_val_out_idx'],
            }
            if 'self_idx' in aux:
                stats['self_idx'] = aux['self_idx']
                stats['self_bias'] = aux['self_bias']
                stats['z_self'] = aux['z_self']
            return z_next, (z_next, diff, aux['entropy'], stats)

        _, (z_qs, diffs, entropies, stats) = lax.scan(macro_step, z, keys)
        return z_qs, diffs, entropies, stats

    # Default path: build no stats and let the scan carry only the historical
    # three outputs. The step still computes aux (z_next depends on it), but
    # the per-layer loss/EMA arrays are not stacked and kept.
    def macro_step(z_cur, key):
        z_next, diff, aux = _step(z_cur, key)
        return z_next, (z_next, diff, aux['entropy'])

    _, (z_qs, diffs, entropies) = lax.scan(macro_step, z, keys)
    return z_qs, diffs, entropies
