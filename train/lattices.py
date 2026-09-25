"""Six specialized memory lattices + routing gate.

Each lattice performs codebook lookup with STE (straight-through estimator).
Training uses mixed EMA/gradient management as specified per lattice type.
All codebooks operate in JAX with pure functional style.
"""
import jax
import jax.numpy as jnp
from jax import lax
import jax.nn as jnn

from train.hyp import (
    poincare_similarity, exp_map, log_map, mobius_add, safe_unit,
)
from train.config import LCMConfig


# ── Shared utilities ────────────────────────────────────────────────────────

def simvq_codebook(params, z):
    """SimVQ linear reparameterized codebook.
    params: {'A': (M,d), 'W': (d,d)} — A is learnable, W is semi-orthogonal.
    """
    C = params['A'] @ params['W']  # (M, d)
    dist = jnp.linalg.norm(z[:, None, :] - C[None, :, :], axis=-1)
    idx = dist.argmin(axis=-1)
    z_q = C[idx]
    z_q = z + lax.stop_gradient(z_q - z)  # STE
    return z_q, idx, dist.min(axis=-1)


def init_simvq(rng, M, d):
    k1, k2 = jax.random.split(rng)
    return {
        'A': jax.random.normal(k1, (M, d)) * 0.01,
        'W': jax.random.normal(k2, (d, d)) * 0.01,
    }


def value_biased_score(z, C, v, avg_dist2, alpha_val):
    """Compute retrieval scores with local value bias.
    score(z, c_j) = -||z - c_j||² + alpha_val · v_j · avg_dist²
    """
    dist2 = jnp.sum((z[:, None, :] - C[None, :, :]) ** 2, axis=-1)  # (B, M)
    return -dist2 + alpha_val * v[None, :] * avg_dist2  # (B, M)


def value_biased_scores(z, C, v, alpha_val):
    """Retrieval scores for the value-biased path (higher = better, ``argmax``).

    Identical in forward value to
    ``value_biased_score(z, C, v, mean(||z - c||²), alpha_val)``, but the
    distance terms are stop_gradient'ed so the only differentiable input is the
    local value vector ``v`` (see ``ste_relax``). The distances are the
    selection *criterion*, not a training target — letting gradient through
    them would drag every codebook toward the current batch centroid.
    """
    dist2 = jnp.sum((z[:, None, :] - C[None, :, :]) ** 2, axis=-1)  # (B, M)
    avg_d2 = lax.stop_gradient(jnp.mean(dist2))
    # The stored scalars are unbounded parameters; tanh makes the *effective*
    # bias saturate at ±1 as documented, with a non-zero gradient everywhere
    # (a hard clip would leave a railed scalar with no way back).
    return -lax.stop_gradient(dist2) + alpha_val * jnp.tanh(v)[None, :] * avg_d2


def ste_relax(scores, C_out, tau=1.0):
    """Softmax-relaxation correction for a straight-through retrieval.

    ``p - sg(p)`` is **exactly zero in the forward pass**, so adding this
    anywhere leaves every forward number bit-identical. Its gradient w.r.t. the
    retrieval scores is ``(∂p/∂scores) @ C_out`` — which is the only path by
    which the local value scalars ``v_j`` can learn at all. ``v_j`` influences
    nothing but *which* code an ``argmax`` picks, so
    ``∂output/∂v ≡ 0`` without this relaxation and ``value_scalars`` stays at
    its zero initialisation for the whole run. Its gradient w.r.t. ``C_out`` is
    ``p - sg(p)``, i.e. numerically zero, so codebook gradients are unchanged.

    Apply this **after** the caller's ``base + stop_gradient(o - base)``
    wrapper: those wrappers block the entire branch, relaxation included.
    """
    p = jax.nn.softmax(
        (scores - jnp.max(scores, axis=-1, keepdims=True)) / tau, axis=-1)
    return (p - lax.stop_gradient(p)) @ C_out


def value_biased_retrieve(z, C, v, alpha_val, C_out=None):
    """Value-biased retrieval with a straight-through selection.

    Forward is bit-identical to ``C_out[argmax(score)]``. The caller is expected
    to add ``ste_relax(scores, C_out)`` after its own STE wrapper to give the
    local value scalars a gradient.

    Args:
        z: (B, d) query.
        C: (M, d) codebook the score is computed against.
        v: (M,) local value scalars in [-1, +1].
        alpha_val: Value bias strength.
        C_out: (M, d) codebook to select from; defaults to ``C``. Hyperbolic
            lattices score in tangent space but retrieve the ``exp_map`` of the
            prototype, so the two differ.

    Returns:
        (o, idx, scores): o (B, d) = ``C_out[idx]``, idx (B,), scores (B, M).
    """
    scores = value_biased_scores(z, C, v, alpha_val)
    idx = jnp.argmax(scores, axis=-1)  # (B,)
    return (C if C_out is None else C_out)[idx], idx, scores


# ── 4.0 Routing Gate ─────────────────────────────────────────────────────────

def init_route_params(rng, n_lattices, d):
    return {
        'C_route': jax.random.normal(rng, (n_lattices, d)) * 0.02,
        'W_route': jax.random.normal(rng, (d, n_lattices)) * (d ** -0.5),
    }


def routing_gate(params, z, tau, hard=False, rng=None):
    """Gumbel-Softmax routing gate.

    Supports optional bias injection via params['bias'] (6,).
    Bias is added to logits before softmax — used by BehaviorExplorer
    for active bias exploration (see e.md §六).
    """
    # Nearest codebook (STE)
    C = params['C_route']
    dist = jnp.linalg.norm(z[:, None, :] - C[None, :, :], axis=-1)
    idx = dist.argmin(axis=-1)
    z_route = C[idx]
    z_route = z + lax.stop_gradient(z_route - z)  # STE

    # Gumbel-Softmax with optional bias
    logits = z_route @ params['W_route']  # (B, n_lattices)
    if 'bias' in params:
        logits = logits + params['bias'][None, :]  # (B, n_lattices)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    soft_mask = jax.nn.softmax((logits + _sample_gumbel(rng, logits.shape)) / tau, axis=-1)

    if hard:
        # Straight-through: hard in forward, soft gradient in backward
        hard_mask = jax.nn.one_hot(soft_mask.argmax(axis=-1), soft_mask.shape[-1])
        soft_mask = lax.stop_gradient(hard_mask - soft_mask) + soft_mask

    return soft_mask, z_route, idx


def _sample_gumbel(rng, shape):
    u = jax.random.uniform(rng, shape, minval=1e-8, maxval=1 - 1e-8)
    return -jnp.log(-jnp.log(u))


def route_commit_loss(z, z_code, beta=0.25):
    """Codebook loss for the routing gate (unit-sphere normalized).

    ``z_code`` MUST be the raw codebook vector ``C_route[idx]``, not
    ``routing_gate``'s STE output. The STE output is
    ``z + stop_gradient(C_route[idx] - z)``, i.e. constant w.r.t. ``C_route``,
    so using it here leaves ``C_route`` with exactly zero gradient and only
    AdamW's decoupled weight decay touching it. Passing the raw vector makes
    this the ordinary VQ codebook loss and gives ``C_route`` the gradient
    ``∂/∂C_route = 2β(zr_n - sg(z_n))`` on the selected row.

    Forward value is identical either way (the STE output *is* ``C_route[idx]``
    in the forward pass), so this only changes the backward pass.
    """
    z_n = safe_unit(z)
    zr_n = safe_unit(z_code)
    return beta * jnp.mean((lax.stop_gradient(z_n) - zr_n) ** 2)


# ── 4.1 Hierarchical Lattice (Λ_hrq) ────────────────────────────────────────

def init_hrq_params(rng, d, M_top, M_fine, n_layers):
    keys = jax.random.split(rng, 1 + n_layers)
    params = {'top': init_simvq(keys[0], M_top, d)}
    params['fine'] = []
    for l in range(n_layers):
        params['fine'].append(init_simvq(keys[1 + l], M_fine, d))
    return params


def hrq_forward(params, z, tau_fallback=0.1, value_scalars=None, alpha_val=0.0):
    """Hierarchical residual quantization on Poincaré ball.

    1. Top-1 hard routing on top-level prototypes (fallback for uncertainty).
    2. Möbius residual through fine layers.
    """
    d = z.shape[-1]
    z_P = exp_map(z)

    # Top-level: similarity routing
    C_top = params['top']['A'] @ params['top']['W']
    C_top_P = exp_map(C_top)

    sims = poincare_similarity(z_P[:, None, :], C_top_P[None, :, :]).squeeze(-1)  # (B, M_top)

    if value_scalars is not None and alpha_val > 0:
        # Score in tangent space, retrieve the ball prototype.
        c_top, top_idx, top_scores = value_biased_retrieve(
            z, C_top, value_scalars, alpha_val, C_out=C_top_P)
    else:
        # poincare_similarity is distance-monotone (larger = farther): nearest = argmin
        top_idx = sims.argmin(axis=-1)
        c_top = C_top_P[top_idx]  # (B, d)
        top_scores = None

    top_sim = jnp.take_along_axis(sims, top_idx[:, None], axis=-1).squeeze(-1)

    # Check if we need fallback (top-1 / top-2 gap < threshold)
    # Ascending: closest first; gap = second-closest minus closest
    sorted_sims = jnp.sort(sims, axis=-1)  # ascending
    gap = sorted_sims[:, 1] - sorted_sims[:, 0]
    use_fallback = gap < tau_fallback

    # Möbius residual through the fine layers, shared by both routing branches
    # (they were duplicated verbatim; a fix to one silently missed the other).
    def fine_residual(r, fine_params):
        for fb in fine_params:
            C_fb_P = exp_map(fb['A'] @ fb['W'])
            vr_sims = poincare_similarity(r[:, None, :], C_fb_P[None, :, :]).squeeze(-1)
            r = mobius_add(r, -C_fb_P[vr_sims.argmin(axis=-1)])
        return r

    def hard_route(z_P, c_top, fine_params):
        return log_map(mobius_add(c_top, fine_residual(mobius_add(z_P, -c_top), fine_params)))

    def fallback_route(z_P, sims, C_top_P, fine_params):
        weights = jax.nn.softmax(-sims, axis=-1)  # (B, M_top)
        c_top_w = jnp.einsum('bm,md->bd', weights, C_top_P)  # weighted
        return log_map(mobius_add(c_top_w,
                                  fine_residual(mobius_add(z_P, -c_top_w), fine_params)))

    o_hrq = jnp.where(use_fallback[:, None],
                      fallback_route(z_P, sims, C_top_P, params['fine']),
                      hard_route(z_P, c_top, params['fine']))

    # Value-scalar gradient path (forward ≡ 0), added after all routing so it
    # cannot be swallowed by an upstream stop_gradient.
    if top_scores is not None:
        o_hrq = o_hrq + ste_relax(top_scores, C_top_P)

    return o_hrq, top_idx, top_sim


# ── 4.2 Sparse Lattice (Λ_sparse) ───────────────────────────────────────────

def init_sparse_params(rng, d, M_sparse):
    return {
        'C': jax.random.normal(rng, (M_sparse, d)) * 0.02,
        'zero_vec': jnp.zeros((1, d)),
    }


def sparse_forward(params, z, training=True, lambda_sparse=1e-4, d_top=None,
                   value_scalars=None, alpha_val=0.0):
    """Sparse lattice with train/inference branching.

    Training: searches without zero vector (pure VQ).
    Inference: includes zero vector for LFQ binary decision.
    """
    if training:
        C_search = params['C']  # (M_sparse, d) no zero
    else:
        C_search = jnp.concatenate([params['zero_vec'], params['C']], axis=0)

    if value_scalars is not None and alpha_val > 0:
        # Value-biased retrieval. Pad value_scalars for zero_vec in non-training
        # mode. The selection is straight-through w.r.t. value_scalars, so the
        # index below may still be overridden by LFQ before materialisation.
        if not training and 'zero_vec' in params:
            vs = jnp.concatenate([jnp.zeros(1), value_scalars], axis=0)
        else:
            vs = value_scalars
        scores = value_biased_scores(z, C_search, vs, alpha_val)
        idx = scores.argmax(axis=-1)
        dist = -scores  # convert to distance-like for LFQ
    else:
        scores = None
        dist = jnp.linalg.norm(z[:, None, :] - C_search[None, :, :], axis=-1)
        idx = dist.argmin(axis=-1)

    # LFQ binary decision (inference only)
    if not training and d_top is not None:
        d_min = dist.min(axis=-1)
        threshold = lambda_sparse * d_top
        zero_idx = jnp.zeros_like(idx)  # index of zero_vec (0 in C_search)
        idx = jnp.where(d_min > threshold, zero_idx, idx)

    o_sparse = z + lax.stop_gradient(C_search[idx] - z)  # STE
    if scores is not None:
        o_sparse = o_sparse + ste_relax(scores, C_search)  # value-scalar gradient
    return o_sparse, idx


def sparse_ema_update(params, z_sum, count, N, m, gamma=0.99, lambda_s=1e-4):
    """EMA update with soft shrinkage."""
    N_new = gamma * N + (1 - gamma) * count
    m_new = gamma * m + (1 - gamma) * z_sum
    C_new = m_new / jnp.clip(N_new, 1.0)[:, None]
    # Soft shrinkage
    C_new = jnp.sign(C_new) * jnp.clip(jnp.abs(C_new) - lambda_s, 0)
    return C_new, N_new, m_new


# ── 4.3 Low-Rank Lattice (Λ_lowrank) ────────────────────────────────────────

def init_lowrank_params(rng, d, M_lr, ranks):
    keys = jax.random.split(rng, 1 + len(ranks))
    r_max = max(ranks)
    params = {
        'A_V': jax.random.normal(keys[0], (d, r_max)) * 0.02,
        'W_V': jax.random.normal(keys[0], (r_max, r_max)) * 0.01,
    }
    params['U'] = []
    for l, r in enumerate(ranks):
        params['U'].append(jax.random.normal(keys[1 + l], (M_lr, r)) * 0.02)
    return params


def lowrank_forward(params, z, ranks, value_scalars=None, alpha_val=0.0):
    """Incremental rank VQ with shared base V."""
    V = params['A_V'] @ params['W_V']  # (d, r_max)
    o_total = jnp.zeros_like(z)
    relax = None
    r = z
    for l, (u_k, r_k) in enumerate(zip(params['U'], ranks)):
        C_k = u_k @ V[:, :r_k].T  # (M_lr, d)
        if value_scalars is not None and alpha_val > 0:
            scores = value_biased_scores(r, C_k, value_scalars, alpha_val)
            idx = scores.argmax(axis=-1)
            step = ste_relax(scores, C_k)
            relax = step if relax is None else relax + step
        else:
            dist = jnp.linalg.norm(r[:, None, :] - C_k[None, :, :], axis=-1)
            idx = dist.argmin(axis=-1)
        c_k = C_k[idx]
        o_total = o_total + c_k
        r = r - c_k
    o_total = z + lax.stop_gradient(o_total - z)  # STE
    if relax is not None:
        o_total = o_total + relax  # value-scalar gradient (forward ≡ 0)
    return o_total


# ── 4.4 Manifold Lattice (Λ_manifold) ───────────────────────────────────────

def init_manifold_params(rng, d, M_man, t_dim):
    k1, k2 = jax.random.split(rng)
    return {
        'C': jax.random.normal(k1, (M_man, d)) * 0.02,
        'T': jax.random.normal(k2, (M_man, d, t_dim)) * 0.01,
    }


def manifold_forward(params, z, value_scalars=None, alpha_val=0.0):
    """Hyperbolic manifold with local tangent space."""
    d = z.shape[-1]
    z_P = exp_map(z)
    C_P = exp_map(params['C'])

    if value_scalars is not None and alpha_val > 0:
        # Value-biased in tangent space; retrieve the ball prototype.
        _, idx, scores = value_biased_retrieve(
            z, params['C'], value_scalars, alpha_val, C_out=C_P)
    else:
        # Nearest neighbor in Poincaré ball
        sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
        idx = sims.argmin(axis=-1)
        scores = None
    c_idx = C_P[idx]  # (B, d)
    T_idx = params['T'][idx]  # (B, d, t)

    # Tangent space projection: T @ T.T @ r
    r = z_P - c_idx  # (B, d)
    # T_idx: (B, d, t), r: (B, d)
    # T.T @ r → einsum('bdt,bd->bt', T_idx, r) → (B, t)
    # T @ (T.T @ r) → einsum('bdt,bt->bd', T_idx, T_T_r) → (B, d)
    T_T_r = jnp.einsum('bdt,bd->bt', T_idx, r)  # (B, t)
    proj = jnp.einsum('bdt,bt->bd', T_idx, T_T_r)  # (B, d)

    o_manifold = log_map(c_idx + proj)
    o_manifold = z + lax.stop_gradient(o_manifold - z)  # STE
    if scores is not None:
        o_manifold = o_manifold + ste_relax(scores, C_P)  # value-scalar gradient
    return o_manifold, idx


def manifold_orth_loss(T, indices, k, lambda_orth=0.01, rng=None):
    """Vectorized orthogonality regularization on sampled tangent spaces.

    Computes ‖T_j^T T_j - I‖² averaged over active + randomly sampled codebooks.
    Vectorized for JIT compatibility.

    ``k`` is the number of *sampled* codebooks (static): the sample shape must
    be a compile-time constant, since ``jax.random.choice`` cannot take a
    shape derived from a traced value. The previous form sized the draw by
    ``k - len(unique(indices))``, and ``indices`` is a traced array under
    ``jax.grad``/``jit`` — ``jnp.unique(...).shape[0]`` is not concrete, so the
    call raised a shape error whenever this was used inside a jitted step.
    Active codebooks are added on top of the draw and de-duplicated.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    k = max(1, min(int(k), T.shape[0]))  # choice(replace=False) needs k <= M
    sampled = jax.random.choice(rng, T.shape[0], (k,), replace=False)
    active = jnp.unique(jnp.asarray(indices, dtype=sampled.dtype))
    target = jnp.unique(jnp.concatenate([active, sampled]))

    # Vectorized: T[target] -> (n, d, t)
    T_target = T[target]  # (n_target, d, t)
    # T_j.T @ T_j for all j: (n, t, d) @ (n, d, t) -> (n, t, t)
    T_T = jnp.einsum('bdt,bde->bte', T_target, T_target)
    I = jnp.eye(T_target.shape[-1])[None, :, :]  # (1, t, t)
    M = T_T - I
    loss = jnp.sum(M ** 2) / target.shape[0]
    return lambda_orth * loss


def manifold_ema_update(C, z_sum, count, N, m, gamma=0.99):
    """EMA update for the manifold codebook.

    ``params['manifold']['C']`` is **tangent-space** coordinates: every consumer
    pushes it onto the ball itself (``manifold_forward`` does
    ``C_P = exp_map(params['C'])``, ``compute_vq_loss`` compares against
    ``log_map(exp_map(C))``). Applying ``exp_map`` here stored ball coordinates
    back into the tangent slot, so the next forward ran the map twice and the
    codebook drifted toward the boundary on every EMA step. Store the raw EMA.
    """
    N_new = gamma * N + (1 - gamma) * count
    m_new = gamma * m + (1 - gamma) * z_sum
    C_new = m_new / jnp.clip(N_new, 1.0)[:, None]
    return C_new, N_new, m_new


# ── 4.5 Binding Lattice (Λ_binding) ─────────────────────────────────────────

def init_binding_params(rng, d, M_bind, n_layers, r_max):
    keys = jax.random.split(rng, 3)
    k_k, k_v, k_b = jax.random.split(keys[2], 3)  # distinct streams per codebook family
    return {
        'A_k': jax.random.normal(keys[0], (r_max, d)) * 0.01,
        'A_v': jax.random.normal(keys[1], (r_max, d)) * 0.01,
        'key_cb': [init_simvq(k, M_bind, d) for k in jax.random.split(k_k, n_layers)],
        'val_cb': [init_simvq(k, M_bind, d) for k in jax.random.split(k_v, n_layers)],
        'bind_cb': [init_simvq(k, M_bind, d) for k in jax.random.split(k_b, n_layers)],
    }


def normalize_fft(x):
    """Unit-circle normalization in frequency domain."""
    X = jnp.fft.rfft(x)
    mag = jnp.abs(X) + 1e-8
    return X / mag


def _residual_vq_chain(codebooks, r, value_scalars, alpha_val):
    """Multi-layer residual VQ over a list of SimVQ codebooks.

    Returns:
        (quantised list, residual inputs list, accumulated ste_relax or None).
        The residual inputs are the per-layer query vectors; they are what the
        per-codebook EMA must accumulate, since each layer quantises its own
        residual and not the raw lattice input.
    """
    quantised, inputs = [], []
    relax = None
    for cb in codebooks:
        inputs.append(r)
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']  # (M, d)
            scores = value_biased_scores(r, C, value_scalars, alpha_val)
            z_q = r + lax.stop_gradient(C[scores.argmax(axis=-1)] - r)
            step = ste_relax(scores, C)
            relax = step if relax is None else relax + step
        else:
            z_q, _, _ = simvq_codebook(cb, r)
        quantised.append(z_q)
        r = r - z_q
    return quantised, inputs, relax


def binding_forward(params, z, V, value_scalars=None, alpha_val=0.0):
    """HRR binding/unbinding with cross-layer superposition.

    Returns:
        (o_bind, residuals): residuals maps 'key'/'val'/'bind' to the list of
        per-layer query vectors, so the EMA pass can update each binding
        codebook in the space it actually quantises.
    """
    # Key/value projections: W_k = V @ A_k, z_k = z @ W_k.T
    # (d, r_max) @ (r_max, d) = (d, d); (B, d) @ (d, d) = (B, d)
    W_k = V @ params['A_k']
    W_v = V @ params['A_v']
    z_k = z @ W_k.T  # (B, d)
    z_v = z @ W_v.T  # (B, d)

    # Multi-layer residual VQ for key / value
    k_q_list, key_inputs, relax_k = _residual_vq_chain(
        params['key_cb'], z_k, value_scalars, alpha_val)
    v_q_list, val_inputs, relax_v = _residual_vq_chain(
        params['val_cb'], z_v, value_scalars, alpha_val)

    # Cross-layer HRR binding (9 pairs for 3 layers)
    b_raw = 0.0
    for k_i in k_q_list:
        for v_j in v_q_list:
            b_raw = b_raw + jnp.fft.irfft(
                normalize_fft(k_i) * normalize_fft(v_j), n=z.shape[-1])

    # Quantize the bound representation
    b_q_list, bind_inputs, relax_b = _residual_vq_chain(
        params['bind_cb'], b_raw, value_scalars, alpha_val)

    o_bind = jnp.sum(jnp.stack(b_q_list), axis=0)
    o_bind = z + lax.stop_gradient(o_bind - z)  # STE
    for step in (relax_k, relax_v, relax_b):
        if step is not None:
            o_bind = o_bind + step  # value-scalar gradient (forward ≡ 0)

    residuals = {'key': key_inputs, 'val': val_inputs, 'bind': bind_inputs}
    return o_bind, residuals


# ── 4.6 Contrast Lattice (Λ_contrast) ───────────────────────────────────────

def init_contrast_params(rng, d, M_contrast, n_layers):
    keys = jax.random.split(rng, 2)
    return {
        'C_a': [init_simvq(k, M_contrast, d)
                for k in jax.random.split(keys[0], n_layers)],
        'C_b': [init_simvq(k, M_contrast, d)
                for k in jax.random.split(keys[1], n_layers)],
    }


def contrast_forward(params, z, value_scalars=None, alpha_val=0.0):
    """Dual codebook contrast lattice."""
    a_q_list, _, relax_a = _residual_vq_chain(
        params['C_a'], z, value_scalars, alpha_val)
    b_q_list, _, relax_b = _residual_vq_chain(
        params['C_b'], z, value_scalars, alpha_val)

    o_contrast = (jnp.sum(jnp.stack(a_q_list), axis=0)
                  + jnp.sum(jnp.stack(b_q_list), axis=0)) / 2.0
    o_contrast = z + lax.stop_gradient(o_contrast - z)  # STE
    for step in (relax_a, relax_b):
        if step is not None:
            o_contrast = o_contrast + step / 2.0  # value-scalar gradient

    return o_contrast


def contrast_info_nce_loss(params, z, tau=0.5):
    """DualVC InfoNCE: each codebook uses the other as negative source."""
    z_detach = lax.stop_gradient(z)
    loss = 0.0

    for layer_idx in range(len(params['C_a'])):
        C_a = params['C_a'][layer_idx]['A'] @ params['C_a'][layer_idx]['W']
        C_b = params['C_b'][layer_idx]['A'] @ params['C_b'][layer_idx]['W']

        # Distances to a and b
        d_a = jnp.linalg.norm(z_detach[:, None, :] - C_a[None, :, :], axis=-1)
        d_b = jnp.linalg.norm(z_detach[:, None, :] - C_b[None, :, :], axis=-1)

        idx_a = d_a.argmin(axis=-1)
        idx_b = d_b.argmin(axis=-1)

        d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1).squeeze(-1)
        d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1).squeeze(-1)

        # InfoNCE: a vs b negatives, b vs a negatives
        # Safe logsumexp formulation: log(exp(x)/sum(exp)) = x - logsumexp(all)
        def _ce_safe(pos, all_vals):
            s = jax.nn.logsumexp(-all_vals / tau, axis=-1)
            x = -pos / tau
            return -jnp.mean(x - s)

        loss_a = _ce_safe(d_a_pos, d_b)
        loss_b = _ce_safe(d_b_pos, d_a)
        loss = loss + jnp.mean(loss_a) + jnp.mean(loss_b)

    return loss


def contrast_value_biased_nce_loss(params, z, v_harm, tau=0.5, tau_val=0.1):
    """Value-biased contrastive loss — harm-weighted negative sampling.

    Negatives are weighted by exp(-||c - v_harm||² / τ_val), making
    safety-critical (harm-proximal) codebook entries dominate the contrast
    signal. This trains the contrast lattice to discriminate positive anchors
    from safety-critical negatives.

    Args:
        params: Contrast lattice params with C_a, C_b codebooks.
        z: Encoder output (B, d).
        v_harm: Harm anchor vector from global value codebook.
        tau: InfoNCE temperature.
        tau_val: Value weighting temperature.

    Returns:
        loss: Scalar loss value.
    """
    z_detach = lax.stop_gradient(z)
    loss = 0.0

    for layer_idx in range(len(params['C_a'])):
        C_a = params['C_a'][layer_idx]['A'] @ params['C_a'][layer_idx]['W']
        C_b = params['C_b'][layer_idx]['A'] @ params['C_b'][layer_idx]['W']

        # Distances to a and b
        d_a = jnp.linalg.norm(z_detach[:, None, :] - C_a[None, :, :], axis=-1)  # (B, M)
        d_b = jnp.linalg.norm(z_detach[:, None, :] - C_b[None, :, :], axis=-1)

        idx_a = d_a.argmin(axis=-1)
        idx_b = d_b.argmin(axis=-1)

        d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1).squeeze(-1)
        d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1).squeeze(-1)

        # Harm weights for codebook entries
        harm_d_a = jnp.linalg.norm(C_a - v_harm[None, :], axis=-1)  # (M,)
        harm_d_b = jnp.linalg.norm(C_b - v_harm[None, :], axis=-1)
        w_a = jnp.exp(-harm_d_a / tau_val)  # (M,) higher weight ≈ closer to harm
        w_b = jnp.exp(-harm_d_b / tau_val)

        # Value-biased InfoNCE: negatives weighted by harm proximity
        # w_neg * exp(-d / τ) — harm-close entries contribute more to denominator
        weighted_neg_a = jnp.sum(w_b[None, :] * jnp.exp(-d_b / tau), axis=-1)
        weighted_neg_b = jnp.sum(w_a[None, :] * jnp.exp(-d_a / tau), axis=-1)

        # Subtract positive to avoid double-counting
        w_pos_a = jnp.take_along_axis(w_b[None, :], idx_b[:, None], axis=-1).squeeze(-1)
        w_pos_b = jnp.take_along_axis(w_a[None, :], idx_a[:, None], axis=-1).squeeze(-1)

        # Stable form of the same InfoNCE:
        #   -log(exp(-p/τ) / (exp(-p/τ) + Σ_{j≠pos} w_j·exp(-d_j/τ)))
        #   = log(1 + Σ_{j≠pos} w_j·exp(-(d_j - d_pos)/τ))
        # All exponent arguments are ≤ 0 (no overflow) and the old
        # denominator-subtraction form could cancel to ≤ 0 in float32
        # (harm weights concentrate → -inf/NaN). log1p → no underflow.
        excl_b = jax.nn.one_hot(idx_b, w_b.shape[0], dtype=jnp.float32)  # (B, M)
        excl_a = jax.nn.one_hot(idx_a, w_a.shape[0], dtype=jnp.float32)
        w_b_excl = w_b[None, :] * (1.0 - excl_b)  # (B, M), positive excluded
        w_a_excl = w_a[None, :] * (1.0 - excl_a)
        loss_a = jnp.log1p(
            jnp.sum(w_b_excl * jnp.exp(-(d_b - d_a_pos[:, None]) / tau), axis=-1))
        loss_b = jnp.log1p(
            jnp.sum(w_a_excl * jnp.exp(-(d_a - d_b_pos[:, None]) / tau), axis=-1))
        loss = loss + jnp.mean(loss_a) + jnp.mean(loss_b)

    return loss


# ── Local Value Scalars ──────────────────────────────────────────────────────

def init_danger_params(rng, M_danger, d):
    """Initialize danger codebook — frozen set of vectors marking unsafe regions.

    Used like gvalue: saved with SHA-256, never updated by optimizer.
    """
    C = jax.random.normal(rng, (M_danger, d)) * 0.02
    return {'C': C}


def init_value_scalars(rng, lattice_sizes):
    """Initialize local value scalars v_j for each lattice.

    Stored unnormalised; ``value_biased_scores`` applies ``tanh`` so the
    *effective* local value bias saturates at ±1 as the design specifies while
    keeping a non-zero gradient everywhere.
    """
    params = {}
    for name, M in lattice_sizes:
        params[name] = jnp.zeros(M)  # initialized to 0
    return params
