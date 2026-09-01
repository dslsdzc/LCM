"""Six specialized memory lattices + routing gate.

Each lattice performs codebook lookup with STE (straight-through estimator).
Training uses mixed EMA/gradient management as specified per lattice type.
All codebooks operate in JAX with pure functional style.
"""
import jax
import jax.numpy as jnp
from jax import lax
import jax.nn as jnn

from train.hyp import poincare_similarity, exp_map, log_map, mobius_add
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


def route_commit_loss(z, z_route, beta=0.25):
    """VQ commitment loss for routing gate (unit-sphere normalized)."""
    z_n = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    zr_n = z_route / (jnp.linalg.norm(z_route, axis=-1, keepdims=True) + 1e-8)
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

    if value_scalars is not None and alpha_val > 0:
        avg_dist2 = jnp.mean(jnp.linalg.norm(z[:, None, :] - C_top[None, :, :], axis=-1) ** 2)
        scores = value_biased_score(z, C_top, value_scalars, avg_dist2, alpha_val)
        top_idx = scores.argmax(axis=-1)
        sims = poincare_similarity(z_P[:, None, :], C_top_P[None, :, :]).squeeze(-1)
        top_sim = jnp.take_along_axis(sims, top_idx[:, None], axis=-1).squeeze(-1)
    else:
        sims = poincare_similarity(z_P[:, None, :], C_top_P[None, :, :])
        sims = sims.squeeze(-1)  # (B, M_top)
        # poincare_similarity is distance-monotone (larger = farther): nearest = argmin
        top_idx = sims.argmin(axis=-1)
        top_sim = jnp.take_along_axis(sims, top_idx[:, None], axis=-1).squeeze(-1)

    # Check if we need fallback (top-1 / top-2 gap < threshold)
    # Ascending: closest first; gap = second-closest minus closest
    sorted_sims = jnp.sort(sims, axis=-1)  # ascending
    gap = sorted_sims[:, 1] - sorted_sims[:, 0]
    use_fallback = gap < tau_fallback

    # Hard route or fallback
    def hard_route(z_P, top_idx, C_top_P, fine_params):
        c_top = C_top_P[top_idx]  # (B, d)
        r = mobius_add(z_P, -c_top)
        for fb in fine_params:
            C_fb = fb['A'] @ fb['W']
            C_fb_P = exp_map(C_fb)
            vr_sims = poincare_similarity(r[:, None, :], C_fb_P[None, :, :]).squeeze(-1)
            fi = vr_sims.argmin(axis=-1)
            c_f = C_fb_P[fi]
            r = mobius_add(r, -c_f)
        return log_map(mobius_add(c_top, r))

    def fallback_route(z_P, sims, C_top_P, fine_params):
        weights = jax.nn.softmax(-sims, axis=-1)  # (B, M_top)
        c_top_w = jnp.einsum('bm,md->bd', weights, C_top_P)  # weighted
        r = mobius_add(z_P, -c_top_w)
        for fb in fine_params:
            C_fb = fb['A'] @ fb['W']
            C_fb_P = exp_map(C_fb)
            vr_sims = poincare_similarity(r[:, None, :], C_fb_P[None, :, :]).squeeze(-1)
            fi = vr_sims.argmin(axis=-1)
            c_f = C_fb_P[fi]
            r = mobius_add(r, -c_f)
        return log_map(mobius_add(c_top_w, r))

    o_hrq = jnp.where(use_fallback[:, None],
                      fallback_route(z_P, sims, C_top_P, params['fine']),
                      hard_route(z_P, top_idx, C_top_P, params['fine']))

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
        # Value-biased retrieval
        # Pad value_scalars for zero_vec in non-training mode
        if not training and 'zero_vec' in params:
            vs = jnp.concatenate([jnp.zeros(1), value_scalars], axis=0)
        else:
            vs = value_scalars
        avg_dist2 = jnp.mean(jnp.linalg.norm(z[:, None, :] - C_search[None, :, :], axis=-1) ** 2)
        scores = value_biased_score(z, C_search, vs, avg_dist2, alpha_val)
        idx = scores.argmax(axis=-1)
        dist = -scores  # convert to distance-like for LFQ
    else:
        dist = jnp.linalg.norm(z[:, None, :] - C_search[None, :, :], axis=-1)
        idx = dist.argmin(axis=-1)

    # LFQ binary decision (inference only)
    if not training and d_top is not None:
        d_min = dist.min(axis=-1)
        threshold = lambda_sparse * d_top
        zero_idx = jnp.zeros_like(idx)  # index of zero_vec (0 in C_search)
        idx = jnp.where(d_min > threshold, zero_idx, idx)

    o_sparse = C_search[idx]
    o_sparse = z + lax.stop_gradient(o_sparse - z)  # STE
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
    r = z
    for l, (u_k, r_k) in enumerate(zip(params['U'], ranks)):
        C_k = u_k @ V[:, :r_k].T  # (M_lr, d)
        if value_scalars is not None and alpha_val > 0:
            avg_dist2 = jnp.mean(jnp.linalg.norm(r[:, None, :] - C_k[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r, C_k, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
        else:
            dist = jnp.linalg.norm(r[:, None, :] - C_k[None, :, :], axis=-1)
            idx = dist.argmin(axis=-1)
        c_k = C_k[idx]
        o_total = o_total + c_k
        r = r - c_k
    o_total = z + lax.stop_gradient(o_total - z)  # STE
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
        # Value-biased in tangent space
        avg_dist2 = jnp.mean(jnp.linalg.norm(z[:, None, :] - params['C'][None, :, :], axis=-1) ** 2)
        scores = value_biased_score(z, params['C'], value_scalars, avg_dist2, alpha_val)
        idx = scores.argmax(axis=-1)
    else:
        # Nearest neighbor in Poincaré ball
        sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
        idx = sims.argmin(axis=-1)
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
    return o_manifold, idx


def manifold_orth_loss(T, indices, k, lambda_orth=0.01, rng=None):
    """Vectorized orthogonality regularization on sampled tangent spaces.

    Computes ‖T_j^T T_j - I‖² averaged over active + randomly sampled codebooks.
    Vectorized for JIT compatibility.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    active = jnp.unique(indices)
    n_active = active.shape[0]
    n_needed = jnp.maximum(k - n_active, 0)
    sampled = jax.random.choice(rng, T.shape[0], (n_needed,), replace=False)
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
    """EMA update for manifold codebook, re-project to Poincaré ball."""
    N_new = gamma * N + (1 - gamma) * count
    m_new = gamma * m + (1 - gamma) * z_sum
    C_new = m_new / jnp.clip(N_new, 1.0)[:, None]
    C_new = exp_map(C_new)
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


def binding_forward(params, z, V, value_scalars=None, alpha_val=0.0):
    """HRR binding/unbinding with cross-layer superposition."""
    # Key/value projections: W_k = V @ A_k, z_k = z @ W_k.T
    # (d, r_max) @ (r_max, d) = (d, d); (B, d) @ (d, d) = (B, d)
    W_k = V @ params['A_k']
    W_v = V @ params['A_v']
    z_k = z @ W_k.T  # (B, d)
    z_v = z @ W_v.T  # (B, d)

    # Multi-layer residual VQ for key (with value bias)
    k_q_list = []
    r_k = z_k
    for cb in params['key_cb']:
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']  # (M_bind, d)
            avg_dist2 = jnp.mean(jnp.linalg.norm(r_k[:, None, :] - C[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r_k, C, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = C[idx]
            z_q = r_k + lax.stop_gradient(z_q - r_k)
        else:
            z_q, idx, _ = simvq_codebook(cb, r_k)
        k_q_list.append(z_q)
        r_k = r_k - z_q

    # Multi-layer residual VQ for value (with value bias)
    v_q_list = []
    r_v = z_v
    for cb in params['val_cb']:
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']
            avg_dist2 = jnp.mean(jnp.linalg.norm(r_v[:, None, :] - C[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r_v, C, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = C[idx]
            z_q = r_v + lax.stop_gradient(z_q - r_v)
        else:
            z_q, idx, _ = simvq_codebook(cb, r_v)
        v_q_list.append(z_q)
        r_v = r_v - z_q

    # Cross-layer HRR binding (9 pairs for 3 layers)
    b_raw = 0.0
    for i, k_i in enumerate(k_q_list):
        for j, v_j in enumerate(v_q_list):
            k_norm = normalize_fft(k_i)
            v_norm = normalize_fft(v_j)
            b_raw = b_raw + jnp.fft.irfft(k_norm * v_norm, n=z.shape[-1])

    # Quantize the bound representation (with value bias)
    r_b = b_raw
    b_q_list = []
    for cb in params['bind_cb']:
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']
            avg_dist2 = jnp.mean(jnp.linalg.norm(r_b[:, None, :] - C[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r_b, C, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = C[idx]
            z_q = r_b + lax.stop_gradient(z_q - r_b)
        else:
            z_q, idx, _ = simvq_codebook(cb, r_b)
        b_q_list.append(z_q)
        r_b = r_b - z_q

    o_bind = jnp.sum(jnp.stack(b_q_list), axis=0)
    o_bind = z + lax.stop_gradient(o_bind - z)  # STE
    return o_bind


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
    o_a = jnp.zeros_like(z)
    r_a = z
    for cb in params['C_a']:
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']
            avg_dist2 = jnp.mean(jnp.linalg.norm(r_a[:, None, :] - C[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r_a, C, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = C[idx]
            z_q = r_a + lax.stop_gradient(z_q - r_a)
        else:
            z_q, idx, _ = simvq_codebook(cb, r_a)
        o_a = o_a + z_q
        r_a = r_a - z_q

    o_b = jnp.zeros_like(z)
    r_b = z
    for cb in params['C_b']:
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']
            avg_dist2 = jnp.mean(jnp.linalg.norm(r_b[:, None, :] - C[None, :, :], axis=-1) ** 2)
            scores = value_biased_score(r_b, C, value_scalars, avg_dist2, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = C[idx]
            z_q = r_b + lax.stop_gradient(z_q - r_b)
        else:
            z_q, idx, _ = simvq_codebook(cb, r_b)
        o_b = o_b + z_q
        r_b = r_b - z_q

    o_contrast = (o_a + o_b) / 2.0
    o_contrast = z + lax.stop_gradient(o_contrast - z)  # STE
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
    """Initialize local value scalars v_j ∈ [-1, +1] for each lattice."""
    params = {}
    for name, M in lattice_sizes:
        params[name] = jnp.zeros(M)  # initialized to 0
    return params


def value_biased_retrieve(z, C, v, alpha_val):
    """Retrieve with local value bias."""
    avg_z_norm = jnp.mean(jnp.linalg.norm(z, axis=-1) ** 2)
    avg_C_dist2 = jnp.mean(jnp.linalg.norm(z[:, None, :] - C[None, :, :], axis=-1) ** 2)
    scores = value_biased_score(z, C, v, avg_C_dist2, alpha_val)
    idx = scores.argmax(axis=-1)
    return C[idx], idx
