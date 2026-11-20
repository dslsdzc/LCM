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

    ``hard=True`` is the inference path: deterministic one-hot argmax with no
    Gumbel noise (R19). ``hard=False`` is the training path and keeps the
    Gumbel noise for exploration.

    Supports optional bias injection via params['bias'] (n_lattices,).
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

    if hard:
        # R19: inference is deterministic — one_hot(argmax(logits)) with no
        # Gumbel noise. The soft mask exists only as the backward proxy.
        deterministic = jax.nn.softmax(logits / tau, axis=-1)
        hard_mask = jax.nn.one_hot(logits.argmax(axis=-1), logits.shape[-1])
        soft_mask = lax.stop_gradient(hard_mask - deterministic) + deterministic
    else:
        # Training keeps the Gumbel noise: it is the only stochastic
        # exploration the gate has.
        if rng is None:
            rng = jax.random.PRNGKey(0)
        soft_mask = jax.nn.softmax(
            (logits + _sample_gumbel(rng, logits.shape)) / tau, axis=-1)

    return soft_mask, z_route, idx


def _sample_gumbel(rng, shape):
    u = jax.random.uniform(rng, shape, minval=1e-8, maxval=1 - 1e-8)
    return -jnp.log(-jnp.log(u))


def route_commit_loss(z, z_code, beta=0.25):
    """R18 routing codebook loss: ``β·mean((sg[z] − z_route)²)``.

    ``z_code`` MUST be the raw codebook vector ``C_route[idx]``, not
    ``routing_gate``'s STE output. The STE output is
    ``z + stop_gradient(C_route[idx] - z)``, i.e. constant w.r.t. ``C_route``,
    so using it here leaves ``C_route`` with exactly zero gradient and only
    AdamW's decoupled weight decay touching it. Passing the raw vector makes
    this the ordinary VQ codebook loss and gives ``C_route`` the gradient
    ``∂/∂C_route = 2β(C_route[idx] - sg[z])`` on the selected row.

    Forward value is identical either way (the STE output *is* ``C_route[idx]``
    in the forward pass), so this only changes the backward pass.
    """
    return beta * jnp.mean((lax.stop_gradient(z) - z_code) ** 2)


# ── 4.1 Hierarchical Lattice (Λ_hrq) ────────────────────────────────────────

def init_hrq_params(rng, d, M_top, M_fine, n_layers):
    keys = jax.random.split(rng, 1 + n_layers)
    params = {'top': init_simvq(keys[0], M_top, d)}
    params['fine'] = []
    for l in range(n_layers):
        params['fine'].append(init_simvq(keys[1 + l], M_fine, d))
    return params


def _hrq_sims(params, z):
    """R6 domain for HRQ's top stage: ``z_P = exp_map(safe_unit(z))``.

    Returns ``(z_P, C_top_P, sims)`` with ``sims`` of shape ``(B, M_top)``.
    ``poincare_similarity`` is distance-monotone (larger = farther).
    """
    z_P = exp_map(safe_unit(z))
    C_top_P = exp_map(params['top']['A'] @ params['top']['W'])
    sims = poincare_similarity(
        z_P[:, None, :], C_top_P[None, :, :]).squeeze(-1)
    return z_P, C_top_P, sims


def _topk_indices(params, z, k):
    """Indices of the ``k`` nearest top-level prototypes, nearest first.

    Test helper mirroring the first stage of ``hrq_forward`` (R1): geometric
    top-k before any value rescoring.
    """
    _, _, sims = _hrq_sims(params, z)
    k = min(int(k), int(sims.shape[-1]))
    return jnp.argsort(sims, axis=-1)[:, :k]


def hrq_top_prototype(params, idx):
    """Ball prototype ``c_top*`` selected by ``hrq_forward``'s ``top_idx``.

    Sparse's LFQ threshold needs ``c_top*`` as an input edge (design R2/C-5);
    HRQ always runs first in ``six_lattice_step`` so the point is available.
    """
    C_top_P = exp_map(params['top']['A'] @ params['top']['W'])
    return C_top_P[idx]


def hrq_forward(params, z, tau_fallback=0.1, value_scalars=None, alpha_val=0.0,
                value_topk=4, tau_proxy=1.0, fine_value_scalars=None):
    """Hierarchical residual quantization on Poincaré ball.

    Two-stage top routing (design R1):

    1. Geometric top-k by ``poincare_similarity(z_P, C_top_P)``.
    2. Inside the top-k only, ``score_j = sim_j - alpha_val·tanh(v_j)·scale``
       with ``scale = sg[mean(sims)] + 1e-6``; ``argmin`` gives the final top-1.
       A high value lowers the score and makes a prototype easier to select.

    The hard forward value is the routed prototype's residual chain; the
    backward pass goes through the top-k softmax mixture, so ``v_j`` and
    ``C_top`` both receive gradient. Fallback is a *relative* gap
    ``(sim_2 - sim_1) / (sim_1 + eps) < tau_fallback`` and blends the top-k with
    ``softmax(-score)``.

    R22: ``value_scalars`` may be either the top-level ``(M_top,)`` vector or
    the whole ``{'top': (M_top,), 'fine': (M_fine,)}`` structure. In the latter
    case the fine residual layers use ``'fine'`` (one shared vector per layer).
    ``fine_value_scalars`` overrides the fine entry when given explicitly.

    Returns:
        (o_hrq, top_idx, top_sim, vq_pairs). ``vq_pairs`` carries R18's
        Euclidean-space training pairs: ``cont`` and ``quant`` are each a list
        with one ``(B, d)`` vector for the top layer followed by one per fine
        layer. ``cont`` is the pre-quantisation residual pointer after
        ``log_map``; ``quant`` is the selected prototype after ``log_map``.
        ``top_sim`` is per-sample ``(B,)`` — it is Sparse's ``d_top``.
    """
    if int(value_topk) <= 0:
        raise ValueError(f"value_topk must be >= 1, got {value_topk}")
    if tau_proxy <= 0:
        raise ValueError(f"tau_proxy must be > 0, got {tau_proxy}")
    M_top = int(params['top']['A'].shape[0])
    M_fine = int(params['fine'][0]['A'].shape[0]) if params['fine'] else None
    if isinstance(value_scalars, dict):
        if fine_value_scalars is None:
            fine_value_scalars = value_scalars.get('fine')
        value_scalars = value_scalars.get('top')
    if value_scalars is not None:
        v_shape = jnp.shape(value_scalars)
        if len(v_shape) != 1 or int(v_shape[0]) != M_top:
            raise ValueError(
                f"value_scalars must have one scalar per top prototype "
                f"({M_top},), got shape {v_shape}")
    if fine_value_scalars is not None and M_fine is not None:
        f_shape = jnp.shape(fine_value_scalars)
        if len(f_shape) != 1 or int(f_shape[0]) != M_fine:
            raise ValueError(
                f"fine_value_scalars must have one scalar per fine prototype "
                f"({M_fine},), got shape {f_shape}")

    z_P, C_top_P, sims = _hrq_sims(params, z)
    B = sims.shape[0]

    # ── Stage 1: geometric top-k (no value term) ───────────────────────────
    k = min(int(value_topk), int(M_top))
    topk_idx = jnp.argsort(sims, axis=-1)[:, :k]              # (B, k)
    sims_topk = jnp.take_along_axis(sims, topk_idx, axis=-1)  # (B, k)
    C_topk_P = C_top_P[topk_idx]                              # (B, k, d)

    # ── Stage 2: value rescoring within the top-k ──────────────────────────
    v = (jnp.zeros((M_top,), dtype=sims.dtype) if value_scalars is None
         else jnp.asarray(value_scalars))
    # Per-sample scale, detached: it only makes the value term dimensionless.
    scale = lax.stop_gradient(jnp.mean(sims, axis=-1, keepdims=True)) + 1e-6
    score_topk = sims_topk - alpha_val * jnp.tanh(v)[topk_idx] * scale

    best = jnp.argmin(score_topk, axis=-1)
    top_idx = jnp.take_along_axis(topk_idx, best[:, None], axis=-1)[:, 0]
    top_sim = jnp.take_along_axis(sims, top_idx[:, None], axis=-1)[:, 0]
    c_top = C_top_P[top_idx]                                  # (B, d)

    # ── STE: hard forward, top-k softmax backward (R1.3) ───────────────────
    # Two corrections to the design's shorthand:
    #   * the literal ``c_top + sg(o_soft - c_top)`` puts stop_gradient on the
    #     soft operand, so d output/d v is exactly zero;
    #   * ``o_soft + sg(c_top - o_soft)`` has the right gradient but its
    #     forward value is only c_top up to one ulp.
    # This form is exact in both directions: the forward correction is
    # ``o_soft - sg(o_soft) == 0`` (no rounding), the backward path is the soft
    # mixture alone, and the hard prototype is detached.
    w_proxy = jax.nn.softmax(-score_topk / tau_proxy, axis=-1)  # (B, k)
    o_soft_topk = jnp.einsum('bk,bkd->bd', w_proxy, C_topk_P)   # (B, d)
    c_top_route = (lax.stop_gradient(c_top)
                   + (o_soft_topk - lax.stop_gradient(o_soft_topk)))

    # ── Relative fallback criterion (R1.4) ────────────────────────────────
    if M_top > 1:
        sorted_sims = jnp.sort(sims, axis=-1)  # ascending
        gap = ((sorted_sims[:, 1] - sorted_sims[:, 0])
               / (sorted_sims[:, 0] + 1e-8))
    else:
        gap = jnp.zeros((B,), dtype=sims.dtype)
    use_fallback = gap < tau_fallback

    # Fallback blends only the top-k candidates, with the value-adjusted score.
    w_fb = jax.nn.softmax(-score_topk, axis=-1)                # (B, k)
    c_top_w = jnp.einsum('bk,bkd->bd', w_fb, C_topk_P)         # (B, d)

    # ── Möbius residual through the fine layers (shared by both branches) ──
    # Also returns the per-layer output-space pairs R18 needs. The loss is in
    # the Euclidean space *after* log_map(c_top ⊕ ·), not the tangent residual:
    # the continuous pointer and the selected code both get c_top added before
    # log_map (design R18, b.md §4.1).
    def fine_residual(r, fine_params, top):
        cont, quant = [], []
        for fb in fine_params:
            C_fb_P = exp_map(fb['A'] @ fb['W'])
            vr_sims = poincare_similarity(r[:, None, :], C_fb_P[None, :, :]).squeeze(-1)
            if fine_value_scalars is not None and alpha_val > 0:
                # R22: the fine layers share one value vector. Same
                # similarity-domain rescoring as the top stage: high value
                # lowers the score, so it is more likely to be selected.
                v_f = jnp.asarray(fine_value_scalars)
                scale_f = lax.stop_gradient(
                    jnp.mean(vr_sims, axis=-1, keepdims=True)) + 1e-6
                fine_scores = vr_sims - alpha_val * jnp.tanh(v_f)[None, :] * scale_f
                c_fb = C_fb_P[fine_scores.argmin(axis=-1)]
                # Forward ≡ 0; gives v_fine a gradient through the argmin.
                # Negate because ste_relax's softmax is higher=better while the
                # similarity score is lower=better.
                c_fb = c_fb + ste_relax(-fine_scores, C_fb_P)
            else:
                c_fb = C_fb_P[vr_sims.argmin(axis=-1)]
            cont.append(log_map(mobius_add(top, r)))
            quant.append(log_map(mobius_add(top, c_fb)))
            r = mobius_add(r, -c_fb)
        return r, cont, quant

    # Hard branch: routed prototype (with the soft backward proxy).
    r_hard, cont_hard, quant_hard = fine_residual(
        mobius_add(z_P, -c_top_route), params['fine'], c_top_route)
    # Fallback branch: top-k softmax-weighted prototype blend.
    r_soft, cont_soft, quant_soft = fine_residual(
        mobius_add(z_P, -c_top_w), params['fine'], c_top_w)

    o_hard = log_map(mobius_add(c_top_route, r_hard))
    o_soft = log_map(mobius_add(c_top_w, r_soft))
    o_hrq = jnp.where(use_fallback[:, None], o_soft, o_hard)

    # Top term follows the branch that actually ran: the hard target is the
    # routed prototype, the fallback target is the softmax blend. Fine-layer
    # pairs are selected elementwise the same way. Targets use the raw
    # prototypes, not the STE copy (same forward value).
    def _pick(hard, soft):
        return jnp.where(use_fallback[:, None], soft, hard)

    top_quant = jnp.where(use_fallback[:, None],
                          log_map(c_top_w), log_map(c_top))
    vq_pairs = {
        'cont': [z] + [_pick(a, b) for a, b in zip(cont_hard, cont_soft)],
        'quant': [top_quant] + [
            _pick(a, b) for a, b in zip(quant_hard, quant_soft)],
    }
    return o_hrq, top_idx, top_sim, vq_pairs


# ── 4.2 Sparse Lattice (Λ_sparse) ───────────────────────────────────────────

def init_sparse_params(rng, d, M_sparse):
    return {
        'C': jax.random.normal(rng, (M_sparse, d)) * 0.02,
        'zero_vec': jnp.zeros((1, d)),
    }


def clamp_norm(C, max_norm):
    """Scale rows with ``‖C_j‖ > max_norm`` down to ``max_norm``.

    R2 wraps every sparse prototype in ``exp_map`` before the hyperbolic
    comparison; without the clamp a code far outside the ball would saturate
    ``exp_map`` at the boundary and the similarity would blow up. Zero rows
    (untouched EMA entries) are left exactly zero.
    """
    norm = jnp.linalg.norm(C, axis=-1, keepdims=True)
    return C * jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))


def sparse_zero_similarity(z):
    """C-parity helper: ``poincare_similarity(z_P, 0)`` in the unified R6 domain.

    Equals ``2‖z_P‖²/(1-‖z_P‖²)``. It exists so the C side (Task 8) has one
    authoritative formula for the ``zero_vec`` candidate's domain. The JAX LFQ
    threshold does **not** call it: b.md §4.2 defines ``d_sparse_min`` over the
    non-zero codebook only, and the zero vector is the *output* of the LFQ
    decision, not a candidate in its comparison.
    """
    z_P = exp_map(safe_unit(z))
    return poincare_similarity(z_P, jnp.zeros_like(z_P)).squeeze(-1)


def sparse_forward(params, z, training=True, lambda_lfq=1.5, d_top=None,
                   c_top=None, value_scalars=None, alpha_val=0.0):
    """Sparse lattice with train/inference branching and LFQ.

    Training: searches without the zero vector (pure VQ), no LFQ.
    Inference: includes the zero vector in the nearest-neighbour competition and
    may force it via LFQ.

    ``d_top`` is HRQ's per-sample ``poincare_similarity(z_P, c_top*)`` of shape
    ``(B,)``; ``c_top`` is the matching prototype. Both are required for LFQ:
    ``c_top=None`` means HRQ did not activate, so the threshold has no defined
    reference and must not be evaluated (design R2, C-5 dependency edge).
    ``d_min`` is measured in the same similarity domain
    (``exp_map(clamp_norm(C, 0.99))``), while retrieval keeps b.md §4.2's
    Euclidean ``argmin`` (value-biased when ``alpha_val > 0``, R22).

    ``d_top`` must be ``(B,)``. The old batch scalar was the defect R2 removed;
    scalar/1-element inputs are rejected here so they cannot silently broadcast
    back into batch-shared behaviour.
    """
    B = z.shape[0]
    if d_top is not None:
        shape = jnp.shape(d_top)
        if len(shape) != 1 or int(shape[0]) != int(B):
            raise ValueError(
                f"d_top must be per-sample with shape ({B},), got {shape}")
    if c_top is not None:
        shape = jnp.shape(c_top)
        if len(shape) != 2 or int(shape[0]) != int(B):
            raise ValueError(
                f"c_top must be a per-sample prototype ({B}, d), got {shape}")

    if training:
        C_search = params['C']  # (M_sparse, d) no zero
    else:
        C_search = jnp.concatenate([params['zero_vec'], params['C']], axis=0)

    scores = None
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
    else:
        dist = jnp.linalg.norm(z[:, None, :] - C_search[None, :, :], axis=-1)
        idx = dist.argmin(axis=-1)

    # LFQ binary decision (inference only, R2). d_min is the hyperbolic
    # similarity to the nearest *non-zero* code, so it shares d_top's domain.
    if (not training) and d_top is not None and c_top is not None:
        z_P = exp_map(safe_unit(z))
        C_P = exp_map(clamp_norm(params['C'], 0.99))
        d_min = poincare_similarity(
            z_P[:, None, :], C_P[None, :, :]).squeeze(-1).min(axis=-1)
        zero_idx = jnp.zeros_like(idx)  # index of zero_vec (0 in C_search)
        # C's `lattice.c` treats d_top <= 0 as "HRQ did not activate" and
        # disables LFQ; JAX always receives a tensor, so mask the comparison
        # explicitly. Without this, d_top == 0 made every sample take the
        # zero_vec branch while C kept the nearest code.
        lfq_on = jnp.asarray(d_top) > 0.0
        idx = jnp.where(lfq_on & (d_min > lambda_lfq * d_top),
                        zero_idx, idx)

    o_sparse = z + lax.stop_gradient(C_search[idx] - z)  # STE
    if scores is not None:
        o_sparse = o_sparse + ste_relax(scores, C_search)  # value-scalar gradient
    return o_sparse, idx


def sparse_ema_update(params, z_sum, count, N, m, gamma=0.99, lambda_s=1e-4):
    """EMA update with soft shrinkage.

    DEPRECATED: no callers. The live EMA path is ``train/ema_state.py``
    (feature-bank cold start needs the decayed-sum recursion); kept because
    tests/imports may still reference this historical helper.
    """
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


def lowrank_forward(params, z, ranks, value_scalars=None, alpha_val=0.0,
                    return_aux=False):
    """Incremental rank VQ with shared base V.

    ``return_aux=True`` additionally returns the per-layer residual inputs and
    selected indices (R18's codebook/commitment targets). The default keeps the
    historical bare-array return for existing callers.
    """
    V = params['A_V'] @ params['W_V']  # (d, r_max)
    o_total = jnp.zeros_like(z)
    relax = None
    r = z
    residuals, indices = [], []
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
        residuals.append(r)
        indices.append(idx)
        c_k = C_k[idx]
        o_total = o_total + c_k
        r = r - c_k
    o_total = z + lax.stop_gradient(o_total - z)  # STE
    if relax is not None:
        o_total = o_total + relax  # value-scalar gradient (forward ≡ 0)
    if return_aux:
        return o_total, {'residuals': residuals, 'indices': indices}
    return o_total


# ── 4.4 Manifold Lattice (Λ_manifold) ───────────────────────────────────────

def init_manifold_params(rng, d, M_man, t_dim):
    k1, k2 = jax.random.split(rng)
    return {
        'C': jax.random.normal(k1, (M_man, d)) * 0.02,
        'T': jax.random.normal(k2, (M_man, d, t_dim)) * 0.01,
    }


def manifold_forward(params, z, value_scalars=None, alpha_val=0.0):
    """Hyperbolic manifold with local tangent space.

    R6 domain: the query is ``exp_map(safe_unit(z))``, matching C's
    ``manifold_forward_c``. Codebooks stay tangent-space parameters and are
    pushed onto the ball with ``exp_map(C)`` only (they are not re-normalised).
    """
    d = z.shape[-1]
    z_P = exp_map(safe_unit(z))
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
    shape derived from a traced value. ``indices`` is a traced array under
    ``jax.grad``/``jit``, so it must not be passed through ``jnp.unique``:
    without a static ``size=`` that call raises a ConcretizationTypeError
    during tracing. Active codebooks are appended to the draw as-is; a
    duplicate index simply contributes the same term twice, which is a
    weighting choice, not a correctness issue, and keeps every tensor shape
    static.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    k = max(1, min(int(k), T.shape[0]))  # choice(replace=False) needs k <= M
    sampled = jax.random.choice(rng, T.shape[0], (k,), replace=False)
    active = jnp.asarray(indices, dtype=sampled.dtype).reshape(-1)
    target = jnp.concatenate([active, sampled])

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

    DEPRECATED: no callers. The live EMA path is ``train/ema_state.py``
    (decayed-sum recursion + untouched-row preservation); kept because
    tests/imports may still reference this historical helper.

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
        (quantised list, residual inputs list, per-layer idx list, accumulated
        ste_relax or None). The residual inputs are the per-layer query vectors;
        the indices are the rows the forward actually selected. Both are needed
        by the EMA pass: recomputing an argmin would diverge from the
        value-biased selection under a nonzero ``value_scalars``.
    """
    quantised, inputs, indices = [], [], []
    relax = None
    for cb in codebooks:
        inputs.append(r)
        if value_scalars is not None and alpha_val > 0:
            C = cb['A'] @ cb['W']  # (M, d)
            scores = value_biased_scores(r, C, value_scalars, alpha_val)
            idx = scores.argmax(axis=-1)
            z_q = r + lax.stop_gradient(C[idx] - r)
            step = ste_relax(scores, C)
            relax = step if relax is None else relax + step
        else:
            z_q, idx, _ = simvq_codebook(cb, r)
        quantised.append(z_q)
        indices.append(idx)
        r = r - z_q
    return quantised, inputs, indices, relax


def _family_value_vector(value_scalars, family, M, lattice):
    """Select one family's value vector from an R22 sub-codebook structure.

    ``value_scalars`` is normally ``{'key': (M,), ...}`` (binding) or
    ``{'a': (M,), ...}`` (contrast). A bare 1-D array is accepted as the legacy
    shared vector so direct callers and older tests keep working.
    """
    if value_scalars is None:
        return None
    if isinstance(value_scalars, dict):
        v = value_scalars.get(family)
    else:
        v = value_scalars
    if v is not None:
        shape = jnp.shape(v)
        if len(shape) != 1 or int(shape[0]) != int(M):
            raise ValueError(
                f"{lattice} {family} value_scalars must be ({M},), got {shape}")
    return v


def binding_forward(params, z, V, value_scalars=None, alpha_val=0.0):
    """HRR bind + conjugate unbind (R20), with cross-layer superposition.

    Forward::

        k_q = Σ_i k^(i),  v_q = Σ_j v^(j)
        b_raw = Σ_{i,j} IFFT(norm_fft(k^(i)) ⊙ norm_fft(v^(j)))
        b_q = RVQ(C_bind, b_raw)
        v_out = NN(C_val, IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q)))
        o_bind = v_out

    ``C_val`` is the concatenation of every ``val_cb`` layer's effective
    codebook ``A @ W`` into ``(n_layers * M_bind, d)``. The unbind stage selects
    the nearest row by plain Euclidean distance; it is a controller decision to
    keep the unbind readout independent of the local value bias (which is
    applied only inside the key/val/bind residual-VQ chains, R22).

    ``value_scalars`` follows R22's binding structure
    ``{'key': (M,), 'val': (M,), 'bind': (M,)}``. A bare ``(M,)`` vector is
    accepted as the legacy shared vector.

    Returns:
        (v_out, residuals, indices, b_q, k_q, val_out_idx): residuals maps
        'key'/'val'/'bind' to the list of per-layer query vectors, so the EMA
        pass can update each binding codebook in the space it actually
        quantises. indices maps the same families to the rows selected by
        *this* forward (value-biased when the family's value vector is active);
        EMA must use them rather than recomputing a plain Euclidean argmin.
        ``b_q`` is the quantised bound representation, ``k_q`` the summed key
        representation, and ``val_out_idx`` the row selected in the
        concatenated ``C_val`` (the observable unbind result).
    """
    M_bind = int(params['key_cb'][0]['A'].shape[0])
    v_key = _family_value_vector(value_scalars, 'key', M_bind, 'binding')
    v_val = _family_value_vector(value_scalars, 'val', M_bind, 'binding')
    v_bind = _family_value_vector(value_scalars, 'bind', M_bind, 'binding')

    # Key/value projections: W_k = V @ A_k, z_k = z @ W_k.T
    # (d, r_max) @ (r_max, d) = (d, d); (B, d) @ (d, d) = (B, d)
    W_k = V @ params['A_k']
    W_v = V @ params['A_v']
    z_k = z @ W_k.T  # (B, d)
    z_v = z @ W_v.T  # (B, d)

    # Multi-layer residual VQ for key / value (each family's own value vector).
    k_q_list, key_inputs, key_idx, relax_k = _residual_vq_chain(
        params['key_cb'], z_k, v_key, alpha_val)
    v_q_list, val_inputs, val_idx, relax_v = _residual_vq_chain(
        params['val_cb'], z_v, v_val, alpha_val)

    # Cross-layer HRR binding (n_layers² pairs).
    b_raw = 0.0
    for k_i in k_q_list:
        for v_j in v_q_list:
            b_raw = b_raw + jnp.fft.irfft(
                normalize_fft(k_i) * normalize_fft(v_j), n=z.shape[-1])

    # Quantize the bound representation.
    b_q_list, bind_inputs, bind_idx, relax_b = _residual_vq_chain(
        params['bind_cb'], b_raw, v_bind, alpha_val)

    b_q = jnp.sum(jnp.stack(b_q_list), axis=0)
    k_q = jnp.sum(jnp.stack(k_q_list), axis=0)

    # ── Unbind: recover v from (k_q, b_q), then retrieve from C_val ────────
    # C_val spans every value-codebook layer; concatenating (rather than
    # summing) keeps each layer's rows addressable in the controller output.
    C_val = jnp.concatenate(
        [cb['A'] @ cb['W'] for cb in params['val_cb']], axis=0)
    recovered = jnp.fft.irfft(
        jnp.conj(normalize_fft(k_q)) * normalize_fft(b_q), n=z.shape[-1])
    dist = jnp.sum((recovered[:, None, :] - C_val[None, :, :]) ** 2, axis=-1)
    val_out_idx = dist.argmin(axis=-1)  # (B,) into the concatenated C_val
    # The readout is a hard codebook row (no gradient through the phase-only
    # `recovered` query; that derivative is O(1/|X|) at init scale).
    v_out = lax.stop_gradient(C_val[val_out_idx])

    # Ruling 16: output-level STE. The forward value is exactly v_out
    # (`z - sg(z)` is exactly zero), while the backward pass is the identity
    # w.r.t. z — matching b.md §4's uniform STE rule for every lattice.
    o_bind = v_out + (z - lax.stop_gradient(z))
    for step in (relax_k, relax_v, relax_b):
        if step is not None:
            o_bind = o_bind + step  # value-scalar gradient (forward ≡ 0)

    residuals = {'key': key_inputs, 'val': val_inputs, 'bind': bind_inputs}
    indices = {'key': key_idx, 'val': val_idx, 'bind': bind_idx}
    return o_bind, residuals, indices, b_q, k_q, val_out_idx


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
    """Dual codebook contrast lattice.

    Each family uses its own value vector (R22):
    ``value_scalars = {'a': (M,), 'b': (M,)}``; a bare ``(M,)`` array is the
    legacy shared vector.

    Returns:
        (o_contrast, indices, residuals). ``indices`` maps ``'a'``/``'b'`` to
        the per-layer selected rows (R21). ``residuals`` maps the same families
        to the per-layer residual inputs R18's codebook/commitment terms
        consume. The two are separate return values so ``indices`` stays exactly
        the R21 interface.
    """
    M = int(params['C_a'][0]['A'].shape[0])
    v_a = _family_value_vector(value_scalars, 'a', M, 'contrast')
    v_b = _family_value_vector(value_scalars, 'b', M, 'contrast')

    a_q_list, a_inputs, a_idx, relax_a = _residual_vq_chain(
        params['C_a'], z, v_a, alpha_val)
    b_q_list, b_inputs, b_idx, relax_b = _residual_vq_chain(
        params['C_b'], z, v_b, alpha_val)

    o_contrast = (jnp.sum(jnp.stack(a_q_list), axis=0)
                  + jnp.sum(jnp.stack(b_q_list), axis=0)) / 2.0
    o_contrast = z + lax.stop_gradient(o_contrast - z)  # STE
    for step in (relax_a, relax_b):
        if step is not None:
            o_contrast = o_contrast + step / 2.0  # value-scalar gradient

    return o_contrast, {'a': a_idx, 'b': b_idx}, {
        'a': a_inputs, 'b': b_inputs}


def contrast_info_nce_loss(params, z, tau=0.1):
    """DualVC InfoNCE with the positive excluded from the negatives (R21).

    Each family's nearest code is the positive; the other family's rows are
    the negatives. ``tau`` defaults to 0.1. The loss is evaluated in the
    numerically stable form ``pos/tau + logsumexp(-neg/tau)`` over
    ``neg != positive_index``.
    """
    z_detach = lax.stop_gradient(z)
    loss = 0.0

    for layer_idx in range(len(params['C_a'])):
        C_a = params['C_a'][layer_idx]['A'] @ params['C_a'][layer_idx]['W']
        C_b = params['C_b'][layer_idx]['A'] @ params['C_b'][layer_idx]['W']

        d_a = jnp.linalg.norm(z_detach[:, None, :] - C_a[None, :, :], axis=-1)
        d_b = jnp.linalg.norm(z_detach[:, None, :] - C_b[None, :, :], axis=-1)

        idx_a = d_a.argmin(axis=-1)
        idx_b = d_b.argmin(axis=-1)

        d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1).squeeze(-1)
        d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1).squeeze(-1)

        def _ce_excluding(pos, neg, idx):
            excluded = jax.nn.one_hot(idx, neg.shape[-1], dtype=bool)
            log_neg = jnp.where(excluded, -jnp.inf, -neg / tau)
            return pos / tau + jax.nn.logsumexp(log_neg, axis=-1)

        loss = loss + jnp.mean(_ce_excluding(d_a_pos, d_b, idx_b))
        loss = loss + jnp.mean(_ce_excluding(d_b_pos, d_a, idx_a))

    return loss


def contrast_value_biased_nce_loss(params, z, v_harm, tau=0.1, tau_val=0.05):
    """Value-biased contrastive loss — harm-weighted negative sampling (R21).

    Negatives are weighted by ``exp(-||c - v_harm||² / tau_val)``: the
    *squared* Euclidean distance, not the norm. The positive index is excluded
    from the negatives for both families. ``tau`` defaults to 0.1 and
    ``tau_val`` to 0.05 (R17).

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

        d_a = jnp.linalg.norm(z_detach[:, None, :] - C_a[None, :, :], axis=-1)
        d_b = jnp.linalg.norm(z_detach[:, None, :] - C_b[None, :, :], axis=-1)

        idx_a = d_a.argmin(axis=-1)
        idx_b = d_b.argmin(axis=-1)

        d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1).squeeze(-1)
        d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1).squeeze(-1)

        # R21: squared harm distance, not the norm. Keep the harm exponent in
        # log space: at init scale (C ~ 1e-4, tau_val=0.05) the weights
        # underflow to exactly 0, and taking log(exp(...)) would produce
        # inf/NaN gradients.
        log_w_a = -jnp.sum((C_a - v_harm[None, :]) ** 2, axis=-1) / tau_val
        log_w_b = -jnp.sum((C_b - v_harm[None, :]) ** 2, axis=-1) / tau_val

        # Stable form:
        #   -log(exp(-p/τ) / (exp(-p/τ) + Σ_{j≠pos} w_j·exp(-d_j/τ)))
        #   = log(1 + Σ_{j≠pos} w_j·exp(-(d_j - d_pos)/τ))
        #   = softplus(logsumexp(log w_j - (d_j - d_pos)/τ))
        # The last form never forms exp(positive), and softplus handles the
        # large-logit regime without overflow.
        def _weighted_term(pos, d_neg, log_w, idx):
            excluded = jax.nn.one_hot(idx, log_w.shape[0], dtype=bool)
            logits = log_w[None, :] - (d_neg - pos[:, None]) / tau
            logits = jnp.where(excluded, -jnp.inf, logits)
            return jax.nn.softplus(jax.nn.logsumexp(logits, axis=-1))

        loss_a = _weighted_term(d_a_pos, d_b, log_w_b, idx_b)
        loss_b = _weighted_term(d_b_pos, d_a, log_w_a, idx_a)
        loss = loss + jnp.mean(loss_a) + jnp.mean(loss_b)

    return loss


# ── Local Value Scalars ──────────────────────────────────────────────────────

def init_danger_params(rng, M_danger, d):
    """Initialize the two danger codebooks — frozen, never trained.

    R13 requires ``C_threats`` and ``C_normal`` to come from different data and
    to be structurally distinct; ``lcm.py`` refuses to load a checkpoint whose
    halves are equal. These are **structural placeholders** for the ≤100-step
    conformance run: the threat half is a fixed random basis, and the normal
    half is a second, independent draw. ``cog_train`` replaces the normal half
    with the first training batch's ``z`` so the normal path is data-aligned;
    neither half has any semantic safety meaning yet. Used like gvalue: saved
    with SHA-256, never updated by the optimizer.
    """
    threat_key, normal_key = jax.random.split(rng)
    return {
        'C': jax.random.normal(threat_key, (M_danger, d)) * 0.02,
        'C_normal': jax.random.normal(normal_key, (M_danger, d)) * 0.02,
    }


def init_value_scalars(cfg: LCMConfig):
    """Initialize local value scalars v_j at R22's per-codebook granularity.

    Stored unnormalised; ``value_biased_scores`` applies ``tanh`` so the
    *effective* local value bias saturates at ±1 as the design specifies while
    keeping a non-zero gradient everywhere.

    Structure (R22)::

        {'hrq': {'top': (M_top,), 'fine': (M_fine,)},
         'sparse': (M_sparse,),
         'lowrank': (M_lr,),
         'manifold': (M_man,),
         'binding': {'key': (M_bind,), 'val': (M_bind,), 'bind': (M_bind,)},
         'contrast': {'a': (M_contrast,), 'b': (M_contrast,)}}

    Residual layers share one vector (HRQ's fine layers, LowRank's rank
    layers); different sub-codebooks (binding's three families, contrast's two
    codebooks) each get their own.
    """
    return {
        'hrq': {
            'top': jnp.zeros(int(cfg.M_top)),
            'fine': jnp.zeros(int(cfg.M_fine)),
        },
        'sparse': jnp.zeros(int(cfg.M_sparse)),
        'lowrank': jnp.zeros(int(cfg.M_lr)),
        'manifold': jnp.zeros(int(cfg.M_man)),
        'binding': {
            family: jnp.zeros(int(cfg.M_bind))
            for family in ('key', 'val', 'bind')
        },
        'contrast': {
            family: jnp.zeros(int(cfg.M_contrast))
            for family in ('a', 'b')
        },
    }
