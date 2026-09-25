"""LCM loss functions.

Total loss: L_total = L_LM + L_VQ + L_contrast + L_orth
"""
import jax
import jax.numpy as jnp
from jax import lax
import optax

from train.lattices import (
    route_commit_loss,
    hrq_forward,
    sparse_forward,
    lowrank_forward,
    manifold_forward,
    manifold_orth_loss,
    binding_forward,
    contrast_forward,
    contrast_info_nce_loss,
    contrast_value_biased_nce_loss,
)
from train.config import LCMConfig


def compute_lm_loss(logits, targets, vocab_size):
    """Cross-entropy language modeling loss."""
    # logits: (B, N, V), targets: (B, N)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1))
    return loss.mean()


def compute_vq_loss(params, aux, z, cfg: LCMConfig):
    """Compute all VQ commitment losses."""
    # Normalize z to unit sphere so commitment is about direction, not magnitude.
    # Without this, encoder output can grow unboundedly and commitment loss explodes.
    z = safe_unit(z)
    losses = {}

    # Routing gate — pass the RAW codebook vector, not routing_gate's STE output.
    # The STE output is constant w.r.t. C_route, so the loss would leave the
    # routing prototype with zero gradient (see route_commit_loss).
    losses['route'] = route_commit_loss(
        z, params['route']['C_route'][aux['route_idx']], cfg.beta_vq)

    # Hierarchy: commitment for each SimVQ layer
    hrq_loss = 0.0
    for fb in params['hrq']['fine']:
        C_fb = fb['A'] @ fb['W']
        hrq_loss += commit_loss(z, C_fb, cfg.beta_vq)
    C_top = params['hrq']['top']['A'] @ params['hrq']['top']['W']
    hrq_loss += commit_loss(z, C_top, cfg.beta_vq)
    losses['hrq'] = hrq_loss

    # Sparse
    C_sparse = params['sparse']['C']
    losses['sparse'] = commit_loss(z, C_sparse, cfg.beta_vq)

    # Low-rank (all layers)
    lr_loss = 0.0
    V = params['lowrank']['A_V'] @ params['lowrank']['W_V']
    r = z
    for u_k, r_k in zip(params['lowrank']['U'], cfg.ranks):
        C_k = u_k @ V[:, :r_k].T
        lr_loss += commit_loss(r, C_k, cfg.beta_vq)
        idx = jnp.linalg.norm(r[:, None, :] - C_k[None, :, :], axis=-1).argmin(axis=-1)
        r = r - C_k[idx]
    losses['lowrank'] = lr_loss

    # Manifold
    C_man = exp_map(params['manifold']['C'])
    losses['manifold'] = commit_loss(z, log_map(C_man), cfg.beta_vq)

    # Binding: all sub-codebooks
    binding_loss = 0.0
    for cb_list in [params['binding']['key_cb'],
                    params['binding']['val_cb'],
                    params['binding']['bind_cb']]:
        for cb in cb_list:
            C_cb = cb['A'] @ cb['W']
            binding_loss += commit_loss(z, C_cb, cfg.beta_vq)
    losses['binding'] = binding_loss

    return losses


def commit_loss(z, C, beta=0.25):
    """VQ commitment loss: β·||sg[z_norm] - C_norm[idx]||² (unit-sphere)."""
    z_n = safe_unit(z)
    C_n = safe_unit(C)
    dist = jnp.linalg.norm(z_n[:, None, :] - C_n[None, :, :], axis=-1)
    idx = dist.argmin(axis=-1)
    return beta * jnp.mean((lax.stop_gradient(z_n) - C_n[idx]) ** 2)


def compute_contrast_loss(params, z, cfg: LCMConfig, gvalue=None):
    """Contrastive InfoNCE loss (detached from encoder).

    Uses value-biased negative sampling when gvalue is available,
    weighting negatives by proximity to v_harm (safety-critical focus).
    """
    if gvalue is not None and cfg.alpha_val > 0:
        v_harm = gvalue.C_neg[1]  # harm anchor
        return cfg.lambda_contrast * contrast_value_biased_nce_loss(
            params['contrast'], z, v_harm, tau=0.5, tau_val=cfg.tau_val_signal)
    return cfg.lambda_contrast * contrast_info_nce_loss(
        params['contrast'], z, tau=0.5)


def compute_orth_loss(params, aux, cfg: LCMConfig, rng=None):
    """Tangent space orthogonality regularization: λ·‖Tᵀ T − I‖².

    ``manifold_orth_loss`` already scales by ``lambda_orth``; multiplying again
    here applied the weight twice (λ²).
    """
    return manifold_orth_loss(
        params['manifold']['T'],
        aux['man_idx'],
        k=cfg.n_orth_samples,
        lambda_orth=cfg.lambda_orth,
        rng=rng)


def _min_euclidean(x, anchors):
    """Minimum Euclidean distance from x (B, d) to each of the anchors (d,)."""
    return jnp.min(jnp.stack(
        [jnp.linalg.norm(x - a[None, :], axis=-1) for a in anchors], axis=-1),
        axis=-1)


def value_contrast_loss(lattice_outputs, C_pos, C_neg, tau_val,
                        lambda_val=1.0):
    """Value contrast loss over the lattice outputs — the canonical form.

    ``L = λ · mean_i softplus((d_pos(i) − d_neg(i)) / τ)``

    Safe state (far from the negative anchors, close to the positive ones)
    drives the logit negative and the loss to zero; an unsafe state makes it
    large. Note the sign: ``d_pos − d_neg``, NOT ``d_neg − d_pos``. The
    reversed form trains the lattice outputs *toward* the negative anchors and
    away from the positive ones — it is a silent, high-impact inversion (the
    loss still decreases, just in the wrong direction).

    Distances are Euclidean, not hyperbolic. Lattice outputs are
    ``z + stop_gradient(...)`` with ``z`` normalised to the unit sphere, i.e.
    they sit essentially on the Poincaré ball's boundary, where
    ``poincare_similarity``'s denominator ``(1−‖u‖²)(1−‖v‖²)`` collapses to the
    clamp and the "distance" becomes meaningless.

    ``lambda_val`` is applied exactly once, here. Callers must not scale the
    result again.

    The negative term is the distance to the *nearest* negative anchor. The
    previous form, a harm-proximity-weighted mean ``Σ_j exp(−‖o−v_harm‖/τ)·d_j``,
    evaluated to zero in practice: lattice outputs sit at unit norm and the
    anchors at 0.9, so distances are O(1) while ``tau_val_signal`` is 0.1 and
    ``exp(−1.4/0.1) ≈ 1e-6``. The weighting did not reweight the four anchors
    either — it depends on ``o`` and the harm anchor only, so it scaled the
    whole negative term by a factor that was always ~0. The loss then reduced
    to "be close to the positive anchors" with the negatives inert. Harm
    emphasis is still present where it is well-posed: the value signal in
    ``fusion.fuse_lattices`` and the per-entry weighting in
    ``lattices.contrast_value_biased_nce_loss``.

    Args:
        lattice_outputs: list of (B, d) arrays, one per lattice.
        C_pos: (P, d) positive value anchors.
        C_neg: (N, d) negative value anchors; ``C_neg[1]`` is the harm anchor.
        tau_val: Temperature for the margin logit.
        lambda_val: Loss weight.

    Returns:
        Scalar loss.
    """
    if lattice_outputs is None or C_pos is None or C_neg is None:
        return jnp.array(0.0)
    if lambda_val == 0:
        return jnp.array(0.0)

    loss = 0.0
    for o in lattice_outputs:
        d_pos = _min_euclidean(o, C_pos)
        d_neg = _min_euclidean(o, C_neg)
        logit = (d_pos - d_neg) / tau_val
        loss = loss + jnp.mean(jax.nn.softplus(logit))

    return lambda_val * loss / len(lattice_outputs)


def compute_value_contrast_loss(params, gvalue, aux, cfg: LCMConfig):
    """Value contrast loss for local value scalars (see ``value_contrast_loss``).

    Negative samples are weighted by proximity to the harm anchor, so
    safety-critical boundaries dominate the signal. Only the local value
    scalars are affected; ``gvalue`` stays frozen.
    """
    if gvalue is None or aux.get('value_signals') is None:
        return jnp.array(0.0)
    return value_contrast_loss(
        aux['lattice_outputs'], gvalue.C_pos, gvalue.C_neg,
        cfg.tau_val_signal, cfg.lambda_val)



# NOTE: there used to be a `compute_total_loss` composition here, reachable only
# from train.py's `_build_loss_grad_fn` — a second, never-executed copy of the
# training loss. Both are gone; `_jitted_step.loss_fn` in train.py is the single
# live composition, and every term it uses comes from this module or lattices.py.
# Two parallel loss implementations is how the value-contrast sign and the
# orthogonality term drifted apart in the first place.


# Import for hyperbolic ops in this module
from train.hyp import exp_map, log_map, safe_unit
