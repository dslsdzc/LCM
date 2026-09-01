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
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    losses = {}

    # Routing gate
    losses['route'] = route_commit_loss(z, aux['z_route'], cfg.beta_vq)

    # Hierarchy: commitment for each SimVQ layer
    hrq_loss = 0.0
    for fb in params['hrq']['fine']:
        C_fb = fb['A'] @ fb['W']
        hrq_loss += _commit_loss(z, C_fb, cfg.beta_vq)
    C_top = params['hrq']['top']['A'] @ params['hrq']['top']['W']
    hrq_loss += _commit_loss(z, C_top, cfg.beta_vq)
    losses['hrq'] = hrq_loss

    # Sparse
    C_sparse = params['sparse']['C']
    losses['sparse'] = _commit_loss(z, C_sparse, cfg.beta_vq)

    # Low-rank (all layers)
    lr_loss = 0.0
    V = params['lowrank']['A_V'] @ params['lowrank']['W_V']
    r = z
    for u_k, r_k in zip(params['lowrank']['U'], cfg.ranks):
        C_k = u_k @ V[:, :r_k].T
        lr_loss += _commit_loss(r, C_k, cfg.beta_vq)
        idx = jnp.linalg.norm(r[:, None, :] - C_k[None, :, :], axis=-1).argmin(axis=-1)
        r = r - C_k[idx]
    losses['lowrank'] = lr_loss

    # Manifold
    C_man = exp_map(params['manifold']['C'])
    losses['manifold'] = _commit_loss(z, log_map(C_man), cfg.beta_vq)

    # Binding: all sub-codebooks
    binding_loss = 0.0
    for cb_list in [params['binding']['key_cb'],
                    params['binding']['val_cb'],
                    params['binding']['bind_cb']]:
        for cb in cb_list:
            C_cb = cb['A'] @ cb['W']
            binding_loss += _commit_loss(z, C_cb, cfg.beta_vq)
    losses['binding'] = binding_loss

    return losses


def _commit_loss(z, C, beta=0.25):
    """VQ commitment loss: β·||sg[z_norm] - C_norm[idx]||² (unit-sphere)."""
    z_n = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    C_n = C / (jnp.linalg.norm(C, axis=-1, keepdims=True) + 1e-8)
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


def compute_orth_loss(params, aux, cfg: LCMConfig):
    """Tangent space orthogonality regularization."""
    return cfg.lambda_orth * manifold_orth_loss(
        params['manifold']['T'],
        aux['man_idx'],
        k=cfg.n_orth_samples,
        lambda_orth=cfg.lambda_orth)


def compute_value_contrast_loss(params, gvalue, aux, cfg: LCMConfig):
    """Value contrast loss for local value scalars.

    Negative samples weighted by proximity to v_harm (safety-critical boundary):
      L_val = λ_val * Σ_i InfoNCE(o_i, C_pos, C_neg_w_harm_weight)

    Uses numerically stable log-softplus formulation to avoid exp underflow.

    Only trains the local value scalars v_j (gvalue stays frozen).
    """
    if gvalue is None or aux.get('value_signals') is None:
        return jnp.array(0.0)

    v_harm = gvalue.C_neg[1]  # the harm vector (index 1 in C_neg)
    loss = 0.0
    n_lattices = len(aux['lattice_outputs'])
    tau_val = cfg.tau_val_signal

    for i in range(n_lattices):
        o = aux['lattice_outputs'][i]  # (B, d)

        # Distances to positive anchors (min over 4 laws)
        d_pos = jnp.stack([poincare_distance(o, gvalue.C_pos[j])
                           for j in range(4)]).min(axis=0)  # (B, 1)
        # Distances to negative anchors, harm-weighted
        d_neg_vals = []
        for j in range(4):
            d_j = poincare_distance(o, gvalue.C_neg[j])  # (B, 1)
            harm_dist = poincare_distance(o, v_harm)  # (B, 1)
            w_harm = jnp.exp(-harm_dist / tau_val)
            d_neg_vals.append(w_harm * d_j)
        d_neg_w = jnp.stack(d_neg_vals, axis=-1).mean(axis=-1)  # (B, 1)

        # Numerically stable InfoNCE: -log(sigmoid((neg-pos)/τ))
        # = softplus(-(neg-pos)/τ) = softplus((d_pos - d_neg) / τ)
        # Safe state (d_neg > d_pos) → logit < 0 → loss ≈ 0;
        # unsafe state (d_neg < d_pos) → loss large.
        logit = (d_pos.squeeze(-1) - d_neg_w.squeeze(-1)) / tau_val
        loss = loss + jnp.mean(jax.nn.softplus(logit))

    return cfg.lambda_val * loss / n_lattices


def compute_safety_margin_loss(gvalue, aux, cfg: LCMConfig):
    """Safety margin regularization — mild penalty near boundary."""
    if gvalue is None or 'lattice_outputs' not in aux:
        return jnp.array(0.0)
    # Apply to fused output z_q (we don't have it here directly, use lattice_outputs)
    # In practice, safety_margin_loss is applied separately in train_step
    return jnp.array(0.0)


def compute_total_loss(params, gvalue, logits, targets, z, aux, cfg: LCMConfig,
                       ewc_loss_val=None):
    """Compute total training loss.

    L_total = L_LM + L_VQ + L_contrast + L_orth + L_val + L_ewc + L_margin

    Args:
        ewc_loss_val: Optional EWC loss from continual learning system.
    """
    loss_lm = compute_lm_loss(logits, targets, cfg.vocab_size)
    vq_losses = compute_vq_loss(params, aux, z, cfg)
    loss_vq = sum(vq_losses.values())
    loss_contrast = compute_contrast_loss(params, z, cfg, gvalue=gvalue)
    loss_orth = compute_orth_loss(params, aux, cfg)
    loss_val = compute_value_contrast_loss(params, gvalue, aux, cfg)

    total = loss_lm + loss_vq + loss_contrast + loss_orth + loss_val

    if ewc_loss_val is not None:
        total = total + ewc_loss_val

    components = {
        'lm': loss_lm,
        'vq': loss_vq,
        'contrast': loss_contrast,
        'orth': loss_orth,
        'val': loss_val,
        'ewc': ewc_loss_val if ewc_loss_val is not None else jnp.array(0.0),
    }
    return total, components


# Import for hyperbolic ops in this module
from train.hyp import exp_map, log_map, poincare_distance
