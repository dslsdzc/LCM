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
from train.param_groups import EMA_PATHS


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


# ── R18: layered codebook / commitment losses ──────────────────────────────
#
# These are two distinct terms, not one name used for both directions:
#
#   codebook loss    β·mean((sg[r] − C[idx])²)   → gradient only to C
#   commitment loss  β·mean((r − sg[C[idx]])²)   → gradient only to r
#
# b.md §4.0.3's residual_vq_commit_loss is the first form. It moves the
# codebook toward the encoder, and does nothing for the encoder. The second
# form is what pulls the encoder (and upper residual layers) toward the
# codebook. EMA-managed codebooks (sparse/manifold/binding) never receive
# gradients — their codebook term is skipped (see cognitive_vq_losses), but
# the commitment term still trains the encoder.

def _codebook_term(r, q, beta):
    """β·mean((sg[r] − q)²) for an already-gathered quantised vector q."""
    return beta * jnp.mean((lax.stop_gradient(r) - q) ** 2)


def _commitment_term(r, q, beta):
    """β·mean((r − sg[q])²) for an already-gathered quantised vector q."""
    return beta * jnp.mean((r - lax.stop_gradient(q)) ** 2)


def codebook_loss(r, C, idx, beta=0.25):
    """VQ codebook loss: ``β·mean((sg[r] − C[idx])²)``.

    Gradient flows only into the codebook ``C``; ``r`` is detached.
    """
    return _codebook_term(r, C[idx], beta)


def commitment_loss(r, C, idx, beta=0.25):
    """VQ commitment loss: ``β·mean((r − sg[C[idx]])²)``.

    Gradient flows only into ``r`` (the encoder / upper residual layer); the
    selected codebook row is detached.
    """
    return _commitment_term(r, C[idx], beta)


def hrq_vq_loss(params, z, aux, cfg: LCMConfig):
    """Per-layer HRQ codebook/commitment losses in post-``log_map`` space.

    R18: HRQ's loss lives in the Euclidean space after
    ``log_map(c_top ⊕ c_fine)``, not in the tangent-space residual ``r``.
    ``hrq_forward`` already returns one continuous/quantised pair per layer
    (top-1 plus each fine layer) in that space, so this consumes the forward
    pass instead of re-running it.

    Args:
        params: HRQ params. Used to validate that the pair count matches the
            codebook structure (top + ``len(params['fine'])``).
        z: (B, d) tangent-space input to the lattice; the continuous top term.
        aux: the step/loop aux dict carrying ``hrq_vq`` — or that dict itself.
            Required keys: ``cont`` and ``quant``, each a list of pairs.
        cfg: LCMConfig; ``cfg.beta_vq`` is the weight.

    Returns:
        (codebook, commitment), each a ``(1 + n_fine,)`` array of per-layer
        scalars. The top-level prototype is the first entry.
    """
    pairs = aux['hrq_vq'] if isinstance(aux, dict) and 'hrq_vq' in aux else aux
    if not isinstance(pairs, dict) or 'cont' not in pairs or 'quant' not in pairs:
        raise KeyError(
            "hrq_vq_loss requires aux['hrq_vq'] with 'cont'/'quant' lists "
            "(produced by hrq_forward)")
    # The top continuous term is the lattice input, by definition; the stored
    # copy is the same array, but using `z` keeps the argument meaningful and
    # matches the fallback branch's shared input.
    cont = [z] + list(pairs['cont'][1:])
    quant = list(pairs['quant'])
    n_fine = len(params['fine'])
    if len(cont) != 1 + n_fine or len(quant) != 1 + n_fine:
        raise ValueError(
            f"HRQ loss pair count {len(cont)}/{len(quant)} does not match "
            f"top + {n_fine} fine layers")
    beta = cfg.beta_vq
    codebook = jnp.stack([_codebook_term(c, q, beta)
                          for c, q in zip(cont, quant)])
    commitment = jnp.stack([_commitment_term(c, q, beta)
                            for c, q in zip(cont, quant)])
    return codebook, commitment


def _ema_owns(ema_paths, *candidates):
    """True when any candidate codebook path is claimed by ``ema_paths``.

    Prefix matching in both directions lets ``('lowrank',)`` claim every
    codebook below it and lets the leaf ``('sparse','C')`` claim exactly that
    codebook. The skip decision is therefore derived from the parameter groups,
    not a second hardcoded list that can drift.
    """
    for candidate in candidates:
        for ema_path in ema_paths:
            n = min(len(candidate), len(ema_path))
            if candidate[:n] == ema_path[:n]:
                return True
    return False


def cognitive_vq_losses(params, stats, cfg: LCMConfig, ema_paths=EMA_PATHS):
    """Compose the whole R18 loss from one forward pass's stats.

    ``stats`` is the last step of ``cog_loop_scan(..., return_stats=True)`` or
    the aux returned by ``six_lattice_step``: it carries the per-layer
    continuous/quantised vectors, residual inputs and selection indices the
    losses need, so nothing is recomputed with a second full forward pass.

    ``ema_paths`` is the set of EMA-owned codebook paths (defaults to
    ``train.param_groups.EMA_PATHS``). A codebook in that set gets no codebook
    term because its gradient is discarded by the parameter grouping; every
    codebook still gets the commitment term, which trains ``r``.

    Returns:
        (total, parts): ``total`` is the scalar added to the training loss
        (codebook + commitment); ``parts`` holds per-lattice and split totals
        for logging. EMA-owned codebooks report a zero codebook term by design.
    """
    z = stats['z_in']
    beta = cfg.beta_vq
    parts = {}

    # HRQ — pure gradient unless a group decision says otherwise: top + every
    # fine layer, both terms per layer.
    hrq_cb, hrq_cm = hrq_vq_loss(params['hrq'], z, stats, cfg)
    hrq_cb_total = jnp.array(0.0)
    if not _ema_owns(ema_paths, ('hrq', 'top')):
        hrq_cb_total = hrq_cb_total + hrq_cb[0]
    if not _ema_owns(ema_paths, ('hrq', 'fine')):
        hrq_cb_total = hrq_cb_total + jnp.sum(hrq_cb[1:])
    parts['hrq_codebook'] = hrq_cb_total
    parts['hrq_commitment'] = jnp.sum(hrq_cm)

    # Sparse — EMA codebook (currently): skip codebook, keep commitment.
    if _ema_owns(ema_paths, ('sparse', 'C')):
        parts['sparse_codebook'] = jnp.array(0.0)
    else:
        parts['sparse_codebook'] = codebook_loss(
            z, params['sparse']['C'], stats['sparse_idx'], beta)
    parts['sparse_commitment'] = commitment_loss(
        z, params['sparse']['C'], stats['sparse_idx'], beta)

    # LowRank — pure gradient unless U/V are EMA: every rank layer.
    lowrank_ema = _ema_owns(
        ema_paths, ('lowrank', 'U'), ('lowrank', 'A_V'), ('lowrank', 'W_V'))
    lowrank_cb = jnp.array(0.0)
    lowrank_cm = jnp.array(0.0)
    V = params['lowrank']['A_V'] @ params['lowrank']['W_V']
    lr_aux = stats['lowrank_aux']
    for u_k, rank, residual, idx in zip(
            params['lowrank']['U'], cfg.ranks,
            lr_aux['residuals'], lr_aux['indices']):
        C_k = u_k @ V[:, :rank].T
        if not lowrank_ema:
            lowrank_cb = lowrank_cb + codebook_loss(residual, C_k, idx, beta)
        lowrank_cm = lowrank_cm + commitment_loss(residual, C_k, idx, beta)
    parts['lowrank_codebook'] = lowrank_cb
    parts['lowrank_commitment'] = lowrank_cm

    # Manifold — EMA codebook: skip codebook; commitment in log_map space.
    C_man = log_map(exp_map(params['manifold']['C']))
    if _ema_owns(ema_paths, ('manifold', 'C')):
        parts['manifold_codebook'] = jnp.array(0.0)
    else:
        parts['manifold_codebook'] = codebook_loss(
            z, C_man, stats['man_idx'], beta)
    parts['manifold_commitment'] = commitment_loss(
        z, C_man, stats['man_idx'], beta)

    # Binding — three sub-codebook families, each layer's own residual.
    binding_cb = jnp.array(0.0)
    binding_cm = jnp.array(0.0)
    for family in ('key', 'val', 'bind'):
        family_ema = _ema_owns(ema_paths, ('binding', f'{family}_cb'))
        for cb, residual, idx in zip(
                params['binding'][f'{family}_cb'],
                stats['binding_residuals'][family],
                stats['binding_indices'][family]):
            C_k = cb['A'] @ cb['W']
            if not family_ema:
                binding_cb = binding_cb + codebook_loss(
                    residual, C_k, idx, beta)
            binding_cm = binding_cm + commitment_loss(
                residual, C_k, idx, beta)
    parts['binding_codebook'] = binding_cb
    parts['binding_commitment'] = binding_cm

    # Contrast — pure gradient unless a family is EMA: C_a and C_b, every
    # layer, both terms.
    contrast_cb = jnp.array(0.0)
    contrast_cm = jnp.array(0.0)
    for family in ('a', 'b'):
        family_ema = _ema_owns(ema_paths, ('contrast', f'C_{family}'))
        fam_aux = stats['contrast_aux'][family]
        for cb, residual, idx in zip(
                params['contrast'][f'C_{family}'],
                fam_aux['residuals'], fam_aux['indices']):
            C_k = cb['A'] @ cb['W']
            if not family_ema:
                contrast_cb = contrast_cb + codebook_loss(
                    residual, C_k, idx, beta)
            contrast_cm = contrast_cm + commitment_loss(
                residual, C_k, idx, beta)
    parts['contrast_codebook'] = contrast_cb
    parts['contrast_commitment'] = contrast_cm

    # Routing — pure gradient, one term each. ``route_commit_loss`` is the
    # codebook direction (its argument is the raw C_route[idx], not the STE
    # output); commitment_loss supplies the encoder-side pull.
    C_route = params['route']['C_route']
    route_idx = stats['route_idx']
    if _ema_owns(ema_paths, ('route', 'C_route')):
        parts['route_codebook'] = jnp.array(0.0)
    else:
        parts['route_codebook'] = route_commit_loss(
            z, C_route[route_idx], beta)
    parts['route_commitment'] = commitment_loss(z, C_route, route_idx, beta)

    codebook = sum(v for k, v in parts.items() if k.endswith('_codebook'))
    commitment = sum(v for k, v in parts.items() if k.endswith('_commitment'))
    parts['codebook'] = codebook
    parts['commitment'] = commitment
    return codebook + commitment, parts


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
