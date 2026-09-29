"""Canonical six-lattice orchestration — the single source for one cognitive step.

Two consumers, one implementation:

  * ``train/model.py::forward`` — one step, inside a full forward pass
  * ``train/cog_loop.py``       — the same step repeated under ``lax.scan``

There used to be two. ``model.forward`` ran the real lattice operations, while
the cognitive loop ran a generic nearest-neighbour over flattened codebooks and
fused the results by reciprocal distance. Those are not the same computation,
and the gap was invisible from the outside: ``manifold_forward`` slides a
Poincare geodesic, ``binding_forward`` binds HRR key/value chains through a
residual VQ stack, ``contrast_forward`` projects onto learned contrastive
directions. A distance-weighted average of nearest codebook entries reproduces
none of that, so a cognitive-training run was optimising a different model from
the one ``model.py`` describes.

Scope is routing + the six lattice forwards + fusion. The encoder that produces
z, the self lattice, and the generation head remain in ``model.forward``;
GValue is accepted through fusion but is None for now.
"""
import jax
import jax.numpy as jnp

from train.fusion import fuse_lattices_with_aux
from train.lattices import (
    binding_forward, contrast_forward, hrq_forward, lowrank_forward,
    manifold_forward, routing_gate, sparse_forward,
)


def six_lattice_step(z, params, cfg, *, training=True, rng=None, gvalue=None,
                     self_output=None, self_bias_weight=None):
    """One cognitive step: route, run the six lattices, fuse.

    Args:
        z: (B, d) current cognitive state.
        params: carries 'route', 'fusion', the six lattice param dicts, and
            optionally 'value_scalars'.
        cfg: LCMConfig.
        gvalue: optional GValueCodebook for value-biased fusion. Passed through
            rather than hardcoded, so a caller that has one keeps its
            modulation; cognitive training passes None this round.
        training: selects sparse_forward's LFQ path and the routing gate's
            hard/soft choice. A loop step is a forward evaluation of the
            lattice stack, so this mirrors whatever the surrounding pass is.
        rng: PRNG key for the routing gate's Gumbel noise.
        self_output: optional (B, d) self-lattice output, appended as a 7th
            fused element. The self lattice itself is run by the caller; this
            function does not compute it.
        self_bias_weight: weight for self_output. Required whenever
            self_output is given, because fusion identifies the self element by
            having more outputs than routing weights.

    Returns:
        z_next: (B, d) fused state.
        aux: dict with 'soft_mask', 'weights' (normalised fusion weights),
            'entropy' (per-sample, nats), 'lattice_outputs', and the
            per-lattice selection indices the canonical forward exposes.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    if self_output is not None and self_bias_weight is None:
        raise ValueError(
            "self_output requires self_bias_weight: fuse_lattices separates "
            "the self element by it having no routing weight, so passing one "
            "without the other silently routes the self output instead")

    soft_mask, z_route, route_idx = routing_gate(
        params['route'], z, cfg.tau_route, hard=not training, rng=rng)

    vs = params.get('value_scalars', {})
    alpha_val = cfg.alpha_val

    # Return arities differ and are not uniform: lowrank_forward and
    # contrast_forward return a bare array, the other four return tuples.
    o_hrq, hrq_idx, hrq_top_sim = hrq_forward(
        params['hrq'], z, cfg.tau_route_fallback,
        value_scalars=vs.get('hrq'), alpha_val=alpha_val)
    # LFQ dynamic threshold: Poincare dis-similarity to the nearest HRQ top
    # prototype. Evaluation only.
    d_top = None if training else (1.0 - jnp.mean(hrq_top_sim))
    o_sparse, sparse_idx = sparse_forward(
        params['sparse'], z, training=training,
        lambda_sparse=cfg.lambda_sparse, d_top=d_top,
        value_scalars=vs.get('sparse'), alpha_val=alpha_val)
    o_lowrank = lowrank_forward(
        params['lowrank'], z, cfg.ranks,
        value_scalars=vs.get('lowrank'), alpha_val=alpha_val)
    o_manifold, man_idx = manifold_forward(
        params['manifold'], z,
        value_scalars=vs.get('manifold'), alpha_val=alpha_val)
    # binding_forward quantises against the shared low-rank base V, derived
    # from lowrank's own factors rather than stored separately.
    V = params['lowrank']['A_V'] @ params['lowrank']['W_V']
    o_binding, binding_residuals = binding_forward(
        params['binding'], z, V,
        value_scalars=vs.get('binding'), alpha_val=alpha_val)
    o_contrast = contrast_forward(
        params['contrast'], z,
        value_scalars=vs.get('contrast'), alpha_val=alpha_val)

    lattice_outputs = [o_hrq, o_sparse, o_lowrank, o_manifold, o_binding,
                       o_contrast]
    if self_output is not None:
        lattice_outputs.append(self_output)

    z_next, weights, entropy = fuse_lattices_with_aux(
        lattice_outputs, soft_mask, params['fusion'],
        gvalue=gvalue, beta_val=cfg.beta_val, tau_val=cfg.tau_val_signal,
        self_bias_weight=self_bias_weight if self_output is not None else None)

    aux = {
        'soft_mask': soft_mask,
        'weights': weights,
        'entropy': entropy,
        'lattice_outputs': lattice_outputs,
        'z_route': z_route,
        'route_idx': route_idx,
        'hrq_idx': hrq_idx,
        'hrq_top_sim': hrq_top_sim,
        'sparse_idx': sparse_idx,
        'man_idx': man_idx,
        # Per-layer query vectors of the binding residual chains — the space
        # each binding codebook actually quantises, and therefore the space its
        # EMA must accumulate.
        'binding_residuals': binding_residuals,
    }
    return z_next, aux
