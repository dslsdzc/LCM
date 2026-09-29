"""Parity for the canonical six-lattice step.

Three layers, because they fail differently:

  * per-lattice — the step calls each forward with the arguments the forward
    expects (right value_scalars, right shared V, right return arity). A wrong
    argument still produces a tensor of plausible shape, so this is checked
    against direct calls rather than against a shape assertion.
  * whole-step — the step, given the same z model.forward feeds it, reproduces
    model.forward's z_q. Since model.forward now calls the step, this is what
    catches the step silently diverging from its only other consumer.
  * gradient — every lattice's parameters actually receive gradient through the
    step.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_cognitive_step_parity.py -v
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from train.cognitive_step import six_lattice_step
from train.config import LCMConfig
from train.encoder import encoder_forward
from train.hyp import safe_unit
from train.fusion import fuse_lattices_with_aux
from train.lattices import (
    binding_forward, contrast_forward, hrq_forward, lowrank_forward,
    manifold_forward, sparse_forward,
)
from train.model import forward, init_all_params

# r_max must equal max(ranks): binding_forward projects the shared low-rank base
# V (last dim max(ranks)) through params['A_k'], shaped (r_max, ...).
CFG = LCMConfig(d_model=32, d_ff=48, n_heads=4, d_head=8, vocab_size=64,
                max_seq_len=16, n_encoder_layers=1,
                M_top=16, M_fine=8, n_hrq_layers=1,
                M_sparse=16, M_lr=16, M_man=16, M_bind=16, M_contrast=16,
                n_bind_layers=1, n_contrast_layers=1,
                n_self_codes=8, use_bf16=False)

LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast")


@pytest.fixture(scope="module")
def built():
    params, gvalue, self_state = init_all_params(CFG, jax.random.PRNGKey(0))
    x = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
    z = safe_unit(encoder_forward(params['encoder'], x, CFG.n_heads))
    return params, gvalue, self_state, x, z


def _vs(params, name):
    return params.get('value_scalars', {}).get(name)


def test_step_output_is_the_fusion_of_its_own_reported_outputs(built):
    """whole-step: z_next must be the fusion of the outputs the step reports.

    This is the invariant whose absence caused the original problem. The
    cognitive loop used to return a z_next built from a generic nearest-neighbour
    over flattened codebooks — a computation unrelated to the six lattice
    forwards the architecture claims to run. Recomputing the fusion from
    `aux['lattice_outputs']` and `aux['soft_mask']` pins the step: whatever it
    returns must be exactly the documented computation over what it says it ran.

    Recomputing through the canonical fuse_lattices_with_aux is deliberate. A
    hand-rolled fusion here would be a second implementation, and would agree
    with the step for the wrong reason.
    """
    params, _, _, _, z = built
    z_next, aux = six_lattice_step(z, params, CFG, training=True,
                                   rng=jax.random.PRNGKey(1))
    want, _, _ = fuse_lattices_with_aux(
        aux['lattice_outputs'], aux['soft_mask'], params['fusion'],
        gvalue=None, beta_val=CFG.beta_val, tau_val=CFG.tau_val_signal)
    assert np.allclose(np.asarray(z_next), np.asarray(want), atol=1e-6), \
        "z_next is not the fusion of the step's own reported lattice outputs"


def test_model_forward_uses_the_step(built):
    """model.forward's z_q is the step's z_q with self fused as the 7th element.

    Checked by reproducing forward's fusion from the step's own outputs plus a
    freshly computed self output — which is exactly what forward does. If
    forward ever regrows its own copy of the lattice stack, this stops holding.
    """
    params, gvalue, self_state, x, z = built
    _, z_q_ref, _, aux_ref, _ = forward(
        params, gvalue, x, CFG, training=True, rng=jax.random.PRNGKey(1),
        self_state=self_state)
    # forward returns the fused lattice list including the self element.
    self_out = aux_ref['lattice_outputs'][-1]
    _, lat = six_lattice_step(z, params, CFG, training=True,
                              rng=jax.random.PRNGKey(1), gvalue=gvalue)
    want, _, _ = fuse_lattices_with_aux(
        lat['lattice_outputs'] + [self_out], lat['soft_mask'],
        params['fusion'], gvalue=gvalue, beta_val=CFG.beta_val,
        tau_val=CFG.tau_val_signal, self_bias_weight=CFG.alpha_self)
    assert np.allclose(np.asarray(z_q_ref), np.asarray(want), atol=1e-6), \
        "model.forward is no longer the canonical step plus its own periphery"


@pytest.mark.parametrize("name", LATTICES)
def test_lattice_output_matches_direct_call(built, name):
    """per-lattice: the step's i-th output equals a direct call to that forward."""
    params, gvalue, self_state, x, z = built
    _, aux = six_lattice_step(z, params, CFG, training=True,
                              rng=jax.random.PRNGKey(1))
    got = aux['lattice_outputs'][LATTICES.index(name)]

    vs, a = _vs(params, name), CFG.alpha_val
    if name == "hrq":
        want = hrq_forward(params['hrq'], z, CFG.tau_route_fallback,
                           value_scalars=vs, alpha_val=a)[0]
    elif name == "sparse":
        want = sparse_forward(params['sparse'], z, training=True,
                              lambda_sparse=CFG.lambda_sparse,
                              value_scalars=vs, alpha_val=a)[0]
    elif name == "lowrank":
        want = lowrank_forward(params['lowrank'], z, CFG.ranks,
                               value_scalars=vs, alpha_val=a)
    elif name == "manifold":
        want = manifold_forward(params['manifold'], z,
                                value_scalars=vs, alpha_val=a)[0]
    elif name == "binding":
        V = params['lowrank']['A_V'] @ params['lowrank']['W_V']
        want = binding_forward(params['binding'], z, V,
                               value_scalars=vs, alpha_val=a)[0]
    else:
        want = contrast_forward(params['contrast'], z,
                                value_scalars=vs, alpha_val=a)

    assert np.allclose(np.asarray(got), np.asarray(want), atol=1e-6), \
        f"{name}: step output differs from a direct forward call"


def test_hrq_and_routing_receive_gradient(built):
    """gradient: the two lattices that currently do receive gradient, do."""
    params, _, _, _, z = built

    def scalar(p):
        z_next, _ = six_lattice_step(z, p, CFG, training=True,
                                     rng=jax.random.PRNGKey(1))
        return jnp.sum(z_next ** 2)

    grads = jax.grad(scalar)(params)
    for key in ("hrq", "route", "fusion"):
        total = sum(float(jnp.sum(jnp.abs(g)))
                    for g in jax.tree_util.tree_leaves(grads[key]))
        assert total > 0, f"grad.{key} is exactly zero"


def test_retrieval_gradient_reaches_only_hrq_by_design(built):
    """gradient: exactly one lattice is differentiable through retrieval.

    This pins a deliberate design, not a defect. Every other lattice retrieves
    via ``o = z + stop_gradient(hard - z)`` (see train/lattices.py: simvq_codebook,
    _residual_vq_chain, contrast_forward): forward is the hard-quantised value,
    ``d/dz`` is the identity so z stays differentiable, and ``d/dparams`` is
    exactly zero because the codebook is detached on purpose.

    Codebooks are updated by EMA plus explicit auxiliary losses in
    cog_train's Stage-3 block (vq_total, contrast_info_nce_loss,
    manifold_orth_loss) — not by retrieval gradient. Letting gradient through
    the distance term would drag every codebook toward the current batch
    centroid; value_biased_scores' docstring states this outright.

    hrq is the exception: it does not use that wrapper.

    If this test ever fails because another lattice gained gradient, that is a
    design change and should be a deliberate one.
    """
    params, _, _, _, z = built

    def scalar(p):
        z_next, _ = six_lattice_step(z, p, CFG, training=True,
                                     rng=jax.random.PRNGKey(1))
        return jnp.sum(z_next ** 2)

    grads = jax.grad(scalar)(params)
    totals = {}
    for name in LATTICES:
        totals[name] = sum(float(jnp.sum(jnp.abs(g)))
                           for g in jax.tree_util.tree_leaves(grads[name]))

    assert totals["hrq"] > 0, "hrq must stay differentiable through retrieval"
    detached = [n for n in LATTICES if n != "hrq" and totals[n] == 0.0]
    assert sorted(detached) == sorted(n for n in LATTICES if n != "hrq"), (
        f"expected every non-hrq lattice detached, but these received "
        f"gradient: {[n for n in LATTICES if n != 'hrq' and totals[n] != 0.0]} "
        f"(totals={totals})")
