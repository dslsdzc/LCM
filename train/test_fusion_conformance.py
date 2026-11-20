"""Canonical fusion conformance (R4/R5).

R4: ``z_q = layer_norm(Σ softmask·alpha·o + self_bias·o_self)``.
R5: no value-signal soft penalty. ``gvalue``/``beta_val``/``tau_val`` stay as
deprecated API parameters so old callers keep working, but they must not change
the weights or ``z_q``.

Run from the repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_fusion_conformance.py -v
"""
import jax
import jax.numpy as jnp

from train.fusion import (
    fuse_lattices_with_aux, init_fusion_params, layer_norm,
)


def _outs(n=6, d=4, scale=1.0):
    return [jnp.ones((1, d)) * (i * scale) for i in range(n)]


def test_gvalue_does_not_change_weights_or_z_q():
    """R5: a non-None gvalue (even a dummy object) is ignored."""
    outs = _outs()
    mask = jnp.ones((1, 6)) / 6
    params = init_fusion_params(jax.random.PRNGKey(0), 6, 4)

    z, w, h = fuse_lattices_with_aux(
        outs, mask, params, gvalue=object(), beta_val=0.5, tau_val=0.1)
    z2, w2, h2 = fuse_lattices_with_aux(outs, mask, params, gvalue=None)

    assert jnp.allclose(w, w2)
    assert jnp.allclose(z, z2)
    assert jnp.allclose(h, h2)


def test_weights_are_only_softmask_times_alpha():
    """R5: the fusion weight is exactly ``soft_mask_i · alpha_i``, normalised."""
    outs = _outs()
    mask = jnp.array([[0.1, 0.2, 0.3, 0.1, 0.2, 0.1]])
    params = init_fusion_params(jax.random.PRNGKey(0), 6, 4)
    params = dict(params, alpha=jnp.array([1.0, 2.0, 0.5, 1.0, 3.0, 1.0]))

    _, weights, _ = fuse_lattices_with_aux(outs, mask, params)
    want = mask * params['alpha'][None, :]
    want = want / want.sum(axis=-1, keepdims=True)
    assert jnp.allclose(weights, want)


def test_self_bias_is_added_before_layer_norm():
    """R4/R15: the self term enters the pre-LayerNorm sum, not after it."""
    outs = _outs()
    mask = jnp.ones((1, 6)) / 6
    params = init_fusion_params(jax.random.PRNGKey(0), 6, 4)
    self_out = jnp.array([[0.5, -0.5, 0.25, -0.25]])
    self_bias = 0.3

    z, _, _ = fuse_lattices_with_aux(
        outs + [self_out], mask, params, self_bias_weight=self_bias)

    stacked = jnp.stack(outs, axis=-1)
    weights = mask * params['alpha'][None, :]
    weights = weights / weights.sum(axis=-1, keepdims=True)
    pre = jnp.einsum('bdn,bn->bd', stacked, weights) + self_bias * self_out
    want = layer_norm(pre, params['ln_scale'], params['ln_bias'])

    assert jnp.allclose(z, want, atol=1e-6)
    # The distinction is observable: post-LayerNorm addition produces a
    # different vector, so this test would fail on the old ordering.
    post = layer_norm(jnp.einsum('bdn,bn->bd', stacked, weights),
                      params['ln_scale'], params['ln_bias'])
    assert not jnp.allclose(z, post + self_bias * self_out, atol=1e-6)
