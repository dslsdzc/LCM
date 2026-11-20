"""Contrast lattice conformance (R21/R22).

R21: ``tau=0.1``, the positive index is excluded from the negatives for both
codebooks, the value-biased variant weights by ``exp(-||c-v_harm||²/tau_val)``
(squared distance), and ``contrast_forward`` exposes per-layer indices.

R22: contrast has one value vector per codebook (``a`` and ``b``).

Run from the repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_contrast_conformance.py -v
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from train.config import LCMConfig
from train.lattices import (
    contrast_forward, contrast_info_nce_loss, contrast_value_biased_nce_loss,
    init_contrast_params, init_value_scalars,
)


def test_temperature_is_accepted():
    p = init_contrast_params(jax.random.PRNGKey(0), 8, 16, 2)
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    loss = contrast_info_nce_loss(p, z, tau=0.1)
    assert jnp.isfinite(loss)
    # R21 default is 0.1, not the old 0.5.
    assert inspect.signature(contrast_info_nce_loss).parameters['tau'].default == 0.1


def _codebooks(params, layer):
    C_a = params['C_a'][layer]['A'] @ params['C_a'][layer]['W']
    C_b = params['C_b'][layer]['A'] @ params['C_b'][layer]['W']
    return C_a, C_b


def _info_nce_reference(params, z, tau, exclude_positive):
    z = jax.lax.stop_gradient(z)
    total = jnp.array(0.0)
    for layer in range(len(params['C_a'])):
        C_a, C_b = _codebooks(params, layer)
        d_a = jnp.linalg.norm(z[:, None, :] - C_a[None, :, :], axis=-1)
        d_b = jnp.linalg.norm(z[:, None, :] - C_b[None, :, :], axis=-1)
        idx_a = d_a.argmin(axis=-1)
        idx_b = d_b.argmin(axis=-1)
        d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1)[:, 0]
        d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1)[:, 0]

        def term(pos, neg, idx):
            if exclude_positive:
                neg = jnp.where(
                    jax.nn.one_hot(idx, neg.shape[-1], dtype=bool),
                    -jnp.inf, -neg / tau)
            else:
                neg = -neg / tau
            return jax.nn.logsumexp(neg, axis=-1) + pos / tau

        total = total + jnp.mean(term(d_a_pos, d_b, idx_b))
        total = total + jnp.mean(term(d_b_pos, d_a, idx_a))
    return total


def test_positive_is_excluded_from_negatives():
    d, M = 8, 12
    p = init_contrast_params(jax.random.PRNGKey(0), d, M, 2)
    z = jax.random.normal(jax.random.PRNGKey(1), (3, d))
    loss = contrast_info_nce_loss(p, z, tau=0.1)

    want = _info_nce_reference(p, z, tau=0.1, exclude_positive=True)
    no_exclusion = _info_nce_reference(p, z, tau=0.1, exclude_positive=False)

    assert jnp.isfinite(loss)
    np.testing.assert_allclose(np.asarray(loss), np.asarray(want), rtol=1e-5)
    # If the positive row were included as its own negative, the loss would
    # differ; this guards against a future "simplification" that drops the mask.
    assert not jnp.allclose(loss, no_exclusion, rtol=1e-5)


def test_far_positive_is_finite_and_lower_than_positive_at_query():
    """The excluded positive must not be a free denominator term.

    With the positive index removed, moving the whole negative bank away from
    the query lowers ``logsumexp(-neg/tau)`` and therefore the loss. If the
    positive row were counted as its own negative, the loss would be pinned by
    that row instead. Both cases must stay finite.
    """
    d = 2
    z = jnp.zeros((1, d))
    C_a = jnp.array([[1.0, 0.0], [3.0, 0.0]])

    def params_with(C_b):
        C_b = jnp.asarray(C_b, dtype=jnp.float32)
        return {
            'C_a': [{'A': C_a, 'W': jnp.eye(d)}],
            'C_b': [{'A': C_b, 'W': jnp.eye(d)}],
        }

    near = params_with([[0.0, 0.0], [0.0, 0.0]])
    far = params_with([[2.0, 0.0], [4.0, 0.0]])

    loss_near = contrast_info_nce_loss(near, z, tau=1.0)
    loss_far = contrast_info_nce_loss(far, z, tau=1.0)

    assert jnp.isfinite(loss_near) and jnp.isfinite(loss_far)
    assert float(loss_far) < float(loss_near)


def test_value_biased_loss_uses_squared_harm_distance():
    d, M = 8, 12
    p = init_contrast_params(jax.random.PRNGKey(2), d, M, 1)
    z = jax.random.normal(jax.random.PRNGKey(3), (2, d))
    v_harm = jax.random.normal(jax.random.PRNGKey(4), (d,))
    tau, tau_val = 0.5, 1.0

    loss = contrast_value_biased_nce_loss(p, z, v_harm, tau=tau, tau_val=tau_val)

    C_a, C_b = _codebooks(p, 0)
    d_a = jnp.linalg.norm(z[:, None, :] - C_a[None, :, :], axis=-1)
    d_b = jnp.linalg.norm(z[:, None, :] - C_b[None, :, :], axis=-1)
    idx_a = d_a.argmin(axis=-1)
    idx_b = d_b.argmin(axis=-1)
    d_a_pos = jnp.take_along_axis(d_a, idx_a[:, None], axis=-1)[:, 0]
    d_b_pos = jnp.take_along_axis(d_b, idx_b[:, None], axis=-1)[:, 0]

    def weights(C, squared):
        diff = C - v_harm[None, :]
        dist = jnp.sum(diff ** 2, axis=-1) if squared else jnp.linalg.norm(diff, axis=-1)
        return jnp.exp(-dist / tau_val)

    def biased_loss(w_a, w_b):
        excl_b = jax.nn.one_hot(idx_b, M)
        excl_a = jax.nn.one_hot(idx_a, M)
        loss_a = jnp.log1p(jnp.sum(
            w_b[None, :] * (1.0 - excl_b)
            * jnp.exp(-(d_b - d_a_pos[:, None]) / tau), axis=-1))
        loss_b = jnp.log1p(jnp.sum(
            w_a[None, :] * (1.0 - excl_a)
            * jnp.exp(-(d_a - d_b_pos[:, None]) / tau), axis=-1))
        return jnp.mean(loss_a) + jnp.mean(loss_b)

    want = biased_loss(weights(C_a, True), weights(C_b, True))
    norm_version = biased_loss(weights(C_a, False), weights(C_b, False))

    np.testing.assert_allclose(np.asarray(loss), np.asarray(want), rtol=1e-5)
    assert not jnp.allclose(loss, norm_version, rtol=1e-5)
    # R21 default is 0.05.
    sig = inspect.signature(contrast_value_biased_nce_loss)
    assert sig.parameters['tau_val'].default == 0.05


def test_contrast_forward_returns_per_layer_indices():
    d, M, n, B = 8, 16, 3, 2
    p = init_contrast_params(jax.random.PRNGKey(5), d, M, n)
    z = jax.random.normal(jax.random.PRNGKey(6), (B, d))

    o, aux = contrast_forward(p, z)

    assert o.shape == (B, d)
    for family in ('a', 'b'):
        assert len(aux[family]) == n
        assert len(aux['residuals'][family]) == n
        for idx in aux[family]:
            assert idx.shape == (B,)
        assert aux['residuals'][family][0].shape == (B, d)


def test_contrast_value_scalars_are_per_codebook():
    """R22: changing C_a's value vector cannot move C_b's selection."""
    d, M = 4, 4
    base = jnp.stack([
        jnp.zeros(d),
        jnp.eye(d)[0],
        jnp.eye(d)[1] * 2,
        -jnp.eye(d)[2] * 3,
    ])
    p = {
        'C_a': [{'A': base, 'W': jnp.eye(d)}],
        'C_b': [{'A': base, 'W': jnp.eye(d)}],
    }
    z = jnp.array([[0.1, 0.0, 0.0, 0.0]])
    vs = {'a': jnp.array([0.0, 0.0, 1.0, 0.0]),
          'b': jnp.array([0.0, 0.0, 0.0, 0.0])}

    _, aux0 = contrast_forward(p, z, value_scalars=vs, alpha_val=50.0)
    vs2 = dict(vs, a=jnp.array([0.0, 1.0, 0.0, 0.0]))
    _, aux1 = contrast_forward(p, z, value_scalars=vs2, alpha_val=50.0)

    assert int(aux0['a'][0][0]) == 2
    assert int(aux1['a'][0][0]) == 1
    assert int(aux0['b'][0][0]) == int(aux1['b'][0][0]) == 0


def test_contrast_feature_bank_replaces_dead_rows_by_farthest_point():
    """R21: the training loop calls ``replace_dead_contrast`` every interval."""
    from train.cog_train import _host_bank_update
    from train.feature_bank import FeatureBank, init_contrast_last_used

    d, M = 2, 3
    C = jnp.array([[5.0, 0.0], [0.25, 0.25], [6.0, 0.0]])
    params = {
        'contrast': {
            'C_a': [{'A': C, 'W': jnp.eye(d)}],
            'C_b': [{'A': C, 'W': jnp.eye(d)}],
        },
    }
    contrast_bank = FeatureBank(capacity=4, dim=d)
    for v in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]):
        contrast_bank.add(jnp.asarray([v]))

    clock = init_contrast_last_used(n_layers=1, M_contrast=M)
    loop_stats = {
        'contrast_aux': {
            'a': [jnp.array([1])],
            'b': [jnp.array([0])],
            'residuals': {'a': [], 'b': []},
        },
    }
    cfg = LCMConfig(d_model=d, M_contrast=M, n_contrast_layers=1)
    new_params, _, new_clock = _host_bank_update(
        params, {}, FeatureBank(capacity=4, dim=d), contrast_bank, clock,
        loop_stats, step=1001, cfg=cfg)

    C_a = new_params['contrast']['C_a'][0]
    C_b = new_params['contrast']['C_b'][0]
    # Dead rows are replaced by the farthest bank vector ([-1, 0]).
    assert float((C_a['A'] @ C_a['W'])[0, 0]) < 0.0
    assert float((C_a['A'] @ C_a['W'])[2, 0]) < 0.0
    # A live row is untouched: b selected row 0, so it is not dead.
    np.testing.assert_allclose(np.asarray((C_b['A'] @ C_b['W'])[0]),
                               [5.0, 0.0])
    # The replacement is stored as A=vector, W=I so the effective row is exact.
    np.testing.assert_allclose(np.asarray(C_a['W']), np.eye(d), atol=0)
    # The clock records both the live selection and the replaced rows.
    assert int(new_clock['a'][0][0]) == 1001
    assert int(new_clock['a'][0][1]) == 1001
    assert int(new_clock['b'][0][0]) == 1001


def test_value_scalar_structure_has_distinct_family_shapes():
    cfg = LCMConfig(
        d_model=8, M_top=7, M_fine=5, M_sparse=3, M_lr=4,
        M_man=6, M_bind=8, M_contrast=9)
    vs = init_value_scalars(cfg)

    assert vs['hrq']['top'].shape == (7,)
    assert vs['hrq']['fine'].shape == (5,)
    assert vs['sparse'].shape == (3,)
    assert vs['lowrank'].shape == (4,)
    assert vs['manifold'].shape == (6,)
    for family in ('key', 'val', 'bind'):
        assert vs['binding'][family].shape == (8,)
    for family in ('a', 'b'):
        assert vs['contrast'][family].shape == (9,)

    # Every family owns a distinct leaf, not a shared (M,) vector.
    assert len({id(v) for v in (
        vs['hrq']['top'], vs['hrq']['fine'], vs['sparse'],
        vs['lowrank'], vs['manifold'],
        vs['binding']['key'], vs['binding']['val'], vs['binding']['bind'],
        vs['contrast']['a'], vs['contrast']['b'])}) == 10
