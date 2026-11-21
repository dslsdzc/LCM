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
            # Proper InfoNCE: the positive logit is part of the denominator.
            # softplus(logsumexp_{j!=pos}(-(d_j - p)/tau)) is >= 0.
            logits = -(neg - pos[:, None]) / tau
            if exclude_positive:
                logits = jnp.where(
                    jax.nn.one_hot(idx, neg.shape[-1], dtype=bool),
                    -jnp.inf, logits)
            return jax.nn.softplus(jax.nn.logsumexp(logits, axis=-1))

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


def test_info_nce_is_nonnegative_when_negatives_are_far():
    """InfoNCE is a negative log-likelihood, so it cannot go below zero.

    The old ``pos/tau + logsumexp(-neg/tau)`` form omitted the positive logit
    from the denominator. With far negatives it returned a large negative
    value, and the optimizer could reduce it without bound by pushing the
    negative bank away. That is the loss divergence seen in the CPU dialogue
    run (total loss ≈ -1360).
    """
    d = 2
    z = jnp.zeros((1, d))
    C_a = jnp.array([[1.0, 0.0], [2.0, 0.0]])
    C_b = jnp.array([[50.0, 0.0], [100.0, 0.0]])
    p = {
        'C_a': [{'A': C_a, 'W': jnp.eye(d)}],
        'C_b': [{'A': C_b, 'W': jnp.eye(d)}],
    }
    loss = contrast_info_nce_loss(p, z, tau=1.0)
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0, f"InfoNCE must be >= 0, got {float(loss)}"


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


def test_value_biased_loss_is_finite_at_init_scale_with_tau_val_005():
    """tau_val=0.05 must not yield inf/NaN when harm weights underflow."""
    p = init_contrast_params(jax.random.PRNGKey(10), 8, 16, 2)
    z = jax.random.normal(jax.random.PRNGKey(11), (2, 8))
    v_harm = jax.random.normal(jax.random.PRNGKey(12), (8,))
    loss = contrast_value_biased_nce_loss(p, z, v_harm, tau_val=0.05)
    assert jnp.isfinite(loss)

    grads = jax.grad(
        lambda pp: contrast_value_biased_nce_loss(
            pp, z, v_harm, tau_val=0.05))(p)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))


def test_contrast_forward_returns_per_layer_indices():
    d, M, n, B = 8, 16, 3, 2
    p = init_contrast_params(jax.random.PRNGKey(5), d, M, n)
    z = jax.random.normal(jax.random.PRNGKey(6), (B, d))

    o, indices, residuals = contrast_forward(p, z)

    assert o.shape == (B, d)
    for family in ('a', 'b'):
        assert len(indices[family]) == n
        assert len(residuals[family]) == n
        for idx in indices[family]:
            assert idx.shape == (B,)
        assert residuals[family][0].shape == (B, d)


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

    _, idx0, _ = contrast_forward(p, z, value_scalars=vs, alpha_val=50.0)
    vs2 = dict(vs, a=jnp.array([0.0, 1.0, 0.0, 0.0]))
    _, idx1, _ = contrast_forward(p, z, value_scalars=vs2, alpha_val=50.0)

    assert int(idx0['a'][0][0]) == 2
    assert int(idx1['a'][0][0]) == 1
    assert int(idx0['b'][0][0]) == int(idx1['b'][0][0]) == 0


def test_contrast_interval_path_uses_usage_between_checks_and_preserves_live_rows():
    """R17/R18: per-step clock, interval-only replacement, only dead rows."""
    from train.cog_train import _contrast_bank_step
    from train.feature_bank import FeatureBank, init_contrast_last_used

    d, M = 2, 3
    C = jnp.array([[5.0, 0.0], [0.25, 0.25], [6.0, 0.0]])
    W = jnp.eye(d) * 2.0  # non-identity W exercises the dead-row solve
    params = {
        'contrast': {
            'C_a': [{'A': C / 2.0, 'W': W}],
            'C_b': [{'A': C / 2.0, 'W': W}],
        },
    }
    A_a_before = np.asarray(params['contrast']['C_a'][0]['A'])
    A_b_before = np.asarray(params['contrast']['C_b'][0]['A'])
    W_before = np.asarray(W)

    contrast_bank = FeatureBank(capacity=4, dim=d)
    for v in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]):
        contrast_bank.add(jnp.asarray([v]))

    clock = init_contrast_last_used(n_layers=1, M_contrast=M)
    cfg = LCMConfig(d_model=d, M_contrast=M, n_contrast_layers=1)

    # Step 500 is between checks: update the clock, do not replace.
    stats_500 = {'contrast_indices': {
        'a': [jnp.array([2])], 'b': [jnp.array([0])]}}
    params, clock = _contrast_bank_step(
        params, contrast_bank, clock, stats_500, step=500,
        do_replace=False, cfg=cfg)
    assert int(clock['a'][0][2]) == 500
    assert int(clock['a'][0][0]) == 0

    # Step 1001 is a check: code 2 was used 501 steps ago, so it is live.
    stats_1001 = {'contrast_indices': {
        'a': [jnp.array([1])], 'b': [jnp.array([0])]}}
    params, clock = _contrast_bank_step(
        params, contrast_bank, clock, stats_1001, step=1001,
        do_replace=True, cfg=cfg)

    C_a = params['contrast']['C_a'][0]
    C_b = params['contrast']['C_b'][0]
    A_a = np.asarray(C_a['A'])
    A_b = np.asarray(C_b['A'])

    # Non-dead rows are bit-identical (A rows and W untouched).
    np.testing.assert_array_equal(A_a[1:], A_a_before[1:])
    np.testing.assert_array_equal(A_b[0], A_b_before[0])
    np.testing.assert_array_equal(np.asarray(C_a['W']), W_before)
    np.testing.assert_array_equal(np.asarray(C_b['W']), W_before)

    # Dead rows become the requested farthest-point vectors: C row == [-1, 0].
    np.testing.assert_allclose(np.asarray(A_a[0] @ C_a['W']), [-1.0, 0.0])
    np.testing.assert_allclose(np.asarray(A_b[1] @ C_b['W']), [-1.0, 0.0])
    np.testing.assert_allclose(np.asarray(A_b[2] @ C_b['W']), [-1.0, 0.0])

    # a row 2 (selected at step 500) was not replaced; b row 0 stayed live.
    np.testing.assert_array_equal(A_a[2], A_a_before[2])
    np.testing.assert_array_equal(A_b[0], A_b_before[0])
    assert int(clock['a'][0][1]) == 1001
    assert int(clock['a'][0][0]) == 1001  # replaced rows are stamped at check


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
