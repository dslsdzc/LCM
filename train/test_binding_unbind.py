"""Binding unbind conformance (R20).

R20::

    k_q = Σ_i k^(i), v_q = Σ_j v^(j)
    b_raw = Σ_{i,j} IFFT(norm_fft(k^(i)) ⊙ norm_fft(v^(j)))
    b_q = RVQ_3layer(C_bind, b_raw)
    v_out = NN(C_val, IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q)))
    o_bind = z + stop_gradient(v_out - z)   # Ruling 16 output-level STE

The controller decision in ``binding_forward`` is that ``C_val`` is the
concatenation of every ``val_cb`` layer's effective codebook and the nearest row
is chosen by plain Euclidean distance (no value bias in the unbind stage).

Run from the repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_binding_unbind.py -v
"""
import jax
import jax.numpy as jnp
import numpy as np

from train.lattices import (
    binding_forward, init_binding_params, normalize_fft,
)


def test_binding_forward_shapes_and_auxiliary_return():
    """The contract: 6-tuple return, b_q/k_q and unbind row exposed."""
    d, M, n, r = 8, 16, 3, 4
    p = init_binding_params(jax.random.PRNGKey(0), d, M, n, r)
    V = jax.random.normal(jax.random.PRNGKey(1), (d, r)) * 0.1
    z = jax.random.normal(jax.random.PRNGKey(2), (2, d)) * 0.2

    o_bind, res, indices, b_q, k_q, val_out_idx = binding_forward(p, z, V)

    assert o_bind.shape == (2, d)
    assert b_q.shape == (2, d)
    assert k_q.shape == (2, d)
    assert val_out_idx.shape == (2,)
    for family in ('key', 'val', 'bind'):
        assert len(res[family]) == n
        assert len(indices[family]) == n
        assert res[family][0].shape == (2, d)


def _crafted_roundtrip_params(d=8, M=16, seed=0):
    """One-layer binding with identity projections and fixed code words.

    ``v_code`` is a delta: its spectrum already has unit magnitude, so the
    phase-only HRR normalization is lossless on it and the round-trip can be
    checked exactly. (An arbitrary vector loses magnitude information in
    ``norm_fft``; the nearest-code assertion below is the weaker contract for
    that case.)
    """
    key = jax.random.PRNGKey(seed)
    k_code = jax.random.normal(key, (d,))
    v_code = jax.nn.one_hot(0, d)

    # Rows 0 are the quantisation targets; the rest are far away so the argmin
    # cannot accidentally pick another row.
    def far_rows(target, offset=10.0):
        rows = [target]
        for i in range(1, M):
            rows.append(target + offset * (i + 1))
        return jnp.stack(rows)

    bind_row = jnp.fft.irfft(
        normalize_fft(k_code) * normalize_fft(v_code), n=d)
    params = {
        'A_k': jnp.eye(d),
        'A_v': jnp.eye(d),
        'key_cb': [{'A': far_rows(k_code), 'W': jnp.eye(d)}],
        'val_cb': [{'A': far_rows(v_code), 'W': jnp.eye(d)}],
        'bind_cb': [{'A': far_rows(bind_row), 'W': jnp.eye(d)}],
    }
    V = jnp.eye(d)
    return params, V, v_code


def test_unbind_recovers_value_code():
    """Round-trip: unbind(bind(k, v), k) selects v's value code."""
    d, M = 8, 16
    params, V, v_code = _crafted_roundtrip_params(d, M)
    z = jnp.stack([v_code, -v_code])  # identity projections: z is the query

    o_bind, _, _, b_q, k_q, val_out_idx = binding_forward(params, z, V)

    # b_q is the bound representation, not the output (R20).
    assert not jnp.allclose(o_bind, b_q, atol=1e-5)
    assert int(val_out_idx[0]) == 0
    # The unbind result is the value codebook row selected by the recovered
    # query; both batch rows used the same key/value pair.
    C_val = jnp.concatenate([
        cb['A'] @ cb['W'] for cb in params['val_cb']], axis=0)
    for b in range(2):
        recovered = jnp.fft.irfft(
            jnp.conj(normalize_fft(k_q[b])) * normalize_fft(b_q[b]), n=d)
        np.testing.assert_allclose(np.asarray(recovered),
                                   np.asarray(v_code), atol=1e-5)
        nearest = int(jnp.argmin(jnp.sum((C_val - recovered[None, :]) ** 2,
                                         axis=-1)))
        assert nearest == 0
        np.testing.assert_allclose(np.asarray(o_bind[b]),
                                   np.asarray(v_code), atol=1e-5)


def test_binding_value_scalars_are_per_family():
    """R22: key/val/bind each use their own value vector."""
    d = 4
    base = jnp.array([[0.0, 0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0, 0.0]])
    params = {
        'A_k': jnp.eye(d),
        'A_v': jnp.eye(d),
        'key_cb': [{'A': base, 'W': jnp.eye(d)}],
        'val_cb': [{'A': base, 'W': jnp.eye(d)}],
        'bind_cb': [{'A': base, 'W': jnp.eye(d)}],
    }
    V = jnp.eye(d)
    z = jnp.array([[0.1, 0.0, 0.0, 0.0]])
    vs = {'key': jnp.array([0.0, 1.0]),
          'val': jnp.array([0.0, 0.0]),
          'bind': jnp.array([0.0, 0.0])}

    _, _, idx0, _, _, _ = binding_forward(
        params, z, V, value_scalars=vs, alpha_val=20.0)
    # Key value bias flips row 0 -> row 1; val has no bias and stays at row 0.
    assert int(idx0['key'][0][0]) == 1
    assert int(idx0['val'][0][0]) == 0

    vs2 = dict(vs, key=jnp.array([0.0, 0.0]))
    _, _, idx1, _, _, _ = binding_forward(
        params, z, V, value_scalars=vs2, alpha_val=20.0)
    assert int(idx1['key'][0][0]) == 0
    # Changing key's value vector must not move the val selection.
    assert int(idx1['val'][0][0]) == int(idx0['val'][0][0]) == 0


def test_binding_output_ste_forward_is_exact_and_z_gradient_is_identity():
    """Ruling 16: forward is exactly v_out, backward is identity w.r.t. z."""
    params, V, _ = _crafted_roundtrip_params(8, 16)
    z = jax.random.normal(jax.random.PRNGKey(9), (2, 8)) * 0.2

    o_bind, _, _, _, _, val_out_idx = binding_forward(params, z, V)
    C_val = jnp.concatenate([
        cb['A'] @ cb['W'] for cb in params['val_cb']], axis=0)
    np.testing.assert_array_equal(np.asarray(o_bind),
                                  np.asarray(C_val[val_out_idx]))

    def scalar(zz):
        return jnp.sum(binding_forward(params, zz, V)[0])

    g = jax.grad(scalar)(z)
    assert jnp.all(jnp.isfinite(g))
    # `o_bind = v_out + (z - sg(z))`: the z path is exactly the identity.
    np.testing.assert_allclose(np.asarray(g), np.ones_like(np.asarray(z)),
                               atol=0, rtol=0)


def test_three_layer_binding_uses_nine_hrr_pairs():
    """n_layers=3: b_raw is the sum of all 9 cross-layer key/value pairs."""
    d, M, n = 8, 8, 3
    key = jax.random.PRNGKey(11)
    k_codes = list(jax.random.split(key, n))
    k_codes = [jax.random.normal(k, (d,)) for k in k_codes]
    v_codes = [jax.random.normal(k, (d,))
               for k in jax.random.split(jax.random.PRNGKey(12), n)]

    def far_rows(target, offset=1000.0):
        return jnp.stack([target] + [target + offset * (i + 1)
                                     for i in range(1, M)])

    expected = 0.0
    for k in k_codes:
        for v in v_codes:
            expected = expected + jnp.fft.irfft(
                normalize_fft(k) * normalize_fft(v), n=d)

    params = {
        'A_k': jnp.eye(d),
        'A_v': jnp.eye(d),
        'key_cb': [{'A': far_rows(k), 'W': jnp.eye(d)} for k in k_codes],
        'val_cb': [{'A': far_rows(v), 'W': jnp.eye(d)} for v in v_codes],
        # Layer 0 quantises the full 9-pair sum. The residual is then zero, so
        # layers 1/2 must select their zero row and add nothing.
        'bind_cb': [
            {'A': far_rows(expected), 'W': jnp.eye(d)},
            {'A': far_rows(jnp.zeros(d)), 'W': jnp.eye(d)},
            {'A': far_rows(jnp.zeros(d)), 'W': jnp.eye(d)},
        ],
    }
    z = jnp.zeros((1, d))

    o_bind, res, indices, b_q, k_q, val_out_idx = binding_forward(
        params, z, jnp.eye(d))

    assert len(res['key']) == len(res['val']) == len(indices['bind']) == n
    np.testing.assert_allclose(np.asarray(k_q[0]),
                               np.asarray(sum(k_codes)), atol=1e-5)
    np.testing.assert_allclose(np.asarray(b_q[0]),
                               np.asarray(expected), atol=1e-5)
    assert o_bind.shape == (1, d)
    assert val_out_idx.shape == (1,)
