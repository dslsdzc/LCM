"""R2: per-sample sparse LFQ in the unified Poincare domain.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_sparse_lfq.py -v
"""
import jax
import jax.numpy as jnp
import pytest

from train.cognitive_step import six_lattice_step
from train.config import LCMConfig
from train.encoder import encoder_forward
from train.hyp import exp_map, poincare_similarity, safe_unit
from train.lattices import (
    clamp_norm, hrq_forward, hrq_top_prototype, init_simvq, init_sparse_params,
    sparse_forward, sparse_zero_similarity,
)
from train.model import init_all_params

CFG = LCMConfig(
    d_model=8, d_ff=12, n_heads=2, d_head=4, vocab_size=16,
    max_seq_len=6, n_encoder_layers=1,
    M_top=6, M_fine=3, n_hrq_layers=1,
    M_sparse=6, M_lr=6, M_man=6, M_bind=6, M_contrast=6,
    n_bind_layers=1, n_contrast_layers=1, n_self_codes=3,
    use_bf16=False,
)


def _sparse_params(M=3, d=4):
    return init_sparse_params(jax.random.PRNGKey(0), d, M)


def test_no_lfq_without_d_top():
    """No HRQ dependency edge → LFQ disabled (nearest non-zero code kept)."""
    p = _sparse_params()
    z = jnp.ones((2, 4))
    o, idx = sparse_forward(p, z, training=False, d_top=None)
    assert not jnp.allclose(o, 0.0)


def test_no_lfq_without_c_top():
    """``c_top=None`` means HRQ did not activate; LFQ must not invent a value."""
    p = _sparse_params()
    z = jnp.ones((2, 4))
    o, _ = sparse_forward(p, z, training=False,
                          d_top=jnp.array([1e-6, 1e-6]), c_top=None)
    assert not jnp.allclose(o, 0.0)


def test_lfq_forces_zero_when_far():
    """Input far from every sparse code relative to d_top → zero_vec."""
    p = _sparse_params()
    z = jnp.ones((2, 4))
    d_top = jnp.array([1e-6, 1e-6])
    o, idx = sparse_forward(p, z, training=False, d_top=d_top, c_top=z)
    assert jnp.allclose(o, 0.0)
    assert jnp.array_equal(idx, jnp.zeros_like(idx))


def test_lfq_is_per_sample():
    """The threshold is (B,), not a batch mean: one sample can zero, one not."""
    z = jnp.ones((2, 4))
    p = {'C': jnp.ones((1, 4)) * 0.99, 'zero_vec': jnp.zeros((1, 4))}
    # d_min(0) ~ 2e-4: sample 0's threshold is 1e-9, sample 1's is huge.
    d_top = jnp.array([1e-9, 1e9])
    o, _ = sparse_forward(p, z, training=False, lambda_lfq=1.5,
                          d_top=d_top, c_top=z)
    assert jnp.allclose(o[0], 0.0)
    assert not jnp.allclose(o[1], 0.0)


def test_lfq_disabled_when_training():
    """Training disables LFQ even when a d_top is supplied."""
    p = _sparse_params()
    z = jnp.ones((2, 4))
    o, _ = sparse_forward(p, z, training=True,
                          d_top=jnp.array([1e-6, 1e-6]), c_top=z)
    assert not jnp.allclose(o, 0.0)


def test_lambda_lfq_scales_the_threshold():
    """``lambda_lfq`` is the dimensionless multiplier in ``d_min > λ·d_top``."""
    z = jnp.ones((2, 4))
    p = {'C': jnp.ones((1, 4)) * 0.99, 'zero_vec': jnp.zeros((1, 4))}
    d_top = jnp.ones((2,))
    # d_min ~ 2.5e-4 for this crafted pair. A multiplier above that keeps the
    # code; one below it zeroes the output.
    kept, _ = sparse_forward(p, z, training=False, lambda_lfq=1.5,
                             d_top=d_top, c_top=z)
    zeroed, _ = sparse_forward(p, z, training=False, lambda_lfq=1e-6,
                               d_top=d_top, c_top=z)
    assert not jnp.allclose(kept, 0.0)
    assert jnp.allclose(zeroed, 0.0)


def test_lfq_uses_real_hrq_top_similarity():
    """The threshold must work at HRQ's real similarity scale.

    Fabricated ``d_top=1.0`` hides the scale mismatch: with the old 0.1
    multiplier and an init-scale ``d_top`` of ~2.7, every sample was zeroed at
    random initialisation. Here ``d_top`` is produced by ``hrq_forward``.
    """
    d = 4
    u = jnp.ones((1, d)) / jnp.sqrt(jnp.float32(d))
    # Prototype on the query's ray at tangent radius 0.5 -> d_top ~ 0.54.
    hrq = {
        'top': {'A': jnp.concatenate([0.5 * u, -0.5 * u], axis=0),
                'W': jnp.eye(d)},
        'fine': [init_simvq(jax.random.PRNGKey(7), 1, d)],
    }
    z = jnp.ones((1, d))
    _, _, d_top, _ = hrq_forward(hrq, z, tau_fallback=0.1,
                                 value_topk=CFG.hrq_value_topk)
    assert d_top.shape == (1,)
    assert 0.1 < float(d_top[0]) < 1.0

    zero_vec = jnp.zeros((1, d))
    near = {'C': 0.99 * u, 'zero_vec': zero_vec}   # same ray, almost at z_P
    far = {'C': 0.20 * u, 'zero_vec': zero_vec}    # same ray, near the origin

    def ratio(p):
        z_P = exp_map(safe_unit(z))
        C_P = exp_map(clamp_norm(p['C'], 0.99))
        d_min = poincare_similarity(
            z_P[:, None, :], C_P[None, :, :]).squeeze(-1).min(axis=-1)
        return d_min, d_min / d_top

    d_near, r_near = ratio(near)
    d_far, r_far = ratio(far)
    assert float(r_near[0]) < CFG.lambda_lfq
    assert float(r_far[0]) > CFG.lambda_lfq
    # The far code is still closer than zero_vec, so a zero output is LFQ's
    # decision — ordinary retrieval would have returned the code.
    zero_sim = sparse_zero_similarity(z)
    assert float(d_far[0]) < float(zero_sim[0])

    o_near, idx_near = sparse_forward(
        near, z, training=False, lambda_lfq=CFG.lambda_lfq,
        d_top=d_top, c_top=z)
    o_far, idx_far = sparse_forward(
        far, z, training=False, lambda_lfq=CFG.lambda_lfq,
        d_top=d_top, c_top=z)
    assert not jnp.allclose(o_near, 0.0)
    assert int(idx_near[0]) != 0
    assert jnp.allclose(o_far, 0.0)
    assert int(idx_far[0]) == 0

    # Control: with LFQ effectively disabled, the same far code is returned.
    o_ctrl, idx_ctrl = sparse_forward(
        far, z, training=False, lambda_lfq=1e9, d_top=d_top, c_top=z)
    assert not jnp.allclose(o_ctrl, 0.0)
    assert int(idx_ctrl[0]) == 1

    # Reviewer's init scale: an orthogonal-ish HRQ codebook gives d_top > 2.
    # The near code must still survive the default 1.5 multiplier there.
    wide = {'top': {'A': jnp.eye(d) * 0.9, 'W': jnp.eye(d)},
            'fine': [init_simvq(jax.random.PRNGKey(8), 1, d)]}
    _, _, d_top_wide, _ = hrq_forward(wide, z, value_topk=CFG.hrq_value_topk)
    assert float(d_top_wide[0]) > 2.0
    o_wide, idx_wide = sparse_forward(
        near, z, training=False, lambda_lfq=CFG.lambda_lfq,
        d_top=d_top_wide, c_top=z)
    assert not jnp.allclose(o_wide, 0.0)
    assert int(idx_wide[0]) != 0


def test_all_zero_sparse_codebook_is_finite():
    """All-zero EMA rows must not divide by zero or produce NaN."""
    z = jnp.ones((2, 4))
    p = {'C': jnp.zeros((3, 4)), 'zero_vec': jnp.zeros((1, 4))}
    d_top = jnp.array([0.5, 0.5])
    for training in (True, False):
        o, idx = sparse_forward(p, z, training=training,
                                d_top=d_top, c_top=z)
        assert jnp.all(jnp.isfinite(o))
        assert jnp.allclose(o, 0.0)
        assert jnp.array_equal(idx, jnp.zeros_like(idx))


def test_d_top_must_be_per_sample():
    """A scalar d_top must fail loudly instead of broadcasting to the batch."""
    p = _sparse_params()
    z = jnp.ones((2, 4))
    with pytest.raises(ValueError, match="per-sample"):
        sparse_forward(p, z, training=False, d_top=1e-6, c_top=z)
    with pytest.raises(ValueError, match="per-sample"):
        sparse_forward(p, z, training=False,
                       d_top=jnp.array([1e-6]), c_top=z)


def test_zero_vec_similarity_is_the_poincare_domain():
    """``sim(z_P, 0) = 2‖z_P‖² / (1 - ‖z_P‖²)`` — same domain as d_sparse_min."""
    z = jnp.ones((2, 4)) * 0.3
    z_P = exp_map(safe_unit(z))
    n2 = jnp.sum(z_P ** 2, axis=-1)
    want = 2 * n2 / (1 - n2)
    got = sparse_zero_similarity(z)
    assert jnp.allclose(got, want, atol=1e-6)

    # And it is the ordinary Poincare similarity against an explicit zero.
    direct = poincare_similarity(
        z_P, jnp.zeros_like(z_P)).squeeze(-1)
    assert jnp.allclose(got, direct, atol=1e-6)


def test_step_passes_per_sample_d_top_to_sparse():
    """six_lattice_step exposes HRQ's per-sample similarity as sparse's d_top.

    Shape ``(B,)`` is the point: the old implementation averaged over the batch
    and inverted the sign (``1 - mean``), which made the LFQ decision both
    sample-independent and backwards.
    """
    params, _, _ = init_all_params(CFG, jax.random.PRNGKey(0))
    x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    z = safe_unit(encoder_forward(params['encoder'], x, CFG.n_heads))

    _, aux = six_lattice_step(z, params, CFG, training=False,
                              rng=jax.random.PRNGKey(1))
    d_top = aux['hrq_top_sim']
    assert d_top.shape == (2,)
    assert not jnp.allclose(d_top[0], d_top[1]), \
        "d_top must differ across samples, not be a batch mean"

    _, _, direct_top_sim, _ = hrq_forward(
        params['hrq'], z, CFG.tau_route_fallback,
        value_scalars=params['value_scalars'].get('hrq'),
        alpha_val=CFG.alpha_val, value_topk=CFG.hrq_value_topk)
    assert jnp.allclose(d_top, direct_top_sim, atol=1e-6)

    c_top = hrq_top_prototype(params['hrq'], aux['hrq_idx'])
    want, _ = sparse_forward(
        params['sparse'], z, training=False, lambda_lfq=CFG.lambda_lfq,
        d_top=d_top, c_top=c_top,
        value_scalars=params['value_scalars'].get('sparse'),
        alpha_val=CFG.alpha_val)
    assert jnp.allclose(aux['lattice_outputs'][1], want, atol=1e-6), \
        "step's sparse output is not the forward with HRQ's d_top/c_top"
    # Ruling 15 / design §6.3 B: random-init inference must not be all zero_vec.
    assert not jnp.allclose(aux['lattice_outputs'][1], 0.0), \
        "sparse inference is still all-zero at random initialisation"
