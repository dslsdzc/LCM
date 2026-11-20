"""R2: per-sample sparse LFQ in the unified Poincare domain.

Run from repo root:
    JAX_PLATFORMS=cpu be/bin/python -m pytest train/test_sparse_lfq.py -v
"""
import jax
import jax.numpy as jnp

from train.cognitive_step import six_lattice_step
from train.config import LCMConfig
from train.encoder import encoder_forward
from train.hyp import exp_map, poincare_similarity, safe_unit
from train.lattices import (
    hrq_forward, hrq_top_prototype, init_sparse_params, sparse_forward,
    sparse_zero_similarity,
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
    o, _ = sparse_forward(p, z, training=False, lambda_lfq=0.1,
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
    kept, _ = sparse_forward(p, z, training=False, lambda_lfq=1.0,
                             d_top=d_top, c_top=z)
    zeroed, _ = sparse_forward(p, z, training=False, lambda_lfq=1e-6,
                               d_top=d_top, c_top=z)
    assert not jnp.allclose(kept, 0.0)
    assert jnp.allclose(zeroed, 0.0)


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
        alpha_val=CFG.alpha_val)
    assert jnp.allclose(d_top, direct_top_sim, atol=1e-6)

    c_top = hrq_top_prototype(params['hrq'], aux['hrq_idx'])
    want, _ = sparse_forward(
        params['sparse'], z, training=False, lambda_lfq=CFG.lambda_lfq,
        d_top=d_top, c_top=c_top,
        value_scalars=params['value_scalars'].get('sparse'),
        alpha_val=CFG.alpha_val)
    assert jnp.allclose(aux['lattice_outputs'][1], want, atol=1e-6), \
        "step's sparse output is not the forward with HRQ's d_top/c_top"
