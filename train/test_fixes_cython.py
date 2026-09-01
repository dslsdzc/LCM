"""Cython path regression tests (final verification pass).

Covers two fixes made after the four fix groups:
  1. init_rng_cy() one-shot guard — repeated calls with seed 0 previously put
     xorshift32 into its absorbing state, making _rand_f32() constant and
     sample_categorical_cy() degenerate.
  2. Incremental encoder layout consistency — numpy (_encoder_recurrent_step)
     and Cython (encoder_recurrent_step_cy) must both consume the (d, d)
     block-diagonal state produced by _encoder_full_with_state.

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_fixes_cython
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from train.encoder import init_encoder_params
from lcm import _encoder_full_with_state, _encoder_recurrent_step

try:
    from train._lcm_cy import init_rng_cy, sample_categorical_cy, encoder_recurrent_step_cy
    HAS_CY = True
except ImportError:
    HAS_CY = False
    print("[WARN] train._lcm_cy not importable — rebuild Cython; skipping cython tests")


def _encoder_fixture():
    d, d_ff, H, L, V, T = 8, 16, 2, 2, 32, 16
    pj = init_encoder_params(jax.random.PRNGKey(0), d, d_ff, H, L, V, T)
    pn = jax.tree.map(lambda a: np.asarray(a), pj)
    return pn, H


def test_encoder_incremental_numpy_vs_cython():
    """3-token incremental encode: numpy and Cython must agree step by step."""
    if not HAS_CY:
        return
    pn, H = _encoder_fixture()
    x = np.array([1, 2, 3], dtype=np.int32)
    new_toks = [4, 5, 6]

    # Two independent copies of the same initial state (block layout)
    _, sn = _encoder_full_with_state(pn, x, H)
    sn_np = {'layers': [{'kv': ls['kv'].copy(), 'k': ls['k'].copy()}
                        for ls in sn['layers']],
             'pool_kv': sn['pool_kv'].copy(), 'pool_k': sn['pool_k'].copy(),
             'embed': sn['embed'].copy(), 'q_pool': sn['q_pool'].copy(),
             'w_proj': sn['w_proj'].copy()}
    sn_cy = {'layers': [{'kv': ls['kv'].copy(), 'k': ls['k'].copy()}
                        for ls in sn['layers']],
             'pool_kv': sn['pool_kv'].copy(), 'pool_k': sn['pool_k'].copy(),
             'embed': sn['embed'].copy(), 'q_pool': sn['q_pool'].copy(),
             'w_proj': sn['w_proj'].copy()}

    from lcm import encoder_recurrent_step_cy as _erc_wrapper
    for t in new_toks:
        z_np = _encoder_recurrent_step(sn_np, t, pn['layers'], H)
        z_cy = _erc_wrapper(pn, sn_cy, t, H)
        assert np.allclose(z_np, z_cy, atol=1e-5, rtol=1e-5), \
            f"token {t}: cython vs numpy mismatch, max {np.abs(z_np - z_cy).max()}"


def test_init_rng_cy_guard():
    """Repeated init_rng_cy(0) must not degenerate the sampler."""
    if not HAS_CY:
        return
    init_rng_cy(0)
    init_rng_cy(0)  # must be a no-op now
    init_rng_cy(0)
    logits = np.zeros(50, dtype=np.float32)  # uniform distribution
    samples = [sample_categorical_cy(logits, 1.0, 0) for _ in range(200)]
    n_uniq = len(set(samples))
    assert n_uniq > 20, f"RNG degenerate: only {n_uniq} unique classes in 200 draws"


if __name__ == '__main__':
    test_init_rng_cy_guard()
    test_encoder_incremental_numpy_vs_cython()
    print('All Cython regression tests passed.')
