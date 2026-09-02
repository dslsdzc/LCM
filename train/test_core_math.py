"""Core-path math tests (TDD round): cognitive loop, data contract, encoders.

Covers the untested live paths:
  1. dag_fuse / soft_retrieve / cog_loop_scan math (STE forward, weights)
  2. WikiDataIter target-shift contract (targets[:, i] == x[:, i+1])
  3. Full encoder forward: JAX (train.encoder) == numpy (lcm.py)

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_core_math
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from train.cog_loop import soft_retrieve, dag_fuse, cog_loop_scan
from train.encoder import init_encoder_params, encoder_forward as jax_encoder_forward
from lcm import encoder_forward as numpy_encoder_forward


# ── 1. Cognitive loop math ──────────────────────────────────────────────────

def test_soft_retrieve_forward_is_hard():
    """STE forward must return the exact nearest codebook entry."""
    z = jnp.array([0.1, 0.9], dtype=jnp.float32)
    cb = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    out, d_min, idx = soft_retrieve(z, cb, tau=0.1)
    assert int(idx) == 1, f"nearest should be entry 1, got {idx}"
    assert float(d_min) == float(jnp.sum((z - cb[1]) ** 2))
    assert np.allclose(np.asarray(out), np.asarray(cb[1]), atol=1e-6), \
        "forward must be the hard nearest entry (STE)"


def test_dag_fuse_weights_normalize():
    """Fusion weights must be reciprocal-distance, normalized to sum 1."""
    z = jnp.array([0.0, 0.0], dtype=jnp.float32)
    cb1 = jnp.array([[1.0, 0.0], [2.0, 0.0]], dtype=jnp.float32)  # d=1
    cb2 = jnp.array([[3.0, 0.0], [4.0, 0.0]], dtype=jnp.float32)  # d=3
    z_next, diff, entropy = dag_fuse(z, [cb1, cb2], [0.5, 0.5], tau=0.1)
    # w1 = 1/1, w2 = 1/3 → z_next = (1*[1,0] + 1/3*[3,0]) / (1 + 1/3) = [1.5, 0]
    assert np.allclose(np.asarray(z_next), [1.5, 0.0], atol=1e-5), \
        f"weighted fusion wrong: {np.asarray(z_next)}"
    assert float(diff) > 0  # z moved
    assert float(entropy) > 0  # two active lattices → positive entropy


def test_cog_loop_scan_converges():
    """Fixed codebooks + fixed z → the scan must converge (diff → 0)."""
    z0 = jnp.array([0.2, 0.2], dtype=jnp.float32)
    cb = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=jnp.float32)
    z_qs, diffs, _ = cog_loop_scan(z0, [cb, cb], max_steps=30, tau=0.1)
    assert float(diffs[-1]) < 1e-3, \
        f"loop did not converge: last diff={float(diffs[-1]):.4f}"
    assert np.allclose(np.asarray(z_qs[-1]), np.asarray(z_qs[-2]), atol=1e-3)


# ── 2. Data contract ────────────────────────────────────────────────────────

def test_wikidataiter_shift_contract():
    """targets[:, i] must equal inputs[:, i+1] (next-token prediction)."""
    from train.data import WikiDataIter
    tokens = np.arange(100, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        dat = os.path.join(tmp, "tokens.dat")
        shp = os.path.join(tmp, "tokens_shape.json")
        tokens.tofile(dat)
        with open(shp, "w") as f:
            json.dump({'n_tokens': 100}, f)
        it = WikiDataIter(dat, shp, B=3, N=8)
        inputs, targets = next(it)
    assert inputs.shape == (3, 8) and targets.shape == (3, 8)
    # In-window shift: targets[:, i] == inputs[:, i+1] for i < N-1.
    assert np.array_equal(targets[:, :-1], inputs[:, 1:]), \
        "targets[:, i] must be inputs[:, i+1]"
    # Out-of-window: targets[:, -1] == x[N] (the next token AFTER the window
    # — what the passive channel predicts; it must NOT be in the inputs).
    assert not np.isin(targets[:, -1], inputs).any(), \
        "targets[:, -1] must be outside the input window"


def test_wikidataiter_window_bounds():
    """Sampled windows must stay inside the token array (no OOB reads)."""
    from train.data import WikiDataIter
    tokens = np.arange(50, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        dat = os.path.join(tmp, "tokens.dat")
        shp = os.path.join(tmp, "tokens_shape.json")
        tokens.tofile(dat)
        with open(shp, "w") as f:
            json.dump({'n_tokens': 50}, f)
        it = WikiDataIter(dat, shp, B=5, N=10)
        for _ in range(20):
            inputs, targets = next(it)
            assert inputs.max() < 50 and targets.max() < 50
            assert inputs.min() >= 0


# ── 3. Encoder full-forward JAX == numpy ────────────────────────────────────

def test_encoder_full_forward_jax_vs_numpy():
    """Full encoder forward must agree between JAX and numpy implementations."""
    d, d_ff, H, L, V, T = 16, 24, 4, 2, 32, 32
    pj = init_encoder_params(jax.random.PRNGKey(0), d, d_ff, H, L, V, T)
    pn = jax.tree.map(lambda a: np.asarray(a), pj)
    x = np.array([1, 5, 3, 9, 2, 7], dtype=np.int32)

    z_jax = np.asarray(jax_encoder_forward(pj, jnp.array(x[None, :]), H))  # (B, d)
    z_np = np.asarray(numpy_encoder_forward(pn, x, H))  # (d,) per lcm.py API
    assert z_jax.shape == (1, d) and z_np.shape == (d,), \
        f"unexpected shapes {z_jax.shape} vs {z_np.shape}"
    diff = float(np.abs(z_jax[0] - z_np).max())
    assert diff < 1e-4, f"jax vs numpy encoder mismatch: {diff}"
    print(f"  [PASS] encoder full forward jax==numpy (max diff {diff:.2e})")


if __name__ == '__main__':
    test_soft_retrieve_forward_is_hard()
    test_dag_fuse_weights_normalize()
    test_cog_loop_scan_converges()
    test_wikidataiter_shift_contract()
    test_wikidataiter_window_bounds()
    test_encoder_full_forward_jax_vs_numpy()
    print('All core math tests passed.')
