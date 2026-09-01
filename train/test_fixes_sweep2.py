"""Second-round fix regression tests (sweep-2 audit).

Covers:
  1. resume keeps the trained z_proj (active channel no longer silently dies)
  2. config entropy_threshold aligned to 2.0 (convergence bonus actually fires)
  3. cog decoder.bin loads as 'cog' format with a linear readout
  4. train.py stage-3 EMA updates per codebook (no centroid collapse)

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_fixes_sweep2
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from train.config import LCMConfig
from train.cog_train import init_cog_params


def test_resume_keeps_trained_z_proj():
    """--resume without lang_ckpt must keep z_proj, disable only qwen."""
    cfg = LCMConfig()
    d, V = 64, 32
    with tempfile.TemporaryDirectory() as tmp:
        params_min = {
            'W_out': np.ones((d, V), dtype=np.float32),
            'z_proj': np.ones((896, d), dtype=np.float32),
            'encoder': {'embed': np.ones((V, d), dtype=np.float32)},
        }
        with open(os.path.join(tmp, "cog_params.pkl"), "wb") as f:
            import pickle
            pickle.dump({'params': params_min, 'step': 5,
                         'self_state': None}, f)
        params, _ = init_cog_params(cfg, jax.random.PRNGKey(0), resume=tmp)
    assert params['z_proj'] is not None, "trained z_proj must survive resume"
    assert params['qwen'] is None, "no lang_ckpt → no active channel"
    print("  [PASS] resume keeps trained z_proj, qwen disabled")


def test_entropy_threshold_aligned():
    cfg = LCMConfig()
    assert cfg.entropy_threshold == 2.0, \
        f"entropy_threshold={cfg.entropy_threshold}, expected 2.0 " \
        "(0.5 made the convergence bonus dead code: fusion entropy ≈ ln(6) ≈ 1.79)"
    print("  [PASS] config entropy_threshold == 2.0")


def test_cog_decoder_linear_readout():
    """cog decoder.bin (bare W_out) → 'cog' format, linear z @ W_out."""
    from lcm import load_decoder, gen_head_forward_old
    d, V = 16, 32
    rng = np.random.default_rng(0)
    w_out = rng.standard_normal((d, V)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        w_out.tofile(os.path.join(tmp, "decoder.bin"))
        dec = load_decoder(tmp, d, V)
    assert dec['format'] == 'cog', f"format={dec['format']}, expected 'cog'"
    z = rng.standard_normal(d).astype(np.float32)
    logits = z @ dec['w_out']
    # The old 'old'-format path applied an ELU the training loss never saw.
    old_logits = gen_head_forward_old(
        {'w_proj': np.eye(d, dtype=np.float32), 'w_out': w_out}, z, None)
    assert not np.allclose(logits, old_logits), \
        "cog readout must be linear, not ELU"
    assert np.allclose(logits, z @ w_out, atol=1e-6)
    print("  [PASS] cog decoder loads as 'cog' with linear readout")


def test_train_py_ema_no_collapse():
    """train.py stage-3 _jitted_ema must update per codebook (no collapse)."""
    from train.train import _jitted_ema
    M, d, B = 3, 8, 4  # 3 codes = 3 cluster centroids
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [-10, -10, -10, -10, -10, -10, -10, -10]],
                       dtype=np.float32)
    params = {'sparse': {'C': centers.copy()},
              'manifold': {'C': centers.copy()}}
    ema_state = {'sparse': {'N': np.ones(M, dtype=np.float32),
                            'm': centers.copy()},
                 'manifold': {'N': np.ones(M, dtype=np.float32),
                              'm': centers.copy()},
                 'binding': {}}
    cfg = {'gamma_sparse': 0.99, 'gamma_man': 0.99, 'lambda_sparse': 1e-4}

    def _min_code_dist(C):
        C = np.asarray(C)
        return min(np.linalg.norm(C[i] - C[j])
                   for i in range(len(C)) for j in range(i + 1, len(C)))

    def _sample():
        c = centers[rng.integers(0, M)]
        return c + rng.normal(0, 0.1, d).astype(np.float32)

    for _ in range(300):
        batch = np.stack([_sample() for _ in range(B)])
        params, ema_state = _jitted_ema(params, ema_state, jnp.array(batch), cfg)
    d_sparse = _min_code_dist(params['sparse']['C'])
    d_man = _min_code_dist(params['manifold']['C'])
    assert d_sparse > 0.1, f"sparse codes collapsed (min dist {d_sparse:.3f})"
    assert d_man > 0.1, f"manifold codes collapsed (min dist {d_man:.3f})"
    print(f"  [PASS] stage-3 EMA per-code update: sparse={d_sparse:.2f} man={d_man:.2f}")


if __name__ == '__main__':
    test_resume_keeps_trained_z_proj()
    test_entropy_threshold_aligned()
    test_cog_decoder_linear_readout()
    test_train_py_ema_no_collapse()
    print('All sweep-2 fix tests passed.')
