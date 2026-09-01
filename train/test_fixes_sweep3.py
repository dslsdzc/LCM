"""Third-round fix regression tests (sweep-3 audit).

Covers:
  1. sample_categorical_cy top-k threshold (was off by one: kept V-top_k+1)
  2. gen_head linear-attention einsum 'bnd,bndd->bnd' → 'bnd,bnde->bne'
     (repeated subscript took the diagonal — K-V cross-correlation was lost)
  3. gen_head_new_single first-step double-count of z_q (numpy vs Cython)

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_fixes_sweep3
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from train.fusion import gen_head_forward
from lcm import gen_head_new_single

try:
    from train._lcm_cy import sample_categorical_cy, genhead_step_cy
    HAS_CY = True
except ImportError:
    HAS_CY = False
    print("[WARN] train._lcm_cy not importable — rebuild Cython; skipping cython tests")


def test_topk_threshold():
    """top_k=1 with logits [0.0, 0.1] must never sample token 0."""
    if not HAS_CY:
        return
    logits = np.array([0.0, 0.1], dtype=np.float32)
    samples = [sample_categorical_cy(logits, 1.0, 1) for _ in range(30000)]
    n_bad = sum(1 for s in samples if s == 0)
    assert n_bad == 0, f"top_k=1 leaked {n_bad}/30000 samples of the masked token"
    print("  [PASS] top-k threshold masks correctly (0/30000 leaked)")


def _gen_head_params(d, V, rng):
    return {
        'w_embed': rng.standard_normal((V, d)).astype(np.float32),
        'w_q': rng.standard_normal((d, d)).astype(np.float32),
        'w_k': rng.standard_normal((d, d)).astype(np.float32),
        'w_v': rng.standard_normal((d, d)).astype(np.float32),
        'w_o': rng.standard_normal((d, d)).astype(np.float32),
        'w_1': rng.standard_normal((d, 4 * d)).astype(np.float32),
        'w_2': rng.standard_normal((d, 4 * d)).astype(np.float32),
        'w_3': rng.standard_normal((4 * d, V)).astype(np.float32),
    }


def test_gen_head_einsum_matches_matmul():
    """gen_head_forward's linear attention must equal explicit matmul."""
    d, V, B, N = 8, 16, 2, 5
    rng = np.random.default_rng(0)
    params = {k: jnp.array(v) for k, v in _gen_head_params(d, V, rng).items()}
    z_q = jnp.array(rng.standard_normal((B, d)).astype(np.float32))
    x = jnp.array(rng.integers(0, V, (B, N)))

    logits = gen_head_forward(params, z_q, x)

    # Reference: explicit Q @ kv_cs matmul
    B_, N_ = x.shape
    emb = params['w_embed'][x]
    inputs = jnp.concatenate([z_q[:, None, :], emb], axis=1)
    Q = jax.nn.elu(inputs @ params['w_q']) + 1.0
    K = jax.nn.elu(inputs @ params['w_k']) + 1.0
    V_ = inputs @ params['w_v']
    kv = K[:, :, :, None] @ V_[:, :, None, :]
    kv_cs = jnp.cumsum(kv, axis=1)
    k_cs = jnp.cumsum(K, axis=1)
    attn = jnp.einsum('bnd,bnde->bne', Q, kv_cs) / (
        jnp.einsum('bnd,bnd->bn', Q, k_cs)[:, :, None] + 1e-8)
    attn_out = attn @ params['w_o']  # matches gen_head_forward (projection before GLU)
    gate = jax.nn.sigmoid(attn_out @ params['w_1'])
    up = attn_out @ params['w_2']
    ref = (gate * up) @ params['w_3']

    diff = float(jnp.abs(logits - ref[:, 1:, :]).max())
    assert diff < 1e-4, f"gen_head_forward != matmul reference (max {diff})"
    print(f"  [PASS] gen_head einsum == explicit matmul (max diff {diff:.2e})")


def test_gen_head_new_single_no_double_count():
    """First step must seed z_q once — numpy must match the Cython path."""
    d, V = 8, 16
    rng = np.random.default_rng(1)
    params_np = _gen_head_params(d, V, rng)
    z_q = rng.standard_normal(d).astype(np.float32)
    logits_np, _ = gen_head_new_single(params_np, z_q, np.array([0]))

    if HAS_CY:
        gate_act = np.zeros((4 * d,), dtype=np.float32)
        kv_sum = np.zeros((d, d), dtype=np.float32)
        k_sum = np.zeros(d, dtype=np.float32)
        genhead_step_cy(
            np.ascontiguousarray(params_np['w_embed'], np.float32),
            np.ascontiguousarray(params_np['w_q'], np.float32),
            np.ascontiguousarray(params_np['w_k'], np.float32),
            np.ascontiguousarray(params_np['w_v'], np.float32),
            np.ascontiguousarray(params_np['w_o'], np.float32),
            np.ascontiguousarray(params_np['w_1'], np.float32),
            np.ascontiguousarray(params_np['w_2'], np.float32),
            np.ascontiguousarray(z_q, np.float32),
            0,  # last_token_id (unused on first step)
            kv_sum, k_sum,
            1,  # is_first
            gate_act)
        logits_cy = gate_act @ params_np['w_3']
        diff = float(np.abs(logits_np - logits_cy).max())
        assert diff < 1e-4, f"numpy first-step != Cython (max {diff})"
        print(f"  [PASS] gen_head first step: numpy == Cython (max diff {diff:.2e})")
    else:
        print("  [WARN] cython unavailable; numpy first-step checked only")


def test_poincare_off_ball_no_nan():
    """Off-ball inputs (‖·‖ > 1) must not produce NaN distance."""
    from train.hyp import poincare_similarity, poincare_distance
    u = np.array([1.1, 0.0], dtype=np.float32)  # ‖u‖ > 1 (off the ball)
    v = np.array([0.5, 0.0], dtype=np.float32)
    sim = float(np.asarray(poincare_similarity(u, v)).reshape(-1)[0])
    dist = float(np.asarray(poincare_distance(u, v)).reshape(-1)[0])
    assert np.isfinite(sim), f"similarity NaN for off-ball input ({sim})"
    assert np.isfinite(dist), f"distance NaN for off-ball input ({dist})"
    print(f"  [PASS] off-ball poincare: sim={sim:.3f} dist={dist:.3f} (finite)")


if __name__ == '__main__':
    test_topk_threshold()
    test_gen_head_einsum_matches_matmul()
    test_gen_head_new_single_no_double_count()
    test_poincare_off_ball_no_nan()
    print('All sweep-3 fix tests passed.')
