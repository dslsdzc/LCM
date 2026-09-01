"""Model-layer correctness regression tests (issue group C).

Covers 8 fixes:
  1. HRQ top-level retrieval direction (argmax → argmin) + fallback gap
  2. value-contrast loss sign (softplus((d_pos - d_neg)/τ))
  3. encoder GLU unified to SiLU across pyx / lcm.py numpy / jax
  4. Qwen RoPE switched from interleaved to rotate-half convention
  5. hyp.log_map numerator/denominator consistency (clipped norm)
  6. _lcm_cy argpartition kth out-of-bounds (match_euclidean_cy /
     match_hamming_cy)
  7. binding lattice per-family key separation
  8. (read-only audit of train/encoder.py — nothing to change)

Run from repo root:  python -m train.test_fixes_model
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

from train.config import LCMConfig
from train import lattices, losses
from train.lattices import (
    hrq_forward, init_simvq, init_binding_params,
    poincare_similarity, exp_map, log_map, mobius_add,
)
from train.hyp import log_map as hyp_log_map
from train.qwen_lm import apply_rope, precompute_freqs
from train.encoder import init_encoder_params, init_encoder_state, encoder_recurrent_step

from lcm import _silu, _glu, _encoder_full_with_state, _encoder_recurrent_step

try:
    from train._lcm_cy import match_euclidean_cy, match_hamming_cy, encoder_recurrent_step_cy
    HAS_CY = True
except ImportError:
    HAS_CY = False


# ── #1 HRQ top-level retrieval direction ─────────────────────────────────────

def _ref_route(z_P, sims, C_top_P, fine_params, weighted):
    """Replica of the hard/fallback branch math (fine layers included)."""
    if weighted:
        weights = jnn.softmax(-sims, axis=-1)
        c = jnp.einsum('bm,md->bd', weights, C_top_P)
    else:
        c = C_top_P[sims.argmin(axis=-1)]
    r = mobius_add(z_P, -c)
    for fb in fine_params:
        C_fb = fb['A'] @ fb['W']
        C_fb_P = exp_map(C_fb)
        vr = poincare_similarity(r[:, None, :], C_fb_P[None, :, :]).squeeze(-1)
        fi = vr.argmin(axis=-1)
        r = mobius_add(r, -C_fb_P[fi])
    return log_map(mobius_add(c, r))


def _hrq_setup():
    d = 2
    C_top = jnp.array([[0.3, 0.0], [0.0, 0.3], [-0.3, 0.0]])
    params = {'top': {'A': C_top, 'W': jnp.eye(d)},
              'fine': [init_simvq(jax.random.PRNGKey(7), 1, d)]}
    return d, C_top, params


def test_hrq_top_retrieval_picks_nearest():
    """Nearest prototype must be selected (argmin), gap must use 2 closest."""
    d, C_top, params = _hrq_setup()
    C_P = exp_map(C_top)

    # z clearly nearest to prototype 0 → hard route, idx 0, top_sim = min sim
    z = jnp.array([[0.2, 0.0]])
    z_P = exp_map(z)
    sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
    o, idx, top_sim = hrq_forward(params, z, tau_fallback=0.1)
    assert int(idx[0]) == 0, f"expected nearest proto 0, got {int(idx[0])}"
    assert jnp.allclose(top_sim, sims[0].min(), atol=1e-6), \
        "top_sim must be the similarity of the nearest (argmin) prototype"
    ref_hard = _ref_route(z_P, sims, C_P, params['fine'], weighted=False)
    assert jnp.allclose(o, ref_hard, atol=1e-5), "hard route output mismatch"
    ref_w = _ref_route(z_P, sims, C_P, params['fine'], weighted=True)
    assert not jnp.allclose(o, ref_w, atol=1e-3), \
        "hard route should NOT equal the fallback (gap is large)"


def test_hrq_fallback_uses_closest_pair():
    """Equidistant top-2 → gap ~0 → fallback branch (weighted blend)."""
    _, C_top, params = _hrq_setup()
    C_P = exp_map(C_top)

    z = jnp.array([[0.2, 0.2]])  # equidistant from proto 0 and proto 1
    z_P = exp_map(z)
    sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
    o, idx, _ = hrq_forward(params, z, tau_fallback=0.1)
    # argmin of (nearly) equal sims → one of the two close protos, never the far one
    assert int(idx[0]) in (0, 1), f"expected near proto, got {int(idx[0])}"
    ref_w = _ref_route(z_P, sims, C_P, params['fine'], weighted=True)
    assert jnp.allclose(o, ref_w, atol=1e-5), "fallback route output mismatch"
    ref_hard = _ref_route(z_P, sims, C_P, params['fine'], weighted=False)
    assert not jnp.allclose(o, ref_hard, atol=1e-3), \
        "fallback branch should NOT equal the hard route here"


def test_hrq_value_scalars_branch_unchanged():
    """value_biased_score is higher=better; argmax picks the value-biased proto."""
    _, C_top, params = _hrq_setup()
    z = jnp.array([[0.2, 0.0]])
    # v=[0,1,0], alpha large → proto 1 (farther, but valuable) wins
    vs = jnp.array([0.0, 1.0, 0.0])
    o, idx, _ = hrq_forward(params, z, tau_fallback=0.1,
                            value_scalars=vs, alpha_val=2.0)
    assert int(idx[0]) == 1, f"value-biased argmax should pick proto 1, got {int(idx[0])}"
    assert jnp.all(jnp.isfinite(o))


# ── #2 value-contrast loss sign ──────────────────────────────────────────────

class _FakeGValue:
    def __init__(self, C_pos, C_neg):
        self.C_pos = C_pos
        self.C_neg = C_neg


def test_value_contrast_loss_direction():
    """Near-positive state → small loss; near-harm state → large loss.

    With the reversed (buggy) sign, the near-harm state gets ~0 loss.
    """
    cfg = LCMConfig()
    p = jnp.array([0.9, 0.0])
    n = jnp.array([-0.9, 0.0])
    gvalue = _FakeGValue(
        C_pos=jnp.broadcast_to(p[None, :], (4, 2)),
        C_neg=jnp.broadcast_to(n[None, :], (4, 2)))

    aux_safe = {'lattice_outputs': [p[None, :]], 'value_signals': jnp.array(1.0)}
    aux_harm = {'lattice_outputs': [n[None, :]], 'value_signals': jnp.array(1.0)}

    loss_safe = float(losses.compute_value_contrast_loss({}, gvalue, aux_safe, cfg))
    loss_harm = float(losses.compute_value_contrast_loss({}, gvalue, aux_harm, cfg))
    assert loss_safe < 0.1, f"near-positive state should have small loss, got {loss_safe}"
    assert loss_harm > 0.5, f"near-harm state should have large loss, got {loss_harm}"
    assert loss_harm > 5 * loss_safe, \
        f"loss ratio wrong: safe={loss_safe}, harm={loss_harm}"


def test_value_logit_softplus_algebra():
    """Fixed algebra: softplus((d_pos - d_neg)/τ) ≈ 0 iff d_neg > d_pos."""
    tau = 0.1
    l_safe = jnn.softplus((jnp.array([0.1]) - jnp.array([1.0])) / tau)[0]
    assert float(l_safe) < 1e-3, f"safe side should vanish, got {float(l_safe)}"
    l_unsafe = jnn.softplus((jnp.array([1.0]) - jnp.array([0.1])) / tau)[0]
    assert float(l_unsafe) > 5, f"unsafe side should be large, got {float(l_unsafe)}"


# ── #3 GLU unified to SiLU ───────────────────────────────────────────────────

def test_silu_numpy_vs_jax():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64,)).astype(np.float32)
    assert np.allclose(_silu(x), np.asarray(jnn.silu(jnp.array(x))), atol=1e-6), \
        "lcm.py _silu must match jax.nn.silu"


def _encoder_step_fixture():
    d, d_ff, H, L, V, T = 8, 16, 2, 1, 32, 16
    pj = init_encoder_params(jax.random.PRNGKey(0), d, d_ff, H, L, V, T)
    pn = jax.tree.map(lambda a: np.asarray(a), pj)
    x = np.array([1, 2, 3], dtype=np.int32)
    new_tok = 4
    # jax
    _, sj = init_encoder_state(pj, jnp.array(x[None, :]), H)
    zj, _ = encoder_recurrent_step(sj, pj['embed'][new_tok][None, :], pj['layers'], H)
    # numpy (lcm.py) — snapshot the pre-step state for the pyx first,
    # since _encoder_recurrent_step mutates it in place.
    _, sn = _encoder_full_with_state(pn, x, H)
    sn0 = {'layers': [{'kv': ls['kv'].copy(), 'k': ls['k'].copy()} for ls in sn['layers']],
           'pool_kv': sn['pool_kv'].copy(), 'pool_k': sn['pool_k'].copy()}
    zn = _encoder_recurrent_step(sn, new_tok, pn['layers'], H)
    return pn, sn0, x, new_tok, H, np.asarray(zj), np.asarray(zn)


def test_encoder_step_jax_vs_numpy_glu():
    """numpy (lcm.py) and jax encoder steps must agree (both SiLU now)."""
    _, _, _, _, _, zj, zn = _encoder_step_fixture()
    assert np.allclose(zj, zn, atol=1e-5, rtol=1e-5), \
        f"jax vs numpy encoder mismatch: max {np.abs(zj - zn).max()}"


def test_encoder_step_cython_glu():
    """Cython encoder step must agree with numpy and jax (SiLU gate).

    With the old ELU+1 gate the pyx output differs by O(1) — this fails.
    """
    if not HAS_CY:
        print("[WARN] train._lcm_cy not importable — rebuild Cython; skipping")
        return
    pn, sn0, x, new_tok, H, zj, zn = _encoder_step_fixture()
    d = pn['q_pool'].shape[0]
    d_h = d // H

    # State from _encoder_full_with_state is already in (d, d) block-diagonal
    # layout (head hh at rows/cols [hh*d_h, (hh+1)*d_h), see lcm.py
    # _kv_cumsum_block) — pass it straight through to the pyx.
    layers_kv = [np.ascontiguousarray(ls['kv'], np.float32) for ls in sn0['layers']]
    layers_k = [np.ascontiguousarray(ls['k'], np.float32) for ls in sn0['layers']]

    z_out = np.zeros(d, dtype=np.float32)
    encoder_recurrent_step_cy(
        np.ascontiguousarray(pn['embed'], np.float32),
        np.ascontiguousarray(pn['q_pool'], np.float32),
        np.ascontiguousarray(pn['w_proj'], np.float32),
        layers_kv, layers_k,
        np.ascontiguousarray(sn0['pool_kv'], np.float32),
        np.ascontiguousarray(sn0['pool_k'], np.float32),
        [np.ascontiguousarray(l['ln1_scale'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['ln1_bias'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['ln2_scale'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['ln2_bias'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_q'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_k'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_v'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_o'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_1'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_2'], np.float32) for l in pn['layers']],
        [np.ascontiguousarray(l['w_3'], np.float32) for l in pn['layers']],
        int(new_tok), int(H), z_out)
    assert np.allclose(z_out, zn, atol=1e-5, rtol=1e-5), \
        f"cython vs numpy encoder mismatch: max {np.abs(z_out - zn).max()}"
    assert np.allclose(z_out, zj, atol=1e-5, rtol=1e-5), \
        f"cython vs jax encoder mismatch: max {np.abs(z_out - zj).max()}"


# ── #4 Qwen RoPE rotate-half ─────────────────────────────────────────────────

def test_rope_rotate_half():
    B, N, H, d_head = 1, 8, 2, 64
    x = jax.random.normal(jax.random.PRNGKey(3), (B, N, H, d_head))
    cos, sin = precompute_freqs(d_head, 16, theta=1000000.0)

    out = apply_rope(x, cos, sin)

    # Reference: HF rotate_half (front/back halves, per-position freq slice)
    nf = d_head // 2
    c = cos[:N, :nf][None, :, None, :]
    s = sin[:N, :nf][None, :, None, :]
    x1, x2 = x[..., :nf], x[..., nf:]
    ref = jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    assert jnp.allclose(out, ref, atol=1e-6), "apply_rope != rotate_half reference"

    # Old interleaved (even/odd) pairing must NOT match → behavior changed
    xe, xo = x[..., ::2], x[..., 1::2]
    nf2 = xe.shape[-1]
    c2 = cos[:N, :nf2][None, :, None, :]
    s2 = sin[:N, :nf2][None, :, None, :]
    old = jnp.stack([xe * c2 - xo * s2, xe * s2 + xo * c2], axis=-1).reshape(*x.shape)
    assert not jnp.allclose(out, old, atol=1e-4), \
        "rotate_half result must differ from the old interleaved pairing"


def test_rope_odd_head_dim_rejected():
    x = jnp.zeros((1, 2, 1, 7))
    cos, sin = precompute_freqs(8, 4)
    try:
        apply_rope(x, cos, sin)
        assert False, "odd head_dim should raise"
    except ValueError:
        pass


# ── #5 hyp.log_map clipped-norm consistency ──────────────────────────────────

def test_log_map_clip_consistency():
    y = 0.9995 * jnp.array([1.0, 0.0])  # ||y|| > 0.999 → norm clamped
    out = hyp_log_map(y)
    assert jnp.all(jnp.isfinite(out)), "log_map must stay finite at the boundary"
    # Manual formula with the CLIPPED norm in both numerator and denominator
    expected = (np.arctanh(0.999) / 0.999) * np.asarray(y)
    assert jnp.allclose(out, expected, atol=1e-4), \
        f"log_map mismatch: {np.asarray(out)} vs {expected}"
    # The old (unclipped denominator) formula differs measurably here
    old = (np.arctanh(0.9995 + 1e-8) / (0.9995 + 1e-8)) * np.asarray(y)
    assert abs(float(out[0]) - float(old[0])) > 0.1, \
        "test not discriminating: old and fixed log_map agree"


# ── #6 _lcm_cy argpartition bounds ───────────────────────────────────────────

def _ref_knn(z_cur, z_next, query, K):
    dists = np.sum((z_cur - query) ** 2, axis=-1)
    top = np.argsort(dists)[:min(K, len(dists))]
    w = 1.0 / (dists[top] + 1e-8)
    return np.sum(w[:, None] * z_next[top], axis=0) / w.sum()


def test_match_euclidean_argpartition_bounds():
    if not HAS_CY:
        print("[WARN] train._lcm_cy not importable — rebuild Cython; skipping")
        return
    N, D = 5, 4
    rng = np.random.default_rng(0)
    z_cur = rng.standard_normal((N, D)).astype(np.float32)
    z_next = rng.standard_normal((N, D)).astype(np.float32)
    valid = np.ones(N, dtype=np.uint8)
    query = rng.standard_normal(D).astype(np.float32)
    # K > nv (out-of-bounds kth before the fix), K == nv, and K < nv
    for K in (10, 5, 3):
        out = match_euclidean_cy(z_cur, z_next, valid, query, K)
        assert out is not None and out.shape == (D,), f"K={K} returned bad result"
        assert np.allclose(out, _ref_knn(z_cur, z_next, query, K), atol=1e-5), \
            f"K={K} result differs from reference"
    assert match_euclidean_cy(z_cur, z_next, valid, query, 0) is None


def test_match_hamming_argpartition_bounds():
    if not HAS_CY:
        print("[WARN] train._lcm_cy not importable — rebuild Cython; skipping")
        return
    N, D = 5, 4
    rng = np.random.default_rng(0)
    z_cur = rng.standard_normal((N, D)).astype(np.float32)
    z_next = rng.standard_normal((N, D)).astype(np.float32)
    valid = np.ones(N, dtype=np.uint8)
    query = rng.standard_normal(D).astype(np.float32)
    sigs = rng.integers(0, 2**63, N).astype(np.int64)
    sig = int(sigs[0])
    for K in (10, 5, 3):
        out = match_hamming_cy(z_cur, z_next, sigs, valid, query, sig, 100, K)
        assert out is not None and out.shape == (D,), f"K={K} returned bad result"
        assert np.allclose(out, _ref_knn(z_cur, z_next, query, K), atol=1e-5), \
            f"K={K} result differs from reference"
    assert match_hamming_cy(z_cur, z_next, sigs, valid, query, sig, 100, 0) is None


# ── #7 binding lattice key separation ────────────────────────────────────────

def test_binding_cb_keys_distinct():
    params = init_binding_params(jax.random.PRNGKey(42), d=8, M_bind=16,
                                 n_layers=3, r_max=4)
    families = [params['key_cb'], params['val_cb'], params['bind_cb']]
    for l in range(3):
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = families[i][l], families[j][l]
                same_A = jnp.allclose(a['A'], b['A'])
                same_W = jnp.allclose(a['W'], b['W'])
                assert not (same_A and same_W), \
                    f"layer {l}: family {i} and {j} codebooks identical"


# ── runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_hrq_top_retrieval_picks_nearest,
        test_hrq_fallback_uses_closest_pair,
        test_hrq_value_scalars_branch_unchanged,
        test_value_contrast_loss_direction,
        test_value_logit_softplus_algebra,
        test_silu_numpy_vs_jax,
        test_encoder_step_jax_vs_numpy_glu,
        test_encoder_step_cython_glu,
        test_rope_rotate_half,
        test_rope_odd_head_dim_rejected,
        test_log_map_clip_consistency,
        test_match_euclidean_argpartition_bounds,
        test_match_hamming_argpartition_bounds,
        test_binding_cb_keys_distinct,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print("All model-layer fix tests passed.")


if __name__ == '__main__':
    main()
