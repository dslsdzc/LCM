"""Formula verification tests — every formula that no other suite covers.

  1. HRR bind/unbind round-trip (numpy cog_loop + C lattice.c via ctypes)
  2. soft_retrieve STE backward == softmax(-d/τ) gradient
  3. value_biased_score formula
  4. cog_loop Poincaré helpers vs standard definitions (dead code, kept as
     experiment starting points — must still be *correct*)
  5. incremental gen_head decoding == teacher-forced gen_head_forward, and the
     Cython fast path == the numpy fallback (two implementations of one
     computation; both must stay in step — see the comment above the tests)

Run from repo root:  JAX_PLATFORMS=cpu be/bin/python -m train.test_formulas
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from train.cog_loop import hrr_bind, hrr_unbind, soft_retrieve
from train.cog_loop import poincare_exp_map, poincare_log_map, poincare_dist
from train.lattices import value_biased_score
from train.fusion import gen_head_forward
from lcm import gen_head_new_single, gen_head_new_single_cy


# ── 1. HRR bind/unbind round-trip ───────────────────────────────────────────

def test_hrr_roundtrip_numpy():
    """unbind(bind(a, b), b) must recover a (up to HRR noise)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    bound = hrr_bind(a, b)
    recovered = hrr_unbind(bound, b)
    # HRR recovery is noisy: the bound vector's cross terms leak. The
    # recovered vector must correlate strongly with a (not with a random c).
    c = rng.standard_normal(64).astype(np.float32)
    corr_a = float(np.dot(recovered, a) / (np.linalg.norm(recovered) * np.linalg.norm(a)))
    corr_c = float(np.dot(recovered, c) / (np.linalg.norm(recovered) * np.linalg.norm(c)))
    assert corr_a > 0.5, f"recovered correlation with a too low: {corr_a:.3f}"
    assert corr_a > corr_c + 0.3, f"recovery not discriminative: {corr_a:.3f} vs {corr_c:.3f}"
    print(f"  [PASS] numpy HRR round-trip (corr a={corr_a:.3f} > c={corr_c:.3f})")


def test_hrr_roundtrip_c():
    """Same property for the C engine's hrr_bind/hrr_unbind."""
    try:
        ctypes.CDLL('libm.so.6', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'infer', 'liblcm.so'))
    D = 128
    lib.hrr_bind.restype = None
    lib.hrr_unbind.restype = None
    rng = np.random.default_rng(1)
    a = rng.standard_normal(D).astype(np.float32)
    b = rng.standard_normal(D).astype(np.float32)
    bound = np.zeros(D, dtype=np.float32)
    out = np.zeros(D, dtype=np.float32)
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    key_ptrs = (ctypes.POINTER(ctypes.c_float) * 1)(a_ptr)
    val_ptrs = (ctypes.POINTER(ctypes.c_float) * 1)(b_ptr)
    lib.hrr_bind(key_ptrs, 1, val_ptrs, 1,
                 bound.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), D)
    lib.hrr_unbind(bound.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   b_ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), D)
    c = rng.standard_normal(D).astype(np.float32)
    corr_a = float(np.dot(out, a) / (np.linalg.norm(out) * np.linalg.norm(a)))
    corr_c = float(np.dot(out, c) / (np.linalg.norm(out) * np.linalg.norm(c)))
    assert corr_a > 0.5 and corr_a > corr_c + 0.3, \
        f"C HRR recovery failed: corr_a={corr_a:.3f} corr_c={corr_c:.3f}"
    print(f"  [PASS] C HRR round-trip (corr a={corr_a:.3f} > c={corr_c:.3f})")


# ── 2. soft_retrieve STE backward ───────────────────────────────────────────

def test_soft_retrieve_backward_is_softmax():
    """STE backward: ∂out/∂z must equal the softmax(-d/τ) weighted gradient."""
    z = jnp.array([0.3, 0.1], dtype=jnp.float32)
    cb = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    tau = 0.2

    def loss(z):
        out, _, _ = soft_retrieve(z, cb, tau=tau)
        return jnp.sum(out * jnp.array([1.0, 2.0]))

    grad = jax.grad(loss)(z)
    # Analytic: forward = hard + soft_avg - stop_gradient(soft_avg), so
    # d(out)/dz = d(soft_avg)/dz = Σ_j w_j · dc_j/dz where w = softmax(-d/τ).
    dists = jnp.sum((z[None, :] - cb) ** 2, axis=-1)
    w = jax.nn.softmax(-dists / tau)
    # dc_j/dz for output weight [1,2]: ∂(Σ_j w_j c_j·g)/∂z
    # d(-d_j/τ)/dz = -2(z - c_j)/τ → ∂w_j/∂z = w_j·(-2(z-c_j)/τ - Σ_k w_k(-2(z-c_k)/τ))
    ref = np.zeros(2, dtype=np.float64)
    g = np.array([1.0, 2.0])
    wn = np.asarray(w)
    zn, cbn = np.asarray(z), np.asarray(cb)
    inner = -2.0 * (zn[None, :] - cbn) / tau  # (M, d): ∂s_j/∂z_i
    mean_inner = np.sum(wn[:, None] * inner, axis=0)  # (d,)
    for j in range(3):
        ref += wn[j] * (inner[j] - mean_inner) * float(g @ cbn[j])  # (d,)
    assert np.allclose(np.asarray(grad), ref, atol=1e-4), \
        f"STE backward mismatch: {np.asarray(grad)} vs {ref}"
    print(f"  [PASS] soft_retrieve STE backward == softmax gradient "
          f"(max diff {float(np.abs(np.asarray(grad) - ref).max()):.2e})")


# ── 3. value_biased_score ───────────────────────────────────────────────────

def test_value_biased_score_formula():
    """score = -‖z - c‖² + α·v·avg_dist² (docstring contract)."""
    z = jnp.array([[0.1, 0.2], [0.5, 0.5]], dtype=jnp.float32)
    C = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    v = jnp.array([1.0, -1.0], dtype=jnp.float32)
    alpha = 0.3
    s = value_biased_score(z, C, v, jnp.array(2.0), alpha)
    zn, cn = np.asarray(z), np.asarray(C)
    avg = 2.0
    for b in range(2):
        for j in range(2):
            d2 = float(np.sum((zn[b] - cn[j]) ** 2))
            ref = -d2 + alpha * float(v[j]) * avg
            assert abs(float(s[b, j]) - ref) < 1e-5, f"({b},{j}): {s[b, j]} vs {ref}"
    print("  [PASS] value_biased_score matches docstring formula")


# ── 4. cog_loop Poincaré helpers (dead code, kept as reference) ─────────────

def test_cog_loop_poincare_standard():
    """cog_loop's Poincaré helpers must match the standard definitions."""
    from train.hyp import poincare_distance as std_dist
    x = jnp.array([0.3, -0.2], dtype=jnp.float32)
    y = jnp.array([-0.1, 0.4], dtype=jnp.float32)
    v = jnp.array([0.2, 0.1], dtype=jnp.float32)

    # exp_x(v) = tanh(λ_x·‖v‖/2)·v/(λ_x·‖v‖), λ_x = 2/(1-‖x‖²)
    lam = 2.0 / (1.0 - float(jnp.sum(x ** 2)))
    vn = float(jnp.linalg.norm(v))
    exp_ref = float(np.tanh(lam * vn / 2.0)) * np.asarray(v) / (lam * vn)
    exp_got = np.asarray(poincare_exp_map(x, v))
    assert np.allclose(exp_got, exp_ref, atol=1e-5), \
        f"cog_loop exp_map wrong: got {exp_got}, ref {exp_ref}"

    # log_x(y) at x=0: atanh(‖y‖)·y/‖y‖
    zero = jnp.zeros_like(x)
    log_ref = float(np.arctanh(min(float(jnp.linalg.norm(y)), 0.999))) \
        * np.asarray(y) / float(jnp.linalg.norm(y))
    log_got = np.asarray(poincare_log_map(zero, y))
    assert np.allclose(log_got, log_ref, atol=1e-4), \
        f"cog_loop log_map wrong: got {log_got}, ref {log_ref}"

    # distance matches the hyp.py reference
    d_got = float(np.asarray(poincare_dist(x, y)).reshape(-1)[0])
    d_ref = float(np.asarray(std_dist(x, y)).reshape(-1)[0])
    assert abs(d_got - d_ref) < 1e-4, f"cog_loop dist wrong: {d_got} vs {d_ref}"
    print("  [PASS] cog_loop Poincaré helpers match standard definitions")


# ── 5. incremental inference == teacher-forced training forward ──────────────
#
# gen_head has TWO independent implementations of the same computation:
#   train/fusion.py::gen_head_forward        — JAX, teacher-forced, used by
#                                              training and every eval rollout
#   lcm.py::gen_head_new_single[_cy]         — numpy/Cython, incremental, used
#                                              by `lcm.py --interact`
# Commit 303a708 ("gen_head 认知状态持续注入") added `target_emb += 0.5 * z_q` to
# the first only. The trained weights were therefore evaluated on a different
# function at inference, with the discrepancy growing along the sequence
# (measured 0.198 → 0.349 max |Δlogit| over positions 1..4).
#
# These tests pin the two implementations together so the next edit to one of
# them cannot silently skip the other.

def _gen_head_fixture():
    d, V = 8, 16
    rng = np.random.default_rng(2)
    params = {
        'w_embed': rng.standard_normal((V, d)).astype(np.float32),
        'w_q': rng.standard_normal((d, d)).astype(np.float32),
        'w_k': rng.standard_normal((d, d)).astype(np.float32),
        'w_v': rng.standard_normal((d, d)).astype(np.float32),
        'w_o': rng.standard_normal((d, d)).astype(np.float32),
        'w_1': rng.standard_normal((d, 4 * d)).astype(np.float32),
        'w_2': rng.standard_normal((d, 4 * d)).astype(np.float32),
        'w_3': rng.standard_normal((4 * d, V)).astype(np.float32),
    }
    z_q = rng.standard_normal(d).astype(np.float32)
    seq = np.array([3, 5, 7, 9], dtype=np.int32)
    return ({k: jnp.array(v) for k, v in params.items()},
            {k: np.asarray(v) for k, v in params.items()}, z_q, seq)


def test_gen_head_incremental_matches_teacher_forcing():
    """Incremental step k+1 must equal teacher-forced position k.

    Teacher forcing returns positions 1..N of [z_q, x_0 .. x_{N-1}], so
    fwd[k] is conditioned on {z_q, x_0..x_k}. The incremental path reaches the
    same state after k+1 calls (the first seeds the cache with z_q), so the
    two arrays must agree elementwise, injection included.
    """
    params_j, params_np, z_q, seq = _gen_head_fixture()
    N = len(seq)
    fwd = np.asarray(gen_head_forward(
        params_j, jnp.array(z_q)[None, :], jnp.array(seq)[None, :],
        training=False))[0]                                   # (N, V)

    cache, inc = None, []
    for t in range(N + 1):
        lg, cache = gen_head_new_single(params_np, z_q, list(seq[:t]), cache)
        inc.append(np.asarray(lg))

    for k in range(N):
        diff = float(np.abs(inc[k + 1] - fwd[k]).max())
        assert diff < 1e-4, (
            f"incremental step {k + 1} != teacher-forced position {k} "
            f"(max |diff| = {diff:.3e}). The two gen_head implementations have "
            f"diverged — check for a change applied to only one of "
            f"train/fusion.py::gen_head_forward and "
            f"lcm.py::gen_head_new_single.")
    # inc[0] is position 0 — the readout conditioned on z_q alone, which the
    # teacher-forced forward computes and then discards (`[:, 1:, :]`).
    assert np.all(np.isfinite(inc[0])), "position-0 readout is not finite"
    print(f"  [PASS] incremental gen_head == teacher-forced forward at all "
          f"{N} shared positions (max |diff| "
          f"{max(float(np.abs(inc[k + 1] - fwd[k]).max()) for k in range(N)):.2e})")


def test_gen_head_cython_matches_numpy():
    """The Cython fast path must produce the same logits as the numpy fallback.

    Rebuilding is not automatic: `train/_lcm_cy*.so` is gitignored, so a stale
    build silently keeps the old behaviour on machines that have it while the
    fallback is correct on machines that do not.
    """
    from lcm import gen_head_new_single_cy
    try:
        import train._lcm_cy  # noqa: F401
        cython_present = True
    except ImportError:
        cython_present = False

    _, params_np, z_q, seq = _gen_head_fixture()
    N = len(seq)
    caches = [None, None]
    outs = [[], []]
    for t in range(N + 1):
        for i, fn in enumerate((gen_head_new_single, gen_head_new_single_cy)):
            lg, caches[i] = fn(params_np, z_q, list(seq[:t]), caches[i])
            outs[i].append(np.asarray(lg))
    if not cython_present:
        print("  [SKIP] gen_head Cython/numpy consistency — _lcm_cy not built")
        return
    diff = max(float(np.abs(outs[0][t] - outs[1][t]).max())
               for t in range(N + 1))
    assert diff < 1e-4, (
        f"Cython genhead_step_cy != numpy gen_head_new_single (max {diff:.3e}). "
        f"Rebuild with `python lcm.py build` after changing either one.")
    print(f"  [PASS] Cython gen_head == numpy gen_head (max |diff| {diff:.2e})")


if __name__ == '__main__':
    test_hrr_roundtrip_numpy()
    test_hrr_roundtrip_c()
    test_soft_retrieve_backward_is_softmax()
    test_value_biased_score_formula()
    test_cog_loop_poincare_standard()
    test_gen_head_incremental_matches_teacher_forcing()
    test_gen_head_cython_matches_numpy()
    print('All formula tests passed.')
