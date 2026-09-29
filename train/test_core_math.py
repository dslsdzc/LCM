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

from train.cog_loop import soft_retrieve, dag_fuse
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


def test_dag_fuse_reaches_a_fixed_point():
    """Fixed codebooks + fixed z → repeated dag_fuse must converge (diff → 0).

    This used to be asserted through cog_loop_scan. The cognitive loop now runs
    the canonical six-lattice step (train/cognitive_step.py) and takes lattice
    params rather than codebooks, so the property is asserted against dag_fuse
    directly — which is what it was always about: soft_retrieve snaps to a
    codebook entry and a snap is stable.
    """
    z = jnp.array([0.2, 0.2], dtype=jnp.float32)
    cb = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=jnp.float32)
    diffs = []
    for _ in range(30):
        z, diff, _ = dag_fuse(z, [cb, cb], [0.5, 0.5], tau=0.1)
        diffs.append(float(diff))
    assert diffs[-1] < 1e-3, \
        f"dag_fuse did not converge: last diff={diffs[-1]:.4f}"
    assert abs(diffs[-1] - diffs[-2]) < 1e-4


# ── 2. Data contract ────────────────────────────────────────────────────────

def _mini_corpus(tmp, n, dtype="uint32"):
    """A one-document corpus with real metadata, for WikiDataIter tests.

    WikiDataIter now takes (data_path, shape_path, spans_path) and reads its
    width from metadata, so a bare {n_tokens: N} sidecar no longer describes a
    readable corpus.
    """
    from train.dataset_meta import save_dataset_meta
    from train.tokenizer_spec import sha256_file

    dat = os.path.join(tmp, "tokens.dat")
    shp = os.path.join(tmp, "tokens_shape.json")
    spn = os.path.join(tmp, "tokens_docs.npy")

    tokens = np.arange(n, dtype=np.dtype(dtype))
    tokens.tofile(dat)
    with open(spn, "wb") as f:
        np.save(f, np.array([[0, n]], dtype=np.int64))
    save_dataset_meta(shp, {
        "n_tokens": int(n), "dtype": dtype,
        "tokenizer_id": "qwen2.5-0.5b",
        "tokenizer_sha256": sha256_file(spn),
        "separator_id": int(tokens[-1]),
        "token_id_max": int(tokens.max()),
        "model_vocab_size": 151936,
        "n_docs": 1,
        "document_spans_sha256": sha256_file(spn),
        "token_data_sha256": sha256_file(dat),
    })
    return dat, shp, spn


def test_wikidataiter_shift_contract():
    """targets[:, i] must equal inputs[:, i+1] (next-token prediction)."""
    from train.data import WikiDataIter
    with tempfile.TemporaryDirectory() as tmp:
        dat, shp, spn = _mini_corpus(tmp, 100)
        it = WikiDataIter(dat, shp, spn, B=3, N=8)
        inputs, targets = next(it)
    assert inputs.shape == (3, 8) and targets.shape == (3, 8)
    # In-window shift: targets[:, i] == inputs[:, i+1] for i < N-1.
    assert np.array_equal(targets[:, :-1], inputs[:, 1:]), \
        "targets[:, i] must be inputs[:, i+1]"
    # Out-of-window: targets[:, -1] is x[start+N] — the token immediately after
    # the window, which is what the passive channel reads out. Asserted as index
    # arithmetic, not set membership: WikiDataIter draws random window starts
    # (`np.random.randint`, unseeded), so on a dense ramp another window's
    # inputs routinely contain this window's successor and an `isin` check
    # failed ~50% of runs (measured 21/40) for reasons unrelated to the code.
    assert np.array_equal(targets[:, -1], inputs[:, -1] + 1), \
        "targets[:, -1] must be the token immediately after inputs[:, -1]"


def test_wikidataiter_window_bounds():
    """Sampled windows must stay inside the token array (no OOB reads)."""
    from train.data import WikiDataIter
    with tempfile.TemporaryDirectory() as tmp:
        dat, shp, spn = _mini_corpus(tmp, 50)
        it = WikiDataIter(dat, shp, spn, B=5, N=10)
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
    test_dag_fuse_reaches_a_fixed_point()
    test_wikidataiter_shift_contract()
    test_wikidataiter_window_bounds()
    test_encoder_full_forward_jax_vs_numpy()
    print('All core math tests passed.')
