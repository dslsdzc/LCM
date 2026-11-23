"""Parity between the calibration helper and the real residual-VQ forward.

The D-9 calibration prototype originally selected key/val rows from raw ``z``
and ignored the residual update, so its ``b_raw`` was not the tensor
``binding_forward`` actually binds. This test anchors the numpy helper to
``train.lattices._residual_vq_chain`` so the calibration cannot silently drift
from the model again.
"""
import numpy as np
import jax.numpy as jnp

from train.lattices import _residual_vq_chain
from scripts.calibrate_coverage import _residual_vq_chain_np


def test_calibration_residual_chain_matches_jax_forward():
    rng = np.random.default_rng(0)
    d, m, layers, b = 8, 16, 2, 12
    codebooks = [
        {
            "A": jnp.asarray(rng.standard_normal((m, d)) * 0.1,
                             dtype=jnp.float32),
            "W": jnp.eye(d, dtype=jnp.float32),
        }
        for _ in range(layers)
    ]
    q = rng.standard_normal((b, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8

    jax_q, _, jax_idx, _ = _residual_vq_chain(
        codebooks, jnp.asarray(q), None, 0.0, normalize_selection=True)
    np_q, _, np_idx = _residual_vq_chain_np(codebooks, q, cosine=True)

    for layer in range(layers):
        np.testing.assert_array_equal(
            np.asarray(jax_idx[layer]), np_idx[layer],
            err_msg=f"layer {layer} selected different rows")
        np.testing.assert_allclose(
            np.asarray(jax_q[layer]), np_q[layer],
            rtol=0.0, atol=1e-5,
            err_msg=f"layer {layer} quantised value mismatch")
