"""Numerical-boundary parity for ``safe_unit`` (JAX vs C).

The binding/self selectors compare unit directions. JAX uses
``x / (||x|| + 1e-8)``; C's selection helpers must use the same additive eps.
A threshold branch (``n > eps ? 1/n : 0``) silently disagrees for tiny but
non-zero vectors and changes the selected row.
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest

from train.hyp import safe_unit

SO_PATH = Path(__file__).resolve().parent.parent / "infer" / "liblcm.so"
D = 256


@pytest.mark.skipif(not SO_PATH.exists(), reason="infer/liblcm.so not built")
def test_c_safe_unit_matches_jax_at_tiny_norms():
    lib = ctypes.CDLL(str(SO_PATH))
    lib.lcm_dim.restype = ctypes.c_int
    d = int(lib.lcm_dim())
    lib.safe_unit_c.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_float]

    import jax.numpy as jnp

    directions = np.zeros((6, d), dtype=np.float32)
    directions[:, 0] = [0.0, 1e-12, 1e-9, 1e-7, 1e-3, 1.0]
    directions[1, 1] = 1e-13   # mostly-zero but two coordinates
    directions[2, 1] = 1e-10

    for x in directions:
        c_out = np.zeros(d, dtype=np.float32)
        lib.safe_unit_c(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d, ctypes.c_float(1e-8))
        j_out = np.asarray(safe_unit(jnp.asarray(x)[None])[0])
        np.testing.assert_allclose(
            c_out, j_out, rtol=1e-5, atol=1e-6,
            err_msg=f"safe_unit mismatch at ||x||={np.linalg.norm(x):.3e}")
