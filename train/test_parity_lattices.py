"""Per-lattice and fusion JAX<->C parity (design §6.2, task 11).

Every lattice is compared on the same fixed seed, fixed ``z`` and fixed
parameters. Discrete outputs (``idx`` / routing mask) must match *exactly*;
continuous outputs must satisfy ``max_abs_err < 1e-4``.

The C side is built from the current sources at ``LCM_D=32`` (the dimension
this harness uses), in a dedicated parity object directory, so the deployed
``liblcm.so`` built for d_model=256 is untouched. Set ``LCM_PARITY_LIB`` to use
an externally built .so instead. A missing library or a failed build is a test
failure with the compiler output attached -- never a silent skip.

Run:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_parity_lattices.py -v
"""
import numpy as np
import pytest

from train.parity_c_compare import (
    run_canonical_fusion_parity,
    run_engine_fusion_parity,
    run_lattice_parity,
)

LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast",
            "routing")

CONTINUOUS_TOL = 1e-4
FUSION_TOL = 1e-5


def _describe(res):
    """Failure text with the first mismatch, not just a boolean."""
    lines = [f"{res['lattice']} JAX<->C parity failed (seed={res['seed']}, "
             f"d={res['d']}, alpha={res['alpha']})"]
    if not res["idx_equal"]:
        m = res.get("idx_mismatch")
        lines.append(f"  discrete mismatch: {m}")
    lines.append(f"  max_abs_err={res['max_abs_err']:.3e} "
                 f"(tol={CONTINUOUS_TOL:.0e})")
    mm = res.get("max_mismatch")
    if mm:
        lines.append(
            f"  first over-tolerance point: field={mm['field']} "
            f"index={mm['index']} jax={mm['jax']:.9g} c={mm['c']:.9g} "
            f"abs_err={mm['abs_err']:.3e}")
    for field in res.get("fields", ()):
        j = np.asarray(res["jax"][field])
        c = np.asarray(res["c"][field])
        if j.shape != c.shape:
            lines.append(f"  {field}: shape {j.shape} vs {c.shape}")
            continue
        err = float(np.max(np.abs(j - c))) if j.size else 0.0
        lines.append(f"  {field}: shape={j.shape} max_abs_err={err:.3e}")
    return "\n".join(lines)


@pytest.mark.parametrize("lattice", LATTICES)
def test_lattice_parity(lattice):
    res = run_lattice_parity(lattice, seed=0)
    assert res["idx_equal"], _describe(res)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_sparse_value_bias_parity(alpha):
    """C's sparse value bias must reproduce JAX with and without the term."""
    res = run_lattice_parity("sparse", seed=0, alpha=alpha)
    assert res["idx_equal"], _describe(res)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res)


def test_sparse_value_bias_actually_changes_selection():
    """The alpha>0 case must be discriminating, not a no-op fixture.

    With this seed the value term flips the sparse selection; on both sides.
    If a future refactor makes alpha inert, the parity tests above would still
    pass while testing nothing, so pin the flip here.
    """
    base = run_lattice_parity("sparse", seed=0, alpha=0.0)
    biased = run_lattice_parity("sparse", seed=0, alpha=1.5)
    assert not np.array_equal(base["jax"]["idx"], biased["jax"]["idx"]), \
        "fixture is not discriminating: JAX sparse idx is unchanged at alpha=1.5"
    assert not np.array_equal(base["c"]["idx"], biased["c"]["idx"]), \
        "C sparse idx is unchanged at alpha=1.5; value bias is inert"


def test_sparse_inference_lfq_parity():
    """Inference LFQ branch: both sides must choose the zero vector.

    ``d_sparse_min`` is far above ``lambda_lfq * d_top`` here, so the decision
    fires on JAX and C alike. Comparing idx=0 and the zero_vec payload pins the
    inference index convention (0 = zero_vec) as well.
    """
    res = run_lattice_parity("sparse", seed=0, training=False, d_top=1e-3)
    assert res["idx_equal"], _describe(res)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res)
    assert int(res["c"]["idx"][0]) == 0, "LFQ did not fire in C"
    assert int(res["jax"]["idx"][0]) == 0, "LFQ did not fire in JAX"


def test_canonical_fusion_parity():
    """R9 canonical fusion: JAX fuse_lattices_with_aux vs C lcm_fuse_only."""
    res = run_canonical_fusion_parity(seed=0)
    assert res["max_abs_err"] < FUSION_TOL, _describe(res)


def test_engine_fusion_parity():
    """R9 engine fusion: JAX cog_loop.dag_fuse vs C distance_weighted_fusion."""
    res = run_engine_fusion_parity(seed=0)
    assert res["max_abs_err"] < FUSION_TOL, _describe(res)
