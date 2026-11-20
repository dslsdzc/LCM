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
    JAX_PLATFORMS=cpu python -m pytest train/test_parity_lattices.py -v
"""
import math

import numpy as np
import pytest

from train.parity_c_compare import (
    _parity_result,
    run_binding_projection_parity,
    run_canonical_fusion_parity,
    run_engine_fusion_parity,
    run_hrq_gap_parity,
    run_hrq_poincare_parity,
    run_lattice_parity,
    run_sparse_value_bias_parity,
)

LATTICES = ("hrq", "sparse", "lowrank", "manifold", "binding", "contrast",
            "routing")

CONTINUOUS_TOL = 1e-4
FUSION_TOL = 1e-5


def _describe(res, tol):
    """Failure text with the first discrete mismatch and the max mismatch."""
    lines = [f"{res['lattice']} JAX<->C parity failed (seed={res['seed']}, "
             f"d={res['d']}, alpha={res['alpha']})"]
    if not res["idx_equal"]:
        m = res.get("idx_mismatch")
        lines.append(f"  discrete mismatch: {m}")
    lines.append(f"  max_abs_err={res['max_abs_err']:.3e} "
                 f"(tol={tol:.0e})")
    mm = res.get("max_mismatch")
    if mm:
        lines.append(
            f"  max mismatch: field={mm['field']} "
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


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64)
                               - np.asarray(b, np.float64))))


@pytest.mark.parametrize("alpha", [0.0, 0.1])
@pytest.mark.parametrize("lattice", LATTICES)
def test_lattice_parity(lattice, alpha):
    res = run_lattice_parity(lattice, seed=0, alpha=alpha)
    assert res["idx_equal"], _describe(res, CONTINUOUS_TOL)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res, CONTINUOUS_TOL)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_sparse_value_bias_parity(alpha):
    """C's sparse value bias must reproduce JAX with and without the term."""
    res = run_lattice_parity("sparse", seed=0, alpha=alpha)
    assert res["idx_equal"], _describe(res, CONTINUOUS_TOL)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res, CONTINUOUS_TOL)


def test_sparse_value_bias_actually_changes_selection():
    """The crafted fixture must flip the selected row on both sides.

    z=0.6·e0, C=[0.7·e0, 0.1·e0], v=[−1,+1]: alpha=0 picks row 0, alpha=2
    picks row 1. This is guaranteed by construction, not by a lucky seed.
    """
    res = run_sparse_value_bias_parity()
    assert res["jax_idx_0"] == 0 and res["c_idx_0"] == 0, \
        f"alpha=0 must select row 0, got JAX={res['jax_idx_0']} " \
        f"C={res['c_idx_0']}"
    assert res["jax_idx_2"] == 1 and res["c_idx_2"] == 1, \
        f"alpha=2 must select row 1, got JAX={res['jax_idx_2']} " \
        f"C={res['c_idx_2']}"
    for alpha in (0.0, 2.0):
        r = res[alpha]
        assert r["idx_equal"], _describe(r, CONTINUOUS_TOL)
        assert r["max_abs_err"] < CONTINUOUS_TOL, _describe(r, CONTINUOUS_TOL)


def test_sparse_inference_lfq_parity():
    """Inference LFQ branch: both sides must choose the zero vector.

    ``d_sparse_min`` is far above ``lambda_lfq * d_top`` here, so the decision
    fires on JAX and C alike. Comparing idx=0 and the zero_vec payload pins the
    inference index convention (0 = zero_vec) as well.
    """
    res = run_lattice_parity("sparse", seed=0, training=False, d_top=1e-3)
    assert res["idx_equal"], _describe(res, CONTINUOUS_TOL)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res, CONTINUOUS_TOL)
    assert int(res["c"]["idx"][0]) == 0, "LFQ did not fire in C"
    assert int(res["jax"]["idx"][0]) == 0, "LFQ did not fire in JAX"


def test_hrq_poincare_not_euclidean():
    """Crafted radii must make Poincaré and Euclidean selection disagree.

    C0 is closer in Euclidean distance; C1 is closer in the Poincaré metric
    because its ball radius is nearly the query's. Both implementations must
    follow the Poincaré winner, so an accidental Euclidean argmin fails.
    """
    res = run_hrq_poincare_parity()
    assert res["euclidean_idx"] == 1, "fixture is not discriminating"
    assert res["poincare_idx"] == 0, "fixture is not discriminating"
    assert res["idx_equal"], _describe(res, CONTINUOUS_TOL)
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res, CONTINUOUS_TOL)
    assert res["jax_idx"] == res["poincare_idx"], \
        f"JAX picked idx {res['jax_idx']}, expected Poincaré {res['poincare_idx']}"
    assert res["c_idx"] == res["poincare_idx"], \
        f"C picked idx {res['c_idx']}, expected Poincaré {res['poincare_idx']}"


def test_hrq_relative_gap_not_absolute():
    """Task 8's two gap cases: the relative gap must choose the branch.

    Case A has absolute gap > tau > relative gap (fallback); Case B has
    relative gap > tau > absolute gap (hard). Each hard/fallback pair must be
    distinguishable, and JAX/C must agree on every branch.
    """
    res = run_hrq_gap_parity()
    a, b = res["case_a"], res["case_b"]

    assert a["abs_gap"] > 0.1 > a["rel_gap"], \
        f"case A fixture: abs={a['abs_gap']} rel={a['rel_gap']}"
    assert b["rel_gap"] > 0.1 > b["abs_gap"], \
        f"case B fixture: abs={b['abs_gap']} rel={b['rel_gap']}"
    assert a["jax_hard_soft"] > 1e-3 and a["c_hard_soft"] > 1e-3, \
        "case A: hard and fallback outputs are not distinguishable"
    assert b["jax_hard_soft"] > 1e-3 and b["c_hard_soft"] > 1e-3, \
        "case B: hard and fallback outputs are not distinguishable"

    assert a["jax_default_soft"] < 1e-6 and a["c_default_soft"] < 1e-6, \
        "case A: relative gap < tau did not select fallback"
    assert a["jax_default_hard"] > 1e-3 and a["c_default_hard"] > 1e-3, \
        "case A: fallback output equals the hard output"
    assert b["jax_default_hard"] < 1e-6 and b["c_default_hard"] < 1e-6, \
        "case B: relative gap > tau did not select hard"
    assert b["jax_default_soft"] > 1e-3 and b["c_default_soft"] > 1e-3, \
        "case B: hard output equals the fallback output"

    for case in (a, b):
        for branch in ("default", "hard", "soft"):
            r = case[branch]
            assert r["idx_equal"], _describe(r, CONTINUOUS_TOL)
            assert r["max_abs_err"] < CONTINUOUS_TOL, \
                _describe(r, CONTINUOUS_TOL)


def test_binding_projection_direction():
    """``z_k = V @ A_k @ z``, not the transposed ``A_k^T V^T z``.

    The two competing key-codebook rows are the correct and transposed
    projections. Selecting the wrong row makes k_q match the transposed row.
    """
    res = run_binding_projection_parity()
    assert _max_abs(res["correct"], res["transposed"]) > 1e-3, \
        "fixture does not separate the correct and transposed projections"
    assert res["jax_correct_err"] < 1e-6, \
        f"JAX selected the wrong row (correct err {res['jax_correct_err']})"
    assert res["c_correct_err"] < 1e-6, \
        f"C selected the wrong row (correct err {res['c_correct_err']})"
    assert res["jax_transposed_err"] > 1e-3
    assert res["c_transposed_err"] > 1e-3
    assert res["max_abs_err"] < CONTINUOUS_TOL, _describe(res, CONTINUOUS_TOL)


def test_nan_cannot_pass_parity():
    """A non-finite difference must be reported as +inf, never pass."""
    case = {"seed": 0, "d": 2}
    res = _parity_result(
        "synthetic", case, 0.0, ("out",), [],
        {"out": np.array([np.nan, 1.0])},
        {"out": np.array([1.0, 1.0])})
    assert res["max_abs_err"] == math.inf
    assert res["max_mismatch"]["index"] == (0,)
    assert not res["max_abs_err"] < CONTINUOUS_TOL


def test_canonical_fusion_parity():
    """R9 canonical fusion: JAX fuse_lattices_with_aux vs C lcm_fuse_only."""
    res = run_canonical_fusion_parity(seed=0)
    assert res["max_abs_err"] < FUSION_TOL, _describe(res, FUSION_TOL)


def test_engine_fusion_parity():
    """R9 engine fusion: JAX cog_loop.dag_fuse vs C distance_weighted_fusion."""
    res = run_engine_fusion_parity(seed=0)
    assert res["max_abs_err"] < FUSION_TOL, _describe(res, FUSION_TOL)
