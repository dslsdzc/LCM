"""Hyperbolic operations on the Poincaré ball model.

All operations use poincare_similarity for internal comparisons (avoids acosh),
and poincare_distance only for human-readable output.
"""
import jax.numpy as jnp
from jax import lax
from jax.lax import clamp


def poincare_similarity(u: jnp.ndarray, v: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Hyperbolic similarity, monotonically equivalent to hyperbolic distance.
    Used for all internal comparisons (argmin, softmax, thresholds).
    Returns 2c * ||u-v||² / ((1-c||u||²)(1-c||v||²)).
    """
    u_norm2 = jnp.linalg.norm(u, axis=-1, keepdims=True) ** 2
    v_norm2 = jnp.linalg.norm(v, axis=-1, keepdims=True) ** 2
    diff_norm2 = jnp.linalg.norm(u - v, axis=-1, keepdims=True) ** 2
    denom = (1 - c * u_norm2) * (1 - c * v_norm2)
    # max-clamp (not additive eps): for off-ball inputs (‖·‖ > 1) denom is
    # negative — additive eps keeps it negative and the similarity turns
    # negative, making poincare_distance NaN. Matches infer/hyp.c.
    return 2 * c * diff_norm2 / jnp.maximum(denom, 1e-8)


def poincare_distance(u: jnp.ndarray, v: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """True hyperbolic distance. Only for human-readable output.
    For internal comparisons use poincare_similarity.
    """
    arg = 1 + poincare_similarity(u, v, c) + 1e-8
    return jnp.arccosh(jnp.maximum(arg, 1.0))  # domain guard (arccosh ≥ 1)


def _norm_safe_grad(x: jnp.ndarray) -> jnp.ndarray:
    """``‖x‖`` with the exact value of ``jnp.linalg.norm`` and a finite gradient
    at the origin.

    The value is taken from ``jnp.linalg.norm`` (a hand-rolled ``sqrt(Σxᵢ²)``
    differs from it by a few ulp — the reduction order is not the same), but
    its gradient ``x/‖x‖`` is 0/0 at the origin. The gradient is routed through
    a squared-norm expression, which is smooth there, using the same
    ``stop_gradient(a - b) + b`` trick the lattices use for their straight-
    through estimators. ``exact - safe`` is exact by Sterbenz (the two agree to
    ~1e-6 relative), so the forward value is bit-identical to ``exact``.
    """
    exact = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe = jnp.sqrt(jnp.maximum(
        jnp.sum(x * x, axis=-1, keepdims=True), 1e-12))
    return lax.stop_gradient(exact - safe) + safe


def safe_unit(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """``x / (‖x‖ + eps)`` — unit normalisation with a finite gradient at 0.

    Same origin hazard as ``exp_map``: at ``x = 0`` the forward value is a clean
    0, but ``∂‖x‖/∂x`` is 0/0 and the ``x · ∂‖x‖`` term turns the whole
    gradient into NaN. This is not hypothetical here — an EMA quantiser writes
    exact zeros into every codebook row the batch never selected
    (``m / clip(N, 1)`` = 0/1), so normalization of such a row is reached on
    every step. Use this instead of ``x / (jnp.linalg.norm(x, ...) + eps)``
    wherever the input can be zero.
    """
    return x / (_norm_safe_grad(x) + eps)


def exp_map(x: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Euclidean vector → Poincaré ball (exponential map).

    The expression is unchanged from ``tanh(n)·x/n``; only the norm is swapped
    for ``_norm_safe_grad``. At exactly ``x = 0`` the quotient form is 0/0: the
    *forward* value comes out as a clean 0 while the gradient is NaN. That
    matters because the EMA writes exact zeros into codebook rows no batch ever
    selects (``m / clip(N, 1)`` = 0/1), and one all-zero row is enough —
    ``manifold_forward``'s gradient w.r.t. such a row measured 24/24 NaN, which
    the global-norm clip then spreads to every parameter.
    """
    n = _norm_safe_grad(x) + 1e-8
    return jnp.tanh(jnp.sqrt(c) * n) * x / (jnp.sqrt(c) * n)


def log_map(y: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Poincaré ball → Euclidean vector (logarithmic map).

    Same origin hazard and same remedy as ``exp_map`` — see the note there.
    """
    n = _norm_safe_grad(y) + 1e-8
    n_clipped = clamp(0.0, n, 0.999)  # Stay within domain of atanh
    return jnp.arctanh(jnp.sqrt(c) * n_clipped) * y / (jnp.sqrt(c) * n_clipped)


def mobius_add(u: jnp.ndarray, v: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Möbius addition. Result stays on the Poincaré ball."""
    u_norm2 = jnp.linalg.norm(u, axis=-1, keepdims=True) ** 2
    v_norm2 = jnp.linalg.norm(v, axis=-1, keepdims=True) ** 2
    uv = (u * v).sum(axis=-1, keepdims=True)
    num = (1 + 2 * c * uv + c * v_norm2) * u + (1 - c * u_norm2) * v
    denom = 1 + 2 * c * uv + c ** 2 * u_norm2 * v_norm2
    return num / (denom + 1e-8)
