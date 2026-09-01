"""Hyperbolic operations on the Poincaré ball model.

All operations use poincare_similarity for internal comparisons (avoids acosh),
and poincare_distance only for human-readable output.
"""
import jax.numpy as jnp
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


def exp_map(x: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Euclidean vector → Poincaré ball (exponential map)."""
    n = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return jnp.tanh(jnp.sqrt(c) * n) * x / (jnp.sqrt(c) * n)


def log_map(y: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    """Poincaré ball → Euclidean vector (logarithmic map)."""
    n = jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8
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
