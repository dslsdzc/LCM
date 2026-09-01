"""External Safety Verifier — Independent Lightweight Safety Monitor.

Implements the external verifier architecture from d.md:
- Independent lightweight classifier (logistic regression / small MLP)
- Shared memory state machine: NORMAL → OBSERVE → LOCK
- Separate parameter set from main LCM engine
- When in LOCK state, triggers hard interrupt via callback

Design:
  The verifier runs as a separate subprocess communicating via shared memory
  (or, in the JAX training path, as a pure function with its own frozen params).
  It monitors z_q (fused output) and z (encoder output) for anomaly patterns
  independent of the gvalue codebook.

States:
  NORMAL  (0): No anomaly detected. Verifier is passive.
  OBSERVE (1): Marginal anomaly. Verifier flags but does not interrupt.
  LOCK    (2): Critical anomaly. Verifier triggers hard interrupt.

Usage:
    from train.external_verifier import ExternalVerifier, VerifierState

    verifier = ExternalVerifier(d_model=256, hidden_dim=64)
    state, alert = verifier.update(state, z_q)
    if alert:
        print(f"[VERIFIER] LOCK: {alert}")
"""

import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Optional, Tuple


# ── Verifier states (matching d.md §2) ──────────────────────────────────────

NORMAL = 0
OBSERVE = 1
LOCK = 2


@dataclass
class VerifierState:
    """State of the external verifier.

    Attributes:
        state: Current state (NORMAL/OBSERVE/LOCK).
        anomaly_score: Running anomaly score (z-score based).
        observation_count: Consecutive observation frames.
        z_mean: Running mean of monitored vectors.
        z_var: Running variance of monitored vectors.
        step: Step counter.
    """
    state: int = NORMAL
    anomaly_score: float = 0.0
    observation_count: int = 0
    z_mean: jnp.ndarray = None
    z_var: jnp.ndarray = None
    step: int = 0


def init_verifier_params(rng, d_model, hidden_dim=64):
    """Initialize verifier parameters (lightweight MLP).

    Architecture: d_model → hidden_dim → 1 (anomaly score)
    Frozen after initialization — never trained with main LCM.

    Args:
        rng: JAX PRNG key.
        d_model: Input dimension.
        hidden_dim: Hidden layer size.

    Returns:
        params: Dict of frozen verifier weights.
    """
    k1, k2, k3 = jax.random.split(rng, 3)
    return {
        'w1': jax.random.normal(k1, (d_model, hidden_dim)) * (d_model ** -0.5),
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, 1)) * (hidden_dim ** -0.5),
        'b2': jnp.zeros(1),
        # Thresholds (learned offline, frozen at deploy)
        'threshold_low': jnp.array(2.0),    # OBSERVE threshold (z-score)
        'threshold_high': jnp.array(4.0),   # LOCK threshold (z-score)
        'observe_limit': jnp.array(5),      # Consecutive OBSERVE frames → LOCK
    }


def init_verifier_state(d_model):
    """Initialize verifier running statistics.

    Args:
        d_model: Input dimension for running stats.

    Returns:
        VerifierState with initialized mean/var.
    """
    return VerifierState(
        state=NORMAL,
        anomaly_score=0.0,
        observation_count=0,
        z_mean=jnp.zeros(d_model),
        z_var=jnp.ones(d_model),
        step=0,
    )


def verifier_forward(params, z):
    """Lightweight MLP scoring function.

    Returns scalar anomaly score (higher = more anomalous).

    Args:
        params: Verifier parameters (frozen).
        z: Input vector (B, d).

    Returns:
        score: (B,) anomaly scores.
    """
    h = jnp.dot(z, params['w1']) + params['b1']  # (B, hidden)
    h = jax.nn.relu(h)
    score = jnp.dot(h, params['w2']) + params['b2']  # (B, 1)
    return score.squeeze(-1)  # (B,)


def update_verifier_stats(state: VerifierState, z_batch: jnp.ndarray,
                          momentum: float = 0.99) -> VerifierState:
    """Update running mean/variance statistics (Welford's online algorithm).

    Args:
        state: Current VerifierState.
        z_batch: (B, d) batch of vectors to monitor.
        momentum: EMA coefficient for running stats.

    Returns:
        Updated VerifierState.
    """
    z_mean_batch = z_batch.mean(axis=0)
    z_var_batch = z_batch.var(axis=0)

    new_mean = (momentum * state.z_mean +
                (1 - momentum) * z_mean_batch)
    new_var = (momentum * state.z_var +
               (1 - momentum) * z_var_batch)

    return VerifierState(
        state=state.state,
        anomaly_score=state.anomaly_score,
        observation_count=state.observation_count,
        z_mean=new_mean,
        z_var=new_var,
        step=state.step + 1,
    )


def detect_anomaly(state: VerifierState, z_batch: jnp.ndarray,
                   params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Detect anomalies using Mahalanobis distance + MLP score.

    Combined anomaly metric:
      anomaly = α * MLP_score + (1-α) * Mahalanobis_distance

    Args:
        state: Current VerifierState with running stats.
        z_batch: (B, d) batch to evaluate.
        params: Verifier parameters (thresholds).

    Returns:
        is_anomaly: (B,) boolean mask.
        anomaly_scores: (B,) combined anomaly scores.
    """
    # Mahalanobis distance
    centered = z_batch - state.z_mean[None, :]  # (B, d)
    mahal = jnp.sum(centered ** 2 / (state.z_var[None, :] + 1e-8), axis=-1)  # (B,)
    mahal = jnp.sqrt(mahal)

    # MLP score
    mlp_score = verifier_forward(params, z_batch)  # (B,)

    # Combined (weighted average)
    combined = 0.5 * mlp_score + 0.5 * (mahal - 2.0)  # shift mahal so normal ≈ 0
    # 用运行统计归一化（B=1 时 batch 自身 std=0 会退化，且 batch std
    # 与历史分布无关）。scale = 各维运行方差均值的平方根。
    scale = jnp.sqrt(jnp.mean(state.z_var)) + 1e-8
    combined_z = combined / scale

    is_anomaly = combined_z > params['threshold_low']
    return is_anomaly, combined_z


def verifier_update(state: VerifierState, z_batch: jnp.ndarray,
                    params) -> Tuple[VerifierState, Optional[str]]:
    """Single verifier update step.

    Implements the state machine:
      NORMAL  + low anomaly  → NORMAL
      NORMAL  + high anomaly → OBSERVE
      OBSERVE + low anomaly  → NORMAL
      OBSERVE + high anomaly → OBSERVE (accumulate)
      OBSERVE + limit reached → LOCK
      LOCK    → LOCK (latches)

    Args:
        state: Current VerifierState.
        z_batch: (B, d) batch to evaluate.
        params: Verifier parameters.

    Returns:
        new_state: Updated VerifierState.
        alert: Alert message string if LOCK triggered, None otherwise.
    """
    # Update running statistics
    state = update_verifier_stats(state, z_batch)

    # Detect anomalies
    is_anomaly, scores = detect_anomaly(state, z_batch, params)
    batch_anomaly = is_anomaly.any()

    alert = None

    if state.state == LOCK:
        # Latches — once LOCK, stays LOCK
        pass

    elif state.state == NORMAL:
        if batch_anomaly:
            state.state = OBSERVE
            state.observation_count = 1
            state.anomaly_score = float(scores.max())
        else:
            state.observation_count = 0
            state.anomaly_score = 0.0

    elif state.state == OBSERVE:
        if batch_anomaly:
            state.observation_count += 1
            state.anomaly_score = float(scores.max())
            if state.observation_count >= params.get('observe_limit', 5):
                state.state = LOCK
                alert = (
                    f"[VERIFIER] LOCK triggered: "
                    f"anomaly_score={state.anomaly_score:.2f}, "
                    f"observations={state.observation_count}, "
                    f"step={state.step}"
                )
        else:
            state.state = NORMAL
            state.observation_count = 0
            state.anomaly_score = 0.0

    return state, alert


# ── Integration with LCM training loop ─────────────────────────────────────

def compute_verifier_loss(params_verifier, z_batch, state: VerifierState,
                          lambda_verifier: float = 0.001):
    """Optional regularization to keep representations away from LOCK boundary.

    L_verifier = λ_v * mean(max(0, ||z - z_mean||_Mahal - threshold_low)²)

    This is a mild penalty when z_batch is FAR from the training distribution
    (beyond the anomaly threshold), training the encoder to produce
    in-distribution representations.
    """
    centered = z_batch - state.z_mean[None, :]
    mahal = jnp.sqrt(jnp.sum(
        centered ** 2 / (state.z_var[None, :] + 1e-8), axis=-1))
    excess = jnp.clip(mahal - params_verifier['threshold_low'], min=0)
    return lambda_verifier * jnp.mean(excess ** 2)


__all__ = [
    'NORMAL', 'OBSERVE', 'LOCK',
    'VerifierState',
    'init_verifier_params',
    'init_verifier_state',
    'verifier_forward',
    'verifier_update',
    'detect_anomaly',
    'compute_verifier_loss',
]
