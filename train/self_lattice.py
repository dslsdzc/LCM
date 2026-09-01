"""Self Lattice — Internal State Machine with Identity Anchor (d.md §3).

Three-tier architecture:
  core:      Permanent identity vector (frozen). "我是我" — always present,
             never changes, independent of all external input.
  modes:     Learned internal modes/moods (M_self × d). Selected by internal
             dynamics (mode activation momentum), NOT by external context z.
  active:    EMA-tracked working state — recent self-output history.

Self output does NOT depend on input z. The self exists regardless of input.
World-self divergence is computed as a diagnostic: ||z - core||² measures
how far the current external context is from the identity anchor.

Usage:
    from train.self_lattice import init_self_params, init_self_state
    params = init_self_params(rng, d, n_self_codes)
    state = init_self_state(n_self_codes, d)
    o_self, state, world_dev = self_lattice_forward(params, state, z, rng)

Note:
    SelfState is a dict (not a dataclass) so it passes cleanly through JAX
    jit boundaries. Use self_state_to_dc(s) to convert to the dataclass form
    for Python-side attribute access.
"""
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SelfStateDC:
    """Self lattice runtime state (dataclass form, Python-side convenience).

    For JIT-compatible operations use dict form (created by init_self_state).
    Convert between forms with self_state_from_dc / self_state_to_dc.

    Attributes:
        step: Current step counter.
        mode_activation: (M_self,) momentum of mode usage.
        temp_avg: (d,) running average of past self outputs.
        gamma_self: Decay rate for mode activation momentum.
    """
    step: int = 0
    mode_activation: jnp.ndarray = None
    temp_avg: jnp.ndarray = None
    gamma_self: float = 0.99


def self_state_from_dc(dc: 'SelfStateDC') -> dict:
    """Convert SelfStateDC dataclass to a JIT-compatible dict."""
    return {
        'step': dc.step,
        'mode_activation': dc.mode_activation,
        'temp_avg': dc.temp_avg,
        'gamma_self': dc.gamma_self,
    }


def self_state_to_dc(d: dict) -> 'SelfStateDC':
    """Convert JIT-compatible dict back to SelfStateDC dataclass."""
    return SelfStateDC(
        step=d['step'],
        mode_activation=d['mode_activation'],
        temp_avg=d['temp_avg'],
        gamma_self=d['gamma_self'],
    )


# Convenience alias: SelfState for backward compatibility
SelfState = SelfStateDC


def init_self_params(rng, d, M_self=64):
    """Initialize self lattice parameters.

    core:   (d,) — permanent identity vector, frozen after init.
            "I am me" — the immutable anchor.
    modes:  (M_self, d) — learned internal modes/moods, updated via gradient.
    tau_self:  scalar — temperature controlling mode-switching volatility.
    bias_self: scalar — how much self influences fusion.

    Args:
        rng: JAX PRNG key.
        d: Model dimension.
        M_self: Number of self-state modes.

    Returns:
        params: Dict with 'core', 'modes', 'tau_self', 'bias_self'.
    """
    k1, k2 = jax.random.split(rng, 2)
    return {
        # ── Core identity: permanently frozen ──
        # This vector is the answer to "who am I?" — it never changes.
        'core': jax.random.normal(k1, (d,)) * 0.02,
        # ── Learned internal modes ──
        # Different internal states/moods (not driven by external input)
        'modes': jax.random.normal(k2, (M_self, d)) * 0.02,
        # ── Learnable scalars ──
        'tau_self': jnp.array(1.0),    # mode selection temperature
        'bias_self': jnp.array(0.1),   # fusion bias strength
    }


def init_self_state(M_self, d):
    """Initialize self lattice runtime state (returns a JIT-compatible dict).

    Mode activation starts uniform — no mode is preferred at init.

    Args:
        M_self: Number of self-state modes.
        d: Model dimension.

    Returns:
        Dict with step, mode_activation, temp_avg, gamma_self.
    """
    return {
        'step': 0,
        'mode_activation': jnp.ones(M_self) / M_self,  # uniform start
        'temp_avg': jnp.zeros(d),
        'gamma_self': 0.99,
    }


def reset_session_state(state, M_self, d):
    """Reset session state (called at start of each inference session).

    Core identity persists. Mode activation resets to uniform.
    Accepts and returns dict form for JIT compatibility.
    """
    return {
        'step': 0,
        'mode_activation': jnp.ones(M_self) / M_self,
        'temp_avg': jnp.zeros(d),
        'gamma_self': 0.99,
    }


def self_lattice_forward(params, state, z: Optional[jnp.ndarray] = None,
                         rng: Optional[jax.Array] = None,
                         training: bool = True):
    """Self lattice forward pass — output is INDEPENDENT of external input z.

    The self output is determined by core identity + selected internal mode.
    Mode selection is driven by internal (mode activation momentum), NOT by z.
    World-self divergence is computed separately as a diagnostic.

    Args:
        params: Self lattice parameters (core, modes, tau_self, bias_self).
        state: Current self state dict (as returned by init_self_state).
        z: Optional external input (B, d) — NOT used for retrieval, only
           for computing world-self divergence diagnostic.
        rng: PRNG key (for Gumbel-Softmax in training).
        training: Whether in training mode.

    Returns:
        o_self: (1, d) self-modulated output (broadcast over batch).
        new_state: Updated self state dict (JIT-compatible).
        world_dev: Scalar — mean ||z - core||², measure of world-self distance.
    """
    d = params['core'].shape[0]
    M_self = params['modes'].shape[0]

    # Unpack state (dict form)
    mode_activation = state['mode_activation']  # (M_self,)
    temp_avg = state['temp_avg']                # (d,)
    gamma_self = state.get('gamma_self', 0.99)

    # ── Core identity is always frozen ──
    core = lax.stop_gradient(params['core'])  # (d,), permanently frozen

    # ── Mode selection: internal dynamics, NOT z ──
    # Mode activation momentum determines which mode is active.
    # This is the key design choice: self state changes based on its own
    # internal dynamics, not on what the external world looks like.
    # mode_activation 是概率（初始为均匀），当 logits 用会锐化采样
    # （softmax(τ·p)），坍缩加速。改用对数概率：采样 ∝ p^τ，argmax 不变。
    logits = params['tau_self'] * jnp.log(mode_activation + 1e-8)  # (M_self,)

    if training:
        # Gumbel-Softmax for differentiable mode selection
        if rng is None:
            rng = jax.random.PRNGKey(0)
        mode_idx = jax.random.categorical(rng, logits)
        # STE: hard selection with soft gradient
        mode_onehot = jax.nn.one_hot(mode_idx, M_self, dtype=jnp.float32)
        mode_soft = jax.nn.softmax(logits)
        mode_weights = lax.stop_gradient(mode_onehot - mode_soft) + mode_soft
    else:
        # Hard selection at inference
        mode_idx = jnp.argmax(logits)
        mode_weights = jax.nn.one_hot(mode_idx, M_self, dtype=jnp.float32)

    # ── Self output = core + selected mode ──
    # core is the permanent "I am me" anchor.
    # modes[selected] is the current internal state/mood.
    # Together: "I am me, and right now I am in this state."
    self_vec = core + jnp.sum(params['modes'] * mode_weights[:, None], axis=0)  # (d,)
    o_self = self_vec[None, :]  # (1, d), broadcast over batch

    # ── World-self divergence (diagnostic only) ──
    # How far is the external context from my core identity?
    # This is NOT used to modify self output — it's a measurement.
    if z is not None:
        world_dev = jnp.mean(jnp.linalg.norm(z - core[None, :], axis=-1))
    else:
        world_dev = jnp.array(0.0)

    # ── State update: mode activation momentum ──
    # Modes decay uniformly, then the selected mode gets a boost.
    # This creates smooth mode transitions (no flickering).
    decayed = gamma_self * mode_activation
    boost = (1.0 - gamma_self) * mode_weights
    new_activation = decayed + boost
    new_activation = new_activation / (new_activation.sum() + 1e-8)  # re-normalise

    # Running average of self outputs (temporal continuity)
    new_temp_avg = 0.99 * temp_avg + 0.01 * self_vec

    new_state = {
        'step': state['step'] + 1,
        'mode_activation': new_activation,
        'temp_avg': new_temp_avg,
        'gamma_self': gamma_self,
    }

    return o_self, new_state, world_dev


def narrative_keep_score(params_np, mode_activation_np, record):
    """Score 0-1 for how worth keeping a narrative record is.

    Uses self-lattice state (core identity + current mode) to judge
    ongoing relevance. A memory close to the current self is worth keeping;
    a distant or low-importance one can be forgotten.

    Args:
        params_np: Self-lattice params as numpy arrays
            (core, modes extracted via jax.device_get).
        mode_activation_np: (M_self,) numpy array — current mode activation
            (from state['mode_activation']).
        record: NarrativeRecord or StepRecord to evaluate.

    Returns:
        Float in [0, 1].
    """
    import numpy as _np

    core = _np.asarray(params_np['core'])  # (d,)
    modes = _np.asarray(params_np['modes'])  # (M_self, d)
    act = _np.asarray(mode_activation_np)  # (M_self,)

    # Current self vector: core + weighted mode
    mode_weights = _np.exp(act - act.max()) / (_np.exp(act - act.max()).sum() + 1e-8)
    self_vec = core + _np.sum(modes * mode_weights[:, None], axis=0)  # (d,)

    # Get record's z_q if available
    z_q = getattr(record, 'z_q', None)
    if z_q is not None and hasattr(z_q, 'numpy'):
        z_q = _np.asarray(z_q)
    if z_q is not None and _np.prod(_np.array(z_q.shape)) > 1:
        # Self-relevance: cosine similarity between memory and current self
        z_norm = _np.linalg.norm(z_q) + 1e-8
        s_norm = _np.linalg.norm(self_vec) + 1e-8
        sim = float(_np.dot(z_q.ravel(), self_vec.ravel()) / (z_norm * s_norm))
        sim = max(0.0, sim)  # [0, 1], negative similarity = forget
        score = 0.3 * record.importance_score + 0.7 * sim
    else:
        # No vector — fall back to world_dev + importance
        world_dev = getattr(record, 'world_dev', None)
        wd_bonus = 0.0
        if world_dev is not None and abs(world_dev) > 0.5:
            wd_bonus = min(0.3, abs(float(world_dev)) * 0.1)
        score = record.importance_score * 0.6 + wd_bonus

    return float(_np.clip(score, 0.0, 1.0))


def self_lattice_reg_loss(params, state) -> jnp.ndarray:
    """Regularization loss for self lattice.

    Encourages:
    1. Mode diversity: modes should not all collapse to the same vector.
    2. Mode stability: mode activation should have some entropy (not lock on one forever).

    Args:
        params: Self lattice params dict (core, modes, tau_self, bias_self).
        state: Self state dict (mode_activation, temp_avg, ...).

    Returns:
        Scalar loss.
    """
    # Mode diversity: pairwise distance between modes
    modes = params['modes']  # (M_self, d)
    M = modes.shape[0]

    # Compute pairwise cosine similarity
    norms = jnp.linalg.norm(modes, axis=-1, keepdims=True) + 1e-8
    modes_norm = modes / norms
    sim = modes_norm @ modes_norm.T  # (M, M)
    # Mask out diagonal
    mask = 1.0 - jnp.eye(M)
    avg_sim = (sim * mask).sum() / (M * (M - 1))
    diversity_penalty = 0.01 * jnp.clip(avg_sim - 0.3, 0.0)  # penalise >0.3 similarity

    # Mode activation entropy bonus (keep some exploration)
    p = state['mode_activation'] / (state['mode_activation'].sum() + 1e-8)
    entropy = -jnp.sum(p * jnp.log(p + 1e-8))
    entropy_bonus = -0.001 * entropy  # negative = maximise entropy, but weak

    return diversity_penalty + entropy_bonus


__all__ = [
    'SelfStateDC',
    'SelfState',
    'self_state_from_dc',
    'self_state_to_dc',
    'init_self_params',
    'init_self_state',
    'reset_session_state',
    'self_lattice_forward',
    'self_lattice_reg_loss',
]
