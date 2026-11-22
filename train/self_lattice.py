"""Self Lattice — codebook nearest-neighbour + cosine familiarity (R15).

``e.md §6`` defines the self lattice as a codebook of the system's own
historical states:

    z_self    = C_self[argmin_j ||unit(z) - unit(C_self[j])||²]  (STE)
    self_bias = alpha_self · max(0, cosine_sim(z, z_self))

The selection uses unit-normalized rows (equivalent to cosine nearest
neighbour); the readout is still the raw codebook row. This prevents a
small-norm core/anchor row from winning every query by norm alone.

``C_self`` has ``M_self`` rows. Row 0 is the frozen identity anchor ("我是我"),
kept from the old ``core`` parameter; rows ``1:`` are the EMA modes
(``gamma_self = 0.99``).  The whole array is EMA-owned in ``param_groups``
because one JAX array cannot be split across parameter groups; the freeze is
enforced by ``train.ema_state.ema_update_self``, which never writes row 0.

The old ``core + modes + mode_activation`` machine is gone as a forward path.
The selection is a plain nearest-neighbour over ``C_self``, not a mode sample.
``mode_activation`` survives only as a cheap diagnostic (``init_self_state`` /
``update_self_state``) and never influences ``z_self``.

Usage:
    from train.self_lattice import init_self_params, self_lattice_forward
    params = init_self_params(rng, d, M_self=64)
    z_self, self_bias = self_lattice_forward(params, z, training=False)
"""
import jax
import jax.numpy as jnp
from jax import lax

from train.hyp import safe_unit

# R15 random fallback: standard deviation of the small random codebook rows
# used when no early encoder-bottleneck samples are available. The scale is
# deliberately small (the same 0.02 the old core/modes used) because the
# encoder output is normalised to the unit sphere and an untrained codebook
# must not dominate fusion.
SELF_INIT_STD = 0.02


def init_self_params(rng, d, M_self=64, *, init_samples=None,
                     alpha_self=0.05):
    """Initialise the self codebook.

    Args:
        rng: JAX PRNG key.
        d: Bottleneck dimension.
        M_self: Total number of ``C_self`` rows, including the frozen core.
        init_samples: Optional ``(K, d)`` array of early encoder bottleneck
            vectors. Up to ``M_self - 1`` rows seed the EMA modes; remaining
            rows fall back to small random vectors. If None, every mode is
            random (std ``SELF_INIT_STD``), which is the documented fresh-run
            behaviour (``e.md §6`` prefers early-training samples when a caller
            has them).
        alpha_self: Initial familiarity strength. Stored as the trainable
            scalar ``bias_self`` and used as the upper bound of ``self_bias``.

    Returns:
        ``{'C_self': (M_self, d), 'bias_self': ()}``.
    """
    if int(M_self) < 2:
        raise ValueError(f"M_self must be >= 2 (core + at least one mode), "
                         f"got {M_self}")
    d = int(d)
    M_self = int(M_self)

    k_core, k_modes = jax.random.split(rng, 2)
    core = jax.random.normal(k_core, (d,), dtype=jnp.float32) * SELF_INIT_STD
    modes = (jax.random.normal(k_modes, (M_self - 1, d), dtype=jnp.float32)
             * SELF_INIT_STD)

    if init_samples is not None:
        samples = jnp.asarray(init_samples, dtype=jnp.float32)
        if samples.ndim != 2 or samples.shape[-1] != d:
            raise ValueError(
                f"init_samples must be (K, {d}), got {tuple(samples.shape)}")
        k = min(int(samples.shape[0]), M_self - 1)
        if k > 0:
            modes = modes.at[:k].set(samples[:k])

    C_self = jnp.concatenate([core[None, :], modes], axis=0)
    return {
        'C_self': C_self,
        'bias_self': jnp.asarray(alpha_self, dtype=jnp.float32),
    }


def init_self_state(M_self, d):
    """Initialise the optional diagnostic state (JIT-compatible dict).

    This is *not* the forward selector any more. It records which codebook row
    was nearest over time, for logging/self-observation. ``self_lattice_forward``
    does not read it.
    """
    return {
        'step': 0,
        'mode_activation': jnp.ones(int(M_self), dtype=jnp.float32)
        / float(M_self),
        'temp_avg': jnp.zeros(int(d), dtype=jnp.float32),
        'gamma_self': 0.99,
    }


def reset_session_state(state, M_self, d):
    """Reset the diagnostic state at a session boundary."""
    return init_self_state(M_self, d)


def self_lattice_forward(params, z, training=False, return_aux=False,
                         alpha_self=None):
    """Nearest codebook row + cosine familiarity.

    Args:
        params: ``{'C_self': (M_self, d), 'bias_self': ()}``. ``bias_self`` is
            the frozen ``cfg.alpha_self`` mirror used when ``alpha_self`` is
            not passed.
        z: ``(B, d)`` current bottleneck vectors (a single ``(d,)`` is promoted
            to ``(1, d)``).
        training: Accepted for call-site symmetry with the other lattices. The
            forward is deterministic: retrieval is a hard argmin and STE is
            used in both modes.
        return_aux: When True, also return the cheap diagnostics
            ``{'idx', 'dist2', 'cosine'}`` (raw cosine, may be negative).
            They are not inputs to any selector.
        alpha_self: The configured familiarity strength (``cfg.alpha_self``).
            Callers pass it explicitly; when None the frozen ``params`` mirror
            is used. It is clamped non-negative before scaling.

    Returns:
        ``z_self``: ``(B, d)`` selected codebook rows. The forward value is the
        exact row; the gradient is the identity in ``z`` (straight-through), so
        commitment/encoder gradients flow through it. The selected codebook row
        itself is stop-gradiented: EMA is the only writer of ``C_self``.
        ``self_bias``: ``(B,)`` familiarity in ``[0, alpha_self]``.
        ``aux``: only when ``return_aux=True``.
    """
    C_self = jnp.asarray(params['C_self'])
    z = jnp.atleast_2d(jnp.asarray(z))
    if z.ndim != 2 or z.shape[-1] != C_self.shape[-1]:
        raise ValueError(
            f"z must be (B, {C_self.shape[-1]}), got {tuple(z.shape)}")

    dist2 = jnp.sum(
        (safe_unit(z)[:, None, :] - safe_unit(C_self)[None, :, :]) ** 2,
        axis=-1)
    idx = jnp.argmin(dist2, axis=-1)          # (B,)
    # Ruling 37: the codebook row is a stop-gradient. Forward is exactly the
    # row; d/dz is the identity so the commitment path still trains the encoder.
    selected = lax.stop_gradient(C_self[idx])  # (B, d)
    z_self = selected + (z - lax.stop_gradient(z))

    # Cosine similarity is computed against the selected row (which equals
    # z_self forward). Clipping makes the [0, alpha_self] range structural,
    # not dependent on floating-point noise.
    z_norm = jnp.linalg.norm(z, axis=-1) + 1e-8
    c_norm = jnp.linalg.norm(selected, axis=-1) + 1e-8
    raw_cosine = jnp.sum(z * selected, axis=-1) / (z_norm * c_norm)
    cosine = jnp.clip(raw_cosine, 0.0, 1.0)
    if alpha_self is None:
        alpha = jnp.asarray(params['bias_self'], dtype=jnp.float32)
    else:
        alpha = jnp.asarray(alpha_self, dtype=jnp.float32)
    alpha = jnp.maximum(alpha, 0.0)
    self_bias = alpha * cosine

    if return_aux:
        aux = {
            'idx': idx,
            'dist2': jnp.min(dist2, axis=-1),
            # Raw cosine (can be negative); self_bias is the clipped value.
            'cosine': raw_cosine,
        }
        return z_self, self_bias, aux
    return z_self, self_bias


def update_self_state(state, idx, z_self, gamma_self=None):
    """Update the optional diagnostic state from a self forward.

    ``mode_activation`` is an exponential moving average of hard nearest-row
    assignments. It is logged, never used to pick ``z_self``.
    """
    M_self = state['mode_activation'].shape[0]
    idx = jnp.asarray(idx, dtype=jnp.int32)
    gamma = state.get('gamma_self', 0.99) if gamma_self is None \
        else gamma_self

    counts = jnp.mean(
        jax.nn.one_hot(idx, M_self, dtype=jnp.float32), axis=0)
    activation = gamma * state['mode_activation'] + (1.0 - gamma) * counts
    activation = activation / (activation.sum() + 1e-8)

    z_self = jnp.atleast_2d(jnp.asarray(z_self))
    new_temp_avg = (gamma * state['temp_avg']
                    + (1.0 - gamma) * jnp.mean(z_self, axis=0))
    return {
        'step': state['step'] + 1,
        'mode_activation': activation,
        'temp_avg': new_temp_avg,
        'gamma_self': gamma,
    }


def narrative_keep_score(params_np, mode_activation_np, record):
    """Score 0-1 for how worth keeping a narrative record is.

    Uses the record's ``z_q`` and the nearest self-codebook row: a memory close
    to a prototype of the system's own states is worth keeping; a distant one
    can be forgotten. ``mode_activation_np`` is retained for the legacy caller
    signature and is deliberately ignored (R15 removed it as a selector).
    """
    import numpy as _np

    C = _np.asarray(params_np['C_self'])  # (M_self, d)
    z_q = getattr(record, 'z_q', None)
    if z_q is not None and hasattr(z_q, 'numpy'):
        z_q = _np.asarray(z_q)
    if z_q is not None and _np.prod(_np.array(z_q.shape)) > 1:
        z_q = z_q.ravel()
        idx = int(_np.argmin(_np.sum((C - z_q[None, :]) ** 2, axis=-1)))
        self_vec = C[idx]
        z_norm = _np.linalg.norm(z_q) + 1e-8
        s_norm = _np.linalg.norm(self_vec) + 1e-8
        sim = float(_np.dot(z_q, self_vec) / (z_norm * s_norm))
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


def self_lattice_reg_loss(params, state=None):
    """Weak diversity regulariser over the EMA modes.

    ``C_self[1:]`` are EMA-owned, so this term does not train them; it is kept
    as a reported diagnostic (and so the existing training consumers keep a
    finite, well-defined value). The frozen core row is excluded: it is an
    anchor, not one mode among peers. ``state`` is accepted for the legacy
    caller signature and ignored.
    """
    modes = jnp.asarray(params['C_self'])[1:]  # (M_self - 1, d)
    M = modes.shape[0]
    if M < 2:
        return jnp.array(0.0)

    norms = jnp.linalg.norm(modes, axis=-1, keepdims=True) + 1e-8
    modes_norm = modes / norms
    sim = modes_norm @ modes_norm.T
    mask = 1.0 - jnp.eye(M)
    avg_sim = (sim * mask).sum() / (M * (M - 1))
    return 0.01 * jnp.clip(avg_sim - 0.3, 0.0)


__all__ = [
    'SELF_INIT_STD',
    'init_self_params',
    'init_self_state',
    'reset_session_state',
    'self_lattice_forward',
    'update_self_state',
    'narrative_keep_score',
    'self_lattice_reg_loss',
]
