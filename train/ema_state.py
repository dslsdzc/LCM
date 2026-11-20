"""EMA state for EMA-owned codebooks (R3/R15).

Four lattices are updated by EMA instead of the optimizer:

    sparse    one codebook, C = params['sparse']['C']
    manifold  one tangent-space codebook, C = params['manifold']['C']
    binding   key/val/bind lists of SimVQ dicts, one codebook per layer
    self      C_self (M_self, d); row 0 is the frozen identity anchor and
              rows 1: are EMA modes (R15). ``ema_update_self`` is the only
              writer of C_self and never writes row 0.

State is a JAX pytree of arrays (``N``, ``m``, ``last_used``), carried through
``train_step`` and saved in the checkpoint. It is not a parameter and never
enters the optimizer:

    N          exponential sum of assignment counts, (M,)
    m          exponential sum of assigned vectors, (M, d)
    last_used  step index of the last assignment, (M,) int32

Recursion
---------
``N <- gamma*N + count`` and ``m <- gamma*m + z_sum``.  ``N`` is a decayed
*count*, not ``(1-gamma)*count``: the feature bank cold-start zeroes both
statistics when it replaces a dead row, and the first assignment must then set
``C`` to that observation. With the ``(1-gamma)`` form from b.md, a zeroed row
would come back at ``(1-gamma)*z`` instead. Rows with ``N == 0`` keep their
existing vector (a row that was never assigned is not evidence that its vector
is zero). This is what makes the 100-step variance check possible.

The two recursions are the same estimator up to a global factor. If
``(N_ref, m_ref)`` is b.md's state, ours is
``(N, m) = (N_ref, m_ref) / (1-gamma)``: both updates are linear and the two
forms differ by exactly that factor, which cancels in ``m / N``. For a warm row
(``N >= 1``) the clip is the same in both. For a cold row the first assignment
gives ``N=count, m=z_sum`` here versus ``(1-gamma)*count, (1-gamma)*z_sum`` in
b.md; the ratio is ``z_sum/count`` in both, which is the observation itself for
a single selected row. We keep the un-scaled form so the feature bank's zeroed
cold start is exact and so an untouched row is not divided by a sub-1 clip.
``test_decayed_sum_form_matches_bmd_scaled_form`` pins the warm-row equivalence.

Manifold coordinates are tangent space by construction: ``manifold_forward``
calls ``exp_map`` on them, so the EMA must not apply ``exp_map`` here. The
"reprojection after EMA" in the specification is exactly this equivalence: the
next forward reprojects, and the stored parameter stays tangent-space.
"""
import jax.numpy as jnp


def _empty_state(M, d):
    return {
        'N': jnp.zeros((M,), dtype=jnp.float32),
        'm': jnp.zeros((M, d), dtype=jnp.float32),
        'last_used': jnp.zeros((M,), dtype=jnp.int32),
    }


def init_ema_state(params, cfg):
    """Build zeroed EMA state for every EMA-owned codebook.

    Shapes come from the config, not the arrays: the state must be
    constructible before any EMA update and checked independently of the
    current parameter values.
    """
    d = int(cfg.d_model)
    state = {
        'sparse': _empty_state(int(cfg.M_sparse), d),
        'manifold': _empty_state(int(cfg.M_man), d),
        'binding': {},
    }
    for family in ('key', 'val', 'bind'):
        state['binding'][family] = [
            _empty_state(int(cfg.M_bind), d)
            for _ in range(int(cfg.n_bind_layers))
        ]
    # R15: C_self[0] is the frozen core and C_self[1:] are EMA modes. The
    # state has one row per C_self row; row 0's statistics stay zero because
    # ``ema_update_self`` masks it. ``getattr`` keeps the helper usable from
    # shape-only tests that predate the self branch.
    M_self = int(getattr(cfg, 'n_self_codes', 0))
    if M_self > 0:
        state['self'] = _empty_state(M_self, d)
    return state


def _scatter_sum(z, idx, M):
    """Scatter-add rows of ``z`` into an ``(M, d)`` buffer by ``idx``."""
    z = jnp.asarray(z)
    return jnp.zeros((M, z.shape[-1]), dtype=z.dtype).at[idx].add(z)


def _scatter_count(idx, M, dtype=jnp.float32):
    """One-hot assignment counts per row."""
    return jnp.zeros((M,), dtype=dtype).at[idx].add(1)


def _ema(C, state, z, idx, gamma):
    """One EMA step. Returns the new effective codebook and state."""
    M = C.shape[0]
    idx = jnp.asarray(idx, dtype=jnp.int32)
    count = _scatter_count(idx, M)
    N = gamma * state['N'] + count
    m = gamma * state['m'] + _scatter_sum(z, idx, M)
    # N == 0 means the row has never been assigned: preserve its vector rather
    # than divide 0/1 and erase the initialisation.
    C_new = jnp.where(
        (N > 0.0)[:, None],
        m / jnp.clip(N, 1.0)[:, None],
        C,
    )
    return C_new, N, m


def _update_last_used(state, idx, step, M):
    if step is None:
        return state['last_used']
    step_value = jnp.asarray(step, dtype=state['last_used'].dtype)
    used = _scatter_count(idx, M, dtype=jnp.int32) > 0
    return jnp.where(used, step_value, state['last_used'])


def _rebuild_state(N, m, last_used):
    return {'N': N, 'm': m, 'last_used': last_used}


def ema_update_sparse(C, state, z, idx, gamma, lambda_shrink, step=None):
    """EMA + soft-threshold shrinkage for the sparse codebook.

    ``idx`` is ``(B,)``; the scatter-add accumulates every batch row into its
    selected code. The soft threshold is applied after the EMA, in the codebook
    unit, as R2 specifies.
    """
    M = C.shape[0]
    idx = jnp.asarray(idx, dtype=jnp.int32)
    C_new, N, m = _ema(C, state, z, idx, gamma)
    C_new = jnp.sign(C_new) * jnp.clip(jnp.abs(C_new) - lambda_shrink, 0.0)
    last_used = _update_last_used(state, idx, step, M)
    return C_new, _rebuild_state(N, m, last_used)


def ema_update_manifold(C, state, z, idx, gamma, step=None):
    """EMA for the manifold codebook. Parameters stay in tangent space."""
    M = C.shape[0]
    idx = jnp.asarray(idx, dtype=jnp.int32)
    C_new, N, m = _ema(C, state, z, idx, gamma)
    last_used = _update_last_used(state, idx, step, M)
    return C_new, _rebuild_state(N, m, last_used)


def ema_update_self(C, state, z, idx, gamma, step=None):
    """EMA for the self codebook, with row 0 (the core) permanently frozen.

    R15: ``C_self[0]`` is the identity anchor and ``C_self[1:]`` are the EMA
    modes. ``param_groups`` owns the whole array as one EMA leaf because a
    single JAX array cannot be split across groups, so this function is where
    the row-level freeze is enforced. It is the only writer of ``C_self``.

    ``idx`` is not recomputed here: it is the row the forward actually selected
    (including row 0 when the core is the nearest prototype). If row 0 is
    selected, its vector and EMA statistics are left untouched.
    """
    M = C.shape[0]
    idx = jnp.asarray(idx, dtype=jnp.int32)
    C_new, N, m = _ema(C, state, z, idx, gamma)
    # Frozen core: never write the vector, never accumulate statistics for it.
    C_new = C_new.at[0].set(C[0])
    N = N.at[0].set(0.0)
    m = m.at[0].set(0.0)
    last_used = _update_last_used(state, idx, step, M)
    return C_new, _rebuild_state(N, m, last_used)


def ema_update_binding(params_binding, state, residuals, indices, gamma,
                       step=None):
    """EMA for the three binding codebook families, every layer.

    Args:
        params_binding: dict with ``key_cb``, ``val_cb``, ``bind_cb``; each is
            a list of SimVQ-style ``{'A', 'W'}`` dicts, one per layer.
        state: ``{'key': [state, ...], 'val': [...], 'bind': [...]}``.
        residuals: ``{'key': [r_0, ...], 'val': [...], 'bind': [...]}``, the
            per-layer query vectors returned by ``binding_forward``. Each is
            ``(B, d)``.
        indices: the per-layer rows ``binding_forward`` actually selected,
            same family/layer layout. They are **not** recomputed here: under a
            nonzero ``value_scalars`` the forward picks by value-biased score,
            and a plain Euclidean argmin would write the EMA into a row the
            forward never quantised.
        gamma: EMA decay.
        step: optional training step recorded in ``last_used``.

    Returns:
        (params_binding, state) — the codebooks are stored factorised as
        ``(C, I)`` after the update. These codebooks have no gradient path
        (they are EMA-owned), so keeping a random SimVQ ``W`` would make the
        effective codebook ``A @ W`` differ from the vector the EMA stored.
    """
    new_params = dict(params_binding)
    new_state = {}

    for family in ('key', 'val', 'bind'):
        key = f'{family}_cb'
        entries = params_binding[key]
        family_residuals = residuals[family]
        family_indices = indices[family]
        family_state = state[family]
        if not (len(entries) == len(family_residuals)
                == len(family_indices) == len(family_state)):
            raise ValueError(
                f"binding {family}: {len(entries)} codebooks, "
                f"{len(family_residuals)} residuals, "
                f"{len(family_indices)} indices, "
                f"{len(family_state)} states")

        new_entries, new_family_state = [], []
        for entry, residual, selected, entry_state in zip(
                entries, family_residuals, family_indices, family_state):
            C = entry['A'] @ entry['W']
            idx = jnp.asarray(selected, dtype=jnp.int32)
            C_new, N, m = _ema(C, entry_state, residual, idx, gamma)
            d = C_new.shape[-1]
            new_entries.append({'A': C_new, 'W': jnp.eye(d, dtype=C_new.dtype)})
            new_family_state.append(_rebuild_state(
                N, m, _update_last_used(entry_state, idx, step, C.shape[0])))

        new_params[key] = new_entries
        new_state[family] = new_family_state

    return new_params, new_state


__all__ = [
    'init_ema_state', 'ema_update_sparse', 'ema_update_manifold',
    'ema_update_self', 'ema_update_binding',
]
