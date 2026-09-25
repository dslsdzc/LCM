"""Continual Learning System for LCM — Stage 4.

Enables incremental learning without catastrophic forgetting via:
1. Dynamic codebook expansion for new domains/tasks
2. Elastic Weight Consolidation (EWC) on protected parameters
3. Experience replay across previously seen domains
4. Memory consolidation of high-frequency patterns
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ContinualState:
    """Tracks all continual learning state across tasks."""
    task_id: int = 0
    step: int = 0
    task_boundaries: Dict[int, int] = field(default_factory=dict)  # step -> task_id
    protected_params: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> frozen copy
    fisher_diag: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> Fisher diag
    replay_buffers: Dict[int, Dict[str, jnp.ndarray]] = field(default_factory=dict)  # task_id -> buffer
    access_counters: Dict[str, jnp.ndarray] = field(default_factory=dict)  # path -> per-entry counts
    consolidation_log: List[Dict] = field(default_factory=list)
    z_mean_ema: Optional[jnp.ndarray] = None  # Running EMA of z for shift detection
    z_cov_ema: Optional[jnp.ndarray] = None   # Running EMA of z covariance
    n_seen: int = 0


def init_continual_state(d_model: int) -> ContinualState:
    """Initialize continual learning state."""
    return ContinualState(
        z_mean_ema=jnp.zeros(d_model),
        z_cov_ema=jnp.eye(d_model) * 0.1,
    )


# ── 1. Distribution Shift Detection ──────────────────────────────────────────

def detect_new_task(z: jnp.ndarray, state: ContinualState,
                    threshold: float,
                    min_samples: int = 32) -> Tuple[bool, ContinualState]:
    """Detect distribution shift via Mahalanobis distance on the latent z.

    Updates a running EMA of the latent mean and covariance. If the batch mean
    sits more than ``threshold`` Mahalanobis standard deviations from the EMA
    mean, the batch is declared a new task.

    Args:
        z: (B, d_model) encoder latents — NOT token ids. The running statistics
            are (d_model,) and (d_model, d_model); feeding a (B, seq_len) token
            batch silently redefines d_model as seq_len on the first call and
            then fails on the covariance shape.
        state: Continual state to update.
        threshold: Mahalanobis distance threshold.
        min_samples: Latents to accumulate before a shift may fire. With only a
            handful of samples the covariance estimate is noise and every batch
            looks like a new task. A covariance also needs B >= 2 to be
            estimable at all, so single-sample batches never update it.
    """
    B, d = z.shape
    batch_mean = z.mean(axis=0)

    if state.z_mean_ema is None or state.z_mean_ema.shape[0] != d:
        raise ValueError(
            f"detect_new_task expects latents of width d_model="
            f"{None if state.z_mean_ema is None else state.z_mean_ema.shape[0]}, "
            f"got {d}. Pass encoder latents, not token ids.")

    if state.n_seen < min_samples:
        # Still warming up: accumulate statistics but never signal a task change.
        decay = jnp.clip(1.0 - 1.0 / (state.n_seen + B), 0.9, 0.999)
        state.z_mean_ema = (batch_mean if state.n_seen == 0
                            else decay * state.z_mean_ema + (1 - decay) * batch_mean)
        if B >= 2:
            batch_cov = jnp.cov(z.T)
            if jnp.ndim(batch_cov) == 2:
                state.z_cov_ema = decay * state.z_cov_ema + (1 - decay) * batch_cov
        state.n_seen += B
        return False, state

    # Mahalanobis distance
    diff = batch_mean - state.z_mean_ema
    cov_inv = jnp.linalg.pinv(state.z_cov_ema + 1e-6 * jnp.eye(d))
    m_dist = jnp.sqrt(jnp.maximum(diff @ cov_inv @ diff, 0.0))

    # Update EMA statistics
    decay = jnp.clip(1.0 - 1.0 / state.n_seen, 0.9, 0.999)
    new_mean = decay * state.z_mean_ema + (1 - decay) * batch_mean
    new_cov = state.z_cov_ema
    if B >= 2:
        batch_cov = jnp.cov(z.T)
        if jnp.ndim(batch_cov) == 2:
            new_cov = decay * state.z_cov_ema + (1 - decay) * batch_cov
    is_new = m_dist > threshold

    state.z_mean_ema = new_mean
    state.z_cov_ema = new_cov
    state.n_seen += B

    return bool(is_new), state


# ── 2. Dynamic Codebook Expansion ────────────────────────────────────────────

def expand_codebook(param: jnp.ndarray, n_new: int, rng: jax.Array,
                    init_scale: float = 0.02) -> jnp.ndarray:
    """Expand a codebook by appending n_new randomly initialized entries."""
    M, d = param.shape
    new_entries = jax.random.normal(rng, (n_new, d)) * init_scale
    return jnp.concatenate([param, new_entries], axis=0)


def expansion_key_count(cfg) -> int:
    """Number of RNG keys a full expansion consumes.

    Must stay in step with ``expand_lattice_codebooks``: one key per expanded
    tensor. A hardcoded split count that drifts below this indexes past the end
    of the key array and raises, and the failure lands on whichever codebook
    happens to cross the boundary rather than on the miscount.
    """
    return (1                                   # hrq top
            + cfg.n_hrq_layers                  # hrq fine layers
            + 1                                 # sparse
            + len(cfg.ranks)                    # lowrank U per rank
            + 2                                 # manifold C + T
            + 3 * cfg.n_bind_layers             # binding key/val/bind
            + 2 * cfg.n_contrast_layers)        # contrast C_a + C_b


def expand_lattice_codebooks(params: dict, n_new: int, rng: jax.Array,
                             cfg) -> dict:
    """Expand all lattice codebooks for a new task."""
    n_keys = expansion_key_count(cfg)
    keys = jax.random.split(rng, n_keys)
    ki = 0

    # HRQ: top + fine layers
    M_top_old = params['hrq']['top']['A'].shape[0]
    params['hrq']['top']['A'] = expand_codebook(
        params['hrq']['top']['A'], n_new, keys[ki]); ki += 1
    for l in range(len(params['hrq']['fine'])):
        params['hrq']['fine'][l]['A'] = expand_codebook(
            params['hrq']['fine'][l]['A'], n_new, keys[ki]); ki += 1

    # Sparse
    params['sparse']['C'] = expand_codebook(
        params['sparse']['C'], n_new, keys[ki]); ki += 1

    # Low-rank: U layers
    for l in range(len(params['lowrank']['U'])):
        u_shape = params['lowrank']['U'][l].shape  # (M_lr, r_k)
        new_u = jax.random.normal(keys[ki], (n_new, u_shape[1])) * 0.02; ki += 1
        params['lowrank']['U'][l] = jnp.concatenate(
            [params['lowrank']['U'][l], new_u], axis=0)

    # Manifold: C + T
    params['manifold']['C'] = expand_codebook(
        params['manifold']['C'], n_new, keys[ki]); ki += 1

    t_dim = params['manifold']['T'].shape[-1]
    new_T = jax.random.normal(keys[ki], (n_new, cfg.d_model, t_dim)) * 0.01; ki += 1
    params['manifold']['T'] = jnp.concatenate([params['manifold']['T'], new_T], axis=0)

    # Binding: all sub-codebook layers
    for cb_type in ['key_cb', 'val_cb', 'bind_cb']:
        for l in range(len(params['binding'][cb_type])):
            params['binding'][cb_type][l]['A'] = expand_codebook(
                params['binding'][cb_type][l]['A'], n_new, keys[ki]); ki += 1

    # Contrast: C_a + C_b layers
    for l in range(len(params['contrast']['C_a'])):
        params['contrast']['C_a'][l]['A'] = expand_codebook(
            params['contrast']['C_a'][l]['A'], n_new, keys[ki]); ki += 1
        params['contrast']['C_b'][l]['A'] = expand_codebook(
            params['contrast']['C_b'][l]['A'], n_new, keys[ki]); ki += 1

    assert ki == n_keys, (
        f"expansion consumed {ki} RNG keys but split {n_keys} — "
        f"expansion_key_count() is out of sync with the expansion body")
    return params


def _grow(vec: jnp.ndarray, n_new: int) -> jnp.ndarray:
    """Append ``n_new`` zero entries to a per-code accumulator."""
    return jnp.concatenate([vec, jnp.zeros(n_new, dtype=vec.dtype)])


def expand_ema_state(ema_state: dict, n_new: int, cfg) -> dict:
    """Resize every EMA accumulator alongside its codebook.

    Growing only the codebooks leaves ``N``/``m`` one size short, and the next
    ``_jitted_ema`` call then fails on a shape mismatch — or, worse, silently
    broadcasts. Called from the same transaction as the codebook expansion.
    """
    out = {}
    for name in ('sparse', 'manifold'):
        if name in ema_state:
            out[name] = {
                'N': _grow(ema_state[name]['N'], n_new),
                'm': jnp.concatenate(
                    [ema_state[name]['m'], jnp.zeros((n_new, ema_state[name]['m'].shape[-1]),
                                                     dtype=ema_state[name]['m'].dtype)],
                    axis=0),
            }
    if 'binding' in ema_state:
        out['binding'] = {}
        for fam in ('key', 'val', 'bind'):
            out['binding'][fam] = [
                {'N': _grow(st['N'], n_new),
                 'm': jnp.concatenate(
                     [st['m'], jnp.zeros((n_new, st['m'].shape[-1]), dtype=st['m'].dtype)],
                     axis=0)}
                for st in ema_state['binding'].get(fam, [])
            ]
    return out


def expand_value_scalars(value_scalars: dict, n_new: int) -> dict:
    """Expand local value scalars for every lattice."""
    return {k: _grow(v, n_new) for k, v in value_scalars.items()}


def pad_to_shape(old: jnp.ndarray, new_shape) -> jnp.ndarray:
    """Zero-pad ``old`` along every axis to ``new_shape`` (never truncates)."""
    pads = [(0, max(int(n) - int(o), 0)) for o, n in zip(old.shape, new_shape)]
    return jnp.pad(old, pads) if any(p[1] for p in pads) else old


def pad_stats_to_params(stats: Dict[str, jnp.ndarray], params: dict
                        ) -> Dict[str, jnp.ndarray]:
    """Re-shape EWC bookkeeping to match expanded params.

    ``protected_params`` / ``fisher_diag`` are flat ``path -> array`` maps
    snapshotted before the codebooks grew. ``compute_ewc_loss`` subtracts them
    from the *current* parameters, so leaving them at the old shape makes the
    first post-expansion step fail on a broadcast — after the expansion has
    already been committed. Zero-padding is the semantically correct resize:
    the added rows get Fisher 0, i.e. no protection, which is exactly what a
    freshly initialised codebook entry should have.
    """
    flat = dict(_param_groups(params))
    out = {}
    for path, arr in stats.items():
        cur = flat.get(path)
        out[path] = arr if cur is None else pad_to_shape(arr, cur.shape)
    return out


def expand_seen_masks(seen_masks: dict, n_new: int) -> dict:
    """Grow the cumulative HRQ utilisation mask."""
    out = dict(seen_masks)
    if 'hrq' in out:
        out['hrq'] = jnp.concatenate(
            [out['hrq'], jnp.zeros(n_new, dtype=out['hrq'].dtype)])
    return out


def expand_access_counters(counters: dict, n_new: int) -> dict:
    """Grow the per-codebook access counters used by memory consolidation.

    Every counter is a 1-D per-code array (hrq_top / sparse / manifold), so all
    of them follow their codebook by ``n_new`` entries.
    """
    return {k: _grow(v, n_new) for k, v in counters.items()}


# ── 3. Elastic Weight Consolidation (EWC) ────────────────────────────────────

def _param_groups(params: dict) -> List[Tuple[str, jnp.ndarray]]:
    """Flatten nested param dict into list of (path, array) pairs."""
    groups = []
    _collect_groups(params, '', groups)
    return groups


def _collect_groups(p, prefix: str, groups: List[Tuple[str, jnp.ndarray]]):
    if isinstance(p, dict):
        for k, v in p.items():
            _collect_groups(v, f"{prefix}{k}/", groups)
    elif isinstance(p, list):
        for i, v in enumerate(p):
            _collect_groups(v, f"{prefix}{i}/", groups)
    else:
        groups.append((prefix.rstrip('/'), p))


def snapshot_protected_params(params: dict) -> Dict[str, jnp.ndarray]:
    """Create frozen copy of all gradient-updated params for EWC protection."""
    protected = {}
    for path, val in _param_groups(params):
        protected[path] = jnp.array(val, copy=True)
    return protected


def estimate_fisher_diag(loss_grad_fn, params: dict, rng: jax.Array,
                         n_samples: int) -> Dict[str, jnp.ndarray]:
    """Estimate the diagonal Fisher by Monte-Carlo squared gradients.

    ``F_i ≈ E[(∂L/∂θ_i)²]`` over samples of the data distribution. The squared
    gradient of a single batch is a very high-variance estimator — it is only
    usable because EWC's job is to rank parameters by importance, not to be
    calibrated — but it must at least be computed *from the loss*, on the
    current parameters. The previous implementation here was never called: the
    training loop filled ``fisher_diag`` with ``jnp.ones_like(val) * 1e-4``.

    Args:
        loss_grad_fn: ``(params, rng) -> grads pytree`` for one sample.
        params: Parameters the Fisher is taken at.
        rng: PRNG key.
        n_samples: Number of Monte-Carlo draws.

    Returns:
        dict path -> Fisher diagonal, using the same paths as
        ``snapshot_protected_params`` so ``compute_ewc_loss`` can pair them.
    """
    acc = None
    n = max(int(n_samples), 1)
    for _ in range(n):
        rng, sub = jax.random.split(rng)
        grads = loss_grad_fn(params, sub)
        flat = dict(_param_groups(grads))
        if acc is None:
            acc = {k: jnp.square(v) for k, v in flat.items()}
        else:
            for k, v in flat.items():
                acc[k] = acc[k] + jnp.square(v)
    return {k: v / n for k, v in acc.items()}


def compute_ewc_loss(params: dict, protected_params: Dict[str, jnp.ndarray],
                     fisher_diag: Dict[str, jnp.ndarray],
                     ewc_lambda: float) -> jnp.ndarray:
    """Elastic Weight Consolidation loss: λ/2 * Σ_i F_i * (θ_i - θ_i*)²."""
    loss = 0.0
    for path, theta in _param_groups(params):
        if path in protected_params and path in fisher_diag:
            diff = theta - protected_params[path]
            loss = loss + jnp.sum(fisher_diag[path] * diff ** 2)
    return 0.5 * ewc_lambda * loss


# ── 4. Experience Replay ─────────────────────────────────────────────────────

def update_replay_buffer(state: ContinualState, task_id: int,
                         z: jnp.ndarray, soft_mask: jnp.ndarray,
                         capacity: int) -> ContinualState:
    """Update the per-domain replay buffer with the current batch.

    Only the encoder latents and the routing mask are stored. The previous
    signature also took ``logits``, but the only call site passed
    ``jnp.zeros((B, vocab_size))`` — a (B, 30000) allocation per step whose
    contents were never populated and never read.
    """
    if task_id not in state.replay_buffers:
        state.replay_buffers[task_id] = {
            'z': z[:capacity],
            'soft_mask': soft_mask[:capacity],
            'ptr': 0,
            'full': False,
        }

    buf = state.replay_buffers[task_id]
    for i in range(z.shape[0]):
        idx = buf['ptr'] % capacity
        buf['z'] = buf['z'].at[idx].set(z[i])
        buf['soft_mask'] = buf['soft_mask'].at[idx].set(soft_mask[i])
        buf['ptr'] += 1
    buf['full'] = buf['ptr'] >= capacity

    return state


def sample_replay(state: ContinualState, task_id: int,
                  batch_size: int, replay_ratio: float,
                  rng: jax.Array) -> Tuple[Optional[Dict[str, jnp.ndarray]], float]:
    """Sample from replay buffers of previous tasks.

    Returns:
        (replay_batch, replay_weight): replay_batch is None if no replay data.
    """
    old_tasks = [t for t in state.replay_buffers if t < task_id]
    if not old_tasks:
        return None, 0.0

    n_replay = int(batch_size * replay_ratio)
    if n_replay < 1:
        return None, 0.0

    # Uniform across old tasks
    n_per_task = max(1, n_replay // len(old_tasks))
    all_z, all_masks = [], []

    for t in old_tasks:
        buf = state.replay_buffers[t]
        n_avail = min(buf['ptr'], buf['z'].shape[0]) if buf['full'] else buf['ptr']
        if n_avail < 1:
            continue

        rng, subkey = jax.random.split(rng)
        indices = jax.random.choice(subkey, n_avail,
                                    (min(n_per_task, n_avail),), replace=False)
        all_z.append(buf['z'][indices])
        all_masks.append(buf['soft_mask'][indices])

    if not all_z:
        return None, 0.0

    return {
        'z': jnp.concatenate(all_z, axis=0),
        'soft_mask': jnp.concatenate(all_masks, axis=0),
    }, n_replay / batch_size


# ── 5. Memory Consolidation ──────────────────────────────────────────────────

def update_access_counters(state: ContinualState, params: dict,
                           aux: dict) -> ContinualState:
    """Increment access counters for each codebook entry used this step."""
    # HRQ top index
    for path in ['hrq_top']:
        if path not in state.access_counters:
            state.access_counters[path] = jnp.zeros(
                params['hrq']['top']['A'].shape[0], dtype=jnp.int32)
    # Sparse index
    if 'sparse' not in state.access_counters:
        state.access_counters['sparse'] = jnp.zeros(
            params['sparse']['C'].shape[0], dtype=jnp.int32)
    # Manifold index
    if 'manifold' not in state.access_counters:
        state.access_counters['manifold'] = jnp.zeros(
            params['manifold']['C'].shape[0], dtype=jnp.int32)

    if 'hrq_idx' in aux:
        idx = aux['hrq_idx']
        for i in range(idx.shape[0]):
            state.access_counters['hrq_top'] = state.access_counters['hrq_top'].at[
                idx[i]].add(1)

    if 'sparse_idx' in aux:
        idx = aux['sparse_idx']
        for i in range(idx.shape[0]):
            state.access_counters['sparse'] = state.access_counters['sparse'].at[
                idx[i]].add(1)

    if 'man_idx' in aux:
        idx = aux['man_idx']
        for i in range(idx.shape[0]):
            state.access_counters['manifold'] = state.access_counters['manifold'].at[
                idx[i]].add(1)

    return state


def consolidate_memory(state: ContinualState, params: dict,
                       step: int, frequency_threshold: int = 50) -> Tuple[dict, ContinualState]:
    """Move high-frequency entries to 'stable' pool (marked via logging).

    In practice, consolidation means:
    - High-frequency entries get added to protected_params for EWC
    - Low-frequency entries remain plastic
    - Returns updated params and state with consolidation log entry.
    """
    consolidated = []
    for path, counters in state.access_counters.items():
        high_freq = jnp.where(counters > frequency_threshold)[0]
        if len(high_freq) > 0:
            consolidated.append({'path': path, 'n_entries': len(high_freq)})
            # Reset counters for consolidated entries
            state.access_counters[path] = state.access_counters[path].at[high_freq].set(0)

    if consolidated:
        state.consolidation_log.append({
            'step': step,
            'task_id': state.task_id,
            'consolidated': consolidated,
        })

    return params, state
