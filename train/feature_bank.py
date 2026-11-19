"""Feature bank: ring buffer of observed vectors and dead-code replacement (R3).

The sparse and contrast codebooks can lose entries: an entry that is never
selected stops receiving EMA/gradient signal, and the codebook slowly stops
covering the data. The feature bank keeps recent observations and replaces dead
entries with real data.

Two replacement policies, per R3/R21:

    sparse    random draw from the bank
    contrast  farthest-point draw from the bank (b.md §4.6)

Both run on the host, every ``bank_check_interval`` steps, outside jit: the
replacement count is dynamic and must not become a traced shape.

``state`` in the replace functions is polymorphic on purpose:

    * an EMA state dict ``{'N', 'm', 'last_used'}`` — the replaced rows are
      cold-started (N/m zeroed, last_used = step), so stale statistics cannot
      bias the new vector;
    * a bare ``last_used`` array — only the usage clock is updated. This is the
      form used by codebooks that are not EMA-managed (contrast).
"""
import numpy as np


class FeatureBank:
    """Fixed-capacity ring buffer of ``dim``-dimensional host vectors."""

    def __init__(self, capacity, dim, seed=0):
        self.capacity = int(capacity)
        self.dim = int(dim)
        if self.capacity <= 0 or self.dim <= 0:
            raise ValueError(
                f"FeatureBank needs positive capacity/dim, got "
                f"{capacity}x{dim}")
        self.data = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self.size = 0
        self.pos = 0
        # A fixed seed makes dead-code replacement reproducible across a
        # save→load→resume cycle; the bank itself is checkpointed too.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    @property
    def vectors(self):
        """The valid part of the buffer, oldest first."""
        if self.size < self.capacity:
            return self.data[:self.size]
        # pos is the next write slot, i.e. the oldest entry once the ring is
        # full. Rotate so callers get chronological order.
        return np.concatenate([self.data[self.pos:], self.data[:self.pos]])

    def add(self, z):
        """Append one ``(d,)`` vector or a ``(B, d)`` batch, wrapping around."""
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]
        if z.shape[-1] != self.dim:
            raise ValueError(
                f"FeatureBank.add expected dim {self.dim}, got {z.shape[-1]}")
        z = z.reshape(-1, self.dim)
        for row in z:
            self.data[self.pos] = row
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def dead_indices(self, last_used, step, t_dead):
        """Rows whose last assignment is more than ``t_dead`` steps old."""
        last = np.asarray(last_used)
        return np.nonzero(np.asarray(step) - last > t_dead)[0].astype(np.int32)

    def _random_replacement(self, dead_vector):
        if self.size == 0:
            return None
        candidates = self.vectors
        distances = np.linalg.norm(candidates - dead_vector, axis=-1)
        # A vector identical to the dead one does not revive it. Exclude exact
        # matches unless the bank contains nothing else.
        eligible = np.nonzero(distances > 1e-12)[0]
        pool = candidates[eligible] if eligible.size else candidates
        return pool[self.rng.integers(len(pool))]

    def _farthest_replacement(self, dead_vector):
        if self.size == 0:
            return None
        candidates = self.vectors
        distances = np.linalg.norm(candidates - dead_vector, axis=-1)
        return candidates[int(np.argmax(distances))]

    @staticmethod
    def _unpack(state):
        if isinstance(state, dict):
            return (np.asarray(state['last_used']),
                    np.asarray(state['N']), np.asarray(state['m']))
        return np.asarray(state), None, None

    @staticmethod
    def _repack(state, last_used, N, m):
        if isinstance(state, dict):
            return {'N': N, 'm': m, 'last_used': last_used}
        return last_used

    def _replace(self, C, state, step, t_dead, pick):
        C = np.array(C, dtype=np.float32, copy=True)
        last_used, N, m = self._unpack(state)
        last_used = np.array(last_used, copy=True)
        dead = self.dead_indices(last_used, step, t_dead)
        replaced = []
        for row in dead:
            replacement = pick(C[row])
            if replacement is None:
                break
            C[row] = replacement
            replaced.append(row)
        if replaced:
            last_used[replaced] = step
            if N is not None:
                N = np.array(N, copy=True)
                m = np.array(m, copy=True)
                N[replaced] = 0.0
                m[replaced] = 0.0
        return C, self._repack(state, last_used, N, m)

    def replace_dead_sparse(self, C, state, step, t_dead):
        """Replace dead rows with random bank vectors (cold-start stats)."""
        return self._replace(C, state, step, t_dead, self._random_replacement)

    def replace_dead_contrast(self, C, state, step, t_dead):
        """Replace dead rows with the farthest bank vector from the dead row."""
        return self._replace(C, state, step, t_dead, self._farthest_replacement)


__all__ = ['FeatureBank']
