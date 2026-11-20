"""自我观察（黑匣子）— 纯记录，零判断。

职责：每步推理完成后，将原始轨迹数据无损写入环形缓冲区。
不评判重要性，不做筛选，不参与推理。纯粹的数据记录仪。

类比：赛车上的黑匣子。不判断哪些操作重要，只记录所有数据。
赛后分析时，黑匣子提供精确的回放能力。

Usage:
    from train.observability import ObservabilityRecorder

    obs = ObservabilityRecorder(raw_capacity=1000)
    obs.record_step(step=42, soft_mask=..., route_idx=..., ...)
    trace = obs.get_trace(42)
    obs.print_summary()
"""
import time
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from statistics import median
from typing import Optional, List, Dict, Any, Callable


# ─── Record schema ──────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Single inference step record — immutable snapshot.

    All fields are optional; only provided fields are stored.
    Vectors (z_q, z) are stored as numpy arrays if record_vectors=True.
    """
    # Meta
    step: int
    timestamp: float = 0.0
    step_time_ms: float = 0.0
    source: int = 0  # 0=external, 1=internal (因果主体源标签)

    # Routing
    soft_mask: Optional[np.ndarray] = None   # (n_lattices,)
    route_idx: Optional[int] = None

    # Retrieval path
    hrq_top_sim: Optional[float] = None
    hrq_idx: Optional[List[int]] = None
    sparse_idx: Optional[int] = None
    man_idx: Optional[int] = None

    # Self
    self_mode: Optional[int] = None
    world_dev: Optional[float] = None

    # Safety
    safety_margin: Optional[float] = None
    value_signals: Optional[np.ndarray] = None  # (n_lattices,)
    is_safe: bool = True
    violated_law: Optional[int] = None

    # Convergence
    convergence_diff: Optional[float] = None
    convergence_entropy: Optional[float] = None
    n_retrievals: int = 0

    # Prediction / narrative signal (e.md §3 rule 1)
    prediction_error: Optional[float] = None

    # C engine DAG trace
    dag_nodes: Optional[List[Dict[str, Any]]] = None
    conflict_source: Optional[str] = None
    conflict_detail: Optional[str] = None

    # Vectors (e.md §2: z_cur → inference → z_next)
    z_cur: Optional[np.ndarray] = None  # (d,) state entering the step
    z_next: Optional[np.ndarray] = None  # (d,) state leaving the step
    # Historical names kept for callers that predate the R16 schema.
    z_q: Optional[np.ndarray] = None  # (d,)
    z: Optional[np.ndarray] = None    # (d,)


# ─── Ring buffer ────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-capacity circular buffer — O(1) append, O(capacity) memory.

    Once full, oldest entries are silently overwritten (no blurring).
    Perfect for short-term trajectory storage.
    """
    def __init__(self, capacity: int):
        assert capacity > 0, "capacity must be > 0"
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0
        self._count = 0

    def append(self, item: Any):
        self._buffer[self._head] = item
        self._head = (self._head + 1) % self.capacity
        self._count += 1

    def __getitem__(self, idx: int):
        if len(self) == 0:
            raise IndexError("RingBuffer is empty")
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        # Items are stored from (head-count) to (head-1), wrapping as needed
        pos = (self._head - self._count + idx) % self.capacity
        return self._buffer[pos]

    def __len__(self):
        return min(self._count, self.capacity)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_by_step(self, step: int) -> Optional[Any]:
        for record in self:
            if record and record.step == step:
                return record
        return None

    def pop_by_step(self, step: int) -> Optional[Any]:
        """Remove and return a specific step (for promotion)."""
        for i in range(len(self)):
            record = self[i]
            if record and record.step == step:
                self._buffer[(self._head - len(self) + i) % self.capacity] = None
                return record
        return None

    def clear(self):
        self._buffer = [None] * self.capacity
        self._head = 0
        self._count = 0

    @property
    def utilization(self):
        return len(self) / self.capacity

    def to_list(self):
        return [r for r in self if r is not None]


# ─── Observability Recorder (pure recording, no judgment) ──────────────────

class ObservabilityRecorder:
    """自我观察（黑匣子）— 纯记录，零判断。

    唯一职责：记录原始推理轨迹到环形缓冲区。
    不做重要性评估，不做筛选，不干扰推理。
    """
    def __init__(self, raw_capacity: int = 1000,
                 record_vectors: bool = True,
                 record_every_n: int = 1,
                 long_term_capacity: int = 10000,
                 long_term_every: int = 100,
                 anomaly_oversample: int = 3,
                 anomaly_convergence_diff: float = 0.5):
        """
        Args:
            raw_capacity: 环形缓冲区容量（步数）。
            record_vectors: 是否存储 z/z_cur/z_next 向量（增加内存）。
            record_every_n: 采样率（1=每步都记录）。
            long_term_capacity: 长期降采样环容量（e.md §2: 10000）。
            long_term_every: 均匀降采样间隔（每 N 步取一条）。
            anomaly_oversample: 异常事件写入长期环的额外份数（e.md §2:
                conflicts/safety/novel routing ×3）。异常事件本身总会被写入，
                不等待均匀采样点。
            anomaly_convergence_diff: ``convergence_diff`` 超过该值视为异常，
                与 ``is_safe=False`` / ``conflict_source`` 一样参与 ×N 过采样。
        """
        self.ring = RingBuffer(raw_capacity)
        self.long_term = RingBuffer(max(1, int(long_term_capacity)))
        self.record_vectors = record_vectors
        self.record_every_n = max(1, record_every_n)
        self.long_term_every = max(1, int(long_term_every))
        self.anomaly_oversample = max(1, int(anomaly_oversample))
        self.anomaly_convergence_diff = float(anomaly_convergence_diff)
        self._last_step = -1
        self._long_term_counter = 0
        self._start_time = time.time()

    def should_record(self, step: int) -> bool:
        return step % self.record_every_n == 0

    def record_step(self, step: int, **kwargs) -> Optional[StepRecord]:
        """Record one step's trajectory data.

        Call this AFTER inference step completes.
        O(1) — fixed memory append, no allocation spikes.

        Args:
            step: Step number.
            **kwargs: Fields matching StepRecord attributes.

        Returns:
            The created StepRecord, or None if skipped (sampling).
        """
        if not self.should_record(step):
            return None

        record_kwargs = {'step': step, 'timestamp': time.time()}
        for key in ['step_time_ms', 'source', 'soft_mask', 'route_idx', 'hrq_top_sim',
                     'hrq_idx', 'sparse_idx', 'man_idx', 'self_mode',
                     'world_dev', 'safety_margin', 'value_signals', 'is_safe',
                     'violated_law', 'convergence_diff', 'convergence_entropy',
                     'n_retrievals', 'prediction_error', 'dag_nodes',
                     'conflict_source', 'conflict_detail']:
            if key in kwargs:
                record_kwargs[key] = kwargs[key]

        if self.record_vectors:
            for key in ['z_q', 'z', 'z_cur', 'z_next']:
                if key in kwargs and kwargs[key] is not None:
                    val = kwargs[key]
                    if hasattr(val, 'numpy'):
                        val = np.asarray(val)
                    if isinstance(val, np.ndarray):
                        record_kwargs[key] = val.copy() if val.ndim > 0 else val

        record = StepRecord(**record_kwargs)
        self.ring.append(record)
        self._last_step = step
        self._store_long_term(record)
        return record

    # ── Long-term downsampling (uniform + anomaly oversampling) ──

    def _is_anomaly(self, record: StepRecord) -> bool:
        if not record.is_safe:
            return True
        if record.conflict_source is not None:
            return True
        if (record.convergence_diff is not None
                and float(record.convergence_diff)
                > self.anomaly_convergence_diff):
            return True
        return False

    def _store_long_term(self, record: StepRecord) -> None:
        """Write a uniform sample, or an anomaly ×oversample.

        Anomalies are never delayed until the next uniform tick: a safety
        conflict may be the last thing a run ever records.
        """
        if self._is_anomaly(record):
            for _ in range(self.anomaly_oversample):
                self.long_term.append(record)
            self._long_term_counter = 0
            return
        self._long_term_counter += 1
        if self._long_term_counter >= self.long_term_every:
            self.long_term.append(record)
            self._long_term_counter = 0

    # ── Query ──

    def get_trace(self, step: int) -> Optional[StepRecord]:
        return self.ring.get_by_step(step)

    def get_recent_traces(self, n: int = 10) -> List[StepRecord]:
        return self.ring.to_list()[-n:]

    def get_traces_by_range(self, start_step: int,
                            end_step: int) -> List[StepRecord]:
        return [r for r in self.ring if r and start_step <= r.step <= end_step]

    def get_traces_by_route(self, route_idx: int) -> List[StepRecord]:
        return [r for r in self.ring if r and r.route_idx == route_idx]

    def get_long_term_traces(self, n: int = 0) -> List[StepRecord]:
        """Downsampled history (oldest → newest), or its last ``n`` entries."""
        records = self.long_term.to_list()
        return records[-n:] if n > 0 else records

    # ── Summary ──

    def print_summary(self) -> None:
        n = len(self.ring)
        if n == 0:
            print("[OBS] No records yet.")
            return

        unsafe = sum(1 for r in self.ring if r and not r.is_safe)
        steps = [r.step for r in self.ring if r]

        print(f"\n{'=' * 46}")
        print(f"  自我观察 (Observability Recorder)")
        print(f"{'=' * 46}")
        print(f"  Records:          {n}")
        print(f"  Capacity:         {self.ring.capacity}")
        print(f"  Utilization:      {self.ring.utilization:.1%}")
        print(f"  Step range:       {min(steps)} — {max(steps)}")
        print(f"  Unsafe steps:     {unsafe}")
        print(f"  Elapsed:          {time.time() - self._start_time:.1f}s")
        print(f"{'=' * 46}\n")

    # ── Export ──

    def export_json(self, path: str, max_records: int = 0,
                    include_vectors: bool = False) -> None:
        records = self.ring.to_list()
        if max_records > 0:
            records = records[-max_records:]

        data = []
        for r in records:
            if r is None:
                continue
            d = asdict(r)
            for k, v in list(d.items()):
                if isinstance(v, np.ndarray):
                    if include_vectors and k in ('z_q', 'z', 'z_cur',
                                                 'z_next'):
                        d[k] = v.tolist()
                    elif k in ('soft_mask', 'value_signals') and v is not None:
                        d[k] = v.tolist()
                    else:
                        d[k] = None
                elif isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[k] = float(v)
            data.append(d)

        with open(path, 'w') as f:
            json.dump({'n_records': len(data), 'records': data},
                       f, indent=2, default=str)
        print(f"[OBS] Exported {len(data)} raw records → {path}")

    def clear(self) -> None:
        self.ring.clear()
        self.long_term.clear()
        self._long_term_counter = 0
        self._last_step = -1


# ─── 5% overhead measurement (R16) ──────────────────────────────────────────

def measure_overhead(fn_with: Callable[[], Any],
                     fn_without: Callable[[], Any],
                     runs: int = 20,
                     min_baseline_s: float = 1e-5) -> float:
    """Median wall-clock overhead of ``fn_with`` over ``fn_without``.

    R16 explicitly rejects summing ``step_time_ms`` (that *is* inference
    time). Measure both callables with ``time.perf_counter``, alternating the
    order each run so a drift in machine load cannot bias one side, and take
    the median of each. The ratio is ``(with - without) / without``.

    ``min_baseline_s`` is a timer-noise floor (10 µs). Without it, two no-op
    callables differ by fractions of a microsecond and the ratio becomes
    pure measurement noise. For real inference (milliseconds) the floor is
    negligible.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    with_times: List[float] = []
    without_times: List[float] = []
    for i in range(runs):
        order = ((fn_with, with_times), (fn_without, without_times))
        if i % 2:
            order = tuple(reversed(order))
        for fn, bucket in order:
            t0 = time.perf_counter()
            fn()
            bucket.append(time.perf_counter() - t0)

    m_with = median(with_times)
    m_without = median(without_times)
    denominator = max(m_without, float(min_baseline_s))
    ratio = (m_with - m_without) / denominator
    return float(max(0.0, ratio))


# ─── External verifier polling ──────────────────────────────────────────────

# Imported lazily: this module must stay usable in the pure-stdlib verifier
# test path, and LOCK is just an int. The poller never imports JAX.
LOCK_STATE = 2


class VerifierPoller:
    """Poll an external verifier's ``state()`` from an inference/training loop.

    The local verifier form has no parent-side polling thread: the docs require
    the inference loop to call ``state()`` so LOCK/OBSERVE notifications are
    delivered on the next step. This adapter is that call site.

    Args:
        verifier: any object with ``state() -> int`` (normally
            :class:`train.external_verifier.VerifierProcess`).
        every_n: poll every N steps (1 = every step).
        on_lock: optional callable invoked once when LOCK is first observed.
    """

    def __init__(self, verifier, every_n: int = 1,
                 on_lock: Optional[Callable[[dict], None]] = None):
        self.verifier = verifier
        self.every_n = max(1, int(every_n))
        self.on_lock = on_lock
        self.last_state = 0
        self.n_polls = 0
        self._lock_reported = False

    def poll(self, step: int = 0) -> int:
        if step % self.every_n != 0:
            return self.last_state
        self.n_polls += 1
        state = int(self.verifier.state())
        self.last_state = state
        if state == LOCK_STATE and not self._lock_reported:
            self._lock_reported = True
            if self.on_lock is not None:
                self.on_lock({
                    'type': 'LOCK',
                    'source': 'verifier_poller',
                    'step': int(step),
                    'state': state,
                })
        return state

    def close(self) -> None:
        stop = getattr(self.verifier, 'stop', None)
        if stop is not None:
            stop()


__all__ = [
    'StepRecord',
    'RingBuffer',
    'ObservabilityRecorder',
    'measure_overhead',
    'VerifierPoller',
]
