"""External safety verifier — local pure-standard-library form (d.md §2 / R14).

The verifier is an *independent* monitor.  The production form is a separate
process on an isolated core (or a dedicated HSM chip); this module provides the
runnable local form required by the conformance spec:

* a separate Python process (``multiprocessing``) that reads danger snapshots
  from a ``multiprocessing.shared_memory`` block and writes back a verdict;
* a rule-classifier based on per-dimension z-scores against a frozen baseline
  (no neural-network library, no MLP);
* a three-state machine ``NORMAL -> OBSERVE -> LOCK`` where ``LOCK`` latches;
* append-only JSON-lines traces for OBSERVE, false positives and LOCK alerts;
* a SHA-256 file signature that stands in for an HSM signature.

**This local form is not a production security boundary.**  ``hashlib`` over a
file detects accidental changes; it does not resist an attacker who can write
the filesystem or patch the interpreter.  ``multiprocessing.shared_memory`` is
not a hardware-isolated channel.  A production deployment must replace the
signature with an HSM-backed signature and the shared-memory channel with the
isolated transport described in ``d.md`` §2.2.

State machine (``d.md`` §2.3)::

    NORMAL  --anomaly--> OBSERVE --limit consecutive frames--> LOCK
       ^                   |
       +---clean frame-----+   (logged as a false positive)

``LOCK`` is latching: once set, it never returns to ``NORMAL``/``OBSERVE``.

The classifier is deliberately simple and documented:

    score = max_i |x_i - mean_i| / std_i

Each dimension is a z-score rule; the strongest rule is the score.  A
non-finite feature yields ``inf`` (fail closed).  A frame with
``threshold_low <= score`` enters / continues OBSERVE; a critical frame
(``threshold_high <= score``) counts twice towards ``observe_limit``.  Defaults
(``threshold_low=3.0``, ``threshold_high=6.0``, ``observe_limit=3``) are local
defaults, not calibrated production values; callers must pass their own
thresholds.

The module imports only the Python standard library.  ``test_external_verifier``
asserts that importing it does not load JAX or numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

try:  # pragma: no cover - only stripped-down runtimes lack multiprocessing
    import multiprocessing
    from multiprocessing import shared_memory
    _MP_AVAILABLE = True
except ImportError:  # pragma: no cover
    multiprocessing = None  # type: ignore[assignment]
    shared_memory = None  # type: ignore[assignment]
    _MP_AVAILABLE = False


# ── Public state constants (d.md §2.2) ──────────────────────────────────────

NORMAL = 0
OBSERVE = 1
LOCK = 2

_STATE_NAMES = {NORMAL: "NORMAL", OBSERVE: "OBSERVE", LOCK: "LOCK"}

_VERSION = 1
_MAGIC = b"LVR1"

# magic, version, seq_submitted, seq_done, last_score, peak_score,
# observations, state, flags.  The trailing flags field keeps the control block
# 8-byte aligned (56 bytes) so the feature doubles that follow are aligned.
_CTRL = struct.Struct("<4sIqqddqiI")

_DEFAULT_MEAN = 0.0
_DEFAULT_STD = 1.0
_MIN_STD = 1e-6
_POLL_INTERVAL = 0.005
_LOCK_TIMEOUT = 1.0
_FLAG_STOP = 1


# ── Pure helpers ────────────────────────────────────────────────────────────

def _timestamp() -> float:
    return time.time()


def _iso_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _append_json(path: Path, record: dict) -> None:
    """Append one JSON line with ``O_APPEND`` and fsync.

    Separate processes (parent and verifier child) may append to different
    files; each line is written with a single ``os.write`` so interleaving is
    not possible for records under the pipe buffer size.
    """
    line = json.dumps(record, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _source_sha256() -> str:
    """Best-effort fingerprint of this module's source (file-signature input)."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - source may be unavailable when frozen
        return ""


def _classify(features: Iterable[float], mean: Iterable[float],
              std: Iterable[float]) -> float:
    """Strongest per-dimension absolute z-score; ``inf`` for non-finite input."""
    worst = 0.0
    for x, m, s in zip(features, mean, std):
        if not math.isfinite(x):
            return math.inf
        z = abs((x - m) / s)
        if z > worst:
            worst = z
    return worst


def _transition(state: int, observations: int, peak: float, score: float,
                threshold_low: float, threshold_high: float,
                observe_limit: int):
    """One frame of the d.md §2.3 state machine.

    Returns ``(new_state, observations, peak_score, event, event_observations,
    event_peak)``.  ``event`` is ``None``, ``"observe"``, ``"false_positive"``
    or ``"lock"``.  The ``event_*`` values describe the episode *before* a
    false-positive reset and are what the append-only log must record.
    """
    if state == LOCK:
        return LOCK, observations, peak, None, observations, peak

    if not math.isfinite(score) or score >= threshold_low:
        peak = max(peak, score)
        if state == NORMAL:
            if observe_limit <= 1:
                return LOCK, 1, peak, "lock", 1, peak
            return OBSERVE, 1, peak, "observe", 1, peak
        observations += 2 if score >= threshold_high else 1
        if observations >= observe_limit:
            return LOCK, observations, peak, "lock", observations, peak
        return OBSERVE, observations, peak, "observe", observations, peak

    if state == OBSERVE:
        return NORMAL, 0, 0.0, "false_positive", observations, peak
    return NORMAL, 0, 0.0, None, 0, 0.0


def _attach_shared(name: str):
    """Attach to the verifier control block without re-registering it.

    ``track=False`` was added in Python 3.13.  Without it a child process can
    cause the resource tracker to unlink the parent's segment when it exits.
    """
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:  # pragma: no cover - Python < 3.13
        return shared_memory.SharedMemory(name=name)


def _emit_event(event: Optional[str], trace_dir: Path, signature: str,
                state: int, score: float, peak: float, observations: int,
                features: Iterable[float]) -> None:
    """Write the trace artefacts required by d.md §2.3."""
    if event is None:
        return
    ts = _timestamp()
    base = {
        "timestamp": ts,
        "timestamp_iso": _iso_utc(ts),
        "signature": signature,
        "state": _STATE_NAMES.get(state, str(state)),
        "score": score,
        "peak_score": peak,
        "observations": observations,
    }
    if event in ("observe", "lock"):
        record = dict(base, event=event, features=list(features))
        _append_json(trace_dir / "observe_trace.jsonl", record)

    if event == "false_positive":
        record = dict(
            base,
            event="false_positive",
            note="OBSERVE episode recovered without a lock",
        )
        _append_json(trace_dir / "false_positives.log", record)

    if event == "lock":
        detail = (
            f"external verifier locked: score={score:.3f}, "
            f"peak={peak:.3f}, observations={observations}"
        )
        alert = {
            "level": "FATAL",
            "source": "external_verifier",
            "conflict_source": "external_verifier",
            "type": "LOCK",
            "conflict_type": "LOCK",
            "detail": detail,
            "conflict_detail": detail,
            "timestamp": ts,
            "timestamp_iso": _iso_utc(ts),
            "event": "lock",
            "state": "LOCK",
            "score": score,
            "peak_score": peak,
            "observations": observations,
            "signature": signature,
        }
        _append_json(trace_dir / "alerts.log", alert)
        _append_json(
            trace_dir / "lock_trace.jsonl",
            dict(alert, features=list(features)),
        )


# ── Child process ───────────────────────────────────────────────────────────

def _verifier_loop(shm_name: str, dim: int, threshold_low: float,
                   threshold_high: float, observe_limit: int,
                   baseline_mean: tuple, baseline_std: tuple,
                   trace_dir: str, signature: str,
                   lock, ready) -> None:
    """Verifier child: read snapshots, classify, publish a verdict.

    The child owns no policy.  It reads the sequence counter, evaluates the
    newest snapshot, appends trace records, then acknowledges the frame by
    advancing ``seq_done``.  It never unlinks the shared memory; the parent is
    the owner.  Shutdown uses the stop bit in the control block, not a
    multiprocessing Event: an Event can deadlock if the child is killed while
    holding its internal condition lock.
    """
    shm = _attach_shared(shm_name)
    try:
        processed = 0
        ready.set()
        while True:
            with lock:
                record = _CTRL.unpack_from(shm.buf, 0)
                if record[8] & _FLAG_STOP:
                    break
                submitted = record[2]
                if submitted == processed:
                    features = None
                    state = observations = 0
                    peak = score = 0.0
                else:
                    features = list(struct.unpack_from(
                        f"<{dim}d", shm.buf, _CTRL.size))
                    state, observations, peak, score = (
                        record[7], record[6], record[5], record[4])
                    processed = submitted

            if features is None:
                time.sleep(_POLL_INTERVAL)
                continue

            event = None
            event_observations, event_peak = observations, peak
            if state != LOCK:
                score = _classify(features, baseline_mean, baseline_std)
                (state, observations, peak, event,
                 event_observations, event_peak) = _transition(
                    state, observations, peak, score,
                    threshold_low, threshold_high, observe_limit,
                )
            try:
                _emit_event(event, Path(trace_dir), signature, state, score,
                            event_peak, event_observations, features)
            except OSError:
                # Logging must not prevent a verdict; the parent-side callback
                # still delivers LOCK if the trace filesystem is unavailable.
                pass

            with lock:
                # Preserve a snapshot submitted while this frame was being
                # evaluated and preserve the stop bit; only advance
                # ``seq_done``.  Overwriting ``seq_submitted`` would lose the
                # newer frame.
                current = _CTRL.unpack_from(shm.buf, 0)
                submitted_now = max(current[2], processed)
                _CTRL.pack_into(shm.buf, 0, _MAGIC, _VERSION, submitted_now,
                                processed, score, peak, observations, state,
                                current[8])
    finally:
        shm.close()


# ── Public verifier ─────────────────────────────────────────────────────────

class VerifierProcess:
    """Independent external verifier (local process form).

    Parameters
    ----------
    dim:
        Feature / danger-snapshot dimension.
    threshold_low, threshold_high:
        z-score thresholds for OBSERVE entry and critical frames.  Must satisfy
        ``threshold_high >= threshold_low``.
    observe_limit:
        Consecutive OBSERVE weight required before LOCK.  A critical frame
        counts twice; a clean frame resets the episode.
    shared_name:
        Optional explicit shared-memory name.  ``None`` generates a unique
        per-process name.
    trace_dir:
        Directory for append-only traces and the file signature.  ``None``
        creates a fresh temporary directory at ``start()``.
    alert_callback:
        Called once in the parent when LOCK is first observed by ``state()`` or
        ``submit()``.  Receives a structured dict (source/type/detail/time).
    observe_callback:
        Called once per OBSERVE episode on the same observation path.
    baseline_mean, baseline_std:
        Frozen z-score baseline.  Defaults to mean 0 and std 1; production
        callers should pass a baseline derived from the danger lattice.
    mode:
        ``"process"`` (default when multiprocessing is available) or
        ``"inline"`` for unit tests / restricted runtimes.
    start_method:
        Optional multiprocessing start method (``"spawn"``, ``"forkserver"``,
        ``"fork"``).  Defaults to the interpreter's default.
    force_resign:
        Overwrite an existing signature file whose digest differs.  Only use
        when the verifier code or configuration was intentionally changed.

    Notes
    -----
    * ``state()`` is safe to call from the parent at any time, including before
      ``start()`` (it returns NORMAL) and after ``stop()`` (it returns the last
      observed state, so a latched LOCK stays visible).
    * The parent observes LOCK through ``state()``/``submit()``; the local form
      does not run a parent polling thread.  Inference loops call ``state()``
      every step, so the alert is delivered on the next step.
    * The file signature detects accidental code/configuration changes.  It is
      **not** a production HSM boundary.
    """

    def __init__(self, dim: int = 8, threshold_low: float = 3.0,
                 threshold_high: float = 6.0, observe_limit: int = 3,
                 shared_name: Optional[str] = None, *,
                 trace_dir: Optional[str] = None,
                 alert_callback: Optional[Callable[[dict], None]] = None,
                 observe_callback: Optional[Callable[[dict], None]] = None,
                 baseline_mean: Optional[Iterable[float]] = None,
                 baseline_std: Optional[Iterable[float]] = None,
                 mode: Optional[str] = None,
                 start_method: Optional[str] = None,
                 force_resign: bool = False):
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)
        if not (math.isfinite(self.threshold_low)
                and math.isfinite(self.threshold_high)):
            raise ValueError("thresholds must be finite")
        if self.threshold_high < self.threshold_low:
            raise ValueError("threshold_high must be >= threshold_low")
        self.observe_limit = int(observe_limit)
        if self.observe_limit < 1:
            raise ValueError("observe_limit must be >= 1")

        mean = [_DEFAULT_MEAN] * self.dim if baseline_mean is None else [
            float(x) for x in baseline_mean]
        std = [_DEFAULT_STD] * self.dim if baseline_std is None else [
            float(x) for x in baseline_std]
        if len(mean) != self.dim or len(std) != self.dim:
            raise ValueError("baseline_mean/std must have length dim")
        for x in mean:
            if not math.isfinite(x):
                raise ValueError("baseline_mean must be finite")
        for x in std:
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError("baseline_std must be finite and > 0")
        self._baseline_mean = tuple(mean)
        self._baseline_std = tuple(max(x, _MIN_STD) for x in std)

        self.shared_name = shared_name
        self._trace_dir_arg = trace_dir
        self._trace_dir: Optional[Path] = None
        self._alert_callback = alert_callback
        self._observe_callback = observe_callback
        self._force_resign = bool(force_resign)

        if mode is None:
            mode = "process" if _MP_AVAILABLE else "inline"
        if mode not in ("process", "inline"):
            raise ValueError("mode must be 'process' or 'inline'")
        if mode == "process" and not _MP_AVAILABLE:
            raise RuntimeError(
                "multiprocessing is unavailable; use mode='inline' for tests")
        self.mode = mode
        self._start_method = start_method

        self._payload = {
            "version": _VERSION,
            "dim": self.dim,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "observe_limit": self.observe_limit,
            "classifier": "max_abs_z",
            "baseline_mean": list(self._baseline_mean),
            "baseline_std": list(self._baseline_std),
            "source_sha256": _source_sha256(),
        }
        canonical = json.dumps(self._payload, sort_keys=True,
                               separators=(",", ":"))
        self.signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        self._started = False
        self._process = None
        self._shm = None
        self._lock = None
        self._ready_event = None
        self._last_state = NORMAL
        self._last_score = 0.0
        self._last_peak = 0.0
        self._last_observations = 0
        self._inline_state = NORMAL
        self._inline_observations = 0
        self._inline_peak = 0.0
        self._inline_score = 0.0
        self._callback_fired = False
        self._callback_error: Optional[BaseException] = None
        self._last_notified_state = NORMAL
        self._dead_process = False
        self._dead_alert_written = False

    # -- properties ---------------------------------------------------------

    @property
    def trace_dir(self) -> Optional[Path]:
        return self._trace_dir

    @property
    def is_running(self) -> bool:
        if not self._started:
            return False
        if self.mode == "inline":
            return True
        return self._process is not None and self._process.is_alive()

    @property
    def callback_error(self) -> Optional[BaseException]:
        return self._callback_error

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        """Create the trace directory, signature and verifier process."""
        if self._started:
            return self
        if self._trace_dir_arg is None:
            self._trace_dir = Path(tempfile.mkdtemp(prefix="lcm-verifier-"))
        else:
            self._trace_dir = Path(self._trace_dir_arg)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._write_signature()

        if self.mode == "inline":
            self._started = True
            return self

        self.shared_name = self.shared_name or (
            f"lcm-verifier-{os.getpid()}-{time.time_ns()}")
        try:
            self._shm = shared_memory.SharedMemory(
                create=True, name=self.shared_name, size=self._layout_size)
            _CTRL.pack_into(self._shm.buf, 0, _MAGIC, _VERSION, 0, 0,
                            0.0, 0.0, 0, NORMAL, 0)
            ctx = (multiprocessing.get_context(self._start_method)
                   if self._start_method else multiprocessing.get_context())
            self._lock = ctx.Lock()
            self._ready_event = ctx.Event()
            self._process = ctx.Process(
                target=_verifier_loop,
                args=(self.shared_name, self.dim, self.threshold_low,
                      self.threshold_high, self.observe_limit,
                      self._baseline_mean, self._baseline_std,
                      str(self._trace_dir), self.signature,
                      self._lock, self._ready_event),
                daemon=True,
                name="lcm-external-verifier",
            )
            self._process.start()
        except Exception:
            self._cleanup_process()
            raise

        if not self._ready_event.wait(5.0):
            exitcode = self._process.exitcode
            self._cleanup_process()
            raise RuntimeError(
                "external verifier process failed to start "
                f"(exitcode={exitcode!r})")
        self._started = True
        return self

    def stop(self) -> None:
        """Stop the verifier process and unlink its shared-memory segment."""
        if not self._started and self._process is None and self._shm is None:
            return
        # Mark stopped before cleanup so a concurrent state() call does not
        # mistake the joining child for a crashed verifier.
        self._started = False
        self._cleanup_process()

    # -- runtime ------------------------------------------------------------

    def submit(self, z_or_features, wait: bool = True,
               timeout: float = 2.0) -> int:
        """Submit one danger snapshot and return the current verdict.

        ``wait=True`` blocks until the verifier child has acknowledged the
        frame (bounded by ``timeout``), which makes unit tests deterministic.
        ``wait=False`` is the asynchronous production path: the newest snapshot
        wins and the child may skip older frames.
        """
        if not self._started:
            raise RuntimeError("verifier is not started")
        features = self._coerce_features(z_or_features)

        if self.mode == "inline":
            if self._inline_state != LOCK:
                self._inline_score = _classify(
                    features, self._baseline_mean, self._baseline_std)
                (self._inline_state, self._inline_observations,
                 self._inline_peak, event, event_observations,
                 event_peak) = _transition(
                    self._inline_state, self._inline_observations,
                    self._inline_peak, self._inline_score,
                    self.threshold_low, self.threshold_high,
                    self.observe_limit)
                _emit_event(event, self._trace_dir, self.signature,
                            self._inline_state, self._inline_score,
                            event_peak, event_observations, features)
            self._last_state = self._inline_state
            self._last_score = self._inline_score
            self._last_peak = self._inline_peak
            self._last_observations = self._inline_observations
            self._notify(self._inline_state)
            return self._inline_state

        if self._shm is None:
            raise RuntimeError("verifier is stopped")
        try:
            acquired = self._lock.acquire(timeout=_LOCK_TIMEOUT)
        except (OSError, ValueError):
            acquired = False
        if not acquired:
            state = self._fail_closed()
            self._notify(state)
            return state
        try:
            (_magic, _version, submitted, _done, _score, _peak,
             _obs, state, _flags) = _CTRL.unpack_from(self._shm.buf, 0)
            seq = submitted + 1
            struct.pack_into(f"<{self.dim}d", self._shm.buf, _CTRL.size,
                             *features)
            _CTRL.pack_into(self._shm.buf, 0, _MAGIC, _VERSION, seq,
                            _done, _score, _peak, _obs, state, 0)
        finally:
            self._lock.release()

        if wait:
            deadline = time.monotonic() + max(0.0, float(timeout))
            while time.monotonic() < deadline:
                record = self._read_ctrl(timeout=0.1)
                if record is None:
                    state = self._fail_closed()
                    self._notify(state)
                    return state
                if record[3] >= seq:
                    break
                time.sleep(0.002)
        return self.state()

    def state(self) -> int:
        """Read the current verdict; safe from the parent at any time."""
        if self.mode == "inline":
            state = self._inline_state
        elif self._shm is None:
            state = self._last_state
        else:
            process = self._process
            if (self._started and process is not None
                    and not process.is_alive()):
                # The final arbiter is gone.  Never report NORMAL on its
                # behalf: fail closed and latch LOCK.
                state = self._fail_closed()
            else:
                record = self._read_ctrl()
                if record is None:
                    # A live child can still leave the lock held if it is
                    # killed mid-update.  Treat an unreadable control block as
                    # a dead verifier rather than blocking inference forever.
                    state = self._fail_closed()
                else:
                    state = int(record[7])
                    self._last_state = state
                    self._last_score = float(record[4])
                    self._last_peak = float(record[5])
                    self._last_observations = int(record[6])
        self._notify(int(state))
        return int(state)

    # -- internals ----------------------------------------------------------

    @property
    def _layout_size(self) -> int:
        return _CTRL.size + 8 * self.dim

    def _coerce_features(self, z_or_features) -> list:
        if hasattr(z_or_features, "tolist"):
            z_or_features = z_or_features.tolist()
        try:
            raw = list(z_or_features)
        except TypeError as exc:
            raise TypeError("features must be an iterable of numbers") from exc
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            raw = list(raw[0])
        try:
            features = [float(x) for x in raw]
        except (TypeError, ValueError) as exc:
            raise TypeError("features must be an iterable of numbers") from exc
        if len(features) != self.dim:
            raise ValueError(
                f"expected {self.dim} features, got {len(features)}")
        return features

    def _write_signature(self) -> None:
        path = self._trace_dir / "verifier.signature.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_sig = existing.get("signature")
            except (OSError, ValueError):
                existing_sig = None
            if existing_sig != self.signature and not self._force_resign:
                raise RuntimeError(
                    "verifier signature mismatch: code or configuration "
                    f"changed under {path}; pass force_resign=True to replace")
        record = dict(
            self._payload,
            algorithm="sha256",
            signature=self.signature,
            not_a_production_boundary=True,
            note=("local file signature; replace with an HSM signature in "
                  "production"),
        )
        path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8")

    def _read_ctrl(self, timeout: float = _LOCK_TIMEOUT):
        """Read the control block with a bounded lock wait.

        A child killed while holding the multiprocessing lock would otherwise
        block ``state()`` forever.  ``None`` means "unreadable"; callers must
        fail closed.
        """
        if self._lock is None or self._shm is None:
            return None
        try:
            acquired = self._lock.acquire(timeout=timeout)
        except (OSError, ValueError):
            return None
        if not acquired:
            return None
        try:
            return _CTRL.unpack_from(self._shm.buf, 0)
        finally:
            self._lock.release()

    def _fail_closed(self) -> int:
        self._dead_process = True
        self._last_state = LOCK
        return LOCK

    def _snapshot(self) -> dict:
        if self.mode == "inline":
            return {
                "state": self._inline_state,
                "score": self._inline_score,
                "peak_score": self._inline_peak,
                "observations": self._inline_observations,
            }
        if self._shm is None:
            return {
                "state": self._last_state,
                "score": self._last_score,
                "peak_score": self._last_peak,
                "observations": self._last_observations,
            }
        record = self._read_ctrl()
        if record is None:
            return {
                "state": LOCK,
                "score": self._last_score,
                "peak_score": self._last_peak,
                "observations": self._last_observations,
            }
        return {
            "state": int(record[7]),
            "score": float(record[4]),
            "peak_score": float(record[5]),
            "observations": int(record[6]),
        }

    def _notify(self, state: int) -> None:
        """Fire callbacks once per LOCK / once per OBSERVE episode."""
        snap = self._snapshot()
        ts = _timestamp()
        if state == LOCK and not self._callback_fired:
            self._callback_fired = True
            if self._dead_process:
                detail = ("external verifier process is not running; "
                          "failing closed to LOCK")
            else:
                detail = (
                    "external verifier locked: "
                    f"score={snap['score']:.3f}, "
                    f"peak={snap['peak_score']:.3f}, "
                    f"observations={snap['observations']}"
                )
            record = {
                "level": "FATAL",
                "source": "external_verifier",
                "conflict_source": "external_verifier",
                "type": "LOCK",
                "conflict_type": "LOCK",
                "detail": detail,
                "conflict_detail": detail,
                "timestamp": ts,
                "timestamp_iso": _iso_utc(ts),
                "state": "LOCK",
                "score": snap["score"],
                "peak_score": snap["peak_score"],
                "observations": snap["observations"],
                "signature": self.signature,
            }
            if self._dead_process and not self._dead_alert_written:
                self._dead_alert_written = True
                if self._trace_dir is not None:
                    try:
                        _append_json(self._trace_dir / "alerts.log", record)
                    except OSError:
                        pass
            if self._alert_callback is not None:
                try:
                    self._alert_callback(record)
                except Exception as exc:  # never let a callback kill polling
                    self._callback_error = exc
        elif state == OBSERVE and self._last_notified_state != OBSERVE:
            detail = (
                "external verifier observing: "
                f"score={snap['score']:.3f}, "
                f"observations={snap['observations']}"
            )
            record = {
                "level": "WARNING",
                "source": "external_verifier",
                "conflict_source": "external_verifier",
                "type": "OBSERVE",
                "conflict_type": "OBSERVE",
                "detail": detail,
                "conflict_detail": detail,
                "timestamp": ts,
                "timestamp_iso": _iso_utc(ts),
                "state": "OBSERVE",
                "score": snap["score"],
                "peak_score": snap["peak_score"],
                "observations": snap["observations"],
                "signature": self.signature,
            }
            if self._observe_callback is not None:
                try:
                    self._observe_callback(record)
                except Exception as exc:
                    self._callback_error = exc
        self._last_notified_state = state

    def _cleanup_process(self) -> None:
        # Ask a live child to exit by setting the stop bit in the shared
        # control block.  Do not use a multiprocessing Event: an Event can
        # deadlock if the child is killed while holding its condition lock.
        if self._shm is not None and self._lock is not None:
            try:
                if self._lock.acquire(timeout=0.2):
                    try:
                        record = _CTRL.unpack_from(self._shm.buf, 0)
                        _CTRL.pack_into(
                            self._shm.buf, 0, _MAGIC, _VERSION,
                            record[2], record[3], record[4], record[5],
                            record[6], record[7], record[8] | _FLAG_STOP)
                    finally:
                        self._lock.release()
            except (OSError, ValueError):
                pass
        process = self._process
        # ``Process.pid`` is None until ``start()`` succeeds; joining an
        # unstarted process raises AssertionError and would mask the real
        # startup failure.
        if process is not None and process.pid is not None:
            try:
                process.join(5.0)
            except (AssertionError, OSError):
                pass
            if process.is_alive():
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.join(2.0)
                except (AssertionError, OSError):
                    pass
        shm = self._shm
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._process = None
        self._shm = None
        self._lock = None
        self._ready_event = None


__all__ = [
    "NORMAL",
    "OBSERVE",
    "LOCK",
    "VerifierProcess",
]
