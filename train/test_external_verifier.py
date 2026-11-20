"""External verifier process (d.md §2 / R14).

Focused, pure-standard-library tests. No JAX or numpy is imported by the
module under test; a subprocess test asserts that explicitly.

Run from repo root:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_external_verifier.py -v
"""
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path

import pytest

from train.external_verifier import LOCK, NORMAL, OBSERVE, VerifierProcess

REPO_ROOT = Path(__file__).resolve().parents[1]


def _verifier(tmp_path, **overrides):
    kwargs = dict(
        dim=8,
        threshold_low=1.0,
        threshold_high=5.0,
        observe_limit=2,
        shared_name=None,
        trace_dir=str(tmp_path / "verifier"),
    )
    kwargs.update(overrides)
    return VerifierProcess(**kwargs)


@contextmanager
def _running(tmp_path, **overrides):
    v = _verifier(tmp_path, **overrides)
    v.start()
    try:
        yield v
    finally:
        v.stop()


def test_state_machine_reaches_lock(tmp_path):
    with _running(tmp_path, threshold_high=2.0) as v:
        for _ in range(3):
            v.submit([10.0] * 8)
        assert v.state() == LOCK


def test_observe_requires_consecutive_frames(tmp_path):
    with _running(tmp_path, threshold_high=100.0, observe_limit=3) as v:
        v.submit([10.0] * 8)
        assert v.state() == OBSERVE
        v.submit([10.0] * 8)
        assert v.state() == OBSERVE
        v.submit([10.0] * 8)
        assert v.state() == LOCK


def test_lock_latches_against_normal_frames(tmp_path):
    with _running(tmp_path, observe_limit=1) as v:
        v.submit([10.0] * 8)
        assert v.state() == LOCK
        # A normal frame must not release the lock.
        v.submit([0.0] * 8)
        assert v.state() == LOCK
        # Repeated polling must not clear it either.
        assert v.state() == LOCK


def test_child_lock_latches_even_if_ctrl_says_normal(tmp_path):
    with _running(tmp_path, observe_limit=1) as v:
        v.submit([10.0] * 8)
        assert v.state() == LOCK
        record = v._read_ctrl()
        assert record[7] == LOCK
        normal = (record[0], record[1], record[2], record[3],
                  0.0, 0.0, 0, NORMAL, record[8])
        v._read_ctrl = lambda timeout=1.0: normal
        assert v.state() == LOCK
        assert v.state() == LOCK


def test_observe_recovers_to_normal_and_logs_false_positive(tmp_path):
    trace_dir = tmp_path / "verifier"
    with _running(tmp_path, observe_limit=5) as v:
        v.submit([10.0] * 8)
        assert v.state() == OBSERVE
        v.submit([0.0] * 8)
        assert v.state() == NORMAL

    lines = (trace_dir / "false_positives.log").read_text(
        encoding="utf-8").splitlines()
    assert lines, "NORMAL recovery must record a false positive"
    record = json.loads(lines[-1])
    assert record["event"] == "false_positive"
    assert record["observations"] == 1
    assert record["peak_score"] >= 10.0


def test_observe_saves_full_trace(tmp_path):
    trace_dir = tmp_path / "verifier"
    frame = [3.0, -4.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    with _running(tmp_path, observe_limit=5) as v:
        v.submit(frame)
        assert v.state() == OBSERVE

    lines = (trace_dir / "observe_trace.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert lines, "OBSERVE must save a trace record"
    record = json.loads(lines[-1])
    assert record["event"] == "observe"
    assert record["features"] == frame
    assert record["score"] >= 4.0
    assert record["signature"]


def test_observe_callback_fires_once_per_episode(tmp_path):
    observed = []
    with _running(tmp_path, observe_limit=5,
                  observe_callback=observed.append) as v:
        v.submit([10.0] * 8)
        assert len(observed) == 1
        v.submit([10.0] * 8)
        assert len(observed) == 1, "one callback per OBSERVE episode"
        v.submit([0.0] * 8)  # false positive -> NORMAL
        v.submit([10.0] * 8)  # new episode
        assert len(observed) == 2
    assert observed[0]["type"] == "OBSERVE"
    assert observed[1]["level"] == "WARNING"


def test_lock_calls_callback_once_and_writes_structured_alert(tmp_path):
    trace_dir = tmp_path / "verifier"
    alerts = []
    with _running(tmp_path, observe_limit=1,
                  alert_callback=alerts.append) as v:
        v.submit([10.0] * 8)
        assert v.state() == LOCK
        assert v.state() == LOCK

    assert len(alerts) == 1, "LOCK callback must fire exactly once"
    alert = alerts[0]
    assert alert["level"] == "FATAL"
    assert alert["source"] == "external_verifier"
    assert alert["conflict_source"] == "external_verifier"
    assert alert["type"] == "LOCK"
    assert alert["conflict_type"] == "LOCK"
    assert alert["detail"]
    assert alert["timestamp"] > 0.0
    assert alert["signature"]

    lines = (trace_dir / "alerts.log").read_text(
        encoding="utf-8").splitlines()
    assert lines
    logged = json.loads(lines[-1])
    assert logged["source"] == "external_verifier"
    assert logged["type"] == "LOCK"


def test_non_finite_input_is_valid_json(tmp_path):
    trace_dir = tmp_path / "verifier"
    with _running(tmp_path, mode="inline", observe_limit=1) as v:
        assert v.submit([float("inf")] + [0.0] * 7) == LOCK

    lines = (trace_dir / "observe_trace.jsonl").read_text(
        encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["score"] == "non-finite"
    assert record["features"][0] == "non-finite"
    # The strict parser would reject Infinity/NaN; sentinel must be a string.
    assert isinstance(record["score"], str)


def test_signature_file_is_written(tmp_path):
    trace_dir = tmp_path / "verifier"
    with _running(tmp_path) as v:
        path = trace_dir / "verifier.signature.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["algorithm"] == "sha256"
        assert payload["signature"] == v.signature
        assert payload["not_a_production_boundary"] is True


def test_signature_mismatch_rejected_unless_force_resign(tmp_path):
    first = _verifier(tmp_path)
    first.start()
    first.stop()

    second = _verifier(tmp_path, threshold_low=2.0)
    with pytest.raises(RuntimeError):
        second.start()
    assert not second.is_running

    third = _verifier(tmp_path, threshold_low=2.0, force_resign=True)
    third.start()
    third.stop()


def test_stop_cleans_up_process_and_shared_memory(tmp_path):
    v = _verifier(tmp_path)
    v.start()
    name = v.shared_name
    try:
        assert name
        assert v.is_running
    finally:
        v.stop()
    assert not v.is_running
    v.stop()  # idempotent

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_stop_preserves_unobserved_lock(tmp_path):
    with _running(tmp_path, observe_limit=1) as v:
        v.submit([10.0] * 8, wait=False)
        deadline = time.monotonic() + 2.0
        record = None
        while time.monotonic() < deadline:
            record = v._read_ctrl()
            if record is not None and record[7] == LOCK:
                break
            time.sleep(0.01)
        assert record is not None and record[7] == LOCK

    # stop() snapshotted before unlinking; a later state() must still see LOCK.
    assert v.state() == LOCK


def test_crashed_verifier_fails_closed_to_lock(tmp_path):
    trace_dir = tmp_path / "verifier"
    alerts = []
    v = _verifier(tmp_path, alert_callback=alerts.append)
    v.start()
    try:
        v._process.terminate()
        v._process.join(5.0)
        assert v.state() == LOCK
        assert v.state() == LOCK
    finally:
        v.stop()

    assert len(alerts) == 1
    assert "not running" in alerts[0]["detail"]
    assert alerts[0]["type"] == "LOCK"
    lines = (trace_dir / "alerts.log").read_text(
        encoding="utf-8").splitlines()
    assert lines


def test_read_error_latches_fail_closed(tmp_path):
    alerts = []
    with _running(tmp_path, alert_callback=alerts.append) as v:
        good = v._read_ctrl()
        calls = {"n": 0}

        def flaky(timeout=1.0):
            calls["n"] += 1
            return None if calls["n"] == 1 else good

        v._read_ctrl = flaky
        assert v.state() == LOCK
        assert v.state() == LOCK
        assert v.state() == LOCK
    assert calls["n"] == 1, "latched LOCK must not read the recovered ctrl"
    assert len(alerts) == 1
    assert "unreadable" in alerts[0]["detail"]
    assert "not running" not in alerts[0]["detail"]


def test_submit_wait_timeout_fails_closed(tmp_path):
    alerts = []
    with _running(tmp_path, alert_callback=alerts.append) as v:
        good = v._read_ctrl()
        # Keep reporting done=0 so the acknowledgement never arrives.
        stalled = (good[0], good[1], good[2], 0, 0.0, 0.0, 0, NORMAL, 0)
        v._read_ctrl = lambda timeout=1.0: stalled
        assert v.submit([0.0] * 8, wait=True, timeout=0.05) == LOCK
        assert v.state() == LOCK
    assert len(alerts) == 1
    assert "did not acknowledge" in alerts[0]["detail"]


def test_invalid_ctrl_fails_closed(tmp_path):
    with _running(tmp_path) as v:
        v._read_ctrl = lambda timeout=1.0: (
            b"BAD!", 999, 0, 0, 0.0, 0.0, 0, 99, 0)
        assert v.state() == LOCK
        assert v.state() == LOCK


def test_inline_mode_supports_tests_without_a_process(tmp_path):
    v = _verifier(tmp_path, mode="inline", observe_limit=2)
    with pytest.raises(RuntimeError):
        v.state()
    v.start()
    try:
        v.submit([10.0] * 8)
        assert v.state() == OBSERVE
        v.submit([10.0] * 8)
        assert v.state() == LOCK
        assert v.is_running
    finally:
        v.stop()
    assert not v.is_running
    assert v.state() == LOCK


def test_restart_resets_per_run_state(tmp_path):
    alerts = []
    v = _verifier(tmp_path, observe_limit=1, alert_callback=alerts.append)
    v.start()
    try:
        v.submit([10.0] * 8)
        assert v.state() == LOCK
        v.stop()

        v.start()
        assert v.state() == NORMAL
        v.submit([0.0] * 8)
        assert v.state() == NORMAL
        v.submit([10.0] * 8)
        assert v.state() == LOCK
    finally:
        v.stop()
    assert len(alerts) == 2, "each session gets its own LOCK callback"


def test_submit_rejects_wrong_dimension(tmp_path):
    v = _verifier(tmp_path, mode="inline")
    v.start()
    try:
        with pytest.raises(ValueError):
            v.submit([1.0] * 7)
    finally:
        v.stop()


def test_module_does_not_import_jax_or_numpy():
    code = (
        "import sys\n"
        "import train.external_verifier\n"
        "loaded = [m for m in ('jax', 'numpy') if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"external_verifier must stay pure stdlib, loaded: {result.stdout!r}"
    )
