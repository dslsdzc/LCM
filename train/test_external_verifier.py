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


def test_signature_file_is_written(tmp_path):
    trace_dir = tmp_path / "verifier"
    with _running(tmp_path) as v:
        path = trace_dir / "verifier.signature.json"
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["algorithm"] == "sha256"
        assert payload["signature"] == v.signature
        assert payload["not_a_production_boundary"] is True


def test_stop_cleans_up_process_and_shared_memory(tmp_path):
    v = _verifier(tmp_path)
    v.start()
    name = v.shared_name
    assert name
    assert v.is_running

    v.stop()
    assert not v.is_running
    v.stop()  # idempotent

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


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


def test_inline_mode_supports_tests_without_a_process(tmp_path):
    v = _verifier(tmp_path, mode="inline", observe_limit=2)
    assert v.state() == NORMAL
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
