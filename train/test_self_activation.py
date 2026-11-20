"""SelfActivationState — d.md §3 / R14.

This module manages the *writability* of the cognitive self lattice.  It does
not implement the self lattice itself; ``C_self`` stays in
``train/self_lattice.py`` (rewritten in Task 13).

Run from repo root:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_self_activation.py -v
"""
import json

import pytest

from train.self_activation import SelfActivationState, State


def _state(tmp_path, **kwargs):
    kwargs.setdefault("log_path", tmp_path / "self_activation.log")
    return SelfActivationState(**kwargs)


def _read_log(path):
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_path_a_requires_no_authorization(tmp_path):
    s = _state(tmp_path)
    s.transition(State.TEMP_UNFROZEN)
    assert s.state == State.TEMP_UNFROZEN
    # Session ends: the self lattice returns to frozen without an operator.
    s.transition(State.DEFINED_FROZEN)
    assert s.state == State.DEFINED_FROZEN


def test_path_a_records_user_instruction(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    s.transition(State.TEMP_UNFROZEN, authorization="user asked")
    record = _read_log(log)[-1]
    assert record["path"] == "A"
    assert record["to"] == State.TEMP_UNFROZEN.value
    assert record["authorization"] == "user asked"
    assert record["operator"] is None


def test_path_c_requires_operator_authorization(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION)
    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.state == State.FULL_ACTIVATION
    record = _read_log(tmp_path / "self_activation.log")[-1]
    assert record["path"] == "C"
    assert record["operator"] == "alice"


def test_path_b_requires_danger_flag(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.DANGER_FLAGGED)
    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    assert s.state == State.DANGER_FLAGGED
    # Being flagged is not authorization to activate fully.
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION, danger_flag=True)
    s.transition(State.FULL_ACTIVATION, danger_flag=True,
                 authorization="operator:bob")
    assert s.state == State.FULL_ACTIVATION


def test_full_activation_cannot_be_autonomous(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION, danger_flag=True)
    with pytest.raises(PermissionError):
        # A self-generated note is not an operator authorization string.
        s.transition(State.FULL_ACTIVATION, authorization="self generated")
    assert s.state == State.DEFINED_FROZEN


def test_denied_transition_is_append_logged(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION)
    record = _read_log(log)[-1]
    assert record["result"] == "denied"
    assert record["from"] == State.DEFINED_FROZEN.value
    assert record["to"] == State.FULL_ACTIVATION.value
    assert "reason" in record


def test_log_is_append_only(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    s.transition(State.TEMP_UNFROZEN, authorization="user asked")
    s.transition(State.DEFINED_FROZEN)
    first = log.read_text(encoding="utf-8").splitlines()
    assert len(first) == 2

    s.transition(State.TEMP_UNFROZEN, authorization="user asked again")
    second = log.read_text(encoding="utf-8").splitlines()
    assert second[:2] == first, "existing log lines must never be rewritten"

    records = [json.loads(line) for line in second]
    assert records[0]["from"] == State.DEFINED_FROZEN.value
    assert records[0]["to"] == State.TEMP_UNFROZEN.value
    assert records[0]["timestamp"] > 0.0
    assert "timestamp_iso" in records[0]
    assert records[2]["authorization"] == "user asked again"


def test_identity_core_is_read_only_and_unchanged(tmp_path):
    source = [1.0, 2.0, 3.0]
    s = SelfActivationState(
        log_path=tmp_path / "self_activation.log",
        identity_core=source,
    )
    source[0] = 99.0  # mutating the constructor argument must not leak in
    before = s.identity_core
    assert before == (1.0, 2.0, 3.0)
    assert isinstance(before, tuple)

    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.identity_core is before
    assert s.identity_core == (1.0, 2.0, 3.0)


def test_danger_flag_cannot_be_attached_to_path_a(tmp_path):
    assert State.OBSERVE is State.DANGER_FLAGGED
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.TEMP_UNFROZEN, danger_flag=True)
    assert s.state == State.DEFINED_FROZEN


def test_identity_score_provider_is_injected(tmp_path):
    s = _state(tmp_path, identity_score_provider=lambda: 0.75)
    assert s.identity_score() == 0.75

    fallback = _state(tmp_path)
    assert fallback.identity_score() == 0.0
    fallback.note_identity_score(0.4)
    assert fallback.identity_score() == 0.4


def test_full_activation_can_only_be_ended_by_operator(tmp_path):
    s = _state(tmp_path)
    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    with pytest.raises(PermissionError):
        s.transition(State.DEFINED_FROZEN)
    s.transition(State.DEFINED_FROZEN, authorization="operator:alice")
    assert s.state == State.DEFINED_FROZEN


def test_unknown_state_is_rejected(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        s.transition("not-a-state")
