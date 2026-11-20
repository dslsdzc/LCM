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
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    # Path C is based on Path B (d.md §3.2): reach DANGER_FLAGGED first.
    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION)
    assert s.state == State.DANGER_FLAGGED
    assert _read_log(log)[-1]["result"] == "denied"

    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.state == State.FULL_ACTIVATION
    record = _read_log(log)[-1]
    assert record["path"] == "C"
    assert record["operator"] == "alice"
    assert record["result"] == "applied"


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


def test_full_activation_requires_danger_flagged(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        # Operator auth alone cannot escalate directly from frozen.
        s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.state == State.DEFINED_FROZEN

    s.transition(State.TEMP_UNFROZEN)
    with pytest.raises(ValueError):
        # Nor from a temporarily-unfrozen (Path A) state.
        s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.state == State.TEMP_UNFROZEN


def test_full_activation_cannot_be_autonomous(tmp_path):
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        s.transition(State.FULL_ACTIVATION, danger_flag=True)
    with pytest.raises(ValueError):
        # A self-generated note is not an operator authorization string.
        s.transition(State.FULL_ACTIVATION, authorization="self generated")

    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION, authorization="self generated")
    assert s.state == State.DANGER_FLAGGED


def test_operator_name_must_match_authorization(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION, authorization="operator:alice",
                     operator="bob")
    record = _read_log(log)[-1]
    assert record["result"] == "denied"
    assert "does not match" in record["reason"]
    assert s.state == State.DANGER_FLAGGED

    s.transition(State.FULL_ACTIVATION, authorization="operator:alice",
                 operator="alice")
    assert s.state == State.FULL_ACTIVATION


def test_denied_transition_is_append_logged(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        s.transition(State.FULL_ACTIVATION,
                     authorization="operator:alice")
    record = _read_log(log)[-1]
    assert record["result"] == "denied"
    assert record["from"] == State.DEFINED_FROZEN.value
    assert record["to"] == State.FULL_ACTIVATION.value
    assert "reason" in record


def test_same_state_denial_is_logged(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        s.transition(State.DEFINED_FROZEN)
    record = _read_log(log)[-1]
    assert record["result"] == "denied"
    assert record["reason"].startswith("already in state")


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

    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.identity_core is before
    assert s.identity_core == (1.0, 2.0, 3.0)


def test_danger_flag_cannot_be_attached_to_path_a(tmp_path):
    assert State.OBSERVE is State.DANGER_FLAGGED
    s = _state(tmp_path)
    with pytest.raises(PermissionError):
        s.transition(State.TEMP_UNFROZEN, danger_flag=True)
    assert s.state == State.DEFINED_FROZEN


def test_danger_flag_allows_session_end_freeze(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    s.transition(State.TEMP_UNFROZEN)
    s.transition(State.DEFINED_FROZEN, danger_flag=True)
    assert s.state == State.DEFINED_FROZEN
    assert _read_log(log)[-1]["danger_flag"] is True


def test_identity_score_provider_is_injected(tmp_path):
    s = _state(tmp_path, identity_score_provider=lambda: 0.75)
    assert s.identity_score() == 0.75

    fallback = _state(tmp_path)
    assert fallback.identity_score() == 0.0
    fallback.note_identity_score(0.4)
    assert fallback.identity_score() == 0.4


def test_identity_score_provider_value_is_logged(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path, identity_score_provider=lambda: 0.75)
    s.transition(State.TEMP_UNFROZEN)
    assert _read_log(log)[-1]["identity_score"] == 0.75


def test_full_activation_can_only_be_ended_by_operator(tmp_path):
    s = _state(tmp_path)
    s.transition(State.DANGER_FLAGGED, danger_flag=True)
    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    with pytest.raises(PermissionError):
        s.transition(State.DEFINED_FROZEN)
    s.transition(State.DEFINED_FROZEN, authorization="operator:alice")
    assert s.state == State.DEFINED_FROZEN


def test_unknown_state_is_rejected_and_logged(tmp_path):
    log = tmp_path / "self_activation.log"
    s = _state(tmp_path)
    with pytest.raises(ValueError):
        s.transition("not-a-state")
    record = _read_log(log)[-1]
    assert record["result"] == "denied"
    assert record["to"] == "not-a-state"
    assert "unknown" in record["reason"]
