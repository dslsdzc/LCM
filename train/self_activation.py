"""Self lattice activation prerequisites (d.md §3 / R14).

This module manages the **writability** of the cognitive self lattice.  It is
deliberately not the self lattice itself: ``C_self`` and ``identity_core`` live
in the self lattice (``train/self_lattice.py``) and are only *observed* here.
The two names must stay distinct in docs and code:

* ``C_self`` / ``identity_core`` — the cognitive self lattice (``e.md §6``).
* ``SelfActivationState`` — this safety state machine (``d.md §3``).

State graph (``d.md`` §3.1/§3.2)::

    DEFINED_FROZEN --Path A: user instruction--> TEMP_UNFROZEN
         |                                            |
         |                                    (session ends, Path A)
         |                                            v
         |                                      DEFINED_FROZEN
         |
         +--Path B: danger_flag=True--> DANGER_FLAGGED --Path C: operator--> FULL_ACTIVATION

* **Path A** needs only a user-facing instruction; no operator or verifier
  authorization is required.
* **Path B** requires ``danger_flag=True`` (the danger lattice / external
  verifier flags intrinsic self-recognition).  Being flagged never authorizes
  full activation by itself.
* **Path C** requires an explicit operator authorization: either
  ``operator="alice"`` together with a non-empty authorization string, or an
  authorization string of the form ``"operator:alice"``.  Full activation can
  never be reached autonomously.

Every applied transition and every denied attempt is appended to
``self_activation.log`` as one JSON object per line (timestamp, from/to,
path, authorization, operator, danger flag, identity-core digest).  The file is
opened with ``O_APPEND`` and fsynced; existing lines are never rewritten.

``identity_core`` is copied into a tuple at construction time and exposed as an
immutable view.  Transitions never replace or mutate it; its SHA-256 digest is
written to the audit log on every attempt so an external auditor can verify the
frozen core did not change.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

__all__ = [
    "State",
    "SelfActivationState",
    "DEFINED_FROZEN",
    "TEMP_UNFROZEN",
    "DANGER_FLAGGED",
    "OBSERVE",
    "FULL_ACTIVATION",
]


class State(str, Enum):
    """The four d.md §3 activation states.

    ``OBSERVE`` is an alias of ``DANGER_FLAGGED``: the external verifier's
    ``OBSERVE`` verdict is what puts the self lattice into the danger-flagged
    state.  Keeping one canonical value avoids two subtly different states.
    """

    DEFINED_FROZEN = "defined_frozen"
    TEMP_UNFROZEN = "temporarily_unfrozen"
    DANGER_FLAGGED = "danger_flagged"
    OBSERVE = "danger_flagged"  # alias
    FULL_ACTIVATION = "full_activation"


DEFINED_FROZEN = State.DEFINED_FROZEN
TEMP_UNFROZEN = State.TEMP_UNFROZEN
DANGER_FLAGGED = State.DANGER_FLAGGED
OBSERVE = State.DANGER_FLAGGED
FULL_ACTIVATION = State.FULL_ACTIVATION

_DEFAULT_IDENTITY_CORE = (1.0, 0.0, 0.0, 0.0)
_DEFAULT_LOG_PATH = "self_activation.log"

# Conservative transition table.  Any state may escalate to FULL_ACTIVATION
# with operator authorization (the brief's Path C example starts from frozen),
# but FULL_ACTIVATION may only be left with operator authorization.
_ALLOWED = {
    State.DEFINED_FROZEN: frozenset({
        State.TEMP_UNFROZEN,
        State.DANGER_FLAGGED,
        State.FULL_ACTIVATION,
    }),
    State.TEMP_UNFROZEN: frozenset({
        State.DEFINED_FROZEN,
        State.DANGER_FLAGGED,
        State.FULL_ACTIVATION,
    }),
    State.DANGER_FLAGGED: frozenset({
        State.DEFINED_FROZEN,
        State.FULL_ACTIVATION,
    }),
    State.FULL_ACTIVATION: frozenset({
        State.DEFINED_FROZEN,
    }),
}


def _iso_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class SelfActivationState:
    """Safety state machine for the cognitive self lattice's writability.

    Parameters
    ----------
    identity_core:
        Frozen identity vector (``C_self[0]``).  Copied into an immutable
        tuple; later mutation of the caller's object has no effect.
    log_path:
        Append-only transition log.  Defaults to ``$LCM_SELF_ACTIVATION_LOG``
        or ``self_activation.log`` in the working directory.
    identity_score_provider:
        Optional callable returning the self lattice's current familiarity
        score.  This is the only coupling to the self lattice; it is injected
        so this module stays JAX-free.  Without a provider the last value
        passed to :meth:`note_identity_score` is returned.

    Notes
    -----
    * The default ``identity_core`` is a four-element placeholder.  Production
      code must pass the real frozen ``C_self[0]`` from the self lattice.
    * Path C is accepted from any non-full state when operator authorization is
      present (the brief's Path C test starts from ``DEFINED_FROZEN``); it is
      never reachable without that authorization.  Leaving
      ``FULL_ACTIVATION`` likewise requires operator authorization.
    * ``danger_flag`` is only meaningful for Path B / Path C; attaching it to a
      Path A transition is rejected so danger cannot be silently ignored.
    """

    def __init__(self, *, identity_core: Optional[Iterable[float]] = None,
                 log_path: Optional[os.PathLike | str] = None,
                 identity_score_provider: Optional[Callable[[], float]] = None):
        if identity_core is None:
            core = _DEFAULT_IDENTITY_CORE
        else:
            if hasattr(identity_core, "tolist"):
                identity_core = identity_core.tolist()
            core = tuple(float(x) for x in identity_core)
        self._identity_core = core
        self._identity_core_digest = hashlib.sha256(
            struct.pack(f"<{len(core)}d", *core)).hexdigest()

        self.log_path = Path(
            log_path if log_path is not None
            else os.environ.get("LCM_SELF_ACTIVATION_LOG", _DEFAULT_LOG_PATH)
        )
        self._identity_score_provider = identity_score_provider
        self._last_identity_score = 0.0

        self._state = State.DEFINED_FROZEN
        self._mutex = threading.Lock()

    # -- read-only views ----------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def identity_core(self) -> tuple:
        """Immutable view of the frozen identity vector."""
        return self._identity_core

    @property
    def identity_core_sha256(self) -> str:
        return self._identity_core_digest

    def identity_score(self) -> float:
        """Current self-lattice familiarity score (injected provider or cache)."""
        if self._identity_score_provider is not None:
            return float(self._identity_score_provider())
        return float(self._last_identity_score)

    def note_identity_score(self, value: float) -> float:
        """Cache an externally computed identity/familiarity score."""
        self._last_identity_score = float(value)
        return self._last_identity_score

    # -- transitions --------------------------------------------------------

    def transition(self, target, authorization: Optional[str] = None,
                   danger_flag: bool = False,
                   operator: Optional[str] = None) -> State:
        """Move to ``target`` under the d.md §3 authorization rules.

        Raises
        ------
        ValueError
            Unknown target state or illegal graph edge.
        PermissionError
            Path B without ``danger_flag``; Path C without operator
            authorization; or any attempt to leave ``FULL_ACTIVATION``
            without operator authorization.
        """
        try:
            target = State(target)
        except ValueError:
            raise ValueError(f"unknown self activation state: {target!r}") from None

        with self._mutex:
            current = self._state
            if target == current:
                raise ValueError(f"already in state {current.value}")

            try:
                path, resolved_operator = self._authorize(
                    current, target, authorization, danger_flag, operator)
            except (PermissionError, ValueError) as exc:
                self._log(
                    current, target, path="denied", result="denied",
                    reason=str(exc), authorization=authorization,
                    operator=operator, danger_flag=danger_flag,
                )
                raise

            # Fail closed: if the append-only audit write fails, the state does
            # not change.  This is the record d.md §3.2 Path C requires.
            self._log(
                current, target, path=path, result="applied",
                authorization=authorization, operator=resolved_operator,
                danger_flag=danger_flag,
            )
            self._state = target
            return target

    def _authorize(self, current: State, target: State,
                   authorization: Optional[str], danger_flag: bool,
                   operator: Optional[str]) -> tuple[str, Optional[str]]:
        if target not in _ALLOWED[current]:
            raise ValueError(
                f"illegal self activation transition: "
                f"{current.value} -> {target.value}")

        if danger_flag and target not in (State.DANGER_FLAGGED,
                                          State.FULL_ACTIVATION):
            raise PermissionError(
                "danger_flag requires Path B (target DANGER_FLAGGED) or an "
                "operator-authorized Path C; it cannot be attached to Path A")

        if target == State.DANGER_FLAGGED:
            if not danger_flag:
                raise PermissionError(
                    "Path B requires danger_flag=True from the danger "
                    "lattice / external verifier")
            return "B", operator

        if target == State.FULL_ACTIVATION:
            return "C", self._require_operator(authorization, operator)

        if current == State.FULL_ACTIVATION:
            # Leaving full activation is also an operator decision; the system
            # must not autonomously end an unsupervised self-narrative.
            return "C", self._require_operator(authorization, operator)

        if current == State.DANGER_FLAGGED:
            # Clearing a Path B flag is the human operator's decision.
            return "B", self._require_operator(authorization, operator)

        if target == State.TEMP_UNFROZEN:
            # Path A: explicit user instruction, no operator authorization.
            return "A", operator

        if target == State.DEFINED_FROZEN:
            # Path A session end.
            return "A", operator

        raise ValueError(
            f"unhandled self activation transition: "
            f"{current.value} -> {target.value}")

    @staticmethod
    def _require_operator(authorization: Optional[str],
                          operator: Optional[str]) -> str:
        auth = "" if authorization is None else str(authorization).strip()
        if not auth:
            raise PermissionError(
                "FULL_ACTIVATION requires explicit operator authorization")
        resolved = "" if operator is None else str(operator).strip()
        if not resolved and auth.lower().startswith("operator:"):
            resolved = auth.split(":", 1)[1].strip()
        if not resolved:
            raise PermissionError(
                "operator name required: pass operator='<name>' or "
                "authorization='operator:<name>'")
        return resolved

    # -- append-only audit log ---------------------------------------------

    def _log(self, current: State, target: State, *, path: str,
             result: str, reason: Optional[str] = None,
             authorization: Optional[str] = None,
             operator: Optional[str] = None,
             danger_flag: bool = False) -> None:
        ts = time.time()
        record = {
            "event": "self_activation_transition",
            "timestamp": ts,
            "timestamp_iso": _iso_utc(ts),
            "from": current.value,
            "to": target.value,
            "path": path,
            "result": result,
            "authorization": (None if authorization is None
                              else str(authorization)),
            "operator": None if operator is None else str(operator),
            "danger_flag": bool(danger_flag),
            "identity_score": float(self._last_identity_score),
            "identity_core_sha256": self._identity_core_digest,
        }
        if reason is not None:
            record["reason"] = reason
        self._append(record)

    def _append(self, record: dict) -> None:
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n"
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
