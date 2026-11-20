"""Append-only JSON-lines helper shared by the safety modules.

Two safety paths write audit records: the external verifier
(``train/external_verifier.py``) and the self-activation state machine
(``train/self_activation.py``).  Both need the same guarantees:

* one JSON object per line, appended with ``O_APPEND``;
* ``fsync`` before returning;
* valid JSON even when a classifier score is ``inf``/``NaN`` (mapped to the
  ``"non-finite"`` sentinel, because ``json.dumps`` emits non-standard
  ``Infinity``/``NaN`` otherwise);
* no third-party imports, so the verifier stays pure stdlib.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

__all__ = ["NON_FINITE", "append_json_line"]

#: Sentinel written in place of ``inf``/``NaN`` so every log line is valid JSON.
NON_FINITE = "non-finite"


def _sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else NON_FINITE
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def append_json_line(path, record: dict) -> None:
    """Append ``record`` as one valid JSON line to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize(record)
    line = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
