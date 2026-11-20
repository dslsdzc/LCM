"""Focused smoke-report contract test (Task 15).

Runs a 3-step tiny smoke, not the 100-step milestone run. The assertions are
structural: the report must contain every required section, the per-step loss
trajectory must be finite and complete, the report files must be written, and
save/load/old-format rejection must work. Individual conformance checks are
reported in the JSON but are not all required to pass on a 3-step random model
(for example, "last-10 < 0.8 * first-10" is undefined with fewer than 10 steps).

Run:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_smoke_conformance.py -q
"""
import json
import math
from pathlib import Path

import numpy as np


_WORDS = [
    "北京", "上海", "首都", "城市", "历史", "文化", "长江", "入海口", "很大",
]


def _articles(n=10, repeats=12):
    line = " ".join(_WORDS * repeats)
    return [line for _ in range(n)]


def test_mini_smoke_report_has_required_keys(tmp_path, registered_tiny):
    from train.conftest import build_tiny_corpus
    from train.smoke_conformance import REQUIRED_REPORT_KEYS, run_smoke

    _meta, (data, shape, spans) = build_tiny_corpus(
        tmp_path, registered_tiny, _articles(), prefix="smoke")

    out = tmp_path / "smoke_out"
    report = run_smoke(
        steps=3, batch=1, seq_len=16, data=data, shape=shape, spans=spans,
        output=str(out), preset="tiny", passive_only=True,
        tokenizer_path=registered_tiny.path, joint=True,
        self_awareness=True, save_every=3)

    assert report["meta"]["steps"] == 3
    assert report["meta"]["joint"] is True
    assert report["meta"]["self_awareness"] is True
    assert report["meta"]["passive_only"] is True
    for key in REQUIRED_REPORT_KEYS:
        assert key in report, f"missing report section {key}"

    per_step = report["loss"]["per_step"]
    assert len(per_step) == 3
    assert all(math.isfinite(x) for x in per_step)
    assert report["loss"]["all_finite"] is True

    # A 3-step run must still exercise the real training/EMA/save paths.
    assert report["ema"]["pass"] is True
    assert report["safety"]["training_is_safe_all_steps"] is True
    assert report["persistence"]["old_checkpoint_rejected"] is True
    assert report["persistence"]["pass"] is True
    assert report["lattice_updates"]["hrq"]["updated_gt_1e-6"] is True

    json_path = out / "smoke_report.json"
    md_path = out / "smoke_report.md"
    assert json_path.exists()
    assert md_path.exists()
    with open(json_path) as f:
        disk = json.load(f)
    for key in REQUIRED_REPORT_KEYS:
        assert key in disk
    assert disk["meta"]["steps"] == 3
    assert "conformance smoke" in md_path.read_text(encoding="utf-8").lower()

    # d_top must be per-sample, never the old batch scalar.
    assert report["sparse_inference"]["d_top_shape"] == [2]
    assert report["sparse_inference"]["d_top_differs_across_samples"] is True
    assert np.isfinite(
        report["sparse_inference"]["inference_nonzero_fraction"])
