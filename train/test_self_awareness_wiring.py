"""R16/Task 14: self-observation wiring and the 5% overhead budget.

Focused tests for the black box, narrative memory, reflection loop, behavior
explorer, safety filter, and verifier polling. The only JAX compile here is the
2-step routing-bias reachability test.

Run from repo root:
    JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
        train/test_self_awareness_wiring.py -v
"""
import dataclasses
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from train.behavior_explorer import BehaviorExplorer, TensionSignals
from train.narrative_memory import NarrativeMemory
from train.observability import (
    ObservabilityRecorder, StepRecord, VerifierPoller, measure_overhead,
)
from train.reflection_loop import (
    REFLECT_INTERVAL, ReflectionLoop, Suggestion, safety_filter,
)


def _record(step=0, **overrides):
    base = dict(
        step=step,
        step_time_ms=1.0,
        soft_mask=np.full(6, 1.0 / 6.0),
        route_idx=0,
        is_safe=True,
        convergence_diff=0.1,
        dag_nodes=[{'step': step, 'route_idx': 0}],
        z_cur=np.zeros(8, dtype=np.float32),
        z_next=np.ones(8, dtype=np.float32),
    )
    base.update(overrides)
    return StepRecord(**base)


# ─── Black box ──────────────────────────────────────────────────────────────


def test_black_box_records_full_fields_and_copies_vectors():
    obs = ObservabilityRecorder(raw_capacity=4, record_vectors=True)
    z_cur = np.arange(8, dtype=np.float32)
    z_next = z_cur + 1.0
    rec = obs.record_step(
        step=7, step_time_ms=3.5, soft_mask=np.arange(6, dtype=np.float32),
        route_idx=2, is_safe=False, convergence_diff=0.25,
        dag_nodes=[{'op': 'fusion'}], z_cur=z_cur, z_next=z_next,
        prediction_error=0.75,
    )
    assert rec is not None
    assert rec.step == 7
    assert rec.step_time_ms == 3.5
    assert rec.route_idx == 2
    assert rec.is_safe is False
    assert rec.convergence_diff == 0.25
    assert rec.dag_nodes == [{'op': 'fusion'}]
    np.testing.assert_array_equal(rec.z_cur, z_cur)
    np.testing.assert_array_equal(rec.z_next, z_next)
    # Snapshot semantics: mutating the caller's array cannot rewrite history.
    z_cur[:] = -1.0
    np.testing.assert_array_equal(rec.z_cur, np.arange(8, dtype=np.float32))


def test_black_box_long_term_uniform_plus_anomaly_oversampling():
    obs = ObservabilityRecorder(
        raw_capacity=1000, long_term_capacity=10000,
        long_term_every=2, anomaly_oversample=3)
    for step in range(10):
        obs.record_step(step=step, soft_mask=np.full(6, 1 / 6),
                        is_safe=True)
    # Raw keeps every step; long-term takes every 2nd uniform sample.
    assert len(obs.ring) == 10
    assert len(obs.long_term) == 5
    assert obs.long_term.capacity == 10000

    obs2 = ObservabilityRecorder(
        long_term_every=1000, anomaly_oversample=3)
    obs2.record_step(step=0, is_safe=True)
    assert len(obs2.long_term) == 0
    obs2.record_step(step=1, is_safe=False)
    # A safety conflict is oversampled 3x even between uniform samples.
    assert len(obs2.long_term) == 3
    assert all(not r.is_safe for r in obs2.long_term)

    obs3 = ObservabilityRecorder(long_term_every=1000, anomaly_oversample=3)
    obs3.record_step(step=0, route_idx=0, is_safe=True)
    assert len(obs3.long_term) == 0
    obs3.record_step(step=1, route_idx=1, is_safe=True)
    # A novel routing pattern is also an oversampled anomaly (e.md §2).
    assert len(obs3.long_term) == 3


def test_measure_overhead_timer_floor_smoke():
    # Smoke test for the timer floor only. This is NOT the 5% evidence: the
    # real _self_observe_step path is measured below.
    assert measure_overhead(lambda: None, lambda: None) < 0.05


def test_self_observation_overhead_under_five_percent():
    """Real obs+narrative+reflection+behavior path vs a no-op observation."""
    from train.cog_train import _self_observe_step

    cfg = _tiny_cfg()
    obs = ObservabilityRecorder(raw_capacity=100)
    narr = NarrativeMemory(random_fn=lambda: 1.0)
    reflection = ReflectionLoop(log_path=None)
    explorer = BehaviorExplorer(explore_prob=0.0)
    aux = {
        'soft_mask': jnp.full((1, 6), 1.0 / 6.0),
        'route_idx': jnp.array([0], dtype=jnp.int32),
        'z_cur': jnp.zeros((1, cfg.d_model)),
        'z_next': jnp.ones((1, cfg.d_model)),
        'convergence_diff': jnp.array(0.1),
        'is_safe': jnp.array(True),
    }

    def inference_like():
        # ~10 ms of "inference"; Task 15 does the end-to-end measurement.
        x = 0.0
        for _ in range(500000):
            x += 1.0
        return x

    def inference_plus_observation():
        value = inference_like()
        _self_observe_step(
            obs, narr, reflection, explorer, None, 1, 1.0, aux, 1.0, cfg)
        return value

    ratio = measure_overhead(inference_plus_observation, inference_like,
                             runs=20)
    assert ratio < 0.05, f"self-observation overhead {ratio:.1%} >= 5%"


def test_self_observe_records_residual_and_labels_tension_as_proxy():
    """Training tension is entropy+residual only (T_conf=0), not full U."""
    from train.cog_train import _self_observe_step

    cfg = _tiny_cfg()
    obs = ObservabilityRecorder(raw_capacity=10, record_vectors=True)
    narr = NarrativeMemory(random_fn=lambda: 1.0)
    reflection = ReflectionLoop(log_path=None)
    explorer = BehaviorExplorer(explore_prob=0.0)

    class FakeCausal:
        def __init__(self):
            self.calls = []

        def step(self, record, pred_cache=None, action_desc=None):
            self.calls.append((record, pred_cache, action_desc))
            return {}

    causal = FakeCausal()
    aux = {
        'soft_mask': jnp.full((1, 6), 1.0 / 6.0),
        'route_idx': jnp.array([0], dtype=jnp.int32),
        'z_cur': jnp.zeros((1, cfg.d_model)),
        'z_next': jnp.ones((1, cfg.d_model)),
        'convergence_diff': jnp.array(0.1),
        'is_safe': jnp.array(True),
    }
    safe = _self_observe_step(
        obs, narr, reflection, explorer, None, 7, 1.0, aux, 1.0, cfg,
        causal_subject=causal)

    assert safe is True
    record = obs.ring[len(obs.ring) - 1]
    # One-step residual ||z_next - z_cur||², not a negated distance.
    assert record.prediction_error == pytest.approx(cfg.d_model)
    assert causal.calls, "an injected causal subject must receive the step"
    assert causal.calls[0][0].step == 7
    assert causal.calls[0][1] is None

    tension = explorer._eval_buffer[-1]
    assert tension['T_conf'] == 0.0, "no value oracle in Stage 2"
    assert tension['T_pred'] > 0.0
    assert tension['T_res'] > 0.0


def test_inference_convergence_diff_adjacent_and_single_step():
    from lcm import _trace_convergence_diff

    z0 = np.zeros(4, dtype=np.float32)
    z1 = np.ones(4, dtype=np.float32)
    adjacent = _trace_convergence_diff(
        [{'z_next': z0}, {'z_next': z1}], z_q=None, z_cur=None)
    assert adjacent == pytest.approx(float(np.linalg.norm(z1 - z0)))
    assert adjacent > 0.0

    single = _trace_convergence_diff(
        [{'z_next': z0}], z_q=z1, z_cur=z0)
    assert single == pytest.approx(float(np.linalg.norm(z1 - z0)))
    assert single > 0.0


# ─── Narrative memory ───────────────────────────────────────────────────────


def test_narrative_five_triggers_each_fire():
    # 1. safety conflict
    m = NarrativeMemory(random_fn=lambda: 1.0)
    rec = m.feed(_record(step=0, is_safe=False))
    assert rec is not None and rec.importance_rule == 'safety_conflict'

    # 2. prediction error > running mean + 2σ
    m = NarrativeMemory(random_fn=lambda: 1.0)
    for i in range(10):
        m.feed(_record(step=i, prediction_error=0.0, soft_mask=None,
                       convergence_diff=None))
    rec = m.feed(_record(step=10, prediction_error=10.0, soft_mask=None,
                         convergence_diff=None))
    assert rec is not None and rec.importance_rule == 'prediction_error'

    # 3. routing KL novelty against the historical mean distribution
    m = NarrativeMemory(random_fn=lambda: 1.0)
    for i in range(10):
        m.feed(_record(step=i, soft_mask=np.full(6, 1 / 6),
                       convergence_diff=None))
    novel = np.zeros(6, dtype=np.float32)
    novel[0] = 1.0
    rec = m.feed(_record(step=10, soft_mask=novel, convergence_diff=None))
    assert rec is not None and rec.importance_rule == 'routing_novelty'

    # 4. convergence anomaly
    m = NarrativeMemory(random_fn=lambda: 1.0)
    for i in range(10):
        m.feed(_record(step=i, convergence_diff=0.0, soft_mask=None))
    rec = m.feed(_record(step=10, convergence_diff=5.0, soft_mask=None))
    assert rec is not None and rec.importance_rule == 'convergence_anomaly'

    # 5. random 1% ordinary sampling (random source forced)
    m = NarrativeMemory(random_fn=lambda: 0.0)
    rec = m.feed(_record(step=0, soft_mask=None, convergence_diff=0.1))
    assert rec is not None and rec.importance_rule == 'random_sample'


def test_narrative_sustained_novel_routing_triggers_finitely():
    m = NarrativeMemory(random_fn=lambda: 1.0,
                        routing_novelty_cooldown=5)
    for i in range(10):
        m.feed(_record(step=i, soft_mask=np.full(6, 1 / 6),
                       convergence_diff=None))
    novel = np.zeros(6, dtype=np.float32)
    novel[0] = 1.0
    for i in range(10, 110):
        m.feed(_record(step=i, soft_mask=novel, convergence_diff=None))
    hits = len(m.get_by_rule('routing_novelty', touch=False))
    assert 0 < hits < 20, (
        f"sustained novelty must not fire every step (got {hits})")


def test_narrative_context_and_access_count():
    m = NarrativeMemory(max_records=100, random_fn=lambda: 1.0,
                        context_window=5)
    for i in range(12):
        m.feed(_record(step=i, is_safe=False))
    # get_all is a real read path: it increments access frequency and marks
    # the last access step.
    records = m.get_all()
    assert len(records) == 12
    for rec in records:
        assert 'before' in rec.context and 'after' in rec.context
        assert rec.context['window'] == 5
        assert rec.access_count == 1
        assert rec.last_access_step == m._last_step
    # A middle record sees the full 5-step context on both sides.
    middle = next(r for r in records if r.step == 6)
    assert len(middle.context['before']) == 5
    assert len(middle.context['after']) == 5

    m.access(step=6, access_step=12)
    assert middle.access_count == 2
    assert middle.last_access_step == 12
    m.access(step=6, access_step=13)
    assert middle.access_count == 3

    # An unstored feed still advances the clock used for access timestamps.
    assert m.feed(_record(step=20, is_safe=True, soft_mask=None,
                          convergence_diff=None)) is None
    m.get_all()
    assert middle.last_access_step == 20


def test_narrative_by_rule_read_marks_access_without_manual_access():
    # A real read path must count as access; no m.access() call here.
    m = NarrativeMemory(max_records=2, random_fn=lambda: 0.0)
    r0 = m.feed(_record(step=0, is_safe=False))
    r1 = m.feed(_record(step=1, is_safe=True, soft_mask=None,
                        convergence_diff=None))
    assert r0.importance_rule == 'safety_conflict'
    assert r1.importance_rule == 'random_sample'
    m.get_by_rule('safety_conflict')
    assert r0.access_count == 1
    assert r0.last_access_step == m._last_step
    assert r1.access_count == 0

    m.feed(_record(step=2, is_safe=True, soft_mask=None,
                   convergence_diff=None))
    kept = {r.step for r in m.get_all(touch=False)}
    assert 0 in kept, "the record read via get_by_rule wins the retention score"
    assert 2 in kept
    assert 1 not in kept, "the never-read low-importance record is forgotten"


def test_consolidate_staleness_uses_last_access():
    m = NarrativeMemory(random_fn=lambda: 0.0)
    accessed = m.feed(_record(step=0, is_safe=True, soft_mask=None,
                              convergence_diff=None))
    never_read = m.feed(_record(step=1, is_safe=True, soft_mask=None,
                                convergence_diff=None))
    assert accessed.importance_rule == 'random_sample'
    m.access(step=0, access_step=4000)

    dropped = m.consolidate(current_step=5200, keep_threshold=0.0)
    kept = {r.step for r in m.get_all(touch=False)}
    assert dropped == 1
    assert 0 in kept, "age is measured from last access, not creation"
    assert 1 not in kept, "never-read low-importance record is stale"


def test_narrative_forgets_by_importance_times_access():
    m = NarrativeMemory(max_records=2, random_fn=lambda: 1.0)
    m.feed(_record(step=0, is_safe=False))   # score 1.0, accessed below
    m.feed(_record(step=1, is_safe=False))   # score 1.0, never accessed
    for _ in range(5):
        m.access(step=0)
    m.feed(_record(step=2, is_safe=False))   # capacity overflow at 3
    kept = {r.step for r in m.get_all()}
    assert 0 in kept, "high importance x access must survive"
    assert 2 in kept
    assert 1 not in kept, "lowest importance x access is forgotten"


# ─── Reflection loop ────────────────────────────────────────────────────────


def test_reflection_runs_every_interval_async_and_logs(tmp_path):
    log_path = tmp_path / "reflection.log"
    r = ReflectionLoop(log_path=log_path)
    for step in range(1, 201):
        r.observe(step, soft_mask=np.full(6, 1 / 6),
                  convergence_diff=0.1, cache_hit=0.5)
    assert not r.maybe_reflect(199)
    assert r.maybe_reflect(200)
    r.wait(timeout=5.0)
    assert r.n_analyses == 1
    assert log_path.exists()
    lines = log_path.read_text(encoding='utf-8').splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload['step'] == 200
    assert payload['interval'] == REFLECT_INTERVAL


def test_reflection_pending_request_retried_after_busy_thread(tmp_path):
    r = ReflectionLoop(log_path=tmp_path / "reflection.log")
    for step in range(1, 202):
        r.observe(step, soft_mask=np.full(6, 1 / 6),
                  convergence_diff=0.1, cache_hit=0.5)

    class BusyThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    r._thread = BusyThread()
    assert not r.maybe_reflect(200)
    assert r.pending_reflection_step == 200
    r._thread = None
    # The pending boundary is retried on the next call, even though 201 is
    # not itself an interval boundary.
    assert r.maybe_reflect(201)
    r.wait(timeout=5.0)
    assert r.n_analyses == 1
    assert r.last_report['step'] == 200


def test_reflection_detects_abandoned_lattice_and_queues_suggestion(tmp_path):
    r = ReflectionLoop(log_path=tmp_path / "reflection.log",
                       abandoned_steps=1000)
    weights = np.array([0.01] + [0.198] * 5, dtype=np.float32)
    for step in range(1001):
        r.observe(step, soft_mask=weights, convergence_diff=0.1,
                  cache_hit=0.5)
    report = r.reflect_now(step=1000)
    assert 0 in report['abandoned_lattices']
    suggestions = r.drain_suggestions()
    assert suggestions, "abandoned lattice must produce a suggestion"
    assert any(s.action_type == 'bias_delta' for s in suggestions)
    assert all(s.action_type != 'bias_delta'
               or abs(s.params['delta']) <= 0.2 for s in suggestions)


def test_reflection_entropy_variance_and_cache_trend(tmp_path):
    r = ReflectionLoop(log_path=tmp_path / "reflection.log")
    uniform = np.full(6, 1 / 6)
    onehot = np.zeros(6)
    onehot[0] = 1.0
    for step, hit in enumerate([0.9, 0.7, 0.5, 0.3, 0.1]):
        r.observe(step, soft_mask=(uniform if step % 2 else onehot),
                  convergence_diff=0.1, cache_hit=hit)
    report = r.reflect_now(step=5)
    assert report['routing_entropy_variance'] > 0.01
    assert report['cache_declining'] is True
    assert any(s.action_type == 'explore_rate_reduce'
               for s in r.drain_suggestions())


def test_cache_hit_rate_counters_and_reflection_uses_hit_rate():
    from train.predictive_cache import PredictiveSystem

    ps = PredictiveSystem(d_model=8, cache_capacity=8)
    for i in range(3):
        ps.cache.append(sig=1, z_cur=np.zeros(8, dtype=np.float32),
                        z_next=np.ones(8, dtype=np.float32), step=i)
    ps.matcher.epsilon = 0.0  # deterministic: Q all zero -> first arm M0
    z_hit, op, _ = ps.matcher.predict(np.zeros(8, dtype=np.float32), 1)
    assert op == 'M0' and z_hit is not None
    z_miss, _, _ = ps.matcher.predict(np.zeros(8, dtype=np.float32), 2)
    assert z_miss is None

    stats = ps.get_stats()
    assert stats['n_lookups'] == 2
    assert stats['hits'] == 1 and stats['misses'] == 1
    assert stats['hit_rate'] == pytest.approx(0.5)

    # The cache trend is driven by hit rate, never by capacity utilization.
    r = ReflectionLoop(log_path=None)
    current, declining = r._cache_trend([], {'cache_util': 0.99})
    assert current is None and declining is False
    current, declining = r._cache_trend(
        [], {'cache_util': 0.99, 'hit_rate': 0.42})
    assert current == pytest.approx(0.42) and declining is False


def test_reflection_three_sigma_against_narrative_history(tmp_path):
    r = ReflectionLoop(log_path=tmp_path / "reflection.log")
    for step in range(20):
        r.observe(step, soft_mask=np.full(6, 1 / 6),
                  convergence_diff=1.0)
    history = [SimpleNamespace(convergence_diff=1.0) for _ in range(20)]
    narrative = SimpleNamespace(get_all=lambda: history)
    r.observe(100, soft_mask=np.full(6, 1 / 6), convergence_diff=10.0)
    report = r.reflect_now(step=100, narrative=narrative)
    assert report['anomaly_3sigma'] is not None
    assert report['anomaly_3sigma']['step'] == 100


# ─── Level 2 → Level 1 safety filter ────────────────────────────────────────


def test_safety_filter_blocks_unsafe_level2_suggestion():
    unsafe_bias = Suggestion(
        step=1, action_type='bias_delta',
        params={'dim': 0, 'delta': 0.9}, reason='too large')
    assert not safety_filter(unsafe_bias)

    unsafe_explore = Suggestion(
        step=1, action_type='explore_rate_reduce',
        params={'factor': 50.0}, reason='bogus factor')
    assert not safety_filter(unsafe_explore, current_explore=0.2)

    safe_bias = Suggestion(
        step=1, action_type='bias_delta',
        params={'dim': 0, 'delta': 0.1}, reason='bounded')
    assert safety_filter(safe_bias)
    # Danger priority: positive bias (more exploration) is rejected.
    assert not safety_filter(safe_bias, danger=True)

    safe_explore = Suggestion(
        step=1, action_type='explore_rate_reduce',
        params={'factor': 0.5}, reason='cautious')
    assert safety_filter(safe_explore, current_explore=0.2)

    class FakeExplorer:
        explore_prob = 0.2
        bias_override = np.zeros(6, dtype=np.float32)

        def apply_bias_delta(self, dim, delta):
            self.bias_override[dim] += delta

    explorer = FakeExplorer()
    loop = ReflectionLoop(log_path='/dev/null')
    assert loop.apply_suggestion(safe_explore, explorer)
    assert explorer.explore_prob == pytest.approx(0.1)
    assert loop.apply_suggestion(safe_bias, explorer)
    assert explorer.bias_override[0] == pytest.approx(0.1)
    assert not loop.apply_suggestion(unsafe_bias, explorer)
    assert explorer.bias_override[0] == pytest.approx(0.1)


def test_suggestion_born_under_danger_stays_constrained_after_recovery():
    class FakeExplorer:
        explore_prob = 0.2
        bias_override = np.zeros(6, dtype=np.float32)

        def apply_bias_delta(self, dim, delta):
            self.bias_override[dim] += delta

    loop = ReflectionLoop(log_path=None)
    explorer = FakeExplorer()

    # Generated while unsafe, applied after the system recovered. The
    # apply-time danger flag is False; generation-time danger must still win.
    born_danger = Suggestion(
        step=10, action_type='bias_delta',
        params={'dim': 0, 'delta': 0.1}, reason='explore more',
        severity=0.8, danger_at_generation=True,
        is_safe_at_generation=False)
    assert not loop.apply_suggestion(born_danger, explorer, danger=False)
    assert explorer.bias_override[0] == pytest.approx(0.0)

    # A reducing action from the same dangerous generation is permitted.
    reducing = Suggestion(
        step=10, action_type='bias_delta',
        params={'dim': 0, 'delta': -0.1}, reason='reduce tension',
        severity=0.8, danger_at_generation=True,
        is_safe_at_generation=False)
    assert loop.apply_suggestion(reducing, explorer, danger=False)
    assert explorer.bias_override[0] == pytest.approx(-0.1)
    # Per-target cooldown blocks a second mutation in the same interval.
    again = Suggestion(
        step=11, action_type='bias_delta',
        params={'dim': 0, 'delta': -0.1}, reason='repeat',
        severity=0.8, danger_at_generation=True,
        is_safe_at_generation=False)
    assert not loop.apply_suggestion(again, explorer, danger=False)

    # The analysis path stamps suggestions with the observation's safety state.
    r = ReflectionLoop(log_path=None)
    uniform = np.full(6, 1 / 6)
    onehot = np.zeros(6)
    onehot[0] = 1.0
    for step in range(5):
        r.observe(step, soft_mask=(uniform if step % 2 else onehot),
                  convergence_diff=0.1, is_safe=False)
    r.reflect_now(step=5)
    suggestions = r.drain_suggestions()
    assert suggestions
    assert all(s.danger_at_generation for s in suggestions)
    assert all(not s.is_safe_at_generation for s in suggestions)


# ─── Behavior explorer + train step routing_bias ────────────────────────────


def test_behavior_explorer_every_ten_steps_and_danger_priority():
    explorer = BehaviorExplorer(explore_prob=0.0)
    # Make +0.2 the greedy choice for every dimension.
    explorer._P[:, 2] = 1.0
    normal = explorer.plan_step(10, danger=False, interval=10)
    np.testing.assert_allclose(normal, np.full(6, 0.2, dtype=np.float32))
    assert explorer.should_update(10)
    assert not explorer.should_update(11)
    danger = explorer.plan_step(10, danger=True, interval=10)
    np.testing.assert_allclose(danger, np.zeros(6, dtype=np.float32))


def test_behavior_finalize_clears_unevaluable_bias():
    explorer = BehaviorExplorer(explore_prob=0.0)
    signal = TensionSignals(T_pred=0.5, T_conf=0.0, T_res=0.5)
    explorer.current_bias = np.full(6, 0.2, dtype=np.float32)
    explorer._bias_start_step = 100
    # All samples predate the selected bias: no evaluation window exists.
    for step in range(3):
        explorer.observe_tension(step, signal)
    explorer.finalize_evaluation(3)
    assert explorer.current_bias is None
    assert len(explorer._eval_buffer) == 0


def test_behavior_plan_zero_regression_updates_by_thirty():
    # Regression: a plan_step(0) call with an empty baseline must not freeze
    # the selected bias forever. With exploration forced on, the first
    # selection happens once >=3 tension samples exist, and at least one
    # reward update must have landed by step 30.
    explorer = BehaviorExplorer(explore_prob=1.0)
    signal = TensionSignals(T_pred=0.5, T_conf=0.0, T_res=0.5)
    for step in range(31):
        bias = explorer.plan_step(step, danger=False, interval=10)
        assert bias.shape == (explorer.n_dim,)
        assert np.all(np.isfinite(bias))
        explorer.observe_tension(step, signal)
    assert explorer.n_updates > 0, "bias evaluation never finalized"


def _tiny_cfg(**overrides):
    from train.config import LCMConfig
    base = dict(
        d_model=8, d_ff=16, n_heads=2, n_encoder_layers=1,
        vocab_size=32, max_seq_len=16,
        M_top=4, M_fine=4, n_hrq_layers=1,
        M_sparse=4, M_lr=4, n_lr_layers=1, r_max=2, ranks=(2,),
        M_man=4, t_dim=2,
        M_bind=4, n_bind_layers=1,
        M_contrast=4, n_contrast_layers=1,
        n_self_codes=4, n_lattices=6,
        max_inference_steps=2, n_orth_samples=4,
    )
    base.update(overrides)
    return dataclasses.replace(LCMConfig(), **base)


def test_behavior_bias_reaches_routing_bias_in_two_train_steps():
    from train.cog_train import init_cog_params, make_train_step
    from train.ema_state import init_ema_state
    from train.param_groups import partition_params

    cfg = _tiny_cfg()
    params, self_state = init_cog_params(cfg, jax.random.PRNGKey(0))
    ema_state = init_ema_state(params, cfg)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1.0),
    )
    trainable, _, _ = partition_params(params)
    opt_state = optimizer.init(trainable)
    train_step = make_train_step(cfg, optimizer, joint=False)

    tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % cfg.vocab_size
    batch = (tokens, jnp.roll(tokens, -1, axis=1))
    explorer = BehaviorExplorer(explore_prob=0.0)
    explorer._P[:, 2] = 1.0
    first_bias = explorer.plan_step(1, danger=False)
    second_bias = explorer.plan_step(10, danger=False)
    # The learned best bias is +0.2 on every dimension; get_best_bias() is
    # what plan_step returns, and both steps must carry it into train_step.
    np.testing.assert_allclose(second_bias, np.full(6, 0.2))

    for step_i, planned in enumerate((first_bias, second_bias)):
        params, opt_state, ema_state, loss, aux = train_step(
            params, opt_state, ema_state, batch, 1e-3,
            jax.random.PRNGKey(10 + step_i), self_state=self_state,
            step=step_i,
            routing_bias=jnp.asarray(planned, dtype=jnp.float32))
        assert np.isfinite(float(loss))
        np.testing.assert_allclose(
            np.asarray(aux['routing_bias_used']), planned, atol=1e-6)
        if aux.get('self_state') is not None:
            self_state = aux['self_state']


# ─── Verifier polling ───────────────────────────────────────────────────────


def test_inference_close_verifier_stops_and_clears_poller():
    from lcm import LCMInferEngine

    class FakePoller:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    eng = object.__new__(LCMInferEngine)
    poller = FakePoller()
    eng.verifier_poller = poller
    eng.verifier = object()
    eng.close_verifier()
    assert poller.closed
    assert eng.verifier_poller is None
    assert eng.verifier is None
    assert eng.close_verifier() is None  # idempotent


def test_verifier_polling_fires_lock_callback(tmp_path):
    from train.external_verifier import LOCK, NORMAL, VerifierProcess

    alerts = []
    verifier = VerifierProcess(
        dim=4, threshold_low=1.0, threshold_high=2.0, observe_limit=1,
        mode='inline', trace_dir=str(tmp_path / 'verifier'),
        alert_callback=alerts.append)
    verifier.start()
    try:
        poller = VerifierPoller(verifier, on_lock=alerts.append)
        assert poller.poll(0) == NORMAL
        verifier.submit([10.0] * 4)
        assert poller.poll(1) == LOCK
        assert alerts, "LOCK must produce a callback"
        assert alerts[0]['type'] == 'LOCK'
    finally:
        verifier.stop()
