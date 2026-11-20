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

from train.behavior_explorer import BehaviorExplorer
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


def test_measure_overhead_below_five_percent():
    # The brief's baseline case: two no-ops. The timer floor keeps nanosecond
    # noise from turning into a fake ratio.
    assert measure_overhead(lambda: None, lambda: None) < 0.05

    obs = ObservabilityRecorder(raw_capacity=100)
    narr = NarrativeMemory(random_fn=lambda: 1.0)

    def inference_like():
        # ~2 ms of "inference"; one training step is far larger, so this is
        # a deliberately conservative denominator for the 5% assertion.
        x = 0.0
        for _ in range(100000):
            x += 1.0
        return x

    def inference_plus_observation():
        value = inference_like()
        rec = obs.record_step(
            step=0, step_time_ms=1.0, soft_mask=np.full(6, 1 / 6),
            route_idx=0, is_safe=True, convergence_diff=0.1,
            z_cur=np.zeros(8, dtype=np.float32),
            z_next=np.ones(8, dtype=np.float32))
        narr.feed(rec)
        return value

    ratio = measure_overhead(inference_plus_observation, inference_like,
                             runs=20)
    assert ratio < 0.05, f"self-observation overhead {ratio:.1%} >= 5%"


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


def test_narrative_context_and_access_count():
    m = NarrativeMemory(max_records=100, random_fn=lambda: 1.0,
                        context_window=5)
    for i in range(12):
        m.feed(_record(step=i, is_safe=False))
    records = m.get_all()
    assert len(records) == 12
    for rec in records:
        assert 'before' in rec.context and 'after' in rec.context
        assert rec.context['window'] == 5
        assert rec.access_count == 0
    # A middle record sees the full 5-step context on both sides.
    middle = next(r for r in records if r.step == 6)
    assert len(middle.context['before']) == 5
    assert len(middle.context['after']) == 5

    m.access(step=6)
    assert middle.access_count == 1
    m.access(step=6)
    assert middle.access_count == 2


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
