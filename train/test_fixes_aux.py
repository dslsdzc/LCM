"""辅助子系统正确性修复回归测试（D组 #1-#13）。

Run: be/bin/python -m train.test_fixes_aux

每项一个测试函数，覆盖 audit 清单 #1-#13 的关键断言。
"""
import numpy as np
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from train.reflection_loop import (SlidingStats, AnomalyDetector, ReflectionLoop,
                                   CorrectionAction)
from train.external_verifier import (init_verifier_params, init_verifier_state,
                                     detect_anomaly, compute_verifier_loss)
from train.causal_subject import (CausalGraph, CausalSubject,
                                  SOURCE_EXTERNAL, SOURCE_INTERNAL)
from train.narrative_memory import ImportanceEvaluator, _check_state_jump
from train.self_lattice import (init_self_params, init_self_state,
                                self_lattice_forward)
from train.gvalue import make_global_value_vectors, GValueCodebook


# ─── #1 SlidingStats Welford：方差每步重复除 (n-1) ──────────────────────────

def test_01_sliding_stats_welford():
    ss = SlidingStats(window=100)
    rng = np.random.RandomState(0)
    values = rng.normal(5.0, 1.0, 500)
    zs = []
    for v in values:
        z = ss.update(v)
        if z is not None:
            zs.append(z)
    std = ss.get_std()
    assert abs(std - 1.0) < 0.15, f"get_std()={std:.4f}, expected ≈1.0"
    zs = np.asarray(zs)
    frac = np.mean(np.abs(zs) < 3.0)
    assert frac > 0.95, f"only {frac:.3f} of z-scores < 3"
    print(f"  [1] std={std:.4f} (≈1.0), {frac:.1%} of z-scores < 3")


# ─── #2 compute_verifier_loss 方向反 ────────────────────────────────────────

def test_02_verifier_loss_direction():
    rng = jax.random.PRNGKey(0)
    d = 16
    params = init_verifier_params(rng, d, hidden_dim=8)
    state = init_verifier_state(d)
    z_norm = jnp.zeros((1, d))                  # 正常点：mahal=0
    z_out = jnp.full((1, d), 10.0)              # 越界点：mahal 大
    l_norm = float(compute_verifier_loss(params, z_norm, state))
    l_out = float(compute_verifier_loss(params, z_out, state))
    assert l_norm < l_out, f"normal loss {l_norm:.6f} should be < outlier {l_out:.6f}"
    assert l_out > 0.0
    print(f"  [2] normal loss={l_norm:.6f} < outlier loss={l_out:.6f}")


# ─── #3 combined_z 用 batch 自身 std（B=1 退化）────────────────────────────

def test_03_combined_z_batch1():
    rng = jax.random.PRNGKey(1)
    d = 32
    params = init_verifier_params(rng, d, hidden_dim=8)
    state = init_verifier_state(d)              # z_var=1 → scale=1
    z = jax.random.normal(rng, (1, d))          # B=1 正常样本
    _, combined_z = detect_anomaly(state, z, params)
    cz = float(combined_z[0])
    assert 0.05 < abs(cz) < 100.0, f"B=1 combined_z={cz:.4g}, expected O(1) not 1e8"
    print(f"  [3] B=1 combined_z={cz:.4f} (O(1), 非 1e8)")


# ─── #4 detect_causal_edge 不写 flat ring buffer ────────────────────────────

def test_04_detect_edge_writes_flat():
    g = CausalGraph(max_edges=16)
    prev = SimpleNamespace(step=0, source=SOURCE_INTERNAL, z_q=np.ones(4))
    cur = SimpleNamespace(step=1, source=SOURCE_INTERNAL, z_q=np.ones(4) * 2.0)
    edge = g.detect_causal_edge(step=1, action_desc='act', prev_records=[prev, cur])
    assert edge is not None
    assert int(g._edge_valid.sum()) == 1, "flat _edge_valid 未写入"
    n_total = g.count_total_results(window=100, current_step=5)
    n_int = g.count_internal_results(window=100, current_step=5)
    assert n_total == 1 and n_int == 1, f"count_edges_cy 读到 ({n_int},{n_total})"
    print(f"  [4] flat write ok: valid={int(g._edge_valid.sum())}, "
          f"counts=({n_int},{n_total})")


# ─── #5 source 语义错 + 条件 1（状态变化）缺失 ──────────────────────────────

def test_05_detect_edge_state_change_condition():
    g = CausalGraph(max_edges=16)

    # 条件 1：无状态变化 → None
    prev = SimpleNamespace(step=0, source=SOURCE_INTERNAL, z_q=np.ones(4))
    cur = SimpleNamespace(step=1, source=SOURCE_INTERNAL, z_q=np.ones(4))
    assert g.detect_causal_edge(step=1, action_desc='act',
                                prev_records=[prev, cur]) is None, \
        "无状态变化不应建边"

    # 有状态变化 → 建边，source/internal_chain 取自 prev 记录
    cur2 = SimpleNamespace(step=1, source=SOURCE_INTERNAL, z_q=np.ones(4) * 2.0)
    edge2 = g.detect_causal_edge(step=1, action_desc='act',
                                 prev_records=[prev, cur2])
    assert edge2 is not None
    assert edge2.source == SOURCE_INTERNAL and edge2.internal_chain
    assert len(g._edges) == 1 and int(g._edge_valid.sum()) == 1

    # prev 是 external → internal_chain=False（即使 current source=internal）
    prev_ext = SimpleNamespace(step=2, source=SOURCE_EXTERNAL, z_q=np.ones(4) * 2.0)
    cur_ext = SimpleNamespace(step=3, source=SOURCE_INTERNAL, z_q=np.ones(4) * 3.0)
    edge3 = g.detect_causal_edge(step=3, action_desc='act',
                                 prev_records=[prev_ext, cur_ext])
    assert edge3 is not None and not edge3.internal_chain
    assert edge3.source == SOURCE_EXTERNAL

    # 记录无状态向量 → 跳过条件 1（向后兼容）
    prev_nv = SimpleNamespace(step=4, source=SOURCE_INTERNAL)
    cur_nv = SimpleNamespace(step=5, source=SOURCE_INTERNAL)
    assert g.detect_causal_edge(step=5, action_desc='act',
                                prev_records=[prev_nv, cur_nv]) is not None
    print("  [5] 状态变化条件 + prev-source 语义 ok")


# ─── #6 __no_action__ 基线永不写入 ─────────────────────────────────────────

def test_06_no_action_baseline():
    cs = CausalSubject()
    rec = SimpleNamespace(step=1, source=SOURCE_EXTERNAL, route_idx=None)
    cs.step(rec)  # 无 action_desc + external → 记录基线
    p = cs.action_stats.prob_given_no_action(rec)
    assert p > 0.0, f"prob_given_no_action={p}, expected > 0"
    print(f"  [6] P(result|no_action)={p:.3f} > 0")


# ─── #7 periodic 低优先级短路 ───────────────────────────────────────────────

def test_07_periodic_fallback():
    ev = ImportanceEvaluator()
    # 满足 periodic 且 safety_near_miss 命中 → 高优先级规则优先
    rec = SimpleNamespace(step=100, safety_margin=0.05, convergence_entropy=None,
                          world_dev=None, convergence_diff=None, sparse_idx=None,
                          user_flagged=False)
    _, rule = ev.evaluate(rec, step=100)
    assert rule == 'safety_near_miss', f"expected safety rule, got '{rule}'"
    # 无规则命中 → periodic 兜底
    rec2 = SimpleNamespace(step=100, safety_margin=0.5, convergence_entropy=None,
                           world_dev=None, convergence_diff=None, sparse_idx=None,
                           user_flagged=False)
    score2, rule2 = ev.evaluate(rec2, step=100)
    assert rule2 == 'periodic' and abs(score2 - 0.3) < 1e-9
    print("  [7] periodic 兜底 ok（safety 规则优先，periodic 作 fallback）")


# ─── #8 _check_state_jump score 饱和 ────────────────────────────────────────

def test_08_state_jump_score_no_saturation():
    rec = SimpleNamespace(convergence_diff=0.75)   # th=0.5，刚过阈值
    is_imp, score = _check_state_jump(rec, {'state_jump': 0.5})
    assert is_imp and abs(score - 0.5) < 1e-9, f"score={score}, expected 0.5"
    print(f"  [8] state_jump score={score:.3f}（0.5/1.0，不饱和）")


# ─── #9 _jitted_ema 把 batch 总和广播到每个码本行 ──────────────────────────

def test_09_ema_no_collapse():
    from train.train_memory import _jitted_ema
    rng = np.random.RandomState(0)
    M, d = 3, 4
    centers = np.array([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 5.0, 0]],
                       dtype=np.float32)
    params = {
        'sparse': {'C': centers.copy()},
        'manifold': {'C': centers.copy()},
    }
    ema_state = {
        'sparse': {'N': jnp.zeros(M), 'm': jnp.zeros((M, d))},
        'manifold': {'N': jnp.zeros(M), 'm': jnp.zeros((M, d))},
        'binding': {},
    }
    for _ in range(300):
        idx = rng.randint(0, M, size=8)
        z = (centers[idx] + rng.normal(0, 0.1, (8, d))).astype(np.float32)
        params, ema_state = _jitted_ema(params, ema_state, jnp.asarray(z))

    C = np.asarray(params['sparse']['C'])
    D = np.linalg.norm(C[:, None] - C[None, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert D.min() > 0.1, f"sparse 码本坍缩: min pair dist={D.min():.4f}"

    Cm = np.asarray(params['manifold']['C'])
    Dm = np.linalg.norm(Cm[:, None] - Cm[None, :], axis=-1)
    np.fill_diagonal(Dm, np.inf)
    assert Dm.min() > 0.1, f"manifold 码本坍缩: min pair dist={Dm.min():.4f}"
    print(f"  [9] 300 步后码间最小距离: sparse={D.min():.4f}, "
          f"manifold={Dm.min():.4f} (>0.1)")


# ─── #10 mode_activation（概率）当 logits ───────────────────────────────────

def test_10_self_mode_sampling_proportional():
    rng = jax.random.PRNGKey(42)
    d, M = 8, 4
    params = init_self_params(rng, d, M_self=M)
    # 峰值激活：mode 0 占 0.7。logits=τ·log(p) → 采样 ∝ p^τ = p。
    # 旧实现 logits=τ·p → softmax 锐化 → mode 0 占比仅 ~0.25。
    state = {'step': 0,
             'mode_activation': jnp.array([0.7, 0.1, 0.1, 0.1]),
             'temp_avg': jnp.zeros(d),
             'gamma_self': 0.99}
    core = np.asarray(params['core'])
    modes = np.asarray(params['modes'])
    counts = np.zeros(M)
    for i in range(2000):
        o_self, _, _ = self_lattice_forward(
            params, state, rng=jax.random.PRNGKey(i), training=True)
        o = np.asarray(o_self).ravel()
        # STE 下 o_self = core + modes[mode_idx]，用最近 mode 反推采样索引
        idx = int(np.argmin(np.sum(
            (o[None, :] - core[None, :] - modes) ** 2, axis=-1)))
        counts[idx] += 1
    props = counts / counts.sum()
    assert 0.55 <= props[0] <= 0.85, f"mode0 占比 {props[0]:.3f}, 期望 ≈0.7"
    assert np.all((props[1:] >= 0.05) & (props[1:] <= 0.25)), f"props {props}"
    print(f"  [10] 2000 次采样 mode 占比: {np.round(props, 3)} (≈[.7,.1,.1,.1])")


# ─── #11 gvalue 边界不一致（margins == 0）──────────────────────────────────

def test_11_gvalue_margin_zero_safe():
    C_pos, C_neg = make_global_value_vectors(d=8)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 8))  # 原点对 ±0.9 锚点距离对称 → margins 恰为 0
    is_safe, margins, law = gv.check_safety_batch(z, safety_margin_relative=0.0)
    assert float(margins[0, 0]) == 0.0
    assert bool(is_safe[0, 0]), f"margins==0 应为 safe，got is_safe={is_safe}"
    assert int(law[0, 0]) == -1
    print(f"  [11] margins==0.0 → safe={bool(is_safe[0, 0])}, "
          f"violated_law={int(law[0, 0])}")


# ─── #12 abs(z) 双向触发 ────────────────────────────────────────────────────

def test_12_safety_margin_direction():
    # 安全边界升高（更安全）→ 不应触发（safety_near_miss 是低侧信号）
    det_up = AnomalyDetector(stats_window=100, z_threshold=3.0)
    base = SimpleNamespace(safety_margin=1.0)
    for i in range(50):
        det_up.feed(i, base)
    jump = SimpleNamespace(safety_margin=5.0)
    n_up = 0
    for i in range(50, 53):
        n_up += len(det_up.feed(i, jump))
    assert n_up == 0, f"margin 升高不应触发，got {n_up} signals"

    # 安全边界骤降 → 应触发
    det_down = AnomalyDetector(stats_window=100, z_threshold=3.0)
    for i in range(50):
        det_down.feed(i, base)
    drop = SimpleNamespace(safety_margin=0.01)
    sigs = []
    for i in range(50, 57):
        sigs.extend(det_down.feed(i, drop))
    n_drop = sum(1 for s in sigs if s.signal_type == 'safety_near_miss')
    assert n_drop >= 1, "margin 骤降应触发 safety_near_miss"
    print(f"  [12] margin↑: 0 触发; margin↓: {n_drop} 触发（方向正确）")


# ─── #13 get_stats 用 epoch 秒查步号 ────────────────────────────────────────

def test_13_get_stats_active_corrections():
    rl = ReflectionLoop()
    for i in range(5):
        rl.feed(i, SimpleNamespace(safety_margin=1.0))
    rl.registry.register(CorrectionAction(
        action_type='test', target='t', params={},
        decay_steps=50, created_step=4, description=''))
    stats = rl.get_stats()
    assert stats['n_active_corrections'] == 1, \
        f"刚写入的修正应活跃，got {stats['n_active_corrections']}"
    # 显式传步号同样正确
    assert rl.get_stats(current_step=10)['n_active_corrections'] == 1
    assert rl.get_stats(current_step=100)['n_active_corrections'] == 0
    print(f"  [13] n_active_corrections={stats['n_active_corrections']} "
          f"（步号判定，非 epoch 秒）")


# ─── 运行 ───────────────────────────────────────────────────────────────────

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print("train.test_fixes_aux — 辅助子系统修复回归（#1-#13）")
    print("=" * 60)
    n_fail = 0
    for t in tests:
        name = t.__name__.replace('test_', '#')
        try:
            t()
            print(f"  PASS {name}")
        except Exception as e:
            n_fail += 1
            print(f"  FAIL {name}: {e}")
    print("=" * 60)
    print(f"  {len(tests) - n_fail}/{len(tests)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
