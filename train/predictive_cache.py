"""预测缓存 — 隐式推理的副产品，通用预测器。

核心机制：预测是记忆的复用，不是专门的预测模型。
每步推理产生 (state_before, op_signature, state_after) 三元组存入缓存。
查询时用多臂老虎机动态选择匹配策略。

设计详见 e.md。
"""
import random
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable

# Cython-accelerated matching (falls back to pure Python if not compiled)
_HAS_CYTHON = False
for _cy_mod in ('_lcm_cy', 'train._lcm_cy'):
    try:
        import importlib
        _cy = importlib.import_module(_cy_mod)
        match_euclidean_cy = _cy.match_euclidean_cy
        match_hamming_cy = _cy.match_hamming_cy
        match_exact_cy = _cy.match_exact_cy
        _HAS_CYTHON = True
        break
    except ImportError:
        continue


# ─── 操作签名 ──────────────────────────────────────────────────────────────────

# C 推理引擎 trace 中可用的字段: weights(7), confidences(7), has_conflict(1)
# 从中构造轻量签名
SIG_WEIGHT_BITS = 7    # 每个权重 > 0.1 标志
SIG_CONF_BITS = 7      # 每个置信度正负标志
SIG_TOP_BITS = 3       # 权重最大的格索引
SIG_SAFE_BITS = 1      # 安全标志
SIG_TOTAL_BITS = SIG_WEIGHT_BITS + SIG_CONF_BITS + SIG_TOP_BITS + SIG_SAFE_BITS  # 18

# 训练路径（JAX）有完整格索引，可以打包更丰富的签名
# 留出扩展位


def pack_trace_sig(weights, confidences, has_conflict):
    """从 C engine trace step 打包轻量签名到 uint64。

    Args:
        weights: (7,) float array — 各格的路由权重。
        confidences: (7,) float array — 各格的置信度。
        has_conflict: bool — 是否有安全冲突。

    Returns:
        int (uint64) 格式的签名。
    """
    sig = 0
    pos = 0

    # Top-1 格 (3 bits)
    top = int(np.argmax(weights)) & 0x7
    sig |= top << pos
    pos += SIG_TOP_BITS

    # 活跃格模式 (7 bits): weight > 0.1
    active = (np.asarray(weights) > 0.1).astype(np.int64)
    for i in range(7):
        sig |= (int(active[i]) & 0x1) << (pos + i)
    pos += SIG_WEIGHT_BITS

    # 置信度符号 (7 bits)
    for i in range(7):
        sig |= (int(confidences[i] > 0) & 0x1) << (pos + i)
    pos += SIG_CONF_BITS

    # 安全标志 (1 bit)
    sig |= (int(has_conflict) & 0x1) << pos

    return sig


def pack_lattice_sig(hrq_idx, sparse_idx, lowrank_idx, man_idx,
                     bind_idx, contrast_idx, safety_flag):
    """从完整格索引打包签名（训练路径使用）。

    位布局（55 bits → uint64）:
        0-7:   HRQ 顶层索引 (8 bits)
        8-16:  Sparse 索引 (9 bits)
        17-26: LowRank 索引 (10 bits)
        27-35: Manifold 索引 (9 bits)
        36-44: Binding 索引 (9 bits, 三层 xor 压缩)
        45-53: Contrast 索引 (9 bits)
        54:    安全标志 (1 bit)
    """
    sig = 0
    sig |= (int(hrq_idx) & 0xFF) << 0
    sig |= (int(sparse_idx) & 0x1FF) << 8
    sig |= (int(lowrank_idx) & 0x3FF) << 17
    sig |= (int(man_idx) & 0x1FF) << 27
    sig |= (int(bind_idx) & 0x1FF) << 36
    sig |= (int(contrast_idx) & 0x1FF) << 45
    sig |= (int(safety_flag) & 0x1) << 54
    return sig


# ─── 缓存条目 ─────────────────────────────────────────────────────────────────

@dataclass
class PredictEntry:
    """单步推理的 (state_before, op_signature, state_after)."""
    sig: int                # uint64 操作签名
    z_cur: np.ndarray       # (d,) float32 — 推理前状态
    z_next: np.ndarray      # (d,) float32 — 推理后状态
    step: int               # 全局步号
    task_outcome: float = 0.0  # 延迟写入: 所属任务的最终指标


# ─── 预测缓存（环形缓冲区）────────────────────────────────────────────────────

class PredictionCache:
    """固定容量的预测缓存，环形缓冲区。

    存储 (state_before, op_signature, state_after) 三元组。
    用于多臂老虎机的三种匹配方法。

    Args:
        capacity: 缓存容量（默认 2048）。
        d_model: 向量维度。
    """
    def __init__(self, capacity: int = 2048, d_model: int = 256):
        self.capacity = capacity
        self.d_model = d_model
        self._buffer = [None] * capacity
        self._head = 0
        self._count = 0

        # Flat arrays for Cython-accelerated matching.
        # Always maintained alongside _buffer (O(1) per append, ~2 MB at 2048×256).
        self._z_cur_buf = np.zeros((capacity, d_model), dtype=np.float32)
        self._z_next_buf = np.zeros((capacity, d_model), dtype=np.float32)
        self._sig_buf = np.zeros(capacity, dtype=np.int64)
        self._valid = np.zeros(capacity, dtype=np.uint8)

        # Real hit/miss counters for the reflection cache-trend analysis.
        # A lookup is a prediction attempt; a hit is a non-None z_pred.
        self.hits = 0
        self.misses = 0
        self.n_lookups = 0

    def record_lookup(self, hit: bool) -> None:
        self.n_lookups += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def hit_rate(self) -> Optional[float]:
        if self.n_lookups == 0:
            return None
        return self.hits / self.n_lookups

    def append(self, sig: int, z_cur: np.ndarray, z_next: np.ndarray,
               step: int):
        entry = PredictEntry(
            sig=sig,
            z_cur=np.ascontiguousarray(z_cur, dtype=np.float32),
            z_next=np.ascontiguousarray(z_next, dtype=np.float32),
            step=step,
        )
        self._buffer[self._head] = entry

        # Flat arrays for Cython
        self._z_cur_buf[self._head] = entry.z_cur
        self._z_next_buf[self._head] = entry.z_next
        self._sig_buf[self._head] = sig
        self._valid[self._head] = 1

        self._head = (self._head + 1) % self.capacity
        self._count += 1

    def __len__(self):
        return min(self._count, self.capacity)

    def __iter__(self):
        for i in range(len(self)):
            yield self._buffer[(self._head - len(self) + i) % self.capacity]

    def to_list(self):
        return [e for e in self if e is not None]

    # ── 三种匹配操作 ──

    def match_exact(self, z_cur: np.ndarray, sig: int) -> Optional[PredictEntry]:
        """M0: 精确匹配 (z_cur, sig) → z_next."""
        if _HAS_CYTHON:
            result = match_exact_cy(
                self._z_cur_buf, self._z_next_buf,
                self._sig_buf, self._valid,
                np.ascontiguousarray(z_cur, dtype=np.float32),
                np.int64(sig))
            if result is None:
                return None
            z_next, _ = result
            return PredictEntry(sig=sig, z_cur=z_cur, z_next=z_next, step=0)

        # Pure-Python fallback
        best = None
        best_dist = float('inf')
        for e in self:
            if e is None or e.sig != sig:
                continue
            d = np.sum((e.z_cur - z_cur) ** 2)
            if d < best_dist:
                best_dist = d
                best = e
        return best

    def match_hamming(self, z_cur: np.ndarray, sig: int,
                       max_dist: int = 3, K: int = 3) -> Optional[np.ndarray]:
        """M1: 签名汉明距离 ≤ max_dist 的条目加权融合 z_next."""
        if _HAS_CYTHON:
            return match_hamming_cy(
                self._z_cur_buf, self._z_next_buf,
                self._sig_buf, self._valid,
                np.ascontiguousarray(z_cur, dtype=np.float32),
                np.int64(sig), max_dist, K)

        # Pure-Python fallback
        candidates = []
        for e in self:
            if e is None:
                continue
            hd = _popcount64(e.sig ^ sig)
            if hd <= max_dist:
                d_euc = np.sum((e.z_cur - z_cur) ** 2)
                candidates.append((d_euc, e.z_next))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:K]

        weights = np.array([1.0 / (d + 1e-8) for d, _ in candidates], dtype=np.float32)
        weights /= weights.sum() + 1e-8

        z_pred = np.zeros_like(candidates[0][1])
        for (_, zn), w in zip(candidates, weights):
            z_pred += w * zn
        return z_pred

    def match_euclidean(self, z_cur: np.ndarray, K: int = 5
                         ) -> Optional[np.ndarray]:
        """M2: 忽略签名，按 z_cur 欧氏距离加权融合 z_next."""
        if _HAS_CYTHON:
            return match_euclidean_cy(
                self._z_cur_buf, self._z_next_buf, self._valid,
                np.ascontiguousarray(z_cur, dtype=np.float32), K)

        # Pure-Python fallback
        candidates = []
        for e in self:
            if e is None:
                continue
            d = np.sum((e.z_cur - z_cur) ** 2)
            candidates.append((d, e.z_next))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:K]

        weights = np.array([1.0 / (d + 1e-8) for d, _ in candidates], dtype=np.float32)
        weights /= weights.sum() + 1e-8

        z_pred = np.zeros_like(candidates[0][1])
        for (_, zn), w in zip(candidates, weights):
            z_pred += w * zn
        return z_pred

    def utilization(self) -> float:
        return len(self) / self.capacity


def _popcount64(x: int) -> int:
    """uint64 汉明重量（Python int 版）。"""
    return x.bit_count() if hasattr(int, 'bit_count') else bin(x & 0xFFFFFFFFFFFFFFFF).count('1')


# ─── 多臂老虎机匹配器 ─────────────────────────────────────────────────────────

class MABMatcher:
    """多臂老虎机动态选择匹配操作。

    三臂:
        M0 — 精确匹配 (z_cur, sig) → z_next
        M1 — 签名汉明距离加权融合
        M2 — 状态欧氏距离加权融合

    学习: ε-greedy + Q 学习，参数扰动。
    设计详见 e.md 第二节。

    Args:
        d_model: 向量维度。
        cache: PredictionCache 实例。
        alpha: Q 学习率。
        epsilon_init: 初始探索率。
    """
    OP_NAMES = ['M0', 'M1', 'M2']

    # 每臂的超参数空间（参数扰动）
    PARAM_SPACE = {
        'M0': [None],
        'M1': [{'max_dist': d, 'K': k} for d in [1, 2, 3] for k in [2, 3, 5]],
        'M2': [{'K': k} for k in [3, 5, 10]],
    }

    def __init__(self, d_model: int, cache: PredictionCache,
                 alpha: float = 0.1, epsilon_init: float = 0.3):
        self.d_model = d_model
        self.cache = cache
        self.alpha = alpha
        self.epsilon = epsilon_init

        # Q 表: (op_name, param_idx) → Q 值
        self._Q = {}
        for op in self.OP_NAMES:
            for pi in range(len(self.PARAM_SPACE[op])):
                self._Q[(op, pi)] = 0.0

        self.steps = 0
        self.last_pick = None  # (op, param_idx)

    def predict(self, z_cur: np.ndarray, sig: int) -> Tuple[Optional[np.ndarray], str, Optional[dict]]:
        """选择并执行匹配操作，返回预测结果。

        Args:
            z_cur: 当前状态 (d,)。
            sig: 当前操作签名（或 0 如果不可用）。

        Returns:
            (z_pred or None, op_name, param_dict or None)。
        """
        if len(self.cache) < 3:
            self.cache.record_lookup(False)
            return None, '', None  # 缓存太冷，不预测

        # ε-greedy 选择操作
        if random.random() < self.epsilon:
            op = random.choice(self.OP_NAMES)
        else:
            # 选 Q 值最高的操作
            op_q = {}
            for op_name in self.OP_NAMES:
                vals = [self._Q[(op_name, pi)]
                        for pi in range(len(self.PARAM_SPACE[op_name]))]
                op_q[op_name] = max(vals) if vals else 0.0
            op = max(op_q, key=op_q.get)

        # 参数扰动：从该操作的参数空间中按 Q 加权采样
        params = self.PARAM_SPACE[op]
        param_q = [self._Q[(op, pi)] for pi in range(len(params))]
        # Softmax 采样
        if random.random() < self.epsilon:
            pi = random.randrange(len(params))
        else:
            exp_q = np.exp(np.array(param_q, dtype=np.float64)
                           - max(param_q)) if max(param_q) != min(param_q) else np.ones(len(param_q))
            probs = exp_q / exp_q.sum()
            pi = np.random.choice(len(params), p=probs)
        param = params[pi]

        self.last_pick = (op, pi)

        # 执行匹配
        if op == 'M0':
            entry = self.cache.match_exact(z_cur, sig)
            z_pred = entry.z_next.copy() if entry is not None else None
        elif op == 'M1':
            z_pred = self.cache.match_hamming(z_cur, sig,
                                              max_dist=param['max_dist'],
                                              K=param['K'])
        elif op == 'M2':
            z_pred = self.cache.match_euclidean(z_cur, K=param['K'])
        else:
            z_pred = None
        self.cache.record_lookup(z_pred is not None)
        return z_pred, op, param

    def update(self, z_pred: np.ndarray, z_actual: np.ndarray,
               task_outcome: Optional[float] = None):
        """用预测结果更新 Q 表。

        Args:
            z_pred: 预测的 z_next。
            z_actual: 实际的 z_next。
            task_outcome: 可选的长期任务指标（由内在动机模块提供）。
        """
        if self.last_pick is None:
            return

        op, pi = self.last_pick

        # 即时奖励: 负欧氏距离
        reward = -float(np.sum((z_pred - z_actual) ** 2))

        # 如果有任务指标，混合奖励
        if task_outcome is not None:
            reward = 0.7 * reward + 0.3 * task_outcome

        # Q 更新
        old_q = self._Q[(op, pi)]
        self._Q[(op, pi)] = (1 - self.alpha) * old_q + self.alpha * reward

        self.steps += 1
        self.last_pick = None

        # ε 衰减
        if self.steps % 500 == 0:
            self.epsilon = max(0.05, self.epsilon * 0.95)

    def get_stats(self) -> dict:
        """返回匹配器的统计信息。"""
        op_counts = defaultdict(int)
        op_avg_q = {}

        for (op, pi), q in self._Q.items():
            op_counts[op] += 1
            if op not in op_avg_q:
                op_avg_q[op] = []
            op_avg_q[op].append(q)

        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'op_q': {op: float(np.mean(vals)) for op, vals in op_avg_q.items()},
            'op_best': {op: float(np.max(vals)) for op, vals in op_avg_q.items()},
        }


# ─── 内在动机（回报涌现）────────────────────────────────────────────────────────

@dataclass
class PredictionLogEntry:
    """一次预测调用的完整记录。"""
    op: str
    param: Optional[dict]
    z_pred: np.ndarray
    z_actual: np.ndarray
    step: int
    task_id: Optional[int] = None
    task_success: Optional[bool] = None
    task_steps: Optional[int] = None
    task_safety_breaches: Optional[int] = None
    user_feedback: float = 0.0
    proxy_reward: bool = True


class IntrinsicMotivation:
    """内在动机模块 — 无预设收益函数，异步回报分析。

    职责：
        1. 记录每次预测调用的原始事实。
        2. 慢速反思回路中，用长期任务完成指标推断预测价值。
        3. 逐步从临时代理奖励切换到纯任务指标。

    设计详见 e.md 第三节。

    Args:
        reflect_interval: 反思间隔（步数）。
        proxy_initial: 初期是否使用预测准确度作为代理奖励。
    """
    def __init__(self, reflect_interval: int = 1000, proxy_initial: bool = True):
        self.reflect_interval = reflect_interval
        self.proxy_enabled = proxy_initial
        self._log: List[PredictionLogEntry] = []
        self._log_capacity = 10000
        self._last_reflect = 0
        self._proxy_corr_streak = 0
        self._q_baseline_offset = {}  # (op, param_idx) → float

    def log(self, op: str, param: Optional[dict], z_pred: np.ndarray,
            z_actual: np.ndarray, step: int,
            task_id: Optional[int] = None,
            task_success: Optional[bool] = None,
            task_steps: Optional[int] = None,
            task_safety_breaches: Optional[int] = None,
            user_feedback: float = 0.0):
        entry = PredictionLogEntry(
            op=op, param=param,
            z_pred=z_pred.copy(), z_actual=z_actual.copy(),
            step=step, task_id=task_id,
            task_success=task_success, task_steps=task_steps,
            task_safety_breaches=task_safety_breaches,
            user_feedback=user_feedback,
            proxy_reward=self.proxy_enabled,
        )
        self._log.append(entry)
        if len(self._log) > self._log_capacity:
            self._log = self._log[-self._log_capacity:]

    def get_proxy_reward(self, z_pred: np.ndarray,
                          z_actual: np.ndarray) -> float:
        """临时代理奖励：负欧氏距离。"""
        return -float(np.sum((z_pred - z_actual) ** 2))

    def reflect(self, current_step: int, matcher: MABMatcher):
        """反思回路：离线分析预测价值，调整 Q 基线偏移。

        每 reflect_interval 步运行一次。

        Args:
            current_step: 当前步号。
            matcher: MABMatcher 实例，用于调整基线。
        """
        if current_step - self._last_reflect < self.reflect_interval:
            return

        self._last_reflect = current_step

        # 只分析最近区间内的日志
        recent = [e for e in self._log
                  if e.step > current_step - self.reflect_interval * 2]

        if len(recent) < 10:
            return

        # 按 op 分组，计算各组成功率
        op_outcomes = defaultdict(list)
        no_pred_outcomes = []
        for e in recent:
            if e.task_success is not None:
                if e.op:
                    op_outcomes[e.op].append(e.task_success)
                else:
                    no_pred_outcomes.append(e.task_success)

        if not no_pred_outcomes:
            return

        baseline_success = np.mean(no_pred_outcomes)
        baseline_steps = np.mean([e.task_steps for e in recent
                                  if e.task_steps is not None])

        adjustments = {}
        for op, outcomes in op_outcomes.items():
            if len(outcomes) < 3:
                continue
            op_success = np.mean(outcomes)
            # 与对照比较
            if op_success > baseline_success + 0.05:
                adjustments[op] = 0.01
            elif op_success < baseline_success - 0.05:
                adjustments[op] = -0.01
            else:
                adjustments[op] = 0.0

        # 检查代理奖励与任务指标的相关性
        if self.proxy_enabled:
            proxy_vals = []
            task_vals = []
            for e in recent:
                if e.task_success is not None and e.proxy_reward:
                    r = -float(np.sum((e.z_pred - e.z_actual) ** 2))
                    proxy_vals.append(r)
                    task_vals.append(1.0 if e.task_success else 0.0)

            if len(proxy_vals) >= 20:
                corr = np.corrcoef(proxy_vals, task_vals)[0, 1] if np.std(proxy_vals) > 0 and np.std(task_vals) > 0 else 0.0
                if corr > 0.7:
                    self._proxy_corr_streak += 1
                else:
                    self._proxy_corr_streak = 0

                if self._proxy_corr_streak >= 3:
                    self.proxy_enabled = False
                    print(f"[MOTIV] 代理奖励→纯任务指标切换 "
                          f"(corr={corr:.2f}, streak=3)")

        print(f"[MOTIV] 反思 @ step {current_step}: "
              f"baseline_success={baseline_success:.2f}  "
              + '  '.join(f"{op}={adj:+.3f}" for op, adj in adjustments.items())
              + (f"  proxy={'on' if self.proxy_enabled else 'off'}" if not self.proxy_enabled else ""))

    @property
    def n_logged(self) -> int:
        return len(self._log)


# ─── 便利组合 ─────────────────────────────────────────────────────────────────

class PredictiveSystem:
    """预测系统组合：缓存 + 匹配器 + 内在动机。"""
    def __init__(self, d_model: int, cache_capacity: int = 2048,
                 reflect_interval: int = 1000):
        self.cache = PredictionCache(capacity=cache_capacity, d_model=d_model)
        self.matcher = MABMatcher(d_model=d_model, cache=self.cache)
        self.motivation = IntrinsicMotivation(reflect_interval=reflect_interval)
        self.d_model = d_model

    def step(self, z_cur: np.ndarray, sig: int, z_actual: np.ndarray,
             step: int, task_id: Optional[int] = None,
             task_success: Optional[bool] = None,
             task_steps: Optional[int] = None,
             task_safety_breaches: Optional[int] = None,
             user_feedback: float = 0.0) -> Tuple[Optional[np.ndarray], str, Optional[dict]]:
        """完整的预测步骤。

        1. 查询缓存 → z_pred
        2. 写入 (sig, z_cur, z_actual) 到缓存
        3. 如果使用了预测，更新 MAB Q 表
        4. 记录到动机日志

        Args:
            z_cur: 推理前状态。
            sig: 操作签名（打包后）。
            z_actual: 推理后状态。
            step: 全局步号。
            task_*: 可选的长期任务指标。

        Returns:
            (z_pred or None, op_name, param or None)。
        """
        # 1. 查询
        z_pred, op, param = self.matcher.predict(z_cur, sig)

        # 2. 写入缓存
        self.cache.append(sig, z_cur, z_actual, step)

        # 3. 更新 MAB
        if z_pred is not None:
            proxy_r = self.motivation.get_proxy_reward(z_pred, z_actual)
            task_outcome = task_success if not self.motivation.proxy_enabled else None
            self.matcher.update(z_pred, z_actual, task_outcome=task_outcome)

            # 日志
            self.motivation.log(
                op=op, param=param,
                z_pred=z_pred, z_actual=z_actual, step=step,
                task_id=task_id,
                task_success=task_success,
                task_steps=task_steps,
                task_safety_breaches=task_safety_breaches,
                user_feedback=user_feedback)
        else:
            # 记录"未预测"用于对照基线
            self.motivation.log(
                op='', param=None,
                z_pred=z_cur, z_actual=z_actual, step=step,
                task_id=task_id,
                task_success=task_success,
                task_steps=task_steps,
                task_safety_breaches=task_safety_breaches,
                user_feedback=user_feedback)

        # 4. 反思（异步）
        self.motivation.reflect(step, self.matcher)

        return z_pred, op, param

    def get_stats(self) -> dict:
        return {
            'cache_util': self.cache.utilization(),
            'n_cache': len(self.cache),
            'hit_rate': self.cache.hit_rate(),
            'hits': self.cache.hits,
            'misses': self.cache.misses,
            'n_lookups': self.cache.n_lookups,
            'matcher': self.matcher.get_stats(),
            'motivation': {
                'n_logged': self.motivation.n_logged,
                'proxy_enabled': self.motivation.proxy_enabled,
            },
        }


__all__ = [
    'PredictEntry',
    'PredictionCache',
    'MABMatcher',
    'IntrinsicMotivation',
    'PredictiveSystem',
    'pack_trace_sig',
    'pack_lattice_sig',
]
