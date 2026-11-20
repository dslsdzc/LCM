"""行为探索 — 内在动机驱动的路由偏置元学习。

核心思想：系统通过随机扰动路由格偏置，观察对内部张力的影响，
自发学习哪些偏置能有效降低张力。完全基于试错，无预设因果关系。

行为空间: b ∈ R^6, 每维离散化为 3 个值 (-0.2, 0, +0.2)。
张力信号: 预测误差 T_pred、价值冲突 T_conf、资源耗竭 T_res。

设计详见 e.md §六。
"""
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ─── 张力信号 ──────────────────────────────────────────────────────────────────

@dataclass
class TensionSignals:
    """系统可观测的内部张力，归一化到 [0, 1]。

    Attributes:
        T_pred: 预测误差张力 — 预测缓存近期误差。
        T_conf: 价值冲突张力 — 各格价值标量差异。
        T_res: 资源耗竭张力 — 检索距离、融合熵、步时。
        combined: 三种张量的线性组合（权重可调）。
    """
    T_pred: float = 0.0
    T_conf: float = 0.0
    T_res: float = 0.0

    @property
    def combined(self) -> float:
        return (self.T_pred + self.T_conf + self.T_res) / 3.0

    def __repr__(self) -> str:
        return (f"T_pred={self.T_pred:.3f} T_conf={self.T_conf:.3f} "
                f"T_res={self.T_res:.3f} U={self.combined:.3f}")


# ─── 偏置探索器 ────────────────────────────────────────────────────────────────

N_LATTICES = 6  # HRQ, Sparse, LowRank, Manifold, Binding, Contrast
DISCRETE_VALUES = [-0.2, 0.0, 0.2]  # 每维 3 个离散值


class BehaviorExplorer:
    """路由偏置探索器 — 多臂老虎机，独立每维收益估计。

    每维独立维护 3 个离散值的平均收益，ε-greedy 选择。
    不学习联合分布（3^6=729 太大），只学每维边际贡献。

    Args:
        n_dim: 偏置维度数（默认 6 = 六个功能格）。
        n_discrete: 每维离散值数（默认 3）。
        alpha: 收益滑动平均系数。
        explore_prob: 进入元学习模式的概率。
        eval_window: 评估张力变化的窗口步数。
        safety_ban_threshold: 安全中断率超过此值自动禁止该偏置。
    """
    LATTICE_NAMES = ['HRQ', 'Sparse', 'LowRank', 'Manifold', 'Binding', 'Contrast']

    def __init__(self, n_dim: int = N_LATTICES,
                 n_discrete: int = len(DISCRETE_VALUES),
                 alpha: float = 0.1,
                 explore_prob: float = 0.2,
                 eval_window: int = 10,
                 safety_ban_threshold: float = 0.3):
        self.n_dim = n_dim
        self.n_discrete = n_discrete
        self.alpha = alpha
        self.explore_prob = explore_prob
        self.eval_window = eval_window
        self.safety_ban_threshold = safety_ban_threshold

        # 收益估计: P[dim][discrete_idx] = mean_benefit
        self._P = np.zeros((n_dim, n_discrete), dtype=np.float64)
        # 采样计数
        self._N = np.zeros((n_dim, n_discrete), dtype=np.int32)
        # 禁用标记: banned[dim][discrete_idx] = True
        self._banned = np.zeros((n_dim, n_discrete), dtype=bool)

        # 当前偏置（正在评估中）
        self.current_bias: Optional[np.ndarray] = None
        self._bias_start_step: int = 0
        self._eval_buffer: deque = deque(maxlen=eval_window * 2)
        # Level-2 suggestions are bounded deltas on top of the learned bias.
        self.bias_override = np.zeros(n_dim, dtype=np.float32)

        # 统计
        self.n_explores = 0
        self.n_updates = 0
        self._last_eval_step = 0

    # ── 偏置选择 ──

    def should_explore(self, step: int) -> bool:
        """当前步是否应该进入元学习模式。"""
        return np.random.random() < self.explore_prob

    def select_bias(self, step: int) -> np.ndarray:
        """ε-greedy 选择偏置向量。

        Returns:
            (n_dim,) float32 偏置向量，值在 DISCRETE_VALUES 中。
        """
        self.n_explores += 1
        self._bias_start_step = step

        bias = np.zeros(self.n_dim, dtype=np.float32)
        for dim in range(self.n_dim):
            if np.random.random() < 0.1:  # 探索
                available = [v for vi, v in enumerate(DISCRETE_VALUES)
                             if not self._banned[dim, vi]]
                val = np.random.choice(available) if available else 0.0
            else:  # 利用
                # 选收益最高的未禁用值
                best_vi = -1
                best_q = -np.inf
                for vi in range(self.n_discrete):
                    if self._banned[dim, vi]:
                        continue
                    q = self._P[dim, vi]
                    if q > best_q:
                        best_q = q
                        best_vi = vi
                val = DISCRETE_VALUES[best_vi] if best_vi >= 0 else 0.0
            bias[dim] = val

        self.current_bias = bias
        return bias.copy()

    def select_greedy_bias(self) -> np.ndarray:
        """始终选当前最佳偏置（无探索）。"""
        bias = np.zeros(self.n_dim, dtype=np.float32)
        for dim in range(self.n_dim):
            best_vi = int(np.argmax([
                self._P[dim, vi] if not self._banned[dim, vi] else -np.inf
                for vi in range(self.n_discrete)
            ]))
            bias[dim] = DISCRETE_VALUES[best_vi]
        return bias

    def get_best_bias(self) -> np.ndarray:
        """Current best bias including bounded Level-2 corrections."""
        return np.clip(self.select_greedy_bias() + self.bias_override,
                       -0.2, 0.2).astype(np.float32)

    @staticmethod
    def should_update(step: int, interval: int = 10) -> bool:
        """e.md §4: behavior exploration runs every M=10 steps, not per step."""
        return interval > 0 and step % interval == 0

    def plan_step(self, step: int, danger: bool = False,
                  interval: int = 10) -> np.ndarray:
        """Choose the routing bias for this step (danger has priority).

        * danger: return a zero bias and abandon any in-flight evaluation;
        * interval boundary: epsilon-greedy exploration may pick a new bias;
        * otherwise: reuse the learned best bias.
        """
        if danger:
            self.current_bias = None
            return np.zeros(self.n_dim, dtype=np.float32)
        # A selected bias is evaluated for the whole interval; do not snap
        # back to the greedy choice on the very next step.
        if (self.current_bias is not None
                and self.should_update(step, interval)):
            self.finalize_evaluation(step)
        if self.current_bias is not None:
            return self.current_bias.copy()
        # Ruling 46: never select a bias before a tension baseline exists.
        # Checked *after* finalization, so the buffer cleared by an
        # unevaluable bias cannot immediately produce another unbaselined
        # selection on the same boundary.
        if (self.should_update(step, interval)
                and self.should_explore(step)
                and len(self._eval_buffer) >= 3):
            return self.select_bias(step)
        return self.get_best_bias()

    def apply_bias_delta(self, dim: int, delta: float) -> bool:
        """Apply a bounded Level-2 bias correction to one dimension."""
        if not (0 <= int(dim) < self.n_dim):
            return False
        delta = float(np.clip(delta, -0.2, 0.2))
        self.bias_override[int(dim)] = float(np.clip(
            self.bias_override[int(dim)] + delta, -0.2, 0.2))
        return True

    # ── 张力反馈 ──

    def observe_tension(self, step: int, tension: TensionSignals,
                        safety_breach: bool = False):
        """记录一步的张力值。

        Args:
            step: 当前步号。
            tension: 当前步的张力信号。
            safety_breach: 该步是否有安全中断。
        """
        self._eval_buffer.append({
            'step': step,
            'tension': tension.combined,
            'T_pred': tension.T_pred,
            'T_conf': tension.T_conf,
            'T_res': tension.T_res,
            'safety_breach': safety_breach,
            'bias': self.current_bias.copy() if self.current_bias is not None else None,
        })

    def finalize_evaluation(self, step: int):
        """结束当前偏置的评估，更新收益估计。

        比较偏置改变前后的平均张力，计算收益。
        应在偏置保持不变 eval_window 步后调用。
        """
        if self.current_bias is None:
            return

        buffer = list(self._eval_buffer)

        # 分离 baseline（偏置改变前）和 evaluation（偏置改变后）
        baseline = [e for e in buffer if e['step'] < self._bias_start_step]
        evaluation = [e for e in buffer
                      if e['step'] >= self._bias_start_step
                      and e['bias'] is not None
                      and np.array_equal(e['bias'], self.current_bias)]

        if len(baseline) < 3 or len(evaluation) < 3:
            # Ruling 46: an unevaluable bias must not stay installed forever.
            # Clear it and the stale buffer; the next selection starts a new
            # baseline instead of freezing `current_bias` at an old choice.
            self.current_bias = None
            self._eval_buffer.clear()
            return

        # 平均张力
        U_before = float(np.mean([e['tension'] for e in baseline]))
        U_after = float(np.mean([e['tension'] for e in evaluation]))

        # 安全检查
        breach_rate = float(np.mean([e['safety_breach'] for e in evaluation]))

        benefit = U_before - U_after  # 正 = 张力下降

        # 更新每维的收益估计
        for dim in range(self.n_dim):
            vi = self._discrete_index(self.current_bias[dim])
            n = self._N[dim, vi]
            old_q = self._P[dim, vi]
            new_q = (1 - self.alpha) * old_q + self.alpha * benefit
            self._P[dim, vi] = new_q
            self._N[dim, vi] = n + 1

        # 安全禁令
        if breach_rate > self.safety_ban_threshold:
            for dim in range(self.n_dim):
                vi = self._discrete_index(self.current_bias[dim])
                self._banned[dim, vi] = True

        self.n_updates += 1
        self.current_bias = None

    @staticmethod
    def _discrete_index(val: float) -> int:
        """找到最接近的离散值索引。"""
        return int(np.argmin(np.abs(np.array(DISCRETE_VALUES, dtype=np.float32) - val)))

    # ── 统计 ──

    def print_summary(self) -> None:
        print(f"\n{'=' * 46}")
        print(f"  行为探索 (Behavior Explorer)")
        print(f"{'=' * 46}")
        print(f"  Explores:     {self.n_explores}")
        print(f"  Updates:      {self.n_updates}")
        print(f"  Explore prob: {self.explore_prob}")
        print(f"  每维收益估计 (最优加粗):")
        for dim in range(self.n_dim):
            vals = []
            for vi, v in enumerate(DISCRETE_VALUES):
                if self._banned[dim, vi]:
                    s = f"{v:+.1f}=BAN"
                else:
                    q = self._P[dim, vi]
                    s = f"{v:+.1f}={q:.4f}"
                vals.append(s)
            best_idx = int(np.argmax([
                self._P[dim, vi] if not self._banned[dim, vi] else -np.inf
                for vi in range(self.n_discrete)
            ]))
            best_str = f"{DISCRETE_VALUES[best_idx]:+.1f}"
            print(f"  {self.LATTICE_NAMES[dim]:8s}: {'  '.join(vals)}  "
                  f"→ best={best_str}")
        print(f"{'=' * 46}\n")


# ─── 张力计算工具 ───────────────────────────────────────────────────────────────

def compute_tension_from_aux(aux: dict, z_cur: np.ndarray,
                              z_q: np.ndarray,
                              pred_error: Optional[float] = None,
                              d_model: int = 256) -> TensionSignals:
    """从模型 forward 的 aux 输出中计算三种张力。

    Args:
        aux: model.forward() 返回的 aux 字典。
        z_cur: 推理前的状态向量。
        z_q: 推理后的状态向量。
        pred_error: 预测缓存误差（如果可用）。
        d_model: 向量维度，用于缩放。

    Returns:
        TensionSignals 实例，各分量归一化到 [0, 1]。
    """
    T_pred = 0.0
    T_conf = 0.0
    T_res = 0.0

    # T_pred: 预测误差。Ruling 39 uses the nonnegative one-step residual
    # ||z_next - z_cur||² in training; legacy callers pass a negative
    # euclidean distance. Normalise both to a nonnegative error scale.
    if pred_error is not None:
        err = -float(pred_error) if pred_error < 0 else float(np.sqrt(
            max(float(pred_error), 0.0)))
        T_pred = float(np.clip(np.tanh(err / d_model), 0.0, 1.0))

    # T_conf: 价值冲突
    value_signals = aux.get('value_signals')
    if value_signals is not None:
        # 各格价值标量的标准差作为冲突度量
        vs = np.asarray(value_signals)
        if vs.ndim > 0 and vs.size > 1:
            std_v = float(np.std(vs))
            T_conf = float(np.clip(np.tanh(std_v), 0.0, 1.0))

    # T_res: 资源耗竭
    soft_mask = aux.get('soft_mask')
    if soft_mask is not None:
        mask = np.asarray(soft_mask)
        if mask.ndim > 0 and mask.size > 1:
            if mask.ndim > 1:
                mask = mask[0]  # (B, n_lattices) → (n_lattices,)
            mask = mask + 1e-8
            entropy = -np.sum(mask * np.log(mask))
            max_ent = np.log(len(mask))
            T_res = float(np.clip(entropy / max_ent, 0.0, 1.0))

    return TensionSignals(T_pred=T_pred, T_conf=T_conf, T_res=T_res)


def compute_tension_from_trace(trace_step: dict,
                                pred_error: Optional[float] = None,
                                d_model: int = 256) -> TensionSignals:
    """从 C 引擎 trace 步中计算张力（推理路径）。

    Args:
        trace_step: get_trace() 返回的单个步 dict。
        pred_error: 预测缓存误差（如果可用）。
        d_model: 向量维度。

    Returns:
        TensionSignals 实例。
    """
    T_pred = 0.0
    T_conf = 0.0
    T_res = 0.0

    if pred_error is not None:
        err = -float(pred_error) if pred_error < 0 else float(np.sqrt(
            max(float(pred_error), 0.0)))
        T_pred = float(np.clip(np.tanh(err / d_model), 0.0, 1.0))

    confs = trace_step.get('confidences')
    if confs is not None:
        c = np.asarray(confs)
        if c.size > 1:
            std_c = float(np.std(c))
            T_conf = float(np.clip(np.tanh(std_c), 0.0, 1.0))

    weights = trace_step.get('weights')
    if weights is not None:
        w = np.asarray(weights) + 1e-8
        if w.size > 1:
            entropy = -np.sum(w * np.log(w))
            max_ent = np.log(len(w))
            T_res = float(np.clip(entropy / max_ent, 0.0, 1.0))

    return TensionSignals(T_pred=T_pred, T_conf=T_conf, T_res=T_res)


__all__ = [
    'TensionSignals',
    'BehaviorExplorer',
    'compute_tension_from_aux',
    'compute_tension_from_trace',
]
