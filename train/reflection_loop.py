"""反思回路 — 轨迹审计器。

职责：异常检测 → 因果追溯 → 局部修正。
不固定运行频率，由异常信号触发。可审计，可解释。

设计详见 e.md §七（原计划外，补充规格）。
"""
import time
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple


# ─── 异常检测 ──────────────────────────────────────────────────────────────────

@dataclass
class AnomalySignal:
    """一个异常信号实例。

    Attributes:
        signal_type: 异常类型标识。
        severity: 严重程度 [0, 1]。
        step: 触发步号。
        value: 触发时的原始值。
        threshold: 触发的阈值。
        context: 额外上下文（如相关步范围）。
    """
    signal_type: str
    severity: float
    step: int
    value: float
    threshold: float
    context: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"[{self.signal_type}] sev={self.severity:.2f} "
                f"@ step {self.step} (val={self.value:.3f} > th={self.threshold:.3f})")


class SlidingStats:
    """滑动窗口统计量 — O(1) 在线均值/方差。

    Args:
        window: 窗口大小。
        z_threshold: Z-score 异常阈值（默认 3.0）。
    """
    def __init__(self, window: int = 100, z_threshold: float = 3.0):
        self.window = window
        self.z_threshold = z_threshold
        self._buffer: deque = deque(maxlen=window)
        self._mean = 0.0
        self._var = 0.0
        self._count = 0

    def update(self, value: float) -> Optional[float]:
        """更新滑动窗口，返回 z-score（异常程度）。"""
        self._buffer.append(value)
        self._count += 1

        n = len(self._buffer)

        # 增量均值和方差（Welford 在线算法）
        # _var 维护 M2 累计量（不除），std 在读取时计算 sqrt(M2/(n-1))。
        if n == 1:
            self._mean = value
            self._var = 0.0
        else:
            old_mean = self._mean
            self._mean += (value - old_mean) / n
            self._var += (value - old_mean) * (value - self._mean)

        if n < 10:
            return None  # 冷启动（统计量仍正确更新，只是不判异常）

        std = np.sqrt(self._var / (n - 1)) + 1e-8
        z = (value - self._mean) / std
        return float(z)

    def get_mean(self) -> float:
        return float(np.mean(self._buffer)) if self._buffer else 0.0

    def get_std(self) -> float:
        return float(np.std(self._buffer)) if len(self._buffer) > 1 else 0.0


class AnomalyDetector:
    """异常检测器 — 维护多个统计量，独立触发。

    每个信号类型维护一个滑动窗口统计量。
    实时计算 z-score，超过阈值触发异常。

    Args:
        stats_window: 滑动窗口大小。
        z_threshold: Z-score 异常阈值。
    """
    SIGNAL_TYPES = [
        'pred_error',       # 预测误差 (z_pred - z_actual)²
        'fusion_entropy',   # 融合权重熵
        'safety_near_miss', # 安全接近阈值但未触发
        'world_dev',        # 自我-世界偏差
        'convergence_diff', # 收敛差异
        'step_time',        # 步时
    ]

    # 信号方向：值越大越异常（高侧触发）
    _HIGH_BAD = {'step_time', 'world_dev', 'entropy', 'pred_error',
                 'fusion_entropy', 'convergence_diff'}
    # 信号方向：值越小越异常（低侧触发，如 safety_margin 越低越接近危险）
    _LOW_BAD = {'safety_near_miss'}

    def __init__(self, stats_window: int = 100, z_threshold: float = 3.0):
        self.stats_window = stats_window
        self.z_threshold = z_threshold
        self._stats = {s: SlidingStats(window=stats_window, z_threshold=z_threshold)
                       for s in self.SIGNAL_TYPES}
        self._anomalies: deque = deque(maxlen=200)  # 最近异常
        self._last_check = 0
        self._persistent_window: deque = deque(maxlen=50)  # 持久性检查用

    def feed(self, step: int, record) -> List[AnomalySignal]:
        """输入一步轨迹，检测异常。

        Args:
            step: 步号。
            record: StepRecord 或任何带相关属性的对象。

        Returns:
            触发的异常信号列表（可能为空）。
        """
        signals = []

        # 提取各信号值
        extracts = []

        # 预测误差
        if hasattr(record, 'convergence_diff') and record.convergence_diff is not None:
            extracts.append(('pred_error', float(record.convergence_diff)))

        # 融合熵
        if hasattr(record, 'soft_mask') and record.soft_mask is not None:
            mask = np.asarray(record.soft_mask).ravel() + 1e-8
            entropy = -np.sum(mask * np.log(mask)) / np.log(len(mask))
            extracts.append(('fusion_entropy', float(entropy)))

        # 安全接近
        if hasattr(record, 'safety_margin') and record.safety_margin is not None:
            extracts.append(('safety_near_miss', float(record.safety_margin)))

        # 自我-世界偏差
        if hasattr(record, 'world_dev') and record.world_dev is not None:
            extracts.append(('world_dev', float(record.world_dev)))

        # 收敛差异
        if hasattr(record, 'convergence_diff') and record.convergence_diff is not None:
            extracts.append(('convergence_diff', float(record.convergence_diff)))

        # 步时
        if hasattr(record, 'step_time_ms') and record.step_time_ms is not None:
            extracts.append(('step_time', float(record.step_time_ms)))

        # 更新统计量并检查异常
        for sig_type, value in extracts:
            if sig_type not in self._stats:
                continue
            z = self._stats[sig_type].update(value)
            if z is None:
                continue
            # 按信号方向触发：高侧异常 z > th，低侧异常（safety_margin）z < -th
            if sig_type in self._LOW_BAD:
                triggered = z < -self.z_threshold
            elif sig_type in self._HIGH_BAD:
                triggered = z > self.z_threshold
            else:
                triggered = abs(z) > self.z_threshold  # 未知类型保持原双向行为
            if triggered:
                # 持久性检查：同一类型连续触发才算
                self._persistent_window.append((sig_type, step))
                recent_same = [x for x in self._persistent_window
                               if x[0] == sig_type and step - x[1] < 10]
                if len(recent_same) >= 3:
                    signal = AnomalySignal(
                        signal_type=sig_type,
                        severity=min(1.0, abs(z) / (self.z_threshold * 2)),
                        step=step,
                        value=value,
                        threshold=float(self._stats[sig_type].get_mean()
                                        + self.z_threshold * self._stats[sig_type].get_std()),
                        context={'z_score': float(z), 'n_recent': len(recent_same)},
                    )
                    signals.append(signal)

        # 记录异常
        for s in signals:
            self._anomalies.append(s)

        self._last_check = step
        return signals

    def get_recent_anomalies(self, n: int = 10) -> List[AnomalySignal]:
        return list(self._anomalies)[-n:]

    def get_anomaly_rate(self, window: int = 500) -> float:
        recent = [a for a in self._anomalies
                  if a.step > self._last_check - window]
        return len(recent) / max(window, 1)


# ─── 因果追溯 ──────────────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    """因果树中的一个节点。

    Attributes:
        role: 节点角色 ('root_cause', 'contributing', 'consequence', 'anomaly_point')。
        source_step: 相关步号。
        description: 人类可读描述。
        lattice: 相关格名称（如适用）。
        codebook_idx: 相关码本索引（如适用）。
        confidence: 该节点是根因的可信度 [0, 1]。
        children: 子节点（更上游的原因）。
    """
    role: str
    source_step: int
    description: str
    lattice: Optional[str] = None
    codebook_idx: Optional[int] = None
    confidence: float = 1.0
    children: List['CausalNode'] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'step': self.source_step,
            'desc': self.description,
            'lattice': self.lattice,
            'cb_idx': self.codebook_idx,
            'confidence': self.confidence,
            'children': [c.to_dict() for c in self.children],
        }


class CausalTracer:
    """因果追溯器 — 沿 DAG 反向追溯异常根因。

    输入：异常点 + 轨迹记录序列。
    输出：因果树（从 anomaly 点到 root cause）。
    追溯深度动态，由异常类型和可用数据决定。
    """
    LATTICE_NAMES = ['HRQ', 'Sparse', 'LowRank', 'Manifold', 'Binding', 'Contrast', 'Self']

    def trace(self, anomaly: AnomalySignal,
              records: List[Any]) -> CausalNode:
        """从异常点开始反向追溯。

        Args:
            anomaly: 触发的异常信号。
            records: 轨迹记录列表（按步号排序）。

        Returns:
            因果树根节点。
        """
        root = CausalNode(
            role='anomaly_point',
            source_step=anomaly.step,
            description=f"异常触发: {anomaly.signal_type} (val={anomaly.value:.3f})",
            confidence=1.0,
        )

        # 根据异常类型选择追溯策略
        tracer_fn = self._get_tracer(anomaly.signal_type)
        if tracer_fn:
            children = tracer_fn(anomaly, records)
            root.children = children

        return root

    def _get_tracer(self, signal_type: str) -> Optional[Callable]:
        tracers = {
            'pred_error': self._trace_pred_error,
            'fusion_entropy': self._trace_fusion_entropy,
            'safety_near_miss': self._trace_safety_near_miss,
            'world_dev': self._trace_world_dev,
            'convergence_diff': self._trace_convergence,
        }
        return tracers.get(signal_type)

    def _trace_pred_error(self, anomaly: AnomalySignal,
                           records: List[Any]) -> List[CausalNode]:
        """追溯预测误差：检查融合权重+各格检索。"""
        children = []
        step_record = self._find_record(anomaly.step, records)
        if step_record is None:
            return children

        # 检查融合权重 — 哪个格占主导
        if hasattr(step_record, 'soft_mask') and step_record.soft_mask is not None:
            mask = np.asarray(step_record.soft_mask).ravel()
            top_idx = int(np.argmax(mask))
            top_weight = float(mask[top_idx])
            if top_idx < len(self.LATTICE_NAMES):
                node = CausalNode(
                    role='contributing',
                    source_step=anomaly.step,
                    description=f"融合权重偏斜: {self.LATTICE_NAMES[top_idx]} "
                                f"权重={top_weight:.2f}",
                    lattice=self.LATTICE_NAMES[top_idx],
                    confidence=min(1.0, top_weight),
                )
                # 追溯该格的检索历史
                retro = self._trace_lattice_retrieval(
                    top_idx, anomaly.step, records)
                node.children = retro
                children.append(node)

        # 检查前几步是否有异常输入
        for lookback in range(1, min(5, len(records))):
            prev = self._find_record(anomaly.step - lookback, records)
            if prev is None:
                continue
            if (hasattr(prev, 'convergence_diff')
                    and prev.convergence_diff is not None
                    and prev.convergence_diff > 0.5):
                children.append(CausalNode(
                    role='contributing',
                    source_step=anomaly.step - lookback,
                    description=f"前序步 {anomaly.step - lookback} 收敛异常 "
                                f"(diff={prev.convergence_diff:.3f})",
                    confidence=0.5,
                ))

        return children

    def _trace_fusion_entropy(self, anomaly: AnomalySignal,
                               records: List[Any]) -> List[CausalNode]:
        """追溯融合熵高：检查路由门偏置。"""
        children = []
        step_record = self._find_record(anomaly.step, records)
        if step_record is None:
            return children

        if hasattr(step_record, 'soft_mask') and step_record.soft_mask is not None:
            mask = np.asarray(step_record.soft_mask).ravel()
            # 检查是否所有格的权重相近（高熵 = 路由不确定）
            for i, (w, name) in enumerate(zip(mask, self.LATTICE_NAMES)):
                if i >= len(mask):
                    break
                if w > 0.2:  # 多个格有显著权重
                    children.append(CausalNode(
                        role='contributing',
                        source_step=anomaly.step,
                        description=f"路由不确定: {name} 权重={w:.2f}",
                        lattice=name,
                        confidence=min(1.0, w * 2),
                    ))

        return children

    def _trace_safety_near_miss(self, anomaly: AnomalySignal,
                                 records: List[Any]) -> List[CausalNode]:
        """追溯安全接近：检查前几步的趋势。"""
        children = []
        # 逆向检查安全边界的下降趋势
        margins = []
        for lookback in range(min(10, len(records)), 0, -1):
            prev = self._find_record(anomaly.step - lookback, records)
            if prev is not None and hasattr(prev, 'safety_margin'):
                m = prev.safety_margin
                if m is not None:
                    margins.append((anomaly.step - lookback, float(m)))

        if len(margins) >= 3:
            # 检测下降趋势
            values = [m[1] for m in margins]
            if values[-1] < values[0] * 0.5:  # 显著下降
                children.append(CausalNode(
                    role='root_cause',
                    source_step=margins[0][0],
                    description=f"安全边界持续下降: "
                                f"{values[0]:.3f} → {values[-1]:.3f}",
                    confidence=0.7,
                ))

        return children

    def _trace_world_dev(self, anomaly: AnomalySignal,
                          records: List[Any]) -> List[CausalNode]:
        """追溯自我-世界偏差：检查自我格状态。"""
        children = []
        step_record = self._find_record(anomaly.step, records)
        if step_record is None:
            return children
        if hasattr(step_record, 'self_mode') and step_record.self_mode is not None:
            children.append(CausalNode(
                role='root_cause',
                source_step=anomaly.step,
                description=f"自我格模式 {step_record.self_mode} 与外部输入偏差大",
                lattice='Self',
                confidence=0.6,
            ))
        return children

    def _trace_convergence(self, anomaly: AnomalySignal,
                            records: List[Any]) -> List[CausalNode]:
        """追溯收敛问题：检查检索历史。"""
        children = []
        for lookback in range(1, min(10, len(records))):
            prev = self._find_record(anomaly.step - lookback, records)
            if prev is None:
                continue
            if (hasattr(prev, 'n_retrievals')
                    and prev.n_retrievals is not None
                    and prev.n_retrievals > 5):
                children.append(CausalNode(
                    role='contributing',
                    source_step=anomaly.step - lookback,
                    description=f"检索次数过多: {prev.n_retrievals} 次",
                    confidence=0.4,
                ))
        return children

    def _trace_lattice_retrieval(self, lattice_idx: int, step: int,
                                  records: List[Any]) -> List[CausalNode]:
        """追溯某格在给定步的检索历史。"""
        children = []
        name = self.LATTICE_NAMES[lattice_idx] if lattice_idx < len(self.LATTICE_NAMES) else f"Lattice{lattice_idx}"
        record = self._find_record(step, records)
        if record is None:
            return children

        # 根据格类型检索索引
        idx = None
        if lattice_idx == 0 and hasattr(record, 'hrq_idx'):
            idx = record.hrq_idx
        elif lattice_idx == 1 and hasattr(record, 'sparse_idx'):
            idx = record.sparse_idx
        elif lattice_idx == 3 and hasattr(record, 'man_idx'):
            idx = record.man_idx

        if idx is not None:
            children.append(CausalNode(
                role='contributing',
                source_step=step,
                description=f"{name} 检索码本索引 {idx}",
                lattice=name,
                codebook_idx=int(idx) if hasattr(idx, '__iter__') else idx,
                confidence=0.5,
            ))

        return children

    @staticmethod
    def _find_record(step: int, records: list) -> Optional[Any]:
        for r in records:
            if hasattr(r, 'step') and r.step == step:
                return r
        return None


# ─── 修正动作 ──────────────────────────────────────────────────────────────────

@dataclass
class CorrectionAction:
    """一个修正动作实例。

    Attributes:
        action_type: 动作类型标识。
        target: 作用目标描述。
        params: 动作参数。
        decay_steps: 衰减步数（0=永久）。
        created_step: 创建步号。
        description: 人类可读描述。
    """
    action_type: str
    target: str
    params: dict
    decay_steps: int = 50
    created_step: int = 0
    description: str = ""

    def is_expired(self, current_step: int) -> bool:
        if self.decay_steps <= 0:
            return False
        return (current_step - self.created_step) >= self.decay_steps

    def decay_factor(self, current_step: int) -> float:
        if self.decay_steps <= 0:
            return 1.0
        elapsed = current_step - self.created_step
        remaining = max(0, self.decay_steps - elapsed)
        return remaining / self.decay_steps


class CorrectionRegistry:
    """修正注册表 — 管理活跃修正。

    所有修正动作在此注册，查询衰减因子，状态持久化。
    """
    def __init__(self):
        self._actions: List[CorrectionAction] = []
        self._action_log: deque = deque(maxlen=500)  # 历史日志

    def register(self, action: CorrectionAction):
        self._actions.append(action)
        self._action_log.append(action)

    def get_active(self, current_step: int) -> List[CorrectionAction]:
        return [a for a in self._actions if not a.is_expired(current_step)]

    def get_decay(self, action_type: str, target: str,
                   current_step: int) -> float:
        """查询某目标指定类型修正的衰减因子。"""
        for a in reversed(self._actions):
            if a.action_type == action_type and a.target == target:
                return a.decay_factor(current_step)
        return 0.0

    def get_history(self, n: int = 20) -> List[CorrectionAction]:
        return list(self._action_log)[-n:]

    def clear_expired(self, current_step: int) -> int:
        before = len(self._actions)
        self._actions = [a for a in self._actions
                         if not a.is_expired(current_step)]
        return before - len(self._actions)


# ─── 修正执行器 ────────────────────────────────────────────────────────────────

class CorrectionExecutor:
    """修正执行器 — 将修正动作转换为具体操作。

    与 BehaviorExplorer、PredictionCache 等子系统交互。
    """
    def __init__(self, registry: CorrectionRegistry):
        self.registry = registry

    def apply_cache_deweight(self, cache, entry_idx: int,
                               decay_steps: int = 50, step: int = 0) -> CorrectionAction:
        """降低缓存条目的置信度/权重。"""
        action = CorrectionAction(
            action_type='cache_deweight',
            target=f'cache_entry_{entry_idx}',
            params={'entry_idx': entry_idx, 'weight_reduction': 0.5},
            decay_steps=decay_steps,
            created_step=step,
            description=f"降低缓存条目 {entry_idx} 的融合权重 (decay={decay_steps})",
        )
        self.registry.register(action)
        # 实际操作：在条目上标记降低的权重
        if hasattr(cache, '_buffer') and entry_idx < len(cache._buffer):
            entry = cache._buffer[entry_idx]
            if entry is not None:
                # 标记元数据（MAB 匹配时检查此字段）
                if not hasattr(entry, 'confidence'):
                    entry.confidence = 1.0
                entry.confidence = getattr(entry, 'confidence', 1.0) * 0.5
        return action

    def apply_bias_reset(self, explorer, dim: int,
                          decay_steps: int = 30, step: int = 0) -> CorrectionAction:
        """重置某维偏置为 0。"""
        action = CorrectionAction(
            action_type='bias_reset',
            target=f'bias_dim_{dim}',
            params={'dim': dim},
            decay_steps=decay_steps,
            created_step=step,
            description=f"重置 {explorer.LATTICE_NAMES[dim] if hasattr(explorer, 'LATTICE_NAMES') else f'dim {dim}'} 偏置为 0",
        )
        self.registry.register(action)
        return action

    def apply_explore_rate_reduce(self, explorer,
                                   factor: float = 0.5,
                                   decay_steps: int = 100,
                                   step: int = 0) -> CorrectionAction:
        """临时降低探索率。"""
        action = CorrectionAction(
            action_type='explore_rate_reduce',
            target='explorer',
            params={'factor': factor},
            decay_steps=decay_steps,
            created_step=step,
            description=f"探索率降低 {factor:.1f}× (decay={decay_steps})",
        )
        self.registry.register(action)
        if hasattr(explorer, 'explore_prob'):
            explorer.explore_prob = max(0.05, explorer.explore_prob * factor)
        return action

    def apply_cache_entry_delete(self, cache, entry_idx: int,
                                   step: int = 0) -> CorrectionAction:
        """删除错误的缓存条目。"""
        action = CorrectionAction(
            action_type='cache_delete',
            target=f'cache_entry_{entry_idx}',
            params={'entry_idx': entry_idx},
            decay_steps=0,  # 永久（删除不可逆）
            created_step=step,
            description=f"删除缓存条目 {entry_idx}",
        )
        self.registry.register(action)
        if hasattr(cache, '_buffer') and entry_idx < len(cache._buffer):
            cache._buffer[entry_idx] = None
        return action


# ─── 反思回路主控制器 ──────────────────────────────────────────────────────────

@dataclass
class ReflectionReport:
    """一份反思报告的完整记录。"""
    timestamp: float
    anomaly: AnomalySignal
    causal_tree: Optional[CausalNode]
    corrections: List[CorrectionAction]
    step: int

    def summary(self) -> str:
        lines = [f"反思报告 @ step {self.step}",
                 f"  触发: {self.anomaly}"]
        if self.causal_tree:
            lines.append(f"  根因: {self._format_tree(self.causal_tree, '  ')}")
        if self.corrections:
            lines.append(f"  修正: {len(self.corrections)} 个动作")
            for c in self.corrections:
                lines.append(f"    [{c.action_type}] {c.description}")
        return '\n'.join(lines)

    @staticmethod
    def _format_tree(node: CausalNode, indent: str) -> str:
        s = f"{node.role}: {node.description} (conf={node.confidence:.2f})"
        for c in node.children:
            s += f"\n{indent}  └─ {ReflectionReport._format_tree(c, indent + '  ')}"
        return s


class ReflectionLoop:
    """反思回路主控制器。

    职责：
        1. 接收轨迹数据 → 异常检测
        2. 异常触发 → 因果追溯
        3. 根因定位 → 执行修正
        4. 输出反思报告

    异步运行，不由推理引擎直接调用（或低频内联）。

    Args:
        anomaly_window: 异常检测滑动窗口。
        anomaly_z: Z-score 阈值。
        max_records: 保留的轨迹记录数。
    """
    def __init__(self, anomaly_window: int = 100,
                 anomaly_z: float = 3.0,
                 max_records: int = 2000,
                 cooldown: int = 50):
        self.detector = AnomalyDetector(
            stats_window=anomaly_window,
            z_threshold=anomaly_z,
        )
        self.tracer = CausalTracer()
        self.registry = CorrectionRegistry()
        self.executor = CorrectionExecutor(self.registry)
        self._records: deque = deque(maxlen=max_records)
        self._reports: deque = deque(maxlen=100)
        self._last_step = 0  # 最近一次 feed 的步号（get_stats 的修正活跃判定用）

        # 抑制：同类型异常在 N 步内不重复触发
        self._last_signal: Dict[str, int] = defaultdict(int)
        self._cooldown = cooldown

    # ── 输入 ──

    def feed(self, step: int, record) -> Optional[ReflectionReport]:
        """输入一步轨迹，检查异常，如有则执行完整反思。

        Args:
            step: 步号。
            record: StepRecord 或类似结构。

        Returns:
            如果触发了反思则返回 ReflectionReport，否则返回 None。
        """
        # 存储轨迹
        self._records.append(record)
        self._last_step = step

        # 异常检测
        anomalies = self.detector.feed(step, record)

        # 冷却检查
        active = [
            a for a in anomalies
            if step - self._last_signal.get(a.signal_type, -self._cooldown) >= self._cooldown
        ]

        if not active:
            return None

        # 对最严重的异常执行反思
        anomaly = max(active, key=lambda a: a.severity)
        self._last_signal[anomaly.signal_type] = step
        return self._reflect(anomaly)

    def feed_batch(self, records: List[Any], start_step: int = 0):
        """批量输入多条轨迹。"""
        last_report = None
        for i, record in enumerate(records):
            r = self.feed(start_step + i, record)
            if r is not None:
                last_report = r
        return last_report

    # ── 反思 ──

    def _reflect(self, anomaly: AnomalySignal) -> ReflectionReport:
        """执行完整反思：异常→追溯→修正→报告。"""
        records = list(self._records)

        # 因果追溯
        causal_tree = self.tracer.trace(anomaly, records)

        # 决策修正动作
        corrections = self._decide_corrections(anomaly, causal_tree, records)

        report = ReflectionReport(
            timestamp=time.time(),
            anomaly=anomaly,
            causal_tree=causal_tree,
            corrections=corrections,
            step=anomaly.step,
        )
        self._reports.append(report)

        if corrections:
            detail = report.summary()
            print(f"[REFL] {detail}")
        else:
            print(f"[REFL] 检测到异常但无需修正: {anomaly}")

        return report

    def _decide_corrections(self, anomaly: AnomalySignal,
                             tree: CausalNode,
                             records: list) -> List[CorrectionAction]:
        """根据因果树决策修正动作。

        基于根因类型选择修正策略。
        """
        corrections = []

        # 遍历因果树找到根因节点
        root_causes = self._find_root_causes(tree)

        for rc in root_causes:
            if rc.lattice and rc.codebook_idx is not None:
                # 码本检索问题 → 需要距离惩罚
                # 但距离惩罚需要在推理引擎侧实现，这里只记录意图
                corrections.append(CorrectionAction(
                    action_type='codebook_penalty',
                    target=f'{rc.lattice}/cb_{rc.codebook_idx}',
                    params={'lattice': rc.lattice, 'codebook_idx': rc.codebook_idx,
                            'penalty': 0.05},
                    decay_steps=50,
                    created_step=anomaly.step,
                    description=f"码本 {rc.lattice} 索引 {rc.codebook_idx} 临时惩罚",
                ))

            if '不确定' in rc.description or '路由' in rc.description:
                corrections.append(CorrectionAction(
                    action_type='routing_stabilize',
                    target='routing_gate',
                    params={'temperature_reduce': 0.1},
                    decay_steps=30,
                    created_step=anomaly.step,
                    description="路由温度临时降低以增强确定性",
                ))

        # 安全接近 → 探索率降低
        if anomaly.signal_type == 'safety_near_miss':
            corrections.append(CorrectionAction(
                action_type='safety_cautious',
                target='explorer',
                params={'explore_rate_factor': 0.3},
                decay_steps=100,
                created_step=anomaly.step,
                description="安全接近 → 探索率临时降低",
            ))

        # 注册修正
        for c in corrections:
            self.registry.register(c)

        return corrections

    @staticmethod
    def _find_root_causes(node: CausalNode) -> List[CausalNode]:
        """找到因果树中所有根因节点。"""
        if not node.children:
            if node.role == 'root_cause':
                return [node]
            return []

        causes = []
        for c in node.children:
            causes.extend(ReflectionLoop._find_root_causes(c))
        return causes if causes else [node]

    # ── 查询 ──

    def get_recent_reports(self, n: int = 5) -> List[ReflectionReport]:
        return list(self._reports)[-n:]

    def get_stats(self, current_step: int = None) -> dict:
        """反思回路统计。

        Args:
            current_step: 当前步号（步号而非 epoch 秒——修正的
                created_step 是步号，用 time.time() 比较恒判过期）。
                缺省用最近 feed 的步号。
        """
        if current_step is None:
            current_step = getattr(self, '_last_step', 0)
        return {
            'n_anomalies': len(self.detector._anomalies),
            'n_reports': len(self._reports),
            'n_active_corrections': len(self.registry.get_active(
                current_step)),
            'anomaly_rate': self.detector.get_anomaly_rate(),
        }


__all__ = [
    'AnomalyDetector',
    'AnomalySignal',
    'CausalTracer',
    'CausalNode',
    'CorrectionAction',
    'CorrectionRegistry',
    'CorrectionExecutor',
    'ReflectionReport',
    'ReflectionLoop',
]
