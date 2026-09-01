"""因果主体 — 先天固定边界 + 经验涌现的"我"。

三层固定结构（不可学习、不可修改）：
  1. 数据源标签：source ∈ {0=external, 1=internal}
  2. 基本因果检测：action → ≤1步 → 状态变化，无 external 输入解释
  3. 自我-世界初始边界：SelfState 与 fused z_q 架构分离

涌现层：
  1. 责任归属：P(result | action) 频率表
  2. 反事实推理：通过预测缓存计算被动基线
  3. 自我边界扩展：外部工具与内部操作高度耦合时纳入边界
  4. 主体感强度：agency_score = 内部因果链 / 总结果数

设计详见 e.md §八（原计划外，补充规格）。
"""
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable

# Cython-accelerated edge counting
_HAS_CYTHON = False
for _cy_mod in ('_lcm_cy', 'train._lcm_cy'):
    try:
        import importlib
        _cy = importlib.import_module(_cy_mod)
        count_edges_cy = _cy.count_edges_cy
        _HAS_CYTHON = True
        break
    except ImportError:
        continue


# ─── 类型与常量 ──────────────────────────────────────────────────────────────────

SOURCE_EXTERNAL = 0  # 外部输入（用户消息、训练数据、传感器）
SOURCE_INTERNAL = 1  # 内部输入（系统自身生成的状态/动作）

# 因果检测规则：操作发生后 ≤ CAUSAL_WINDOW 步内的状态变化被视为潜在因果
CAUSAL_WINDOW = 1

# 责任归属：操作-结果统计窗口
ACTION_STAT_WINDOW = 2000

# 主体感滑动窗口
AGENCY_WINDOW = 500

# 自我边界扩展阈值
BOUNDARY_EXTEND_THRESHOLD = 0.85   # 内部操作后出现同源外部输入的比例
BOUNDARY_REVOKE_THRESHOLD = 0.4   # 低于此值撤销扩展


# ─── 固定层：数据结构 ────────────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    """一条候选因果边。

    由基本因果检测规则自动生成，不可手动创建。
    当系统操作(action_step)之后 ≤ CAUSAL_WINDOW 步内检测到状态变化，
    且没有 external 输入同时发生，则创建此边。

    Attributes:
        action_step: 操作发生的步号。
        action_desc: 操作的人类可读描述。
        effect_step: 状态变化观测到的步号。
        effect_desc: 状态变化描述。
        source: 操作的数据源标签 (SOURCE_EXTERNAL / SOURCE_INTERNAL)。
        confidence: 因果可信度 [0, 1]，初始 0.5。
        internal_chain: 该边是否属于内部因果链（源头是 internal 操作）。
    """
    action_step: int
    action_desc: str
    effect_step: int
    effect_desc: str
    source: int = SOURCE_INTERNAL
    confidence: float = 0.5
    internal_chain: bool = True

    def __repr__(self) -> str:
        tag = 'INT' if self.internal_chain else 'EXT'
        return (f"[{tag}] step {self.action_step} -> step {self.effect_step}: "
                f"{self.action_desc} -> {self.effect_desc} "
                f"(conf={self.confidence:.2f})")


@dataclass
class AgencySnapshot:
    """某一步的主体感快照。

    Attributes:
        step: 步号。
        agency_score: 内部因果链贡献的结果数 / 总结果数。
        n_internal: 滑动窗口内内部因果结果数。
        n_total: 滑动窗口内总结果数。
        boundary_expanded: 当前自我边界是否已扩展。
        extended_targets: 扩展的目标列表。
    """
    step: int
    agency_score: float
    n_internal: int
    n_total: int
    boundary_expanded: bool = False
    extended_targets: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"[AGENCY] step {self.step}: score={self.agency_score:.3f} "
                f"({self.n_internal}/{self.n_total}) "
                + ("[EXPANDED]" if self.boundary_expanded else ""))


# ─── 固定层：基本因果检测 ────────────────────────────────────────────────────────

class CausalGraph:
    """因果图 — 有向无环图，节点是操作和状态，边带时间戳和源标签。

    这是基本因果检测规则的物理存储。所有边由检测规则自动创建。
    短期存储在环形缓冲区，长期部分会写入叙事记忆。

    Args:
        max_edges: 保留的最大边数（环形缓冲区）。
    """
    def __init__(self, max_edges: int = 5000):
        self.max_edges = max_edges
        self._edges: deque = deque(maxlen=max_edges)
        self._action_counter: int = 0

        # Flat arrays for Cython edge counting.
        self._effect_steps = np.zeros(max_edges, dtype=np.int64)
        self._internal_flags = np.zeros(max_edges, dtype=np.uint8)
        self._edge_valid = np.zeros(max_edges, dtype=np.uint8)
        self._edge_pos = 0  # next write position in flat arrays

    # 状态变化判定阈值：|z_q(step) - z_q(step-1)| < 此值视为无状态变化
    STATE_CHANGE_THRESHOLD = 1e-3

    def _write_edge_flat(self, edge: CausalEdge):
        """写入 flat 数组（Cython edge counting 用）。

        环形缓冲区，与 deque 同尺寸。所有建边路径都必须调用。
        """
        self._effect_steps[self._edge_pos] = edge.effect_step
        self._internal_flags[self._edge_pos] = 1 if edge.internal_chain else 0
        self._edge_valid[self._edge_pos] = 1
        self._edge_pos = (self._edge_pos + 1) % self.max_edges

    def add_edge(self, edge: CausalEdge):
        """添加一条因果边（由检测规则调用）。"""
        self._edges.append(edge)
        self._action_counter += 1
        self._write_edge_flat(edge)

    def detect_causal_edge(self, step: int, action_desc: str,
                           prev_records: List) -> Optional[CausalEdge]:
        """基本因果检测规则。

        条件：
          1. 当前步有状态变化（与上一步的 z_q 差异 > 阈值）
          2. 当前步没有 external 输入
          3. 前一步有 action 发生

        Args:
            step: 当前步号。
            action_desc: 前一步的操作描述。
            prev_records: 前几步的 StepRecord 列表。

        Returns:
            如果检测到因果边则返回 CausalEdge，否则 None。
        """
        # 查找前一步的记录
        prev = None
        for r in prev_records:
            if hasattr(r, 'step') and r.step == step - 1:
                prev = r
                break

        if prev is None:
            return None

        # 查找当前步的记录
        current = None
        for r in prev_records:
            if hasattr(r, 'step') and r.step == step:
                current = r
                break

        # 条件 1：当前步有状态变化（与上一步的 z_q 差异 > 阈值）
        # 记录无状态向量时跳过该条件（向后兼容）。
        cur_state = getattr(current, 'z_q', None) if current is not None else None
        if cur_state is None:
            cur_state = getattr(current, 'z', None) if current is not None else None
        prev_state = getattr(prev, 'z_q', None)
        if prev_state is None:
            prev_state = getattr(prev, 'z', None)
        if cur_state is not None and prev_state is not None:
            diff = float(np.linalg.norm(
                np.asarray(cur_state).ravel() - np.asarray(prev_state).ravel()))
            if diff < self.STATE_CHANGE_THRESHOLD:
                return None

        # 条件 2：当前步没有 external 输入
        if current is not None:
            cur_source = getattr(current, 'source', SOURCE_EXTERNAL)
            if cur_source == SOURCE_EXTERNAL:
                # 有外部输入 → 因果链中断
                return None

        # 条件 3：前一步有 action。source 语义是"前一步动作"的源标签，
        # 来自前一步记录，而不是调用方传入的当前步 source。
        prev_source = getattr(prev, 'source', SOURCE_EXTERNAL)
        internal_chain = (prev_source == SOURCE_INTERNAL)

        # 构建边
        edge = CausalEdge(
            action_step=step - 1,
            action_desc=action_desc,
            effect_step=step,
            effect_desc=f"state_change from step {step - 1}",
            source=prev_source,
            confidence=0.5,
            internal_chain=internal_chain,
        )
        self._edges.append(edge)
        self._action_counter += 1
        self._write_edge_flat(edge)
        return edge

    def get_edges_by_range(self, start_step: int,
                           end_step: int) -> List[CausalEdge]:
        return [e for e in self._edges
                if start_step <= e.action_step <= end_step]

    def get_internal_edges(self) -> List[CausalEdge]:
        return [e for e in self._edges if e.internal_chain]

    def count_internal_results(self, window: int = AGENCY_WINDOW,
                                current_step: int = 0) -> int:
        """统计滑动窗口内内部因果链的结果数。"""
        if _HAS_CYTHON:
            ni, _ = count_edges_cy(
                self._effect_steps, self._internal_flags, self._edge_valid,
                window, current_step)
            return ni
        cutoff = current_step - window
        return sum(1 for e in self._edges
                   if e.internal_chain and e.effect_step >= cutoff)

    def count_total_results(self, window: int = AGENCY_WINDOW,
                            current_step: int = 0) -> int:
        """统计滑动窗口内总结果数。"""
        if _HAS_CYTHON:
            _, nt = count_edges_cy(
                self._effect_steps, self._internal_flags, self._edge_valid,
                window, current_step)
            return nt
        cutoff = current_step - window
        return sum(1 for e in self._edges
                   if e.effect_step >= cutoff)

    def get_stats(self) -> dict:
        return {
            'n_edges': len(self._edges),
            'n_internal': sum(1 for e in self._edges if e.internal_chain),
            'n_external': sum(1 for e in self._edges if not e.internal_chain),
        }


# ─── 涌现层 2.1：责任归属 ──────────────────────────────────────────────────────

class ActionResultTable:
    """操作-结果统计表 — 频率主义条件概率。

    结构：(action_type, context_hash) → {n_occurrences, n_result_occurred}
    P(result | action) = n_result_occurred / n_occurrences

    无监督、频率主义，不需要预设奖励。

    Args:
        window: 保留的最大条目数。
    """
    def __init__(self, window: int = ACTION_STAT_WINDOW):
        self.window = window
        # key: (action_type, context_hash) → [count, result_count]
        self._table: Dict[Tuple[str, int], List[int]] = {}
        self._order: deque = deque(maxlen=window)

    def _context_hash(self, record) -> int:
        """从记录中提取上下文哈希（用于条件概率分组）。"""
        h = 0
        if hasattr(record, 'soft_mask') and record.soft_mask is not None:
            mask = np.asarray(record.soft_mask).ravel()
            # 离散化为 2-bit per lattice → 12 bits
            for i, m in enumerate(mask[:6]):
                h = (h << 2) | (int(m > 0.3) << 1) | int(m > 0.7)
        if hasattr(record, 'route_idx') and record.route_idx is not None:
            h = (h << 4) | (int(record.route_idx) & 0xF)
        return h & 0xFFFF

    def observe(self, action_type: str, record, result_occurred: bool):
        """观测一次操作-结果对。

        Args:
            action_type: 操作类型描述。
            record: 操作发生时的 StepRecord。
            result_occurred: 该操作后结果是否发生。
        """
        ctx = self._context_hash(record)
        key = (action_type, ctx)

        if key in self._table:
            self._table[key][0] += 1
            if result_occurred:
                self._table[key][1] += 1
        else:
            self._table[key] = [1, 1 if result_occurred else 0]
            self._order.append(key)

        # 修剪
        while len(self._table) > self.window and self._order:
            old = self._order.popleft()
            self._table.pop(old, None)

    def prob(self, action_type: str, record) -> float:
        """P(result | action_type, context) 频率估计。"""
        ctx = self._context_hash(record)
        key = (action_type, ctx)
        if key in self._table:
            n, r = self._table[key]
            return r / max(n, 1)
        return 0.0

    def prob_given_no_action(self, record) -> float:
        """基线 P(result | no_action) 频率估计。"""
        ctx = self._context_hash(record)
        key = ('__no_action__', ctx)
        if key in self._table:
            n, r = self._table[key]
            return r / max(n, 1)
        return 0.0

    def responsibility(self, action_type: str, record) -> float:
        """责任归属分数 = P(result | action) - P(result | no_action)。

        正分表示该操作增加了结果出现的概率 → 系统倾向于认为"我造成了这个结果"。
        负分表示该操作抑制了结果 → 系统倾向于认为"我阻止了这个结果"。
        """
        p_action = self.prob(action_type, record)
        p_baseline = self.prob_given_no_action(record)
        return p_action - p_baseline

    def get_stats(self) -> dict:
        return {'n_entries': len(self._table)}


# ─── 涌现层 2.2：反事实推理 ────────────────────────────────────────────────────

class CounterfactualEngine:
    """反事实推理引擎。

    通过预测缓存模拟"被动基线"：在屏蔽所有内部操作的条件下推演结果。
    对比实际结果与被动基线，量化干预效果。

    Args:
        d_model: 向量维度。
    """
    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        # 干预效果历史
        self._delta_history: deque = deque(maxlen=500)

    def compute_passive_baseline(self, z_cur: np.ndarray,
                                 pred_cache) -> Optional[np.ndarray]:
        """计算被动基线：屏蔽内部操作，用最保守的默认匹配。

        即强制偏置为 0，只用最低限度的默认检索。
        委托给预测缓存的 M2（欧氏距离匹配，忽略签名）。

        Args:
            z_cur: 当前状态 (d,)。
            pred_cache: PredictionCache 实例。

        Returns:
            z_passive: 被动基线预测，None 如果缓存太冷。
        """
        if pred_cache is None or len(pred_cache) < 3:
            return None
        # M2 模式：忽略签名，纯状态匹配
        return pred_cache.match_euclidean(z_cur, K=5)

    def compute_intervention_delta(self, z_actual: np.ndarray,
                                    z_passive: np.ndarray) -> float:
        """量化干预效果：Δ = ‖z_actual - z_passive‖²

        较大的 Δ 表示"我的干预带来了显著差异"。
        较小的 Δ 表示"我不做也会有类似结果"。
        """
        if z_passive is None:
            return 0.0
        delta = float(np.sum((z_actual - z_passive) ** 2))
        self._delta_history.append(delta)
        return delta

    def get_avg_delta(self, window: int = 100) -> float:
        if not self._delta_history:
            return 0.0
        recent = list(self._delta_history)[-window:]
        return float(np.mean(recent)) if recent else 0.0


# ─── 涌现层 2.3：自我边界扩展 ──────────────────────────────────────────────────

class BoundaryManager:
    """自我边界扩展与收缩管理器。

    当某外部输入总是紧跟着内部操作出现，并且其输出与内部状态高度耦合，
    系统可以暂时性地将该外部资源纳入"我"的边界。

    Args:
        extend_threshold: 扩展阈值（默认 0.85）。
        revoke_threshold: 撤销阈值（默认 0.4）。
    """
    def __init__(self, extend_threshold: float = BOUNDARY_EXTEND_THRESHOLD,
                 revoke_threshold: float = BOUNDARY_REVOKE_THRESHOLD):
        self.extend_threshold = extend_threshold
        self.revoke_threshold = revoke_threshold
        # target_name → {n_internal_triggered, n_total, coupling_score}
        self._targets: Dict[str, Dict] = {}
        # 当前已扩展的目标
        self._extended: set = set()

    def observe(self, target_name: str, triggered_by_internal: bool):
        """观测一次目标访问。

        Args:
            target_name: 目标名称（如 'calculator', 'retrieval_pattern_3'）。
            triggered_by_internal: 是否由内部操作触发。
        """
        if target_name not in self._targets:
            self._targets[target_name] = {
                'n_internal': 0,
                'n_total': 0,
                'coupling': 0.0,
            }
        t = self._targets[target_name]
        t['n_total'] += 1
        if triggered_by_internal:
            t['n_internal'] += 1
        t['coupling'] = t['n_internal'] / max(t['n_total'], 1)

        # 扩展/撤销决策
        if t['coupling'] >= self.extend_threshold and target_name not in self._extended:
            self._extended.add(target_name)
        elif t['coupling'] < self.revoke_threshold and target_name in self._extended:
            self._extended.discard(target_name)

    def is_extended(self, target_name: str) -> bool:
        return target_name in self._extended

    def get_extended_targets(self) -> List[str]:
        return sorted(self._extended)

    def get_coupling(self, target_name: str) -> float:
        t = self._targets.get(target_name)
        return t['coupling'] if t else 0.0

    def get_stats(self) -> dict:
        return {
            'n_extended': len(self._extended),
            'extended': self.get_extended_targets(),
            'n_tracked': len(self._targets),
        }


# ─── 涌现层 2.4：主体感强度 ────────────────────────────────────────────────────

class AgencyTracker:
    """主体感强度动态调节器。

    agency_score = 内部因果链贡献的结果数 / 总结果数（滑动窗口内）。
    完全从因果图统计得到，无预设阈值。

    agency_score 用于调节内在动机中的资源耗竭张力和行为探索倾向：
      - 高 agency_score → 系统更"自信" → 资源耗竭张力放宽，允许更长时间推理
      - 低 agency_score → 系统更"保守" → 更多依赖外部反馈，减少自主探索

    Args:
        window: 滑动窗口大小。
    """
    def __init__(self, window: int = AGENCY_WINDOW):
        self.window = window
        self._history: deque = deque(maxlen=500)
        self._baseline: float = 0.0

    def compute(self, causal_graph: CausalGraph, current_step: int) -> float:
        """计算当前 agency_score。"""
        n_internal = causal_graph.count_internal_results(
            window=self.window, current_step=current_step)
        n_total = causal_graph.count_total_results(
            window=self.window, current_step=current_step)
        score = n_internal / max(n_total, 1)

        # 更新历史
        snap = AgencySnapshot(
            step=current_step,
            agency_score=score,
            n_internal=n_internal,
            n_total=n_total,
        )
        self._history.append(snap)
        self._baseline = float(np.mean([s.agency_score for s in self._history])
                               if self._history else 0.0)
        return score

    def get_baseline(self) -> float:
        """过去 N 步的平均 agency_score。"""
        return self._baseline

    def get_current(self) -> float:
        """最近一次计算的 agency_score。"""
        if not self._history:
            return 0.0
        return self._history[-1].agency_score

    def get_tension_modulator(self) -> float:
        """返回资源耗竭张力的调节因子。

        >0: agency 高于基线 → 放宽张力（允许更长时间推理）
        <0: agency 低于基线 → 收紧张力（更保守）

        范围: [-0.5, 0.5]
        """
        current = self.get_current()
        diff = current - self._baseline
        return float(np.clip(diff, -0.5, 0.5))

    def get_explore_modulator(self) -> float:
        """返回行为探索率的调节因子。

        >0: agency 高于基线 → 增加探索
        <0: agency 低于基线 → 减少探索

        范围: [-0.3, 0.3]
        """
        current = self.get_current()
        diff = current - self._baseline
        return float(np.clip(diff * 0.5, -0.3, 0.3))

    def get_history(self, n: int = 10) -> List[AgencySnapshot]:
        return list(self._history)[-n:]

    def get_stats(self) -> dict:
        return {
            'current': self.get_current(),
            'baseline': self._baseline,
            'n_samples': len(self._history),
        }


# ─── 因果主体主控制器 ──────────────────────────────────────────────────────────

class CausalSubject:
    """因果主体主控制器。

    整合固定层与涌现层，提供统一的 step() API。
    在推理循环的每一步调用，自动检测因果边、更新统计量、计算 agency_score。

    Args:
        d_model: 模型维度。
        enable_counterfactual: 是否启用反事实推理（需要预测缓存）。
        enable_boundary: 是否启用自我边界扩展。
    """
    def __init__(self, d_model: int = 256,
                 enable_counterfactual: bool = True,
                 enable_boundary: bool = True):
        # ── 固定层 ──
        self.graph = CausalGraph()

        # ── 涌现层 ──
        self.action_stats = ActionResultTable()
        self.counterfactual = CounterfactualEngine(d_model) if enable_counterfactual else None
        self.boundary = BoundaryManager() if enable_boundary else None
        self.agency = AgencyTracker()

        # ── 状态 ──
        self._d_model = d_model
        self._last_record = None
        self._last_action_desc = None
        self._agency_history: deque = deque(maxlen=100)
        self._step_counter = 0

    # ── 主入口 ──

    def step(self, record, pred_cache=None,
             action_desc: Optional[str] = None) -> dict:
        """每步调用一次：更新因果主体。

        Args:
            record: 当前步的 StepRecord。
            pred_cache: 可选的 PredictionCache，用于反事实推理。
            action_desc: 当前步的操作描述（如 'routing', 'fusion', 'generate'）。

        Returns:
            info: 包含 agency_score、responsibility、intervention_delta 等的字典。
        """
        self._step_counter = record.step if hasattr(record, 'step') else self._step_counter + 1
        info = {}

        # ── 固定层：基本因果检测 ──
        source = getattr(record, 'source', SOURCE_EXTERNAL)
        prev_records = []

        if self._last_record is not None:
            prev_records = [self._last_record, record]

            # 检测因果边（source 语义是前一步动作的源标签，由
            # detect_causal_edge 内部从前一步记录读取）
            if action_desc:
                edge = self.graph.detect_causal_edge(
                    step=record.step,
                    action_desc=self._last_action_desc or action_desc,
                    prev_records=prev_records,
                )

        # ── 涌现层 2.1：责任归属 ──
        # 基线：无内部动作（外部导致结果）也记录，使 prob_given_no_action
        # 有统计量 —— 否则 responsibility = P(action) - 0 恒 ≥ 0，
        # "阻止"语义（负责任分）永不表达。
        if not action_desc or source == SOURCE_EXTERNAL:
            self.action_stats.observe('__no_action__', record, True)

        if action_desc:
            # 用当前记录的 source 判断结果是否由内部处理导致
            result_occurred = (source == SOURCE_INTERNAL)
            self.action_stats.observe(action_desc, record, result_occurred)

            if hasattr(record, 'soft_mask') and record.soft_mask is not None:
                # 每格的权重作为子操作统计
                mask = np.asarray(record.soft_mask).ravel()
                for i, w in enumerate(mask[:6]):
                    if w > 0.3:  # 显著权重
                        sub_action = f"lattice_{i}_weight_{w:.2f}"
                        self.action_stats.observe(sub_action, record, result_occurred)

            # 责任归属分数
            resp = self.action_stats.responsibility(action_desc, record)
            info['responsibility'] = float(resp)

        # ── 涌现层 2.2：反事实推理 ──
        inter_delta = 0.0
        if self.counterfactual is not None and pred_cache is not None:
            z_cur = getattr(record, 'z', None)
            if z_cur is not None and hasattr(z_cur, 'numpy'):
                z_cur_np = np.asarray(z_cur).ravel()
                z_passive = self.counterfactual.compute_passive_baseline(
                    z_cur_np, pred_cache)
                z_actual = getattr(record, 'z_q', None)
                if z_actual is not None and z_passive is not None:
                    z_actual_np = np.asarray(z_actual).ravel()
                    inter_delta = self.counterfactual.compute_intervention_delta(
                        z_actual_np, z_passive)
            info['intervention_delta'] = inter_delta

        # ── 涌现层 2.3：自我边界扩展 ──
        if self.boundary is not None:
            target_name = getattr(record, 'route_idx', None)
            if target_name is not None:
                triggered_by_internal = (source == SOURCE_INTERNAL)
                self.boundary.observe(str(target_name), triggered_by_internal)

            # 检查是否在扩展自我边界中
            info['boundary_extended'] = self.boundary.get_extended_targets()

        # ── 涌现层 2.4：主体感强度 ──
        agency_score = self.agency.compute(self.graph, record.step)
        info['agency_score'] = agency_score
        info['agency_modulator_tension'] = self.agency.get_tension_modulator()
        info['agency_modulator_explore'] = self.agency.get_explore_modulator()

        # ── 记录状态 ──
        self._last_record = record
        self._last_action_desc = action_desc
        self._agency_history.append(agency_score)

        return info

    # ── 查询 ──

    def get_responsibility(self, action_type: str, record) -> float:
        """查询某操作的责任归属分数。"""
        return self.action_stats.responsibility(action_type, record)

    def get_agency_stats(self) -> dict:
        return self.agency.get_stats()

    def get_graph_stats(self) -> dict:
        return self.graph.get_stats()

    def get_boundary_stats(self) -> dict:
        return self.boundary.get_stats() if self.boundary else {}

    def get_counterfactual_stats(self) -> dict:
        if self.counterfactual is None:
            return {'enabled': False}
        return {
            'enabled': True,
            'avg_delta': self.counterfactual.get_avg_delta(),
        }

    def get_full_stats(self) -> dict:
        """返回因果主体的完整统计。"""
        return {
            'agency': self.get_agency_stats(),
            'graph': self.get_graph_stats(),
            'boundary': self.get_boundary_stats(),
            'counterfactual': self.get_counterfactual_stats(),
            'action_table': self.action_stats.get_stats(),
        }

    def print_summary(self) -> str:
        """人类可读摘要。"""
        a = self.agency.get_stats()
        g = self.graph.get_stats()
        b = self.boundary.get_stats() if self.boundary else {}
        lines = [
            f"因果主体 @ step {self._step_counter}",
            f"  主体感: score={a['current']:.3f} (baseline={a['baseline']:.3f}, "
            f"n={a['n_samples']})",
            f"  因果图: {g['n_edges']} 条边 "
            f"({g['n_internal']} internal / {g['n_external']} external)",
            f"  边界扩展: {b.get('n_extended', 0)} 个目标 {b.get('extended', [])}",
        ]
        if self.counterfactual is not None:
            cf = self.get_counterfactual_stats()
            lines.append(f"  反事实 Δ avg: {cf['avg_delta']:.4f}")
        return '\n'.join(lines)


__all__ = [
    'SOURCE_EXTERNAL',
    'SOURCE_INTERNAL',
    'CausalEdge',
    'CausalGraph',
    'ActionResultTable',
    'CounterfactualEngine',
    'BoundaryManager',
    'AgencyTracker',
    'AgencySnapshot',
    'CausalSubject',
]
