"""叙事记忆（长期结构化记忆）— 选择性存储重要事件。

叙事记忆不负责原始数据记录（那由自我观察/黑匣子负责）。
它只做一件事：从原始轨迹中筛选出重要事件，存入长期结构化叙事。

筛选标准由硬编码规则 + 可配置阈值决定，不依赖学习（可审计）。

Usage:
    from train.observability import StepRecord
    from train.narrative_memory import NarrativeMemory

    narr = NarrativeMemory(thresholds={'high_entropy': 0.7})

    # Feed it records from the black box
    for record in obs.get_recent_traces(100):
        narr.feed(record)

    # Query
    timeline = narr.get_timeline()
    important = narr.get_by_rule('safety_near_miss')

    # Manual promotion
    narr.promote(record)

    # Export
    narr.export_json("narrative.json")
"""
import random
import json
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Callable


# ─── Importance rules (auditable, hardcoded, configurable thresholds) ───────

@dataclass
class ImportanceRule:
    """A single importance rule — hardcoded, auditable.

    Attributes:
        name: Rule identifier.
        priority: 1 (highest) to 7 (lowest).
        description: Human-readable explanation.
        check_fn: Callable(record, thresholds) -> (is_important, score).
    """
    name: str
    priority: int
    description: str
    check_fn: Callable[..., Tuple[bool, float]]


# ── Built-in rules ──────────────────────────────────────────────────────────

def _check_user_marked(record, thresholds):
    return (record.user_flagged, 1.0 if record.user_flagged else 0.0)

def _check_safety(record, thresholds):
    th = thresholds.get('safety_near_miss', 0.1)
    if record.safety_margin is not None and 0 < record.safety_margin < th:
        score = max(0, 1.0 - record.safety_margin / th)
        return True, float(np.clip(score, 0.0, 1.0))
    return False, 0.0

def _check_entropy(record, thresholds):
    th = thresholds.get('high_entropy', 0.7)
    if record.convergence_entropy is not None and record.convergence_entropy > th:
        score = (record.convergence_entropy - th) / (1.0 - th + 1e-8)
        return True, float(np.clip(score, 0.0, 1.0))
    return False, 0.0

def _check_world_dev(record, thresholds):
    th = thresholds.get('world_dev', 0.5)
    if record.world_dev is not None and record.world_dev > th:
        score = (record.world_dev - th) / th
        return True, float(np.clip(score, 0.0, 1.0))
    return False, 0.0

def _check_state_jump(record, thresholds):
    th = thresholds.get('state_jump', 0.5)
    if record.convergence_diff is not None and record.convergence_diff > th:
        # 仿照 _check_world_dev：刚过阈值时 score 从 0 起，不饱和
        score = (record.convergence_diff - th) / th
        return True, float(np.clip(score, 0.0, 1.0))
    return False, 0.0

def _check_rare(record, thresholds):
    if record.sparse_idx is not None:
        return True, 0.5
    return False, 0.0

def _check_periodic(record, thresholds):
    return False, 0.0  # handled separately by step % N


DEFAULT_RULES: List[ImportanceRule] = [
    ImportanceRule("user_marked",      1, "用户显式标记", _check_user_marked),
    ImportanceRule("safety_near_miss", 2, "安全接近中断", _check_safety),
    ImportanceRule("high_entropy",     3, "融合熵高（犹豫不决）", _check_entropy),
    ImportanceRule("high_world_dev",   4, "自我-世界偏差大", _check_world_dev),
    ImportanceRule("state_jump",       5, "自我状态跳变", _check_state_jump),
    ImportanceRule("rare_retrieval",   6, "检索到罕见概念", _check_rare),
    ImportanceRule("periodic",         7, "定期抽样", _check_periodic),
]

DEFAULT_THRESHOLDS: Dict[str, float] = {
    'safety_near_miss': 0.1,
    'high_entropy': 0.7,
    'world_dev': 0.5,
    'state_jump': 0.5,
    'periodic': 100,  # every N steps
}


def _above_mean_sigma(value: float, history: deque,
                      sigma: float, min_history: int = 5) -> bool:
    """``value > mean(history) + sigma * std(history)`` with cold-start guard."""
    if len(history) < min_history:
        return False
    values = np.asarray(history, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    return value > mean + sigma * std


def _context_summary(record) -> Dict[str, Any]:
    """Small, JSON-safe trajectory slice — vectors stay in the record itself."""
    mask = getattr(record, 'soft_mask', None)
    if mask is not None:
        mask = np.asarray(mask).ravel().astype(float).tolist()
    return {
        'step': int(getattr(record, 'step', 0)),
        'is_safe': bool(getattr(record, 'is_safe', True)),
        'route_idx': getattr(record, 'route_idx', None),
        'convergence_diff': getattr(record, 'convergence_diff', None),
        'soft_mask': mask,
    }


# ─── Importance Evaluator ──────────────────────────────────────────────────

class ImportanceEvaluator:
    """Evaluate importance rules against a StepRecord.

    Rules checked in priority order. Highest-priority match wins.
    This makes the system fully auditable — you can always trace
    WHY a step was considered important.
    """
    def __init__(self, rules: Optional[List[ImportanceRule]] = None,
                 thresholds: Optional[Dict[str, float]] = None):
        self.rules = sorted(rules or DEFAULT_RULES, key=lambda r: r.priority)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def evaluate(self, record,
                 step: Optional[int] = None) -> Tuple[float, str]:
        """Evaluate importance. Returns (score, rule_name).

        Args:
            record: StepRecord (or any object with matching attrs).
            step: Step number (used for periodic check).

        Returns:
            score: 0.0 = not important, >0 = importance level.
            rule_name: Which rule triggered (empty string if none).
        """
        # Periodic sampling by step number（优先级 7 最低）：
        # 只作兜底 —— 规则循环之后、无规则命中时返回，避免覆盖高优先级规则。
        period = self.thresholds.get('periodic', 100)
        step_num = step if step is not None else getattr(record, 'step', 0)

        for rule in self.rules:
            if rule.name == 'periodic':
                continue
            try:
                is_important, score = rule.check_fn(record, self.thresholds)
                if is_important and score > 0:
                    return float(np.clip(score, 0.0, 1.0)), rule.name
            except Exception:
                continue

        if step_num > 0 and step_num % period == 0:
            return 0.3, 'periodic'

        return 0.0, ""


# ─── Narrative Record (augmented StepRecord for long-term storage) ─────────

@dataclass
class NarrativeRecord:
    """A StepRecord enriched with narrative metadata."""
    step: int
    timestamp: float = 0.0

    # Importance
    importance_score: float = 0.0
    importance_rule: str = ""

    # Source
    user_flagged: bool = False
    promoted: bool = False      # manually promoted from raw buffer
    pinned: bool = False        # protected from consolidation (model-driven forgetting bypass)

    # e.md §3: narrative metadata
    context: Dict[str, Any] = field(default_factory=dict)  # before/after window
    access_count: int = 0       # subsequent reads; forgetting score factor
    last_access_step: Optional[int] = None  # used by the stale-record rule

    # Original data (same fields as StepRecord)
    soft_mask: Optional[np.ndarray] = None
    route_idx: Optional[int] = None
    hrq_top_sim: Optional[float] = None
    hrq_idx: Optional[List[int]] = None
    sparse_idx: Optional[int] = None
    man_idx: Optional[int] = None
    self_mode: Optional[int] = None
    world_dev: Optional[float] = None
    safety_margin: Optional[float] = None
    value_signals: Optional[np.ndarray] = None
    is_safe: bool = True
    violated_law: Optional[int] = None
    convergence_diff: Optional[float] = None
    convergence_entropy: Optional[float] = None
    n_retrievals: int = 0
    prediction_error: Optional[float] = None
    dag_nodes: Optional[List[Dict[str, Any]]] = None
    conflict_source: Optional[str] = None
    conflict_detail: Optional[str] = None
    z_cur: Optional[np.ndarray] = None
    z_next: Optional[np.ndarray] = None
    z_q: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    @classmethod
    def from_step_record(cls, record, importance_score=0.0,
                         importance_rule="", user_flagged=False,
                         promoted=False, pinned=False,
                         context=None, access_count=0,
                         last_access_step=None) -> "NarrativeRecord":
        """Promote a StepRecord to NarrativeRecord."""
        return cls(
            step=record.step,
            timestamp=getattr(record, 'timestamp', 0.0),
            importance_score=importance_score,
            importance_rule=importance_rule,
            user_flagged=user_flagged or getattr(record, 'user_flagged', False),
            promoted=promoted,
            pinned=pinned,
            context=dict(context or {}),
            access_count=int(access_count),
            last_access_step=last_access_step,
            soft_mask=getattr(record, 'soft_mask', None),
            route_idx=getattr(record, 'route_idx', None),
            hrq_top_sim=getattr(record, 'hrq_top_sim', None),
            hrq_idx=getattr(record, 'hrq_idx', None),
            sparse_idx=getattr(record, 'sparse_idx', None),
            man_idx=getattr(record, 'man_idx', None),
            self_mode=getattr(record, 'self_mode', None),
            world_dev=getattr(record, 'world_dev', None),
            safety_margin=getattr(record, 'safety_margin', None),
            value_signals=getattr(record, 'value_signals', None),
            is_safe=getattr(record, 'is_safe', True),
            violated_law=getattr(record, 'violated_law', None),
            convergence_diff=getattr(record, 'convergence_diff', None),
            convergence_entropy=getattr(record, 'convergence_entropy', None),
            n_retrievals=getattr(record, 'n_retrievals', 0),
            prediction_error=getattr(record, 'prediction_error', None),
            dag_nodes=getattr(record, 'dag_nodes', None),
            conflict_source=getattr(record, 'conflict_source', None),
            conflict_detail=getattr(record, 'conflict_detail', None),
            z_cur=getattr(record, 'z_cur', None),
            z_next=getattr(record, 'z_next', None),
            z_q=getattr(record, 'z_q', None),
            z=getattr(record, 'z', None),
        )


# ─── Narrative Memory ──────────────────────────────────────────────────────

class NarrativeMemory:
    """叙事记忆 — 长期结构化记忆。

    从黑匣子的原始轨迹中筛选重要事件，存入长期叙事。
    筛选由硬编码规则判定，可审计、可配置。

    与自我观察（ObservabilityRecorder）的分工：
      - 黑匣子：记录一切（纯数据）
      - 叙事记忆：判断什么值得记住（纯选择）

    Usage:
        narr = NarrativeMemory()
        for record in obs.ring:          # 遍历黑匣子
            narr.feed(record)             # 叙事记忆自动筛选
        timeline = narr.get_timeline()    # 查看时间线
    """
    def __init__(self, max_records: int = 10000,
                 rules: Optional[List[ImportanceRule]] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 context_window: int = 5,
                 prediction_sigma: float = 2.0,
                 convergence_sigma: float = 2.0,
                 routing_kl_threshold: float = 0.5,
                 routing_novelty_cooldown: int = 20,
                 history_window: int = 1000,
                 random_fn: Optional[Callable[[], float]] = None,
                 random_seed: Optional[int] = None):
        """
        Args:
            max_records: 最大记录数（e.md §3: 10000）。
            rules: legacy 重要性规则（保留给旧调用方；feed 使用 e.md §3
                的五条规则）。
            thresholds: legacy 规则阈值。
            context_window: NarrativeRecord.context 的前后窗口（默认 5）。
            prediction_sigma: 预测误差触发阈值 mean + Nσ（默认 2）。
            convergence_sigma: 收敛异常触发阈值 mean + Nσ（默认 2）。
            routing_kl_threshold: routing KL novelty 阈值（nats）。
            history_window: 运行统计窗口。
            random_fn: 可注入的 [0,1) 随机源（测试/可复现用）。
            random_seed: 未注入 random_fn 时 random.Random 的种子。
        """
        self.max_records = max(1, int(max_records))
        self.evaluator = ImportanceEvaluator(rules, thresholds)
        self.context_window = max(1, int(context_window))
        self.prediction_sigma = float(prediction_sigma)
        self.convergence_sigma = float(convergence_sigma)
        self.routing_kl_threshold = float(routing_kl_threshold)
        self.routing_novelty_cooldown = max(0, int(routing_novelty_cooldown))
        self.random_fn = random_fn or random.Random(random_seed).random

        self._narrative: List[NarrativeRecord] = []
        self._total_evaluated = 0
        self._total_stored = 0
        self._last_step = -1

        # e.md §3 dynamic thresholds. History excludes the candidate record,
        # so the first outlier is measured against settled ordinary data.
        self._prediction_history: deque = deque(maxlen=max(10, int(history_window)))
        self._convergence_history: deque = deque(maxlen=max(10, int(history_window)))
        self._routing_mean: Optional[np.ndarray] = None
        self._routing_count = 0
        self._last_routing_novelty_step = -(10 ** 9)

        # context: raw before/after window and the open records that can still
        # accept "after" entries.
        self._raw_history: deque = deque(maxlen=self.context_window)
        self._open_context: deque = deque(maxlen=self.context_window)

    # ── Feed records from black box ──

    def feed(self, record, step: Optional[int] = None,
             user_flagged: bool = False) -> Optional[NarrativeRecord]:
        """Evaluate a StepRecord and store if important.

        Args:
            record: StepRecord from the black box.
            step: Step number (used for periodic check; defaults to record.step).
            user_flagged: Mark as user-important.

        Returns:
            NarrativeRecord if stored, None if not important.
        """
        if record is None:
            return None

        self._total_evaluated += 1
        step_num = step if step is not None else getattr(record, 'step', 0)
        # The read timestamp is the current trajectory step, even when this
        # particular feed is not important enough to store.
        self._last_step = step_num

        # e.md §3's five triggers, in priority order. The legacy evaluator is
        # kept for old callers but is no longer the narrative filter.
        score, rule = self._classify(record, step_num)
        flagged = user_flagged or getattr(record, 'user_flagged', False)

        # "after" context is the raw trajectory, not only the stored events:
        # every feed (ordinary or important) advances the open windows.
        summary = _context_summary(record)
        for previous in self._open_context:
            after = previous.context.setdefault('after', [])
            if len(after) < self.context_window:
                after.append(summary)

        if score <= 0 and not flagged:
            self._remember_raw(record)
            return None

        context = {
            'window': self.context_window,
            'before': [_context_summary(r) for r in self._raw_history],
            'after': [],
        }
        nrec = NarrativeRecord.from_step_record(
            record, importance_score=score, importance_rule=rule,
            user_flagged=flagged, context=context)
        self._narrative.append(nrec)
        self._total_stored += 1

        self._open_context.append(nrec)
        self._remember_raw(record)

        # Cap size — pinned entries are never dropped by FIFO
        if len(self._narrative) > self.max_records:
            self._cap_size()

        return nrec

    # ── e.md §3 triggers ──

    def _classify(self, record, step_num: int) -> Tuple[float, str]:
        """Return ``(importance, rule_name)`` for the five R16 triggers."""
        # 1. Safety conflict has absolute priority.
        if getattr(record, 'is_safe', True) is False:
            return 1.0, 'safety_conflict'

        # 2. High prediction error: value > running mean + 2σ.
        pred = getattr(record, 'prediction_error', None)
        if pred is not None and np.isfinite(float(pred)):
            triggered = _above_mean_sigma(
                float(pred), self._prediction_history, self.prediction_sigma)
            self._prediction_history.append(float(pred))
            if triggered:
                return 0.8, 'prediction_error'

        # 3. Routing novelty: KL(current || historical mean) > threshold.
        mask = getattr(record, 'soft_mask', None)
        if mask is not None:
            p = np.asarray(mask, dtype=np.float64).ravel()
            if p.size and np.all(np.isfinite(p)) and p.sum() > 0:
                p = p / p.sum()
                had_history = (self._routing_mean is not None
                               and self._routing_count >= 10)
                # Update the baseline BEFORE the threshold check. A sustained
                # novel distribution then becomes ordinary instead of firing
                # every single step (the old late-update order).
                self._routing_count += 1
                if self._routing_mean is None:
                    self._routing_mean = p.copy()
                else:
                    self._routing_mean = (
                        self._routing_mean * (self._routing_count - 1) + p
                    ) / self._routing_count
                if (had_history
                        and step_num - self._last_routing_novelty_step
                        >= self.routing_novelty_cooldown):
                    q = self._routing_mean
                    kl = float(np.sum(p * np.log(
                        (p + 1e-12) / (q + 1e-12))))
                    if kl > self.routing_kl_threshold:
                        self._last_routing_novelty_step = step_num
                        return 0.6, 'routing_novelty'

        # 4. Convergence anomaly: value > running mean + 2σ.
        conv = getattr(record, 'convergence_diff', None)
        if conv is not None and np.isfinite(float(conv)):
            triggered = _above_mean_sigma(
                float(conv), self._convergence_history, self.convergence_sigma)
            self._convergence_history.append(float(conv))
            if triggered:
                return 0.7, 'convergence_anomaly'

        # 5. Low-probability ordinary sampling: 1%.
        if float(self.random_fn()) < 0.01:
            return 0.1, 'random_sample'

        return 0.0, ''

    def _remember_raw(self, record) -> None:
        self._raw_history.append(record)

    # ── Access / forgetting score ──

    @staticmethod
    def forgetting_score(record: NarrativeRecord) -> float:
        """e.md §3: importance × access frequency (unpinned only)."""
        return float(record.importance_score) * (1.0 + float(record.access_count))

    def _touch(self, records, current_step: Optional[int] = None,
               count: int = 1) -> None:
        """Record a real read: increments access frequency and last access."""
        step = self._last_step if current_step is None else int(current_step)
        for rec in records:
            rec.access_count += int(count)
            rec.last_access_step = step

    def access(self, step: Optional[int] = None, index: Optional[int] = None,
               count: int = 1,
               access_step: Optional[int] = None) -> Optional[NarrativeRecord]:
        """Record an explicit subsequent access, used by forgetting."""
        target = None
        if index is not None:
            target = self._narrative[index]
        elif step is not None:
            for rec in reversed(self._narrative):
                if rec.step == step:
                    target = rec
                    break
        elif self._narrative:
            target = self._narrative[-1]
        if target is not None:
            target.access_count += int(count)
            target.last_access_step = (
                self._last_step if access_step is None else int(access_step))
        return target

    def get_step(self, step: int,
                 touch: bool = True) -> Optional[NarrativeRecord]:
        for rec in reversed(self._narrative):
            if rec.step == step:
                if touch:
                    self._touch([rec], current_step=self._last_step)
                return rec
        return None

    # Alias used by some call sites.
    touch = access

    # ── Manual promotion (user-triggered from raw buffer) ──

    def promote(self, record) -> Optional[NarrativeRecord]:
        """Manually promote a StepRecord to long-term narrative.

        Args:
            record: StepRecord (e.g. from obs.ring.pop_by_step()).

        Returns:
            NarrativeRecord if stored, None if None.
        """
        if record is None:
            return None
        nrec = NarrativeRecord.from_step_record(
            record, importance_score=1.0, importance_rule='user_marked',
            promoted=True, pinned=True,
            context={'window': self.context_window,
                     'before': [_context_summary(r) for r in self._raw_history],
                     'after': []})  # promoted entries are pinned by default
        self._narrative.append(nrec)
        self._total_stored += 1
        if len(self._narrative) > self.max_records:
            self._cap_size()
        return nrec

    # ── Size management (model-driven forgetting) ──

    def _cap_size(self):
        """Drop lowest-scoring unpinned entries when over max_records.

        Pinned entries are never dropped. Unpinned entries are scored by
        their importance_score (lower = more forgettable). Drops the
        lowest-scoring entries until under max_records.
        """
        if len(self._narrative) <= self.max_records:
            return

        unpinned = [(i, r) for i, r in enumerate(self._narrative) if not r.pinned]

        # e.md §3: retention score = importance × access frequency.
        # Ties drop the older record (stable secondary key on step).
        unpinned_sorted = sorted(
            unpinned, key=lambda x: (self.forgetting_score(x[1]), x[1].step))

        # Drop lowest-scoring unpinned entries
        n_drop = len(self._narrative) - self.max_records
        drop_indices = set(i for i, _ in unpinned_sorted[:n_drop])

        self._narrative = [r for i, r in enumerate(self._narrative)
                           if i not in drop_indices]

    def pin(self, step: int) -> bool:
        """Protect a narrative record from forgetting.

        Args:
            step: Step number of the record to pin.

        Returns:
            True if found and pinned, False otherwise.
        """
        for r in self._narrative:
            if r.step == step:
                r.pinned = True
                return True
        return False

    def unpin(self, step: int) -> bool:
        """Remove pin protection from a narrative record.

        Args:
            step: Step number of the record to unpin.

        Returns:
            True if found and unpinned, False otherwise.
        """
        for r in self._narrative:
            if r.step == step:
                r.pinned = False
                return True
        return False

    def consolidate(self, scorer_fn=None, keep_threshold: float = 0.05,
                    current_step: Optional[int] = None):
        """Model-driven forgetting — score each unpinned entry, drop below threshold.

        Args:
            scorer_fn: Optional callable(record) -> float [0, 1] that returns
                how worth keeping this record is. If None, uses built-in
                importance decay: score = importance_score * exp(-0.01 * age).
            keep_threshold: Entries with score < this value are dropped.
                Default 0.05. Pinned entries are never dropped regardless.

        Returns:
            Number of entries dropped.
        """
        if not self._narrative:
            return 0

        if current_step is None:
            current_step = max(r.step for r in self._narrative)
        before = len(self._narrative)

        def _last_access(r):
            return r.step if r.last_access_step is None else r.last_access_step

        if scorer_fn is None:
            # Default: importance × access frequency, decayed since last read.
            def _default_scorer(r):
                age = current_step - _last_access(r)
                decay = np.exp(-0.01 * max(age, 0))
                return self.forgetting_score(r) * decay
            scorer_fn = _default_scorer

        kept = []
        for r in self._narrative:
            if r.pinned:
                kept.append(r)
            elif (current_step - _last_access(r) > 5000
                  and r.importance_score < 0.2):
                # e.md §3: stale, low-importance records are demoted/deleted
                # without waiting for capacity pressure. "Stale" is measured
                # from the last read, not from creation.
                continue
            else:
                score = scorer_fn(r)
                if score >= keep_threshold:
                    kept.append(r)

        self._narrative = kept
        n_dropped = before - len(kept)

        # If still over max_records (unlikely — all surviving entries pinned),
        # drop oldest unpinned first, then oldest pinned if necessary
        if len(self._narrative) > self.max_records:
            n_over = len(self._narrative) - self.max_records
            # Separate pinned and unpinned
            with_pin = [(i, r) for i, r in enumerate(self._narrative) if r.pinned]
            without_pin = [(i, r) for i, r in enumerate(self._narrative) if not r.pinned]
            # Drop oldest unpinned
            without_pin = without_pin[n_over:]
            n_over -= (len([r for r in self._narrative if not r.pinned])
                       - len(without_pin))
            # If still over, drop oldest pinned
            if n_over > 0 and with_pin:
                with_pin = with_pin[n_over:]
            # Reassemble in original order
            keep_indices = set(i for i, _ in with_pin + without_pin)
            self._narrative = [r for i, r in enumerate(self._narrative)
                               if i in keep_indices]

        return n_dropped

    # ── Batch feed ──

    def feed_all(self, records, step_offset: int = 0):
        """Feed multiple records at once."""
        n = 0
        for i, record in enumerate(records):
            if self.feed(record, step=step_offset + i) is not None:
                n += 1
        return n

    # ── Query ──

    def __len__(self):
        return len(self._narrative)

    def __iter__(self):
        return iter(self._narrative)

    def __getitem__(self, idx):
        return self._narrative[idx]

    def get_all(self, touch: bool = True,
                current_step: Optional[int] = None) -> List[NarrativeRecord]:
        records = list(self._narrative)
        if touch:
            self._touch(records, current_step=current_step)
        return records

    def get_recent(self, n: int = 10, touch: bool = True) -> List[NarrativeRecord]:
        records = self._narrative[-n:]
        if touch:
            self._touch(records, current_step=self._last_step)
        return records

    def get_by_rule(self, rule_name: str,
                    touch: bool = True) -> List[NarrativeRecord]:
        records = [r for r in self._narrative
                   if r.importance_rule == rule_name]
        if touch:
            self._touch(records, current_step=self._last_step)
        return records

    def get_by_range(self, start_step: int, end_step: int,
                     touch: bool = True) -> List[NarrativeRecord]:
        records = [r for r in self._narrative
                   if start_step <= r.step <= end_step]
        if touch:
            self._touch(records, current_step=self._last_step)
        return records

    def get_unsafe(self, touch: bool = True) -> List[NarrativeRecord]:
        records = [r for r in self._narrative if not r.is_safe]
        if touch:
            self._touch(records, current_step=self._last_step)
        return records

    def get_timeline(self, touch: bool = True) -> List[Dict[str, Any]]:
        """Timeline view (for reflection / visualization)."""
        if touch:
            self._touch(self._narrative, current_step=self._last_step)
        return [{
            'step': r.step,
            'rule': r.importance_rule,
            'score': r.importance_score,
            'route': r.route_idx,
            'self_mode': r.self_mode,
            'safe': r.is_safe,
            'flagged': r.user_flagged,
            'pinned': r.pinned,
            'access_count': r.access_count,
            'last_access_step': r.last_access_step,
        } for r in self._narrative]

    def print_narrative(self, n: int = 10) -> None:
        """Print most recent narrative entries."""
        records = self._narrative[-n:] if n > 0 else self._narrative
        if not records:
            print("[NARR] No narrative records yet.")
            return

        print(f"\n{'─' * 60}")
        print(f"  最新叙事记录 (最近 {len(records)} 条)")
        print(f"{'─' * 60}")
        for r in records[-20:]:
            flag = " ⚑" if r.user_flagged else ""
            safe = " ✗" if not r.is_safe else ""
            pin = " 📌" if r.pinned else ""
            print(f"  step {r.step:>4} | {r.importance_rule:18s} | "
                  f"score={r.importance_score:.2f}{flag}{safe}{pin}")
        print(f"{'─' * 60}")
        print(f"  共 {self._total_stored}/{self._total_evaluated} 步被存入叙事 "
              f"(筛选率 {self._total_stored/max(self._total_evaluated,1):.1%})")

    # ── Summary ──

    def print_summary(self) -> None:
        n = len(self._narrative)
        if n == 0:
            print("[NARR] No narrative records.")
            return

        rule_counts = defaultdict(int)
        for r in self._narrative:
            rule_counts[r.importance_rule] += 1

        unsafe = sum(1 for r in self._narrative if not r.is_safe)

        print(f"\n{'=' * 46}")
        print(f"  叙事记忆 (Narrative Memory)")
        print(f"{'=' * 46}")
        print(f"  Records:          {n}")
        print(f"  Max records:      {self.max_records}")
        print(f"  Total evaluated:  {self._total_evaluated}")
        print(f"  Total stored:     {self._total_stored}")
        print(f"  Filter rate:      {self._total_stored/max(self._total_evaluated,1):.1%}")
        print(f"  Unsafe events:    {unsafe}")
        print(f"  By rule:")
        for rule, cnt in sorted(rule_counts.items(),
                                 key=lambda x: -x[1]):
            print(f"    {rule:18s}: {cnt}")
        print(f"{'=' * 46}\n")

    # ── Export ──

    def export_json(self, path: str, include_vectors: bool = False) -> None:
        """Export narrative to JSON for external analysis."""
        data = []
        for r in self._narrative:
            d = asdict(r)
            for k, v in list(d.items()):
                if isinstance(v, np.ndarray):
                    if include_vectors and k in ('z_q', 'z', 'z_cur',
                                                 'z_next'):
                        d[k] = v.tolist()
                    elif k in ('soft_mask', 'value_signals') and v is not None:
                        d[k] = v.tolist()
                    else:
                        d[k] = None
                elif isinstance(v, (np.integer,)):
                    d[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    d[k] = float(v)
            data.append(d)

        with open(path, 'w') as f:
            json.dump({
                'n_records': len(data),
                'total_evaluated': self._total_evaluated,
                'total_stored': self._total_stored,
                'thresholds': self.evaluator.thresholds,
                'records': data,
            }, f, indent=2, default=str)
        print(f"[NARR] Exported {len(data)} narrative records → {path}")

    def clear(self):
        self._narrative.clear()
        self._total_evaluated = 0
        self._total_stored = 0
        self._last_step = -1
        self._prediction_history.clear()
        self._convergence_history.clear()
        self._routing_mean = None
        self._routing_count = 0
        self._last_routing_novelty_step = -(10 ** 9)
        self._raw_history.clear()
        self._open_context.clear()


__all__ = [
    'ImportanceRule',
    'ImportanceEvaluator',
    'NarrativeRecord',
    'NarrativeMemory',
]
