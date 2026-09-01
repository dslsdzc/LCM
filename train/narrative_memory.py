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
import time
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
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
    dag_nodes: Optional[List[Dict[str, Any]]] = None
    conflict_source: Optional[str] = None
    conflict_detail: Optional[str] = None
    z_q: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    @classmethod
    def from_step_record(cls, record, importance_score=0.0,
                         importance_rule="", user_flagged=False,
                         promoted=False, pinned=False) -> "NarrativeRecord":
        """Promote a StepRecord to NarrativeRecord."""
        return cls(
            step=record.step,
            timestamp=getattr(record, 'timestamp', 0.0),
            importance_score=importance_score,
            importance_rule=importance_rule,
            user_flagged=user_flagged or getattr(record, 'user_flagged', False),
            promoted=promoted,
            pinned=pinned,
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
            dag_nodes=getattr(record, 'dag_nodes', None),
            conflict_source=getattr(record, 'conflict_source', None),
            conflict_detail=getattr(record, 'conflict_detail', None),
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
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            max_records: 最大记录数（超出后丢弃最旧）。
            rules: 重要性规则列表（默认使用内置7条）。
            thresholds: 覆盖默认阈值。
        """
        self.max_records = max_records
        self.evaluator = ImportanceEvaluator(rules, thresholds)
        self._narrative: List[NarrativeRecord] = []
        self._total_evaluated = 0
        self._total_stored = 0
        self._last_step = -1

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

        # Evaluate importance
        score, rule = self.evaluator.evaluate(record, step=step_num)
        flagged = user_flagged or getattr(record, 'user_flagged', False)

        if score <= 0 and not flagged:
            return None

        # Promote to narrative record
        nrec = NarrativeRecord.from_step_record(
            record, importance_score=score, importance_rule=rule,
            user_flagged=flagged)
        self._narrative.append(nrec)
        self._total_stored += 1
        self._last_step = step_num

        # Cap size — pinned entries are never dropped by FIFO
        if len(self._narrative) > self.max_records:
            self._cap_size()

        return nrec

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
            promoted=True, pinned=True)  # promoted entries are pinned by default
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

        pinned = [(i, r) for i, r in enumerate(self._narrative) if r.pinned]
        unpinned = [(i, r) for i, r in enumerate(self._narrative) if not r.pinned]

        # Sort unpinned by importance_score ascending
        unpinned_sorted = sorted(unpinned, key=lambda x: x[1].importance_score)

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

    def consolidate(self, scorer_fn=None, keep_threshold: float = 0.05):
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

        current_step = max(r.step for r in self._narrative)
        before = len(self._narrative)

        if scorer_fn is None:
            # Default: importance decay over time
            def _default_scorer(r):
                age = current_step - r.step
                decay = np.exp(-0.01 * max(age, 0))
                return r.importance_score * decay
            scorer_fn = _default_scorer

        kept = []
        for r in self._narrative:
            if r.pinned:
                kept.append(r)
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

    def get_all(self) -> List[NarrativeRecord]:
        return list(self._narrative)

    def get_recent(self, n: int = 10) -> List[NarrativeRecord]:
        return self._narrative[-n:]

    def get_by_rule(self, rule_name: str) -> List[NarrativeRecord]:
        return [r for r in self._narrative if r.importance_rule == rule_name]

    def get_by_range(self, start_step: int,
                     end_step: int) -> List[NarrativeRecord]:
        return [r for r in self._narrative
                if start_step <= r.step <= end_step]

    def get_unsafe(self) -> List[NarrativeRecord]:
        return [r for r in self._narrative if not r.is_safe]

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Timeline view (for reflection / visualization)."""
        return [{
            'step': r.step,
            'rule': r.importance_rule,
            'score': r.importance_score,
            'route': r.route_idx,
            'self_mode': r.self_mode,
            'safe': r.is_safe,
            'flagged': r.user_flagged,
            'pinned': r.pinned,
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
                    if include_vectors and k in ('z_q', 'z'):
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


__all__ = [
    'ImportanceRule',
    'ImportanceEvaluator',
    'NarrativeRecord',
    'NarrativeMemory',
]
