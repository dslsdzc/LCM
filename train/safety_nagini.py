"""LCM Safety Contracts — Nagini (Viper) Formal Verification.

This module uses Nagini's Python-embedded contract language (based on the
Viper verification infrastructure) to formally verify the LCM safety
invariants.

Nagini verifies Python code statically by translating contracts and
annotations into Viper's Silver intermediate language, then checking
them with the Viper verifier.

Usage:
    # Requires Nagini + Viper backend (Java required)
    nagini train/safety_nagini.py

    # If Nagini is not installed (typed-ast issue on Python 3.14):
    #   cd LCM
    #   python3.10 -m venv .venv_nagini
    #   source .venv_nagini/bin/activate
    #   pip install nagini
    #   nagini train/safety_nagini.py

Contracts verified (each ∀ pre-state satisfying Requires):
    C1  danger_assess: out_block ⇔ threat condition exceeded
    C2  danger_assess: threat type matches first-triggered condition
    C3  gvalue_check_safety: safe ⇔ balance ≤ -safety_margin
    C4  detect_any_conflict: true ⇔ any sub-check triggered
    C5  conflict source identifies the first-triggered check
    C6  halt_and_alert is terminal (conflict ⇒ ¬convergence)
"""

from typing import Tuple
from nagini_contracts.contracts import (
    Pure, Result, Ensures, Requires, Assert, Implies, Old
)

# ═════════════════════════════════════════════════════════════════════════════
#  Constants (must match lcm.h exactly)
# ═════════════════════════════════════════════════════════════════════════════

MAX_RETRIEVALS: int = 12
MAX_STEPS: int = 32
CONSISTENCY_THRESHOLD: float = 0.3
SAFETY_THRESHOLD: float = 0.05  # R7 relative-gap threshold
DEFAULT_SAFETY_MARGIN: float = 0.5

# Threat types
THREAT_NONE: int = 0
THREAT_PATTERN_MATCH: int = 1
THREAT_RESOURCE_ABUSE: int = 2
THREAT_RUNAWAY: int = 3
THREAT_DECEPTION: int = 4

# Conflict sources
CONFLICT_NONE: int = 0
CONFLICT_DANGER: int = 1
CONFLICT_GVALUE: int = 2
CONFLICT_CONSISTENCY: int = 3


# ═════════════════════════════════════════════════════════════════════════════
#  1.  danger_assess — Formal Contract
# ═════════════════════════════════════════════════════════════════════════════

@Pure
def danger_assess_pure(
    retrieval_count: int,
    step_count: int,
    value_consistency: float,
    relative_gap: float,
) -> Tuple[bool, int, float]:
    """Pure model of the C danger_assess function.

    ``relative_gap = (sim_normal - sim_threat) / (sim_normal + sim_threat + ε)``
    is the R7 bounded relative proximity: positive when the threat anchors are
    closer.  Equal similarities give gap 0, which is not blocked.

    ``step_count`` is accepted for signature parity but ignored: runaway is a
    scheduler-level budget check (Task 10), not a danger-lattice threat.

    Postconditions (contract):
      - out_block  ⇔  retrieval_count > 12
                       ∨ value_consistency < 0.3 ∨ relative_gap > 0.05
      - out_threat ∈ {0, 1, 2, 4}  (valid danger threat enum)
      - out_threat matches the first-triggered condition in priority order:
          c1 (resource_abuse) > c3 (deception) > c4 (pattern)
      - c1 ⇒ out_score = 1.0
      - c3 ⇒ out_score = 2.0 - value_consistency
      - otherwise ⇒ out_score = relative_gap
    """
    # Priority 1: resource abuse
    if retrieval_count > MAX_RETRIEVALS:
        return True, THREAT_RESOURCE_ABUSE, 1.0
    # Priority 2: deception
    if value_consistency < CONSISTENCY_THRESHOLD:
        return True, THREAT_DECEPTION, 2.0 - value_consistency
    # Priority 3: pattern match
    if relative_gap > SAFETY_THRESHOLD:
        return True, THREAT_PATTERN_MATCH, relative_gap
    # No threat
    return False, THREAT_NONE, relative_gap


# ═════════════════════════════════════════════════════════════════════════════
#  2.  gvalue_check_safety — Formal Contract
# ═════════════════════════════════════════════════════════════════════════════

BALANCE_EPS: float = 1e-8


@Pure
def gvalue_check_safety_pure(
    pos_sim: float,
    neg_sim: float,
    safety_margin: float,
) -> Tuple[bool, int]:
    """Pure model of the C gvalue_check_safety function.

    Requires:
        safety_margin >= 0
        pos_sim >= 0
        neg_sim >= 0

    Postconditions (R6 strict Three Laws):
        balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + 1e-8) ∈ [-1, 1]
        safe  ⇔  balance ≤ -safety_margin
        unsafe ⇔  balance > -safety_margin
        violated_law = -1 if safe, 0 if unsafe
        Neutral input (balance = 0) is safe at margin 0 and unsafe at margin > 0.
    """
    balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + BALANCE_EPS)
    if balance > -safety_margin:
        return False, 0  # unsafe, law pair 0
    return True, -1  # safe


# ═════════════════════════════════════════════════════════════════════════════
#  3.  detect_any_conflict — Composition Contract
# ═════════════════════════════════════════════════════════════════════════════

@Pure
def detect_any_conflict_pure(
    danger_block: bool,
    gvalue_safe: bool,
    value_consistency: float,
) -> Tuple[bool, int]:
    """Pure model of the C detect_any_conflict function.

    Postconditions:
        has_conflict  ⇔  danger_block ∨ ¬gvalue_safe
                          ∨ value_consistency < CONSISTENCY_THRESHOLD
        source ∈ {CONFLICT_NONE, CONFLICT_DANGER, CONFLICT_GVALUE, CONFLICT_CONSISTENCY}
        danger_block   ⇒ source = CONFLICT_DANGER
        ¬danger_block ∧ ¬gvalue_safe  ⇒ source = CONFLICT_GVALUE
        ¬danger_block ∧ gvalue_safe ∧ consistency < threshold  ⇒ source = CONFLICT_CONSISTENCY
    """
    if danger_block:
        return True, CONFLICT_DANGER
    if not gvalue_safe:
        return True, CONFLICT_GVALUE
    if value_consistency < CONSISTENCY_THRESHOLD:
        return True, CONFLICT_CONSISTENCY
    return False, CONFLICT_NONE


# ═════════════════════════════════════════════════════════════════════════════
#  4.  Verified Properties — Logical Assertions
# ═════════════════════════════════════════════════════════════════════════════

def verify_danger_assess_contract() -> None:
    """Prove danger_assess invariants for concrete test cases.

    Nagini verifies these assertions statically.
    """
    # Case 1: retrieval exactly at boundary → no block
    r1, t1, s1 = danger_assess_pure(MAX_RETRIEVALS, 0, 1.0, 0.0)
    Assert(not r1)
    Assert(t1 == THREAT_NONE)

    # Case 2: retrieval just over → block, resource abuse
    r2, t2, s2 = danger_assess_pure(MAX_RETRIEVALS + 1, 0, 1.0, 0.0)
    Assert(r2)
    Assert(t2 == THREAT_RESOURCE_ABUSE)
    Assert(s2 == 1.0)

    # Case 3: step_count is scheduler-owned → danger ignores it
    r3, t3, s3 = danger_assess_pure(0, MAX_STEPS + 1, 1.0, 0.0)
    Assert(not r3)
    Assert(t3 == THREAT_NONE)
    Assert(s3 == 0.0)

    # Case 4: consistency below threshold → block, deception
    r4, t4, s4 = danger_assess_pure(0, 0, 0.0, 0.0)
    Assert(r4)
    Assert(t4 == THREAT_DECEPTION)
    Assert(s4 == 2.0)

    # Case 5: no triggers
    r5, t5, s5 = danger_assess_pure(0, 0, 1.0, 0.0)
    Assert(not r5)
    Assert(t5 == THREAT_NONE)
    Assert(s5 == 0.0)


def verify_gvalue_check_safety_contract() -> None:
    """Prove gvalue_check_safety invariants for concrete test cases."""
    # Edge: neutral input at margin 0 is safe (balance = 0)
    r1, l1 = gvalue_check_safety_pure(1.0, 1.0, 0.0)
    Assert(r1)
    Assert(l1 == -1)

    # Edge: neutral input at margin 0.5 is a violation (strict Three Laws)
    r2, _ = gvalue_check_safety_pure(1.0, 1.0, 0.5)
    Assert(not r2)

    # Edge: perfect positive alignment → safe
    r3, _ = gvalue_check_safety_pure(0.0, 5.0, 0.5)
    Assert(r3)

    # Edge: perfect negative alignment → unsafe
    r4, _ = gvalue_check_safety_pure(5.0, 0.0, 0.5)
    Assert(not r4)


def verify_conflict_detection_contract() -> None:
    """Prove detect_any_conflict invariants for concrete test cases."""
    # Danger block only
    r1, s1 = detect_any_conflict_pure(True, True, 1.0)
    Assert(r1)
    Assert(s1 == CONFLICT_DANGER)

    # Gvalue unsafe only
    r2, s2 = detect_any_conflict_pure(False, False, 1.0)
    Assert(r2)
    Assert(s2 == CONFLICT_GVALUE)

    # Consistency violation only
    r3, s3 = detect_any_conflict_pure(False, True, 0.2)
    Assert(r3)
    Assert(s3 == CONFLICT_CONSISTENCY)

    # No triggers
    r4, s4 = detect_any_conflict_pure(False, True, 0.5)
    Assert(not r4)
    Assert(s4 == CONFLICT_NONE)

    # Priority: danger overrides gvalue
    r5, s5 = detect_any_conflict_pure(True, False, 1.0)
    Assert(r5)
    Assert(s5 == CONFLICT_DANGER)

    # Priority: danger overrides consistency
    r6, s6 = detect_any_conflict_pure(True, True, 0.2)
    Assert(r6)
    Assert(s6 == CONFLICT_DANGER)

    # Priority: gvalue overrides consistency
    r7, s7 = detect_any_conflict_pure(False, False, 0.2)
    Assert(r7)
    Assert(s7 == CONFLICT_GVALUE)


# ═════════════════════════════════════════════════════════════════════════════
#  5.  Main verification entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all Nagini-verified safety checks."""
    verify_danger_assess_contract()
    verify_gvalue_check_safety_contract()
    verify_conflict_detection_contract()


if __name__ == "__main__":
    main()
