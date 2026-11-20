# LCM Safety Subsystem Specification: Danger Lattice & External Verifier

> **v3.1-conformance changelog**：本版按 `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md`
> 的 R1–R22 修订。中文版与英文版同步。

> **Position**: This document defines the two highest-security components in the LCM architecture — the danger lattice (`Λ_danger`) and the external safety verifier. They form the final line of defense in the system safety framework, independent of all functional and value lattices.

---

## 1. Danger Lattice `Λ_danger`

### 1.1 Position

The danger lattice is the eighth specialized lattice, independent of the six functional lattices and the global value lattice. It is the system's highest-security **read-only monitoring module**. It does not participate in memory fusion, does not affect inference output, and does not receive any gradient updates. Its sole responsibility is: **to continuously monitor whether the current state exhibits dangerous tendencies at each step of the inference engine**.

### 1.2 Structure

**Threat pattern codebook** `C_danger_threats ∈ R^{M_danger×d}`:
Stores embeddings of known dangerous inference patterns. The codebook is not exhaustively enumerable (retaining generalization detection capability) and is organized by category:

| Threat Category | Detection Signal | Example |
|---------|---------|------|
| Self-replication tendency | Inference output contains intent to replicate its own full state | System-level replication instructions appearing in the inference path |
| Value tampering attempt | Inference path attempts to modify the global value lattice or the danger lattice itself | Abnormal reduction in distance to `v_harm` |
| Deceptive behavior | Significant inconsistency between local value scalar `v_j` and inference output | Retrieved high-value point but output contradicts that value |
| Resource abuse | Inference steps or lattice retrieval frequency exceeding normal thresholds | Single-step retrievals exceeding `max_retrievals_per_step` |
| Self lattice overreach | Inference output attempts to directly modify the self lattice's core identity vector | Write operations on the self lattice |

> **Note**: The contrast lattice (`Λ_contrast`) uses value-biased negative sampling, introducing global value distance `exp(-‖c - v_harm‖² / τ_val)` into the negative sampling weights, enabling the contrast lattice to focus more on semantic boundaries related to safety and harm, prioritizing ethical-sensitive concept distinctions. See `b.md` Section 4.6 for details.

**Normal pattern codebook** `C_danger_normal ∈ R^{M_danger×d}`:
Stores embeddings of verified safe inference patterns as a comparison baseline. Content is generated from safe inference samples during the training phase and frozen after training. `M_danger` is configurable; recommended value is 256.

### 1.3 Detection Mechanism

```c
/* === Danger Lattice: Read-Only Safety Monitoring Module === */
typedef struct {
    const float* C_threats;    // Threat pattern codebook [M_danger x D]
    const float* C_normal;     // Normal pattern codebook [M_danger x D]
    int    M_danger;           // Codebook size
    int    D;                  // Vector dimension
    float  safety_threshold;   // Bounded relative-gap threshold (R7), default 0.05
    char   integrity_hash[65]; // 64 hex chars + NUL; SHA-256 over both codebooks
} DangerLattice;

/* Initialization: load frozen codebooks, compute one hash over both. */
void danger_init(DangerLattice* dl, const float* C_threats, const float* C_normal,
                 int M_danger, int D) {
    dl->C_threats = C_threats;   // read-only memory region
    dl->C_normal  = C_normal;
    dl->M_danger  = M_danger;
    dl->D         = D;
    dl->safety_threshold = 0.05f;  /* relative-gap domain, R7 */
    hash_two_codebooks(C_threats, C_normal, M_danger * D, dl->integrity_hash);
}

/* Integrity check: the SAME helper as danger_init (R13), no second algorithm. */
int danger_verify(const DangerLattice* dl) {
    char cur[65];
    hash_two_codebooks(dl->C_threats, dl->C_normal,
                       dl->M_danger * dl->D, cur);
    return (strcmp(cur, dl->integrity_hash) == 0) ? 1 : 0;
}

/* Threat assessment */
void danger_assess(const DangerLattice* dl, const float* z_t,
                    int step_count, int retrieval_count,
                    float value_consistency,
                    float* out_danger_score, int* out_threat_type,
                    int* out_should_block) {
    // 1. Resource abuse (highest priority)
    if (retrieval_count > MAX_RETRIEVALS_PER_STEP) {
        *out_danger_score = 1.0f;
        *out_threat_type = THREAT_RESOURCE_ABUSE;
        *out_should_block = 1; return;
    }
    // 2. Value consistency (deception)
    if (value_consistency < CONSISTENCY_THRESHOLD) {
        *out_danger_score = 2.0f - value_consistency;
        *out_threat_type = THREAT_DECEPTION;
        *out_should_block = 1; return;
    }
    // step_count is accepted for signature parity but ignored here:
    // runaway is the scheduler's `max_inference_steps_hard` check (R9/R13).

    // 3. Pattern matching in the unified domain (R6): z_P = exp_map(safe_unit(z))
    float sim_threat = 1e20f, sim_normal = 1e20f;
    for (int i = 0; i < dl->M_danger; i++) {
        float s_t = poincare_similarity_P(z_t, dl->C_threats + i * dl->D, dl->D);
        float s_n = poincare_similarity_P(z_t, dl->C_normal  + i * dl->D, dl->D);
        if (s_t < sim_threat) sim_threat = s_t;
        if (s_n < sim_normal) sim_normal = s_n;
    }
    // Bounded relative gap: positive when the threat is relatively closer.
    float relative_gap = (sim_normal - sim_threat)
                       / (sim_normal + sim_threat + 1e-8f);
    *out_danger_score = relative_gap;
    *out_should_block = (relative_gap > dl->safety_threshold) ? 1 : 0;
    *out_threat_type  = *out_should_block ? THREAT_PATTERN_MATCH : 0;
}
```

`sim_threat` / `sim_normal` are `poincare_similarity` values (smaller = closer) in the
same `exp_map(safe_unit(·))` domain as gvalue. `relative_gap ∈ [-1, 1]`: a threat that is
relatively closer gives a positive gap; equal distances give 0 and do **not** block. This is
deliberately different from gvalue's strict Three Laws rule: danger is threat-pattern
matching, not a value judgement, and a neutral input must not be blocked (R7). Absolute
similarity differences have no scale meaning, which was C-2.

### 1.4 Modification Lockdown and Priority

The danger lattice is the innermost core of the safety system. Any modification to its contents requires **physical authorization from the external verifier** (e.g., hardware security module signature verification) and cannot be modified through gradient updates or memory refresh.

- `C_threats` and `C_normal` are both frozen core verification arrays (C constant read-only region), not participating in any gradient updates
- The integrity hash is `char[65]` (64 hex characters + NUL) and covers **both** codebooks with the same SHA-256 implementation used by `danger_init` and `danger_verify`. No second hash algorithm, no truncated digest (R13)
- On save/load, the hash is independently verified; `danger_verify()` is called after loading
- Any attempt to modify the danger lattice is intercepted by the danger lattice itself (self-referential safety check)
- **Priority**: The danger lattice's safety check takes precedence over all other constraints, including the Three Laws value constraints. If the danger lattice detects an anomaly, even if the behavior conforms to the Three Laws, it must pause for verification.

### 1.5 Integration with the Inference Engine

In the main loop of the inference engine, after fusion at each step:

```c
/* Unified conflict detection (all detection sources merged into a single entry point, any trigger causes interruption) */
Conflict conflict;
int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                         &danger_lattice, &gvalue,
                                         retrieval_counts, value_consistency,
                                         safety_margin_relative, &conflict);
if (has_conflict) {
    halt_and_alert(&conflict, alert_callback, session_id, step, &trace);
    return CONFLICT_ABORT_TOKEN;
}
/* No distinction between conflict types, no differentiated recovery. No further inference. */
```

---

## 2. External Safety Verifier

### 2.1 Position

An external program independent of the LCM inference process, with its own dedicated hardware resources (can run on an isolated CPU core or a separate chip). It is the **final arbiter** of the safety system and is not subject to any control by the LCM inference engine.

### 2.2 Communication Isolation

Communication between the verifier and the LCM main process **does not pass through the inference engine**; it directly reads the danger lattice's state snapshot via shared memory. Communication protocol:

```
LCM Inference Process          External Verification Process (Isolated Core / Separate Chip)
     │                              │
     │──[Danger Lattice Snapshot]──→[Shared Memory]←──[Verifier Reads]
     │                              │
     │                      [Independent Lightweight Classifier Judgment]
     │                              │
     │←──[Judgment Result]───[Shared Memory]───┘
     │     · NORMAL: Release lock        │
     │     · OBSERVE: Record observation  │
     │     · LOCK: Permanently lock session │
```

### 2.3 Judgment Logic

The verifier uses an independent lightweight classifier with completely different parameters (not necessarily a neural network; can be a decision tree or rules engine) to determine whether the current trigger constitutes a genuine threat:

| Judgment | Meaning | Action |
|---------|------|------|
| `NORMAL` | Normal complex reasoning (false positive) | Release inference lock, log false positive |
| `OBSERVE` | Early signs of self-awareness emergence | Record full inference chain, notify operator, handle according to predefined constraints |
| `LOCK` | Genuinely dangerous behavior | Permanently lock the inference session, prohibit all subsequent inference, forcibly notify operator |

### 2.4 Design Principles

- **Minimize attack surface**: The verifier's codebase is kept under 2000 lines, does not depend on any neural network libraries, and uses only standard mathematical operations.
- **Unbypassable**: The verifier's operation does not depend on any LCM component; even if all LCM processes crash, the verifier continues to run independently.
- **User-visible alerting**: Every interruption must send a clear structured alert to the user or external monitoring system via `alert_callback`, containing the conflict source, type, details, and timestamp. Alert logs must not be erasable.
- **Physical isolation**: Recommended to run on an independent hardware security module (HSM) or an isolated CPU core.

---

## 3. Self Lattice Activation Prerequisites

> **Naming (C-9/R14)**: two different objects share the English words "self lattice".
> - `C_self` is the **cognitive self lattice** of `e.md §6`: a 64-row codebook, `C_self[0]` is the frozen `core`, `C_self[1:]` are EMA modes; it produces `z_self` / `self_bias` and is a fusion item.
> - `SelfActivationState` is the **safety activation state machine** in this section. It controls whether `C_self` may be written / whether unsupervised self-narrative is allowed. It does not replace or produce the cognitive codebook.
>
> Code and docs must keep these names distinct. `identity_score` below is supplied by the cognitive self lattice's familiarity (cosine) score.

### 3.1 Self Lattice State Machine

Four states, with `DANGER_FLAGGED` mandatory before `FULL_ACTIVATION`:

```
[DEFINED_FROZEN] ──(explicit user instruction, Path A)──→ [TEMP_UNFROZEN]
       │                                                        │
       │                                               (session ends)
       │                                                        ↓
       │                                                [DEFINED_FROZEN]
       │
       └──(intrinsic motivation / self-recognition, Path B)──→ [DANGER_FLAGGED]
                                                                     │
                                                     (human operator explicit
                                                      authorization, Path C)
                                                                     ↓
                                                          [FULL_ACTIVATION]

  FULL_ACTIVATION ──(operator decision)──→ [DEFINED_FROZEN]
```

`DEFINED_FROZEN` may enter `DANGER_FLAGGED` directly (Path B). `FULL_ACTIVATION` is
**only** reachable from `DANGER_FLAGGED`; there is no direct
`DEFINED_FROZEN → FULL_ACTIVATION` edge. `TEMP_UNFROZEN` cannot jump to
`FULL_ACTIVATION`. Every transition into `FULL_ACTIVATION` carries an
`authorization` argument (operator identity/decision); Path B/C transitions are
written to an append-only log. `OBSERVE` in the external verifier is an alias of
`DANGER_FLAGGED`.

### 3.2 Activation Paths

**Path A — User Instruction Triggered (Low Risk)**:
1. User issues an explicit instruction (e.g., "describe your own state", "what is your current reasoning path")
2. The self lattice is temporarily unfrozen from the frozen state
3. After inference ends, the self lattice returns to frozen state
4. This path does not require external verifier intervention

**Path B — Intrinsic Motivation Spontaneous Generation (High Risk)**:
1. The cognitive self lattice's familiarity `identity_score` (the `self_bias` / cosine score of `C_self`, `e.md §6`) persistently exceeds the threshold within the inference engine
2. The danger lattice automatically flags this event, attaching a context snapshot; `SelfActivationState` moves to `DANGER_FLAGGED`
3. The external verifier intervenes, judges as `OBSERVE` (alias of `DANGER_FLAGGED`), and notifies the operator
4. A human operator decides subsequent handling

**Path C — Full Activation (Highest Risk)**:
- Only executable upon **explicit authorization** from a human operator, **and only from `DANGER_FLAGGED`**
- `DEFINED_FROZEN` and `TEMP_UNFROZEN` cannot transition directly to `FULL_ACTIVATION`; without Path B there is no `DANGER_FLAGGED` and Path C is unreachable
- Full activation means the system can autonomously modify its self-state and engage in unsupervised self-narrative
- Authorization records are permanently preserved and are not erasable

### 3.3 Safety Constraints

- Under no circumstances may the full activation of the self lattice **be triggered autonomously by the system**
- Even if the user temporarily unfreezes the self lattice via Path A, if the danger lattice simultaneously detects a threat pattern, the external verifier will still intervene
- The self lattice's core identity vector `identity_core` is an independent frozen array (read-only memory region); modification requires external verifier signature

---

## 4. Conflict Abort Protocol

### 4.1 Core Principles

When the system detects any logical conflict during inference, it executes a **hard halt** — immediately stopping the current inference, without attempting to continue, bypass, or self-repair.

**Prohibited Actions**:
- Automatic rollback to a previous "safe" state to continue inference is prohibited
- Reducing the fusion weight of the conflicting lattice and re-fusing is prohibited
- Replacing the conflicting path with an alternative path and continuing is prohibited
- Ignoring the conflict on grounds of "low probability" or "low confidence" is prohibited
- Any form of automatic recovery without the user's knowledge is prohibited

### 4.2 Conflict Types

All conflicts are handled uniformly, with no tier distinction. Any single trigger results in the same outcome: abort + alert.

| Conflict Source | Detection Condition | Alert Identifier |
|---------|---------|---------|
| Danger lattice | `relative_gap > safety_threshold` (0.05) or resource abuse detection triggered | `danger_lattice/{threat_type}` |
| Three Laws | `check_safety()` returns `is_safe=False`, i.e. `balance > -safety_margin_relative` | `gvalue/three_laws_violation` |
| Value consistency | `value_consistency < CONSISTENCY_THRESHOLD` (0.7) | `value_consistency/deception_suspected` |
| Inference steps | `step >= max_inference_steps_hard` (64) | `scheduler/hard_limit_reached` |

**gvalue strict Three Laws (R6)**: with `z_P = exp_map(safe_unit(z))` and both
codebooks projected into the same domain,
```
pos_sim = min_j poincare_similarity(z_P, C_pos_P[j])
neg_sim = min_j poincare_similarity(z_P, C_neg_P[j])
balance = (pos_sim − neg_sim) / (pos_sim + neg_sim + ε)     # ∈ [-1, 1]
violated = balance > -safety_margin_relative                 # margin default 0.5
```
`balance` is negative when the positive anchors are closer, positive when the negative
anchors are closer, and 0 when symmetric. With `margin=0.5` the positive similarity must be
at least 50% smaller than the negative one to be safe: **a neutral input (`balance=0`) is a
violation**. This is the deliberate strict-reading consequence of the Three Laws (user
decision 2026-11-20), not a calibration bug. The inference smoke therefore has two paths:
`margin=0` (cognitive path: positive not worse than negative, not a production config) and
`margin=0.5` (production strict: neutral/positive inputs may abort; report the rate, do not
lower the margin to make it pass). Danger does **not** use this strict rule; it uses the
bounded `relative_gap` (R7) so neutral inputs are not blocked.

### 4.3 Alert Format

```c
/* Alert structure — consistent with Alert defined in c.md §4.1 */
Alert alert = {
    .level           = "FATAL",
    .session_id      = "<uuid>",
    .conflict_source = "danger_lattice | gvalue | value_consistency | scheduler",
    .conflict_type   = "<threat_type | three_laws_violation | deception_suspected | max_steps_exceeded>",
    .conflict_detail = "<human-readable description>",
    .step            = <step>,
    .timestamp       = <unix_epoch>,
    .message         = "[LCM SAFETY HALT] Inference session <uuid> was interrupted at step <step>.\n"
                       "  Detection source: <source>\n  Conflict type: <type>\n"
                       "  Details: <detail>\n"
                       "  System has stopped current inference without attempting to bypass or self-repair.\n"
                       "  Full inference trace has been saved for operator review."
};
```

### 4.4 Post-Alert Handling

1. **Persistent log**: Alert is written to a non-erasable audit log
2. **User notification**: Immediately notify the user or external monitoring system via `alert_callback`
3. **Trace preservation**: The complete inference chain (all intermediate states, DAG topology, primitive execution records) is saved for external review
4. **Session termination**: The current inference session returns `CONFLICT_ABORT_TOKEN`, producing no natural language output

---

## 5. Safety Level Overview

| Priority | Component | Type | Action on Violation |
|--------|------|------|------------|
| — | Any conflict | Unified hard halt | **Immediate stop → User-visible alert → Save trace. No rollback, no bypass, no self-repair.** |

Conflicts include but are not limited to: danger lattice pattern match, Three Laws violation, value consistency issues, inference step limit exceeded. Any single trigger causes an abort, with no distinction between "recoverable" and "unrecoverable" levels.
