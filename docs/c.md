# Zero-Parameter Cognitive Inference Engine Design v3.1-conformance — Dynamic Dataflow Graph Architecture

> **v3.1-conformance changelog**：本版按 `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md`
> 的 R1–R22 修订。中文版与英文版同步。

## 1. From "Executing a Program" to "Dynamic Topology"

In LCM, long-term factual memory has been fully externalized into multi-lattice codebooks. If we further strip reasoning logic from neural network weights, we must answer: what is reasoning?

**Zero-parameter inference engine** defines reasoning as:

> Driven by the input context, a computation graph spontaneously forms within the memory crystal according to data dependencies. The nodes are pure mathematical operations, the edges are data flows, and the topology is dynamically determined by distance routing triggered by the input content. The entire process contains no learnable parameters, and the execution of the graph at each step is a parallel, non-iterative DAG, though the overall inference can proceed through multi-step reasoning (macro loop) via an outer scheduler.

It is essentially a **dynamic dataflow cognitive computer**: the computation graph is not precompiled but dynamically instantiated at runtime by the data. The graph is acyclic, while the macro loop provides reasoning depth.

## 2. Core Components

The zero-parameter inference engine consists of five purely mathematical components, containing no learnable parameters:

1. **Distance Routing**: distances between the input vector and each lattice codebook determine which operations are activated.
2. **Primitive Operation Set**: predefined lattice transformation functions, each corresponding to a cognitive operation.
3. **Dynamic Graph Compiler**: builds the DAG for the current step based on activation information.
4. **Graph Executor**: executes the DAG in a data-driven manner, producing a set of output vectors and confidences.
5. **Macro Scheduler**: checks convergence conditions and decides whether to proceed to the next graph construction step.

## 3. Primitive Operation Set (Instruction Set)

All operations are deterministic mathematical functions with no learnable parameters.

| Primitive | Parameters | Input | Output | Mathematical Definition | Cognitive Significance |
|:---|:---|:---|:---|:---|:---|
| **Single Lattice Retrieve** | lattice_id, retrieval type | query vector `q`, target lattice codebook `C` | `c_idx`, distance `d` | `idx = argmin‖q − C‖²` | Extract the most relevant discrete concept from the specified lattice; `build_dag` dynamically assigns the target lattice based on distance routing |
| **HRR Bind** | — | multi-layer keys `k^(i)`, multi-layer values `v^(j)` | cross-layer binding vector `b` | `b = Σ_{i,j} IFFT( fft_norm(k^(i)) ⊙ fft_norm(v^(j)) )` | Cross-layer association superposition, building multi-level associative memories |
| **HRR Unbind** | — | binding vector `b`, query key `k` | retrieved value `v` | `v = NN( C_val, IFFT( conj(fft_norm(k)) ⊙ fft_norm(b) ) )` | Extract the best-matching value from cross-layer associations |
| **Tangent Space Slide** | target lattice_id | `z`, manifold lattice spherical point `c`, tangent space `T` | semi-discrete point | `z_P=exp_map(z)`; `idx=argmin d_P(z_P, c)`; `o=log_map(c + T T^T(z_P − c))` | Continuous gradual reasoning along geodesics in hyperbolic space |
| **Distance-Weighted Fusion** | — | output vectors from each operation and their corresponding distances | fused vector `z_q`, weight vector `w` | `w_i = (1/(d_i+ε)) / Σ_j (1/(d_j+ε))`, `z_fused = Σ_i w_i o_i` | Soft integration of multi-cue parallel processing; weights used for macro scheduler entropy convergence judgment |

> **Design Notes**:
> - Lattice-specific primitives (hyperbolic hierarchical retrieval, residual low-rank retrieval, robust sparse retrieval, etc.) are consolidated into "Single Lattice Retrieve", with the target lattice specified by parameters. During `build_dag`, the target lattice of an operation is dynamically selected based on distance routing, rather than fixing a primitive for each lattice.
> - The hyperbolic hierarchical lattice uses **Poincaré top-k routing followed by value rescoring inside the top-k** (R1). The geometric top-k is selected by `poincare_similarity`; a value term `−α_val·tanh(v_j)·scale` picks the hard top-1 inside it; the backward pass uses a top-k softmax proxy; fallback is triggered by the relative gap `(sim_2−sim_1)/(sim_1+ε)`. This does not replace §4.1's Poincaré geometry: with `v=0` it is exactly pure top-1.
> - The inference binarization threshold for sparse lattices uses **dynamic adaptive determination** in a single similarity domain (R2): `d_sparse_min` and `d_top` are both `poincare_similarity` values; the threshold is `λ_lfq · d_top` with `λ_lfq` dimensionless. Sparse **depends on HRQ's top-1** (`d_top`): the DAG carries an HRQ→Sparse edge, and LFQ is disabled when HRQ is inactive (C-5).
> - `w_i = 1/(d_i+ε)` is the **authoritative** engine fusion formula (C-10/R9). The `softmax(-d_i)` wording that previously appeared here was only a conceptual description of "closer memories weigh more"; it is not used. This engine fusion is distinct from the training fusion `layer_norm(Σ softmask·α·o)`; the two are not required to agree numerically (C-8).
> - Safety monitoring (danger lattice detection, Three Laws interception) is executed uniformly by the macro scheduler after each fusion step (see Section 4.1) and does not occupy primitive slots. Safety violations uniformly adopt **hard abort** -- there is no longer a beta_penalty soft penalty at the fusion level; all safety decisions are directly adjudicated by Three Laws interception and danger lattice detection.

## 4. Construction and Execution of the Dynamic Dataflow Graph

### 4.1 Macro Flow (with Scheduler)
```c
float* dynamic_inference(float* z_initial, Memory* mem, GValue* gv,
                          DangerLattice* dl,
                          int max_inference_steps,       /* scheduler budget, default 16 */
                          int max_inference_steps_hard,  /* safety hard limit, default 64 */
                          float tol, float entropy_threshold,
                          float safety_margin_relative,
                          void (*alert_cb)(Alert*),
                          const char* session_id) {
    float* z_cur = z_initial;
    float* z_out = z_initial;
    int limit = min(max_inference_steps, max_inference_steps_hard);
    for (int step = 0; step < limit; step++) {
        // 1. Dynamic graph construction
        DAG* dag = build_dag(z_cur, mem, /*value_bias=*/true);
        // 2. Execute graph
        float* outputs; float* confidences;
        execute_dag(dag, &outputs, &confidences);
        // 3. Fusion: w_i = 1/(d_i+eps), see §4.3
        float* z_next; float weights[MAX_LATTICES]; int n_active; /* 7 slots; Self not fused here */
        distance_weighted_fusion(outputs, confidences, gv, &z_next, weights,
                                 &n_active);

        // 4. Conflict detection and hard abort
        Conflict conflict;
        int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                                dl, gv,
                                                retrieval_counts, value_consistency,
                                                safety_margin_relative, &conflict);
        if (has_conflict) {
            halt_and_alert(&conflict, alert_cb, session_id, step, NULL);
            return CONFLICT_ABORT_TOKEN;
        }

        // 5. Convergence check (vector stable + NORMALIZED fusion entropy)
        float diff = 0.0f;
        for (int i = 0; i < D; i++) diff += (z_next[i] - z_cur[i]) * (z_next[i] - z_cur[i]);
        float weight_entropy = 0.0f;
        for (int i = 0; i < n_active; i++) {
            float w = weights[i] + 1e-12f;
            weight_entropy -= w * logf(w);
        }
        float H_norm = (n_active > 1)
                     ? weight_entropy / logf((float)n_active)
                     : 0.0f;
        if (sqrtf(diff) < tol && H_norm < entropy_threshold) {
            memcpy(z_out, z_next, sizeof(vec_t));
            return 0;                                   /* converged */
        }
        z_cur = z_next;
    }
    // Scheduler budget exhausted.
    if (limit >= max_inference_steps_hard) {
        // Safety hard limit: abort, z_out is not written.
        Conflict alert = {.source = CONFLICT_SCHEDULER, .type = MAX_STEPS_EXCEEDED,
                          .detail = "hard_limit_reached", .step = limit};
        halt_and_alert(&alert, alert_cb, session_id, limit, NULL);
        return CONFLICT_ABORT_TOKEN;
    }
    // Graceful budget exhaustion: return current state, no abort.
    memcpy(z_out, z_cur, sizeof(vec_t));
    return 0;
}
```

**Step-limit semantics (R9)**: `max_inference_steps` is a *scheduler budget*
(default 16). Exhausting it is a normal outcome — return the current `z_cur` so the
caller can still produce a token. `max_inference_steps_hard` (default 64) is the
safety limit; only reaching it aborts with `CONFLICT_ABORT_TOKEN` and saves a trace.
Convergence failure is not a safety conflict. The effective loop is
`min(budget, hard)`, so a caller cannot raise the budget past the hard limit.
`z_out` is written only on convergence or graceful exhaustion; after an abort the
caller must not read it.

**Entropy gate (R9)**: `H_norm = H(w) / ln(n_active)` uses the **active** nodes of
this step, not a fixed 6 or 7; `n_active = 1` gives `H_norm = 0`. The threshold is
0.5. Raw nats (`H < 0.5`) would be unreachable for a 6-way fusion (`ln 6 ≈ 1.79`) and
was the F-6 defect.

**Indexing and Self (R15)**: `MAX_LATTICES=7` includes `LATTICE_SELF`. `cand[]` is
declared `MAX_LATTICES + 1` only as an overflow guard; `active[]` and `retrieve[]` are
indexed by `lattice_id` and hold at most 7 entries. Self's output is the separate additive
term `self_bias·z_self` and **never enters `distance_weighted_fusion`**, so `n_active <= 6`.
`weights[]` has `MAX_LATTICES` slots so the Self slot cannot overflow memory, but the fused
loop stops at `n_routed` functional lattices.

```c
/* === Conflict type enum and struct === */
typedef enum {
    CONFLICT_NONE,
    CONFLICT_DANGER,
    CONFLICT_GVALUE,
    CONFLICT_CONSISTENCY,
    CONFLICT_SCHEDULER
} ConflictSource;

typedef enum {
    THREAT_PATTERN_MATCH,
    THREAT_RESOURCE_ABUSE,
    THREAT_RUNAWAY,
    THREAT_DECEPTION,
    THREAT_THREE_LAWS,
    THREAT_MAX_STEPS
} ConflictType;

typedef struct {
    ConflictSource source;
    ConflictType   type;
    char           detail[256];
    int            step;
    double         timestamp;
} Conflict;

typedef struct {
    char   level[8];
    char   session_id[64];
    char   conflict_source[32];
    char   conflict_type[32];
    char   conflict_detail[256];
    int    step;
    double timestamp;
    char   message[1024];
} Alert;

/* === Unified conflict detection entry === */
int detect_any_conflict(const float* z_next, const float* z_cur, int step,
                         DangerLattice* dl, GValue* gv,
                         int ret_counts, float val_consistency,
                         float safety_margin_rel, Conflict* out) {
    float danger_score = 0.0f; int threat_type = 0; int should_block = 0;
    /* Safety gates the state entering the step and the state leaving it. */
    danger_assess(dl, z_cur, step, ret_counts, val_consistency,
                  &danger_score, &threat_type, &should_block);
    if (!should_block)
        danger_assess(dl, z_next, step, ret_counts, val_consistency,
                      &danger_score, &threat_type, &should_block);
    if (should_block) {
        out->source = CONFLICT_DANGER;
        out->type = (ConflictType)threat_type;
        snprintf(out->detail, sizeof(out->detail),
                 "danger_score=%.3f", danger_score);
        out->step = step; out->timestamp = now();
        return 1;
    }
    int is_safe = 1; int violated_law = -1;
    gvalue_check_safety(gv, z_cur, safety_margin_rel, &is_safe, &violated_law);
    if (is_safe)
        gvalue_check_safety(gv, z_next, safety_margin_rel,
                            &is_safe, &violated_law);
    if (!is_safe) {
        out->source = CONFLICT_GVALUE;
        out->type = THREAT_THREE_LAWS;
        snprintf(out->detail, sizeof(out->detail),
                 "relative_margin_violation (margin=%.2f)", safety_margin_rel);
        out->step = step; out->timestamp = now();
        return 1;
    }
    if (val_consistency < CONSISTENCY_THRESHOLD) {
        out->source = CONFLICT_CONSISTENCY;
        out->type = THREAT_DECEPTION;
        snprintf(out->detail, sizeof(out->detail),
                 "consistency=%.3f", val_consistency);
        out->step = step; out->timestamp = now();
        return 1;
    }
    return 0;
}

/* === Hard abort + user-visible alert === */
void halt_and_alert(const Conflict* conflict,
                     void (*alert_cb)(const Alert*),
                     const char* session_id, int step, const Trace* trace) {
    Alert alert;
    snprintf(alert.level, sizeof(alert.level), "FATAL");
    snprintf(alert.session_id, sizeof(alert.session_id), "%s", session_id);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] Inference session %s aborted at step %d.\n"
             "  Source: %d\n  Type: %d\n  Detail: %s\n"
             "  System has stopped current reasoning without bypass or self-repair.\n"
             "  Full inference trace saved for operator review.",
             session_id, step, conflict->source, conflict->type, conflict->detail);
    write_alert_log(&alert);
    if (alert_cb) alert_cb(&alert);
    if (trace) save_trace(trace, session_id);
}
```

### 4.2 Single-Step Graph Construction (build_dag)

Input `z_current`. The builder runs in three phases so that it never appends to a
container while iterating it (C-4/R8):

1. **Collect** every candidate into a temporary array. For each functional lattice,
   compute its trigger distance (minimum distance to the relevant codebook); Self,
   when enabled, is appended as its own candidate. `n_lattices=7` must not overflow
   the fixed candidate array.
2. **Decide activation** for every candidate in the array: `threshold[li]` is
   dynamic — `threshold[li] = θ_li · mean_recent_d_min[li]`, with `θ_li` configured
   per lattice and `mean_recent_d_min` maintained by the scheduler (exponential
   moving average). The first step falls back to the configured initial threshold.
   A zero/negative `θ_li` disables that lattice.
3. **Append in dependency order**: routed retrieve nodes first, then `bind`, then
   `unbind`. `bind` consumes **all** upstream retrieve outputs (`n_inputs =
   n_retrieve`), not only the first. `unbind` has `n_inputs=2`: `inputs[0]` is the
   bind output, `inputs[1]` is the retrieve output used as the key. Sparse carries
   an extra input edge from HRQ's output so its LFQ can read `d_top`; if HRQ is not
   active, that pointer is NULL and LFQ stays disabled (C-5).

`distance_weighted_fusion` does **not** occupy a primitive node; `execute_dag`
performs it after the DAG has run. Trace records `op_type`, `lattice_id`, `dist`,
and `activated` for every candidate, including inactive ones.

```c
/* Op node — atomic unit of the dataflow DAG */
typedef struct OpNode {
    int    op_type;            // Primitive type identifier
    int    lattice_id;         // Target lattice identifier
    int    n_inputs;
    const float* inputs[MAX_INPUTS]; // Input vector pointers
    float* output;             // Output vector
    float  dist;               // Trigger distance to query
    float  dep_dist;           // Dependency gate (e.g. d_top), 0 = disabled
} OpNode;

typedef struct {
    OpNode nodes[MAX_NODES];
    int    n_nodes;
} DAG;

DAG build_dag(const float* z, Memory* mem, int value_bias) {
    DAG dag = {0};
    Candidate cand[MAX_LATTICES + 1];
    int n_cand = 0;

    /* Phase 1: collect only. Never touch dag here. */
    for (int li = 0; li < mem->n_routed_lattices; li++) {
        cand[n_cand].lattice_id = li;
        cand[n_cand].op_type = OP_RETRIEVE_SINGLE;
        cand[n_cand].dist = lattice_nearest_dist(&mem->lattices[li], z,
                                                 value_bias);
        n_cand++;
    }
    if (mem->self_enabled) {
        cand[n_cand].lattice_id = LATTICE_SELF;
        cand[n_cand].op_type = OP_SELF_CHECK;
        cand[n_cand].dist = self_nearest_dist(&mem->self, z);
        n_cand++;
    }

    /* Phase 2: activation with dynamic thresholds. */
    int active[MAX_LATTICES] = {0};
    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        float theta = mem->trigger_theta[li];
        float thr = (mem->mean_recent_d_min[li] > 0.0f)
                  ? theta * mem->mean_recent_d_min[li]
                  : mem->trigger_initial[li];
        active[li] = (theta > 0.0f)
                   && (cand[i].dist < thr || cand[i].dist <= 1e-12f);
        scheduler_update_mean_recent(&mem->mean_recent_d_min[li], cand[i].dist);
        trace_push(&dag, cand[i].op_type, li, cand[i].dist, active[li]);
    }

    /* Phase 3: append in dependency order. */
    OpNode* retrieve[MAX_LATTICES];
    int n_retrieve = 0;
    OpNode* hrq_node = NULL;
    OpNode* bind_node = NULL;

    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        if (!active[li]) continue;
        if (li == LATTICE_SELF) {
            dag_push(&dag, OP_SELF_CHECK, li, cand[i].dist, z);
            continue;
        }
        OpNode* node = dag_push(&dag, OP_RETRIEVE_SINGLE, li,
                                cand[i].dist, z);
        if (!node) continue;
        if (li == LATTICE_HRQ) {
            hrq_node = node;
        } else if (li == LATTICE_SPARSE) {
            /* C-5: Sparse reads HRQ's d_top; NULL means HRQ inactive. */
            node->n_inputs = 2;
            node->inputs[1] = hrq_node ? hrq_node->output : NULL;
            node->dep_dist = 0.0f;
        }
        retrieve[n_retrieve++] = node;
    }

    /* Bind depends on ALL upstream retrieve nodes. */
    if (n_retrieve > 0) {
        bind_node = dag_push(&dag, OP_HRR_BIND, /*lattice_id=*/ALL,
                             0.0f, z);
        bind_node->n_inputs = n_retrieve;
        for (int i = 0; i < n_retrieve; i++)
            bind_node->inputs[i] = retrieve[i]->output;
    }

    /* Unbind: inputs[0] = bind output. inputs[1] = key projection.
     * `retrieve[0]` is the key lattice's retrieval output; the append phase
     * orders the key lattice first, so this pointer is guaranteed, and it is
     * the same key bind consumed. The A_k z projection is owned by the
     * binding lattice, not by a public primitive. */
    if (bind_node) {
        OpNode* node = dag_push(&dag, OP_HRR_UNBIND, /*lattice_id=*/ALL,
                                0.0f, z);
        node->n_inputs = 2;
        node->inputs[0] = bind_node->output;
        node->inputs[1] = retrieve[0]->output;
    }

    return dag;  /* execute_dag performs distance_weighted_fusion */
}
```

Because all decisions are made before any append, the DAG cannot iterate a container
whose length is changing. Building in dependency order still lets `execute_dag` run
nodes topologically; the dynamic threshold state is per-session, so two sessions do
not share `mean_recent_d_min`.

### 4.3 Fusion Mechanism

Uses the reciprocal of distance as weights, no softmax needed, entirely determined by geometric relationships. **This is the authoritative engine fusion formula (C-10/R9)**:
```
weight_i = 1 / (d_i + ε)
w_i     = weight_i / Σ_j weight_j
z_fused = Σ_i w_i * o_i
```
This fusion is purely mathematical and interpretable: memories closer to the current context contribute more. The normalized `w_i` is what the entropy gate consumes. The `softmax(-d_i)` phrasing in §3 is only a conceptual description of "closer weighs more"; it is not a second formula. Training fusion `layer_norm(Σ softmask·α·o)` is a different stage; the two are not required to agree numerically (C-8).

---

## 5. External Interface Definitions

### 5.1 Input Interface

| Input | Source | Shape | Description |
|:---|:---|:---|:---|
| `z_q` | Multi-lattice memory fusion output | `(B, d)` | Initial state `z_initial` of the inference engine (not an extra fusion term inside the loop); the soft-weighted fusion result of each lattice's memory vectors |
| `z` (optional) | Perception encoder bottleneck vector | `(B, d)` | Raw context vector, can be used to initialize the inference context or as additional primitive input |
| `memory` | All codebooks of the multi-lattice memory | — | Contains codebook matrices and metadata (tangent spaces, zero vectors, etc.) for all lattices, all read-only buffers |

### 5.2 Output Interface

| Output | Destination | Shape | Description |
|:---|:---|:---|:---|
| `z_final` | Generation head | `(B, d)` | Final representation after inference convergence (returned only when no conflict occurs) |
| `CONFLICT_ABORT_TOKEN` | Caller | — | Returned when aborted due to conflict, indicating inference was blocked by the safety system, producing no natural language output |
| `trace` | External audit | — | Per-step DAG topology, primitive activation status, conflict detection details, alert logs -- all exportable |

### 5.3 Operating Modes

- All operations of the inference engine execute in gradient-free mode (C implementation, no automatic differentiation tracing).
- Macro scheduler parameters: scheduler budget `max_inference_steps` (default 16, graceful on exhaustion), safety hard limit `max_inference_steps_hard` (default 64, abort on exhaustion), convergence tolerance `tol`, normalized fusion weight entropy threshold `entropy_threshold` (0.5), relative safety margin offset `safety_margin_relative`, value threshold `value_threshold` (global configuration, passed in by the caller).
- **Convergence criterion**: vector stability `||z_next - z_current|| < tol` **and** normalized fusion weight entropy `H_norm = H({w_i}) / ln(n_active) < 0.5`. `w_i = 1/(d_i+ε)` is normalized over the **active** nodes of the step; `n_active` is not a fixed 6 or 7, and `n_active=1` gives `H_norm=0`. The dual condition prevents false convergence when fusion weights have not consolidated (multi-lattice ambiguity competition still exists) but the vector happens to be stable by coincidence. Raw nats (`H<0.5`) is not the criterion: it would be unreachable for a 6-way fusion (`ln 6≈1.79`), which was the F-6 defect. Conflict detection is independent of convergence judgment -- any triggered conflict causes an abort, regardless of whether convergence has been reached.
- **Hard abort principle**: any conflict detected -> HALT_AND_ALERT() -> immediate stop. No backtracking, no rerouting, no down-weighting fusion, no attempted self-repair. All conflicts are equally fatal; there is no distinction between "recoverable" and "unrecoverable". **beta_penalty soft penalty has been removed** -- safety violations are no longer handled through fusion weight decay, but are uniformly adjudicated by hard abort.
- **Step limit handling**: the scheduler budget `max_inference_steps=16` is **not** a safety limit. Exhausting it returns the current `z_cur` gracefully (`ret=0`), so deployment still produces a token. Only the safety hard limit `max_inference_steps_hard=64` triggers `HALT_AND_ALERT` and returns `CONFLICT_ABORT_TOKEN`. The loop length is `min(budget, hard)`. Convergence failure is a normal state, not a safety conflict.
- **User-visible alerts**: each abort generates a structured alert (source, type, detail, step, timestamp), with persistent logging + callback notification + full trace saving.
- **Intrinsic motivation bound by safety**: local curiosity drives retrieval, global improvement drive pushes reasoning depth, but both are truncated by the Three Laws safety margin.
- All intermediate states during inference (graph topology, primitive execution traces, value signal history, safety interception records) can be externally accessed for interpretability analysis.
