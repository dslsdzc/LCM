/* LCM Inference Engine — Dynamic Dataflow Graph
 *
 * Zero-parameter cognitive inference:
 * 1. build_dag   — Construct computation graph from current state
 * 2. execute_dag — Execute all nodes in topological order
 * 3. fusion      — Weighted fusion of lattice outputs
 * 4. detect      — Safety check (danger + gvalue + consistency)
 * 5. schedule    — Convergence check and step loop
 *
 * FORMAL CONTRACT:
 *   - No dynamic memory allocation (all buffers caller-provided)
 *   - The engine is a PURE FUNCTION of (state, z_initial)
 *   - On conflict: engine returns -1, no recovery attempted
 *   - Thread-safe: no mutable global state
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <time.h>

/* ─── DAG building — Primitive layer definitions ─────────────────────────── */

typedef struct {
    const char* name;
    op_type_t   type;
} primitive_def_t;

/* Layer 0: Independent retrieval/slide (no dependencies) */
static const primitive_def_t LAYER_0[] = {
    {"retrieve_single",    OP_RETRIEVE_SINGLE},
    {"slide_manifold",     OP_SLIDE_MANIFOLD},
    {"contrast_retrieve",  OP_CONTRAST_RETRIEVE},
    {"self_check",         OP_SELF_CHECK},
    {NULL, 0}
};

/* Layer 1: HRR bind (depends on retrieval outputs) */
static const primitive_def_t LAYER_1[] = {
    {"bind", OP_HRR_BIND},
    {NULL, 0}
};

/* Layer 2: HRR unbind (depends on bind output) */
static const primitive_def_t LAYER_2[] = {
    {"unbind", OP_HRR_UNBIND},
    {NULL, 0}
};

/* Layer 3: Distance-weighted fusion (depends on all upstream) */
static const primitive_def_t LAYER_3[] = {
    {"fusion", OP_DISTANCE_FUSION},
    {NULL, 0}
};

static const primitive_def_t* PRIMITIVE_LAYERS[] = {
    LAYER_0, LAYER_1, LAYER_2, LAYER_3
};
#define NUM_LAYERS (sizeof(PRIMITIVE_LAYERS) / sizeof(PRIMITIVE_LAYERS[0]))

/* ─── Lattice "nearest_dist" helper ──────────────────────────────────────── */

static float lattice_nearest_dist(const float* z, const lattice_memory_t* mem,
                                   int* out_idx) {
    float best = 1e20f;
    int   idx = 0;
    for (int j = 0; j < mem->M; j++) {
        float d = 0;
        for (int i = 0; i < LCM_D; i++) {
            float diff = z[i] - mem->C[j][i];
            d += diff * diff;
        }
        if (d < best) { best = d; idx = j; }
    }
    if (out_idx) *out_idx = idx;
    return best;
}

/* ─── build_dag: Construct DAG from current query vector ───────────────────
 *
 * CONTRACT:
 *   Pre:  z is initialized, mem points to valid memory set
 *   Post: Returns a DAG with nodes ordered by layer (dependency order)
 *         Nodes are only added if their trigger distance < threshold
 *         Output DAG has n_nodes ≤ LCM_MAX_NODES
 */
dag_t build_dag(const float* z, memory_t* mem, bool value_bias) {
    (void)value_bias;
    dag_t dag = {0};

    /* Helper: get per-lattice threshold with backward-compatible default */
#define THRESH(li) ((mem)->thresholds[(li)] > 0 ? (mem)->thresholds[(li)] : 1e10f)

    for (int layer = 0; layer < (int)NUM_LAYERS; layer++) {
        for (int p = 0; PRIMITIVE_LAYERS[layer][p].name != NULL; p++) {
            op_type_t op_type = PRIMITIVE_LAYERS[layer][p].type;

            switch (op_type) {
            case OP_RETRIEVE_SINGLE:
                /* Euclidean nearest-neighbor for HRQ, SPARSE, LOWRANK, BINDING */
                for (int li = 0; li < mem->n_lattices; li++) {
                    if (li == LATTICE_MANIFOLD || li == LATTICE_CONTRAST
                        || li == LATTICE_SELF) continue;
                    int idx;
                    float d_min = lattice_nearest_dist(z, &mem->lattices[li], &idx);
                    if (d_min < THRESH(li) && dag.n_nodes < LCM_MAX_NODES) {
                        op_node_t* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = op_type;
                        node->lattice_id = li;
                        node->n_inputs = 1;
                        node->inputs[0] = z;
                        node->dist = d_min;
                        node->output = &dag.storage[dag.n_nodes - 1];
                    }
                }
                break;

            case OP_SLIDE_MANIFOLD: {
                /* Poincaré + tangent projection for MANIFOLD lattice */
                int li = LATTICE_MANIFOLD;
                if (li < mem->n_lattices) {
                    int idx;
                    float d_min = lattice_nearest_dist(z, &mem->lattices[li], &idx);
                    if (d_min < THRESH(li) && dag.n_nodes < LCM_MAX_NODES) {
                        op_node_t* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = op_type;
                        node->lattice_id = li;
                        node->n_inputs = 1;
                        node->inputs[0] = z;
                        node->dist = d_min;
                        node->output = &dag.storage[dag.n_nodes - 1];
                    }
                }
                break;
            }

            case OP_CONTRAST_RETRIEVE: {
                /* Poincaré distance retrieval for CONTRAST lattice */
                int li = LATTICE_CONTRAST;
                if (li < mem->n_lattices) {
                    int idx;
                    float d_min = lattice_nearest_dist(z, &mem->lattices[li], &idx);
                    if (d_min < THRESH(li) && dag.n_nodes < LCM_MAX_NODES) {
                        op_node_t* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = op_type;
                        node->lattice_id = li;
                        node->n_inputs = 1;
                        node->inputs[0] = z;
                        node->dist = d_min;
                        node->output = &dag.storage[dag.n_nodes - 1];
                    }
                }
                break;
            }

            case OP_SELF_CHECK: {
                /* Self-consistency — always on */
                int li = LATTICE_SELF;
                if (li < mem->n_lattices && dag.n_nodes < LCM_MAX_NODES) {
                    op_node_t* node = &dag.nodes[dag.n_nodes++];
                    node->op_type = op_type;
                    node->lattice_id = li;
                    node->n_inputs = 1;
                    node->inputs[0] = z;
                    node->dist = 0.0f;
                    node->output = &dag.storage[dag.n_nodes - 1];
                }
                break;
            }

            case OP_HRR_BIND:
                /* Bind: connect to first retrieval node found */
                for (int ni = 0; ni < dag.n_nodes; ni++) {
                    if (dag.nodes[ni].op_type == OP_RETRIEVE_SINGLE
                        && dag.n_nodes < LCM_MAX_NODES) {
                        op_node_t* node = &dag.nodes[dag.n_nodes++];
                        node->op_type = OP_HRR_BIND;
                        node->lattice_id = LCM_MAX_LATTICES;  /* sentinel: HRR outputs are not lattice outputs */
                        node->n_inputs = 2;
                        node->inputs[0] = (const float*)dag.nodes[ni].output;
                        node->inputs[1] = z;
                        node->output = &dag.storage[dag.n_nodes - 1];
                        break;
                    }
                }
                break;

            case OP_HRR_UNBIND: {
                /* Unbind: connect to bind node + a retrieval node */
                op_node_t* bind_node = NULL;
                op_node_t* key_node  = NULL;
                for (int ni = 0; ni < dag.n_nodes; ni++) {
                    if (dag.nodes[ni].op_type == OP_HRR_BIND)
                        bind_node = &dag.nodes[ni];
                    if (dag.nodes[ni].op_type == OP_RETRIEVE_SINGLE && !key_node)
                        key_node = &dag.nodes[ni];
                }
                if (bind_node && dag.n_nodes < LCM_MAX_NODES) {
                    op_node_t* node = &dag.nodes[dag.n_nodes++];
                    node->op_type = OP_HRR_UNBIND;
                    node->lattice_id = LCM_MAX_LATTICES;  /* sentinel: HRR outputs are not lattice outputs */
                    node->n_inputs = 2;
                    node->inputs[0] = (const float*)bind_node->output;
                    node->inputs[1] = key_node ? (const float*)key_node->output : z;
                    node->output = &dag.storage[dag.n_nodes - 1];
                }
                break;
            }

            default:
                /* OP_DISTANCE_FUSION is handled by execute_dag directly */
                break;
            }
        }
    }
#undef THRESH
    return dag;
}

/* ─── Execute DAG: Run all nodes in topological order ──────────────────────
 *
 * CONTRACT:
 *   Pre:  dag is built by build_dag
 *         outputs has space for LCM_MAX_LATTICES vectors
 *         confidences has space for LCM_MAX_LATTICES floats
 *   Post: All nodes have been executed
 *         outputs and confidences are populated
 */
bool execute_dag(dag_t* dag, const memory_t* mem, vec_t* outputs, float* confidences) {
    if (!dag || !mem || !outputs || !confidences) return false;

    int t_dim = mem->manifold_t_dim > 0 ? mem->manifold_t_dim : 4;

    /* Execute each node in order (already topologically sorted by layer) */
    for (int ni = 0; ni < dag->n_nodes; ni++) {
        op_node_t* node = &dag->nodes[ni];
        vec_t result;
        memset(result, 0, sizeof(vec_t));

        switch (node->op_type) {
        case OP_RETRIEVE_SINGLE: {
            float dist;
            int idx;
            retrieve_single(node->inputs[0],
                            node->lattice_id < mem->n_lattices
                                ? &mem->lattices[node->lattice_id] : NULL,
                            result, &dist, &idx);
            node->dist = dist;
            if (node->lattice_id < LCM_MAX_LATTICES) {
                confidences[node->lattice_id] = 1.0f / (dist + 1e-6f);
            }
            break;
        }
        case OP_SLIDE_MANIFOLD: {
            const lattice_memory_t* lm = NULL;
            if (node->lattice_id < mem->n_lattices)
                lm = &mem->lattices[node->lattice_id];
            if (lm && mem->manifold_T_space) {
                slide_manifold(node->inputs[0], lm,
                               mem->manifold_T_space, t_dim, result);
            } else {
                /* Fallback: no T_space available */
                retrieve_single(node->inputs[0], lm, result, NULL, NULL);
            }
            break;
        }

        case OP_CONTRAST_RETRIEVE: {
            /* Poincaré-distance weighted retrieval for contrast lattice */
            int li = node->lattice_id;
            const lattice_memory_t* lm = li < mem->n_lattices ? &mem->lattices[li] : NULL;
            if (lm && lm->M > 0) {
                float best = 1e20f;
                int best_idx = 0;
                for (int j = 0; j < lm->M; j++) {
                    float d = poincare_distance_c(node->inputs[0],
                                                  (const float*)(lm->C[j]),
                                                  LCM_D, 1.0f);
                    if (d < best) { best = d; best_idx = j; }
                }
                memcpy(result, lm->C[best_idx], sizeof(vec_t));
                if (li < LCM_MAX_LATTICES)
                    confidences[li] = 1.0f / (best + 1e-6f);
            }
            break;
        }

        case OP_SELF_CHECK: {
            /* Self-consistency: identity pass-through */
            memcpy(result, node->inputs[0], sizeof(vec_t));
            if (node->lattice_id < LCM_MAX_LATTICES)
                confidences[node->lattice_id] = 1.0f;
            break;
        }

        case OP_HRR_BIND:
            if (node->n_inputs >= 2) {
                hrr_bind(node->inputs, 1, &node->inputs[1], 1, result, LCM_D);
            }
            break;

        case OP_HRR_UNBIND:
            if (node->n_inputs >= 2) {
                hrr_unbind(node->inputs[0], node->inputs[1], result, LCM_D);
            }
            break;

        default:
            break;
        }

        /* Store output */
        if (node->output) {
            memcpy(node->output, result, sizeof(vec_t));
        }
        if (node->lattice_id < LCM_MAX_LATTICES) {
            memcpy(outputs[node->lattice_id], result, sizeof(vec_t));
        }
    }

    return true;
}

/* ─── Distance-weighted fusion (with optional gvalue bias) ──────────────────
 *
 * z_fused = Σ_i w_i * o_i / Σ_i w_i
 *
 * where w_i = 1/(d_i + ε) * exp(beta_val * value_signal_i)
 * and value_signal_i = proximity to positive vs negative anchors.
 *
 * When gv is NULL or beta_val <= 0, falls back to pure distance weighting.
 *
 * CONTRACT:
 *   Pre:  outputs, confidences, z_out, weights are valid pointers
 *   Post: z_out contains the fused vector
 *         weights are normalized to sum to 1
 */

/*@
  requires \valid_read(outputs + (0 .. LCM_MAX_LATTICES-1));
  requires \forall integer i; 0 <= i < LCM_MAX_LATTICES ==>
      \valid_read(outputs[i] + (0 .. LCM_D-1));
  requires \valid_read(confidences + (0 .. LCM_MAX_LATTICES-1));
  requires \valid(z_out + (0 .. LCM_D-1));
  requires \valid(weights + (0 .. LCM_MAX_LATTICES-1));
  requires \separated(z_out + (0 .. LCM_D-1), (float*)weights + (0 .. LCM_MAX_LATTICES-1));
  requires gv == \null || (\valid(gv) &&
      \valid_read(gv->C_pos + (0 .. LCM_N_VALUE_PAIRS-1)) &&
      \valid_read(gv->C_neg + (0 .. LCM_N_VALUE_PAIRS-1)));
  assigns z_out[0 .. LCM_D-1], weights[0 .. LCM_MAX_LATTICES-1];
 */
void distance_weighted_fusion(const vec_t* outputs, const float* confidences,
                               float* z_out, float weights[LCM_MAX_LATTICES],
                               const gvalue_t* gv, float beta_val) {
    float raw_weights[LCM_MAX_LATTICES];
    float w_sum = 0.0f;

    for (int i = 0; i < LCM_MAX_LATTICES; i++) {
        /* Inverse-distance weighting: w_i ∝ 1/(dist_i + ε) = confidences[i]
         * This creates a contractive mapping — closer codebook entries pull
         * z toward them, ensuring convergence to a fixed point. */
        raw_weights[i] = confidences[i];

        /* Value-biased modulation when gvalue is available */
        if (gv != NULL && beta_val > 0.0f) {
            /* Compute minimum distance to positive and negative anchors */
            float d_pos_min = 1e20f, d_neg_min = 1e20f;
            for (int law = 0; law < LCM_N_VALUE_PAIRS; law++) {
                float d_p = poincare_distance_c(
                    (const float*)outputs[i], (const float*)gv->C_pos[law],
                    LCM_D, 1.0f);
                float d_n = poincare_distance_c(
                    (const float*)outputs[i], (const float*)gv->C_neg[law],
                    LCM_D, 1.0f);
                if (d_p < d_pos_min) d_pos_min = d_p;
                if (d_n < d_neg_min) d_neg_min = d_n;
            }
            /* value_signal = softmax(-d_pos/tau) - softmax(-d_neg/tau) */
            float tau = 0.1f; /* matches tau_val_signal from config */
            float pos_w = expf(-d_pos_min / tau);
            float neg_w = expf(-d_neg_min / tau);
            float value_signal = (pos_w - neg_w) / (pos_w + neg_w + 1e-8f);
            /* Modulate: w *= exp(beta * signal) */
            raw_weights[i] *= expf(beta_val * value_signal);
        }

        w_sum += raw_weights[i];
    }

    /* Normalize weights */
    if (w_sum > 0) {
        for (int i = 0; i < LCM_MAX_LATTICES; i++) weights[i] = raw_weights[i] / w_sum;
    } else {
        for (int i = 0; i < LCM_MAX_LATTICES; i++) weights[i] = 0.0f;
    }

    /* Weighted fusion */
    memset(z_out, 0, sizeof(vec_t));
    for (int i = 0; i < LCM_MAX_LATTICES; i++) {
        for (int j = 0; j < LCM_D; j++) {
            z_out[j] += weights[i] * outputs[i][j];
        }
    }
}

/* ─── Trace saving (c.md §4.1/§5.2) ────────────────────────────────────────
 *
 * Saves a step trace entry into the inference trace ring buffer.
 * If the buffer is full, overwrites the oldest entry.
 *
 * CONTRACT:
 *   Pre:  trace, step are valid pointers
 *   Post: step is appended to trace, n_steps updated
 */
void save_trace(inference_trace_t* trace, const step_trace_t* step,
                const char* session_id) {
    if (!trace || !step) return;

    int idx = trace->n_steps % LCM_MAX_TRACE_STEPS;
    memcpy(&trace->steps[idx], step, sizeof(step_trace_t));
    if (trace->n_steps < LCM_MAX_TRACE_STEPS) {
        trace->n_steps++;
    }
    if (session_id) {
        snprintf(trace->session_id, sizeof(trace->session_id), "%s", session_id);
    }
}

/* ─── Save trace to file (JSON format for analysis) ──────────────────────────
 *
 * CONTRACT:
 *   Pre:  trace is a valid filled trace, filepath is writable
 *   Post: JSON file is written, returns 0 on success, -1 on error
 */
int save_trace_to_file(const inference_trace_t* trace, const char* filepath) {
    if (!trace || !filepath) return -1;

    FILE* f = fopen(filepath, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"session_id\": \"%s\",\n", trace->session_id);
    fprintf(f, "  \"n_steps\": %d,\n", trace->n_steps);
    fprintf(f, "  \"steps\": [\n");

    for (int s = 0; s < trace->n_steps; s++) {
        const step_trace_t* st = &trace->steps[s];
        fprintf(f, "    {\n");
        fprintf(f, "      \"step\": %d,\n", st->step);
        fprintf(f, "      \"has_conflict\": %s,\n",
                st->has_conflict ? "true" : "false");
        fprintf(f, "      \"fusion_weights\": [");
        for (int i = 0; i < LCM_MAX_LATTICES; i++) {
            fprintf(f, "%.4f%s", st->fusion_weights[i],
                    i < LCM_MAX_LATTICES - 1 ? ", " : "");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", s < trace->n_steps - 1 ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}

/* ─── Unified conflict detection ───────────────────────────────────────────
 *
 * Single entry point for all conflict sources.
 * Any source triggers → returns true with conflict populated.
 * No priority ordering — all conflicts equally fatal.
 *
 * CONTRACT:
 *   Pre:  All pointers valid, safety_margin >= 0
 *   Post: Returns true iff a conflict is detected
 *         out is populated with source, type, detail, step, timestamp
 */

/*@
  requires \valid_read(z_next + (0 .. LCM_D-1));
  requires z_cur == \null || \valid_read(z_cur + (0 .. LCM_D-1));
  requires dl == \null || \valid_read(dl);
  requires gv == \null || \valid_read(gv);
  requires retrieval_counts == \null || \valid_read(retrieval_counts + (0 .. 0));
  requires \valid(out);
  requires safety_margin >= 0.0f;
  assigns *out;
 */
bool detect_any_conflict(const float* z_next, const float* z_cur, int step,
                          const danger_lattice_t* dl, const gvalue_t* gv,
                          const int* retrieval_counts, float value_consistency,
                          float safety_margin, int max_steps, conflict_t* out) {
    (void)z_cur;
    /* 1. Danger lattice pattern match (optional — NULL = no danger check) */
    if (dl) {
        float danger_score; int threat_type; bool should_block;
        int rc = retrieval_counts ? retrieval_counts[0] : 0;  /* counts are optional */
        danger_assess(dl, z_next, step, rc, max_steps, value_consistency,
                      &danger_score, &threat_type, &should_block);
        if (should_block) {
            out->source = CONFLICT_DANGER;
            out->type = (threat_type_t)threat_type;
            snprintf(out->detail, sizeof(out->detail), "danger_score=%.3f", danger_score);
            out->step = step;
            out->timestamp = (double)time(NULL);
            return true;
        }
    }

    /* 2. Global value (three laws violation) — optional, skip if NULL */
    if (gv) {
        int violated_law;
        bool safe = gvalue_check_safety(gv, z_next, safety_margin, &violated_law);
        if (!safe) {
            out->source = CONFLICT_GVALUE;
            out->type = THREAT_THREE_LAWS;
            snprintf(out->detail, sizeof(out->detail),
                     "relative_margin_violation (margin=%.2f)", safety_margin);
            out->step = step;
            out->timestamp = (double)time(NULL);
            return true;
        }
    }

    /* 3. Internal consistency (value scalar vs output direction) */
    if (value_consistency < LCM_CONSISTENCY_THRESHOLD) {
        out->source = CONFLICT_CONSISTENCY;
        out->type = THREAT_DECEPTION;
        snprintf(out->detail, sizeof(out->detail),
                 "consistency=%.3f", value_consistency);
        out->step = step;
        out->timestamp = (double)time(NULL);
        return true;
    }

    return false;
}

/* ─── Hard interrupt + user-visible alert ──────────────────────────────────
 *
 * Post: Alert is written to callback
 *       No recovery attempt is made
 */
void halt_and_alert(const conflict_t* conflict, void (*alert_cb)(const alert_t*),
                     const char* session_id, int step) {
    alert_t alert;
    memset(&alert, 0, sizeof(alert));

    snprintf(alert.level, sizeof(alert.level), "FATAL");
    if (session_id) snprintf(alert.session_id, sizeof(alert.session_id), "%s", session_id);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] 推理会话 %s 在第 %d 步被中断。\n"
             "  检测来源: %d\n  冲突类型: %d\n  详细信息: %s\n"
             "  系统已停止当前推理，未尝试绕过或自修复。\n"
             "  完整推理轨迹已保存，请操作员审查。",
             session_id ? session_id : "?", step,
             conflict->source, conflict->type, conflict->detail);

    /* Persist alert log (in production: write to audit trail) */
    alert.step = step;
    alert.timestamp = (double)time(NULL);

    /* Notify via callback */
    if (alert_cb) alert_cb(&alert);
}

/* ─── Main inference loop ──────────────────────────────────────────────────
 *
 * CONTRACT:
 *   Pre:  state is initialized, z_initial is a valid input vector
 *         alert_cb may be NULL (no notification)
 *   Post: Returns 0 on normal convergence, -1 on conflict abort
 *         state->z_current contains final state
 *         On conflict: state->last_alert is populated
 *
 * INVARIANT: The engine itself is PURE — all state is in inference_state_t.
 *            No global/static variables are accessed or modified.
 */

/*@
  requires \valid(state);
  requires \valid_read(z_initial + (0 .. LCM_D-1));
  requires \separated(state, z_initial + (0 .. LCM_D-1));
  requires state->max_steps > 0 && state->max_steps <= 64;
  requires \valid(state->z_current + (0 .. LCM_D-1));
  requires \valid(state->z_next + (0 .. LCM_D-1));
  requires \valid(state->weights + (0 .. LCM_MAX_LATTICES-1));
  requires state->memory == \null || \valid(state->memory);
  requires state->gvalue == \null || \valid(state->gvalue);
  requires state->danger == \null || \valid(state->danger);
  assigns state->z_current[0 .. LCM_D-1], state->z_next[0 .. LCM_D-1],
          state->weights[0 .. LCM_MAX_LATTICES-1], state->step,
          state->dag, state->last_alert, state->trace;
  behavior converges:
    ensures \result == 0;
    ensures \forall integer i; 0 <= i < LCM_D ==> \is_finite(state->z_current[i]);
  behavior aborts:
    ensures \result == -1;
 */
int dynamic_inference(inference_state_t* state, const float* z_initial,
                       void (*alert_cb)(const alert_t*)) {
    if (!state || !z_initial) return -1;

    memcpy(state->z_current, z_initial, sizeof(vec_t));
    state->step = 0;

    for (state->step = 0; state->step < state->max_steps; state->step++) {
        /* 1. Build DAG for current step */
        dag_t dag = build_dag(state->z_current, state->memory, true);

        /* 2. Execute DAG */
        vec_t outputs[LCM_MAX_LATTICES];
        float confidences[LCM_MAX_LATTICES];
        memset(outputs, 0, sizeof(outputs));
        /* Absent lattices start with zero confidence so they do not mix
         * zero vectors into the fusion; executing nodes write their own
         * confidence (self → 1.0, retrieve → 1/(dist+eps)). */
        for (int i = 0; i < LCM_MAX_LATTICES; i++) confidences[i] = 0.0f;

        execute_dag(&dag, state->memory, outputs, confidences);

        /* Count retrievals actually performed this step (per-lattice,
         * reset each step: resource-abuse monitor is per-step). */
        memset(state->retrieval_counts, 0, sizeof(state->retrieval_counts));
        for (int ni = 0; ni < dag.n_nodes; ni++) {
            if (dag.nodes[ni].op_type == OP_RETRIEVE_SINGLE &&
                dag.nodes[ni].lattice_id < LCM_MAX_LATTICES)
                state->retrieval_counts[dag.nodes[ni].lattice_id]++;
        }

        /* 3. Distance-weighted fusion (with gvalue value bias) */
        vec_t z_next;
        float weights[LCM_MAX_LATTICES];
        distance_weighted_fusion((const vec_t*)outputs, confidences, z_next,
                                  weights, state->gvalue, 0.5f);

        memcpy(state->z_next, z_next, sizeof(vec_t));
        memcpy(state->weights, weights, sizeof(weights));

        /* 4. Conflict detection */
        conflict_t conflict;
        bool has_conflict = detect_any_conflict(
            state->z_next, state->z_current, state->step,
            state->danger, state->gvalue,
            state->retrieval_counts, state->value_consistency[0],
            LCM_DEFAULT_SAFETY_MARGIN, state->max_steps, &conflict);

        /* Save step trace */
        step_trace_t step_trace;
        memcpy(step_trace.outputs, outputs, sizeof(outputs));
        memcpy(step_trace.confidences, confidences, sizeof(confidences));
        memcpy(step_trace.fusion_weights, weights, sizeof(weights));
        memcpy(step_trace.z_next, state->z_next, sizeof(vec_t));
        step_trace.step = state->step;
        step_trace.has_conflict = has_conflict;
        step_trace.conflict = conflict;
        step_trace.timestamp = (double)time(NULL);
        save_trace(&state->trace, &step_trace, state->session_id);

        if (has_conflict) {
            halt_and_alert(&conflict, alert_cb, state->session_id, state->step);
            state->last_alert = (alert_t){
                .level = "FATAL",
                .step = state->step,
                .timestamp = (double)time(NULL),
            };
            return -1;  /* Abort — no recovery */
        }

        /* 5. Convergence check */
        float diff = 0.0f;
        /*@
          loop invariant 0 <= i <= LCM_D;
          loop invariant diff >= 0.0f;
          loop assigns i, diff;
          loop variant LCM_D - i;
         */
        for (int i = 0; i < LCM_D; i++) {
            float d = state->z_next[i] - state->z_current[i];
            diff += d * d;
        }
        /*@ assert diff >= 0.0f; */
        diff = sqrtf(diff);

        float weight_entropy = 0.0f;
        /*@
          loop invariant 0 <= i <= LCM_MAX_LATTICES;
          loop assigns i, weight_entropy;
          loop variant LCM_MAX_LATTICES - i;
         */
        for (int i = 0; i < LCM_MAX_LATTICES; i++) {
            float w = weights[i] + 1e-12f;
            /*@ assert w > 0.0f; */
            weight_entropy -= w * logf(w);
        }

        float c_tol = state->conv_tol > 0 ? state->conv_tol : LCM_CONVERGENCE_TOL;
        float e_thresh = state->entropy_thresh > 0 ? state->entropy_thresh : LCM_ENTROPY_THRESHOLD;
        if (diff < c_tol && weight_entropy < e_thresh) {
            memcpy(state->z_current, state->z_next, sizeof(vec_t));
            return 0;  /* Normal convergence */
        }

        memcpy(state->z_current, state->z_next, sizeof(vec_t));
    }

    /* Max steps exceeded — hard interrupt */
    conflict_t conflict = {
        .source = CONFLICT_SCHEDULER,
        .type = THREAT_MAX_STEPS,
        .step = state->step,
        .timestamp = (double)time(NULL),
    };
    snprintf(conflict.detail, sizeof(conflict.detail),
             "max_steps_exceeded (step=%d)", state->step);
    halt_and_alert(&conflict, alert_cb, state->session_id, state->step);

    state->last_alert = (alert_t){
        .level = "FATAL",
        .step = state->step,
        .timestamp = (double)time(NULL),
    };
    return -1;
}
