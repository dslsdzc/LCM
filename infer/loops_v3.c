/* LCM v3 dynamic DAG + macro loop + trace (R8/R9/R13, task 10).
 *
 * This file owns the schema-v2/v3 inference path:
 *   build_dag_v3   — dynamic threshold, HRQ→Sparse edge, bind/unbind wiring
 *   execute_dag_v3 — dispatch to the full per-lattice forwards
 *   lcm_fuse_canonical — shared with lcm_fuse_only (R9 canonical fusion)
 *   lcm_infer_loop_v3 — scheduler budget 16, hard limit 64, safety abort
 *   trace JSON + alert log
 *
 * No heap allocation, no JSON input, no global mutable state. The last trace
 * is thread-local for the export API, matching the v1 lcm_get_trace design.
 */
#define _POSIX_C_SOURCE 200809L

#include "lcm.h"
#include <errno.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define LCM_V3_VALUE_TOPK 4   /* hrq_value_topk default (R1) */
#define LCM_V3_BUCKET_EPS 1e-6f

/* ─── Small helpers ───────────────────────────────────────────────────────── */

static float v3_d2(const float* a, const float* b) {
    float d = 0.0f;
    for (int i = 0; i < LCM_D; i++) {
        float t = a[i] - b[i];
        d += t * t;
    }
    return d;
}

static float v3_euclid_min(const float* z, const float* cb, int M) {
    float best = INFINITY;
    for (int j = 0; j < M; j++) {
        float d2 = v3_d2(z, cb + (size_t)j * LCM_D);
        if (d2 < best) best = d2;
    }
    return sqrtf(best);
}

/* Manifold trigger is measured in the R6 domain: exp_map(safe_unit(·)). */
static float v3_manifold_min(const float* z, const manifold_mem_t* m) {
    vec_t zu, zp, cu, cp;
    safe_unit_c(z, zu, LCM_D, 1e-8f);
    exp_map_c(zu, zp, LCM_D, 1.0f);
    float best = INFINITY;
    for (int j = 0; j < m->M; j++) {
        const float* row = m->C + (size_t)j * LCM_D;
        safe_unit_c(row, cu, LCM_D, 1e-8f);
        exp_map_c(cu, cp, LCM_D, 1.0f);
        float sim = poincare_similarity_c(zp, cp, LCM_D, 1.0f);
        if (sim < best) best = sim;
    }
    return sqrtf(best);
}

/* LowRank's layer-0 codebook is U_0 @ V[:, :r_0]^T; compute rows on demand. */
static float v3_lowrank_min(const float* z, const lowrank_mem_t* m) {
    if (m->n_layers <= 0 || m->ranks == NULL) return INFINITY;
    int rank = m->ranks[0];
    if (rank <= 0 || rank > m->r_max) return INFINITY;
    float best = INFINITY;
    for (int row = 0; row < m->M_lr; row++) {
        const float* u = m->U + (size_t)row * (size_t)m->r_max;
        float d2 = 0.0f;
        for (int j = 0; j < LCM_D; j++) {
            const float* v = m->V + (size_t)j * (size_t)m->r_max;
            float acc = 0.0f;
            for (int k = 0; k < rank; k++) acc += u[k] * v[k];
            float t = z[j] - acc;
            d2 += t * t;
        }
        if (d2 < best) best = d2;
    }
    return sqrtf(best);
}

static float v3_contrast_min(const float* z, const contrast_mem_t* m) {
    float a = v3_euclid_min(z, m->C_a, m->M);
    float b = v3_euclid_min(z, m->C_b, m->M);
    return a < b ? a : b;
}

static float v3_self_min(const float* z, const fusion_mem_t* f) {
    int rows = f->M_self + 1;   /* row 0 is the frozen core */
    return v3_euclid_min(z, f->self_C, rows);
}

/* ─── build_dag_v3 (R8) ───────────────────────────────────────────────────── */

typedef struct {
    int   lattice_id;
    op_type_t op_type;
    float dist;
} v3_candidate_t;

static void trace_push(dag_t* dag, op_type_t op, int lattice_id, float dist,
                       int activated) {
    if (dag->n_trace_nodes >= LCM_MAX_NODES) return;
    dag_trace_node_t* t = &dag->trace_nodes[dag->n_trace_nodes++];
    t->op_type = (int)op;
    t->lattice_id = lattice_id;
    t->dist = dist;
    t->activated = activated;
}

static op_node_t* dag_push(dag_t* dag, op_type_t op, int lattice_id,
                           float dist, const float* in0) {
    if (dag->n_nodes >= LCM_MAX_NODES) return NULL;
    op_node_t* n = &dag->nodes[dag->n_nodes];
    memset(n, 0, sizeof(*n));
    n->op_type = op;
    n->lattice_id = lattice_id;
    n->n_inputs = 1;
    n->inputs[0] = in0;
    n->dist = dist;
    n->dep_dist = 0.0f;
    n->output = &dag->storage[dag->n_nodes];
    dag->n_nodes++;
    return n;
}

int build_dag_v3(const float* z, lcm_memory_v3* mem, dag_t* dag) {
    if (!z || !mem || !dag) return -1;
    if (mem->D != LCM_D || mem->n_lattices <= 0
        || mem->n_lattices > LCM_MAX_LATTICES) return -1;

    memset(dag, 0, sizeof(*dag));

    /* Layer 1: collect every candidate distance first. Never append to the
     * DAG while iterating it (the v1 implementation did). */
    v3_candidate_t cand[LCM_MAX_LATTICES];
    int n_cand = 0;

    for (int li = 0; li < mem->n_lattices; li++) {
        float dist;
        switch (li) {
        case LATTICE_HRQ:
            dist = v3_euclid_min(z, mem->hrq.C_top, mem->hrq.M_top);
            break;
        case LATTICE_SPARSE:
            dist = v3_euclid_min(z, mem->sparse.C, mem->sparse.M);
            break;
        case LATTICE_LOWRANK:
            dist = v3_lowrank_min(z, &mem->lowrank);
            break;
        case LATTICE_MANIFOLD:
            dist = v3_manifold_min(z, &mem->manifold);
            break;
        case LATTICE_BINDING:
            dist = v3_euclid_min(z, mem->binding.bind_cb,
                                 mem->binding.M);
            break;
        case LATTICE_CONTRAST:
            dist = v3_contrast_min(z, &mem->contrast);
            break;
        default:
            dist = INFINITY;
            break;
        }
        cand[n_cand].lattice_id = li;
        cand[n_cand].op_type = OP_RETRIEVE_SINGLE;
        cand[n_cand].dist = dist;
        n_cand++;
    }
    if (mem->fusion.M_self > 0 && mem->fusion.self_C) {
        cand[n_cand].lattice_id = LATTICE_SELF;
        cand[n_cand].op_type = OP_SELF_CHECK;
        cand[n_cand].dist = v3_self_min(z, &mem->fusion);
        n_cand++;
    }

    float sum = 0.0f;
    int n_finite = 0;
    for (int i = 0; i < n_cand; i++) {
        if (isfinite(cand[i].dist)) { sum += cand[i].dist; n_finite++; }
    }
    float mean_d = n_finite > 0 ? sum / (float)n_finite : 0.0f;
    /* R8's mean_recent_d_min is scheduler state. Keep one EMA here; a zero
     * value means "first step", so the first step is thresholded against the
     * current input's own mean distance. */
    float scale = mean_d;
    if (mem->mean_recent_d_min > 0.0f)
        scale = 0.7f * mem->mean_recent_d_min + 0.3f * mean_d;
    mem->mean_recent_d_min = scale;

    int active[LCM_MAX_LATTICES] = {0};
    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        float theta = mem->trigger_theta[li];
        float thr = theta > 0.0f ? theta * scale : 0.0f;
        active[li] = theta > 0.0f
            && (cand[i].dist < thr || cand[i].dist <= 1e-12f);
        trace_push(dag, cand[i].op_type, li, cand[i].dist, active[li]);
    }

    /* Layer 2: append in dependency order. HRQ first so Sparse can point at
     * its output; bind later consumes every routed retrieve node. */
    op_node_t* retrieve[LCM_MAX_LATTICES];
    int n_retrieve = 0;
    op_node_t* hrq_node = NULL;
    op_node_t* bind_node = NULL;

    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        if (!active[li]) continue;
        if (li == LATTICE_SELF) {
            op_node_t* node = dag_push(dag, OP_SELF_CHECK, li,
                                       cand[i].dist, z);
            if (node) {
                node->dep_dist = 0.0f;
                node->inputs[1] = NULL;
            }
            continue;
        }
        op_node_t* node = dag_push(dag, OP_RETRIEVE_SINGLE, li,
                                   cand[i].dist, z);
        if (!node) continue;

        if (li == LATTICE_HRQ) {
            hrq_node = node;
        } else if (li == LATTICE_SPARSE) {
            /* C-5: the LFQ gate is HRQ's d_top. Inactive HRQ ⇒ 0, which
             * disables LFQ inside sparse_forward_c. */
            node->n_inputs = 2;
            node->inputs[1] = hrq_node ? (const float*)hrq_node->output : NULL;
            node->dep_dist = hrq_node
                ? (hrq_node->dist > 0.0f ? hrq_node->dist : LCM_V3_BUCKET_EPS)
                : 0.0f;
        }

        if (li < mem->n_lattices && n_retrieve < LCM_MAX_LATTICES)
            retrieve[n_retrieve++] = node;
    }

    /* Bind depends on ALL upstream retrieve nodes (R8). */
    if (n_retrieve > 0 && dag->n_nodes < LCM_MAX_NODES) {
        bind_node = dag_push(dag, OP_HRR_BIND, LCM_MAX_LATTICES, 0.0f, z);
        if (bind_node) {
            bind_node->n_inputs = n_retrieve;
            for (int i = 0; i < n_retrieve; i++)
                bind_node->inputs[i] = (const float*)retrieve[i]->output;
            trace_push(dag, OP_HRR_BIND, LCM_MAX_LATTICES, 0.0f, 1);
        }
    }

    /* Unbind: inputs[0] = bind output, inputs[1] = key projection. */
    if (bind_node && dag->n_nodes < LCM_MAX_NODES) {
        op_node_t* node = dag_push(dag, OP_HRR_UNBIND, LCM_MAX_LATTICES,
                                   0.0f, z);
        if (node) {
            node->n_inputs = 2;
            node->inputs[0] = (const float*)bind_node->output;
            node->inputs[1] = (const float*)retrieve[0]->output;
            trace_push(dag, OP_HRR_UNBIND, LCM_MAX_LATTICES, 0.0f, 1);
        }
    }

    return n_cand;
}

/* ─── execute_dag_v3 ──────────────────────────────────────────────────────── */

static void v3_self_output(const float* z, const fusion_mem_t* f, float* out) {
    int rows = f->M_self + 1;
    int best = 0;
    float best_d2 = INFINITY;
    for (int r = 0; r < rows; r++) {
        float d2 = v3_d2(z, f->self_C + (size_t)r * LCM_D);
        if (d2 < best_d2) { best_d2 = d2; best = r; }
    }
    memcpy(out, f->self_C + (size_t)best * LCM_D, sizeof(vec_t));
}

bool execute_dag_v3(dag_t* dag, const lcm_memory_v3* mem, float alpha_val,
                    vec_t* outputs, float* confidences, int* active) {
    if (!dag || !mem || !outputs || !confidences) return false;

    for (int i = 0; i < LCM_MAX_LATTICES; i++) {
        if (active) active[i] = 0;
    }

    for (int ni = 0; ni < dag->n_nodes; ni++) {
        op_node_t* node = &dag->nodes[ni];
        vec_t result;
        memset(result, 0, sizeof(result));

        if (node->op_type == OP_HRR_BIND) {
            if (node->n_inputs >= 2) {
                hrr_bind(node->inputs, node->n_inputs,
                         node->inputs, node->n_inputs, result, LCM_D);
            }
        } else if (node->op_type == OP_HRR_UNBIND) {
            if (node->n_inputs >= 2)
                hrr_unbind(node->inputs[0], node->inputs[1], result, LCM_D);
        } else if (node->op_type == OP_SELF_CHECK
                   && node->lattice_id == LATTICE_SELF) {
            v3_self_output(node->inputs[0], &mem->fusion, result);
            confidences[LATTICE_SELF] =
                1.0f / (sqrtf(v3_d2(node->inputs[0], result))
                        + LCM_V3_BUCKET_EPS);
            if (active) active[LATTICE_SELF] = 1;
            memcpy(outputs[LATTICE_SELF], result, sizeof(vec_t));
        } else if (node->op_type == OP_RETRIEVE_SINGLE) {
            int li = node->lattice_id;
            if (li < 0 || li >= mem->n_lattices) continue;
            switch (li) {
            case LATTICE_HRQ:
                hrq_forward_v31(node->inputs[0], &mem->hrq, alpha_val,
                                LCM_V3_VALUE_TOPK, mem->tau_route_fallback,
                                result, NULL, NULL);
                break;
            case LATTICE_SPARSE:
                sparse_forward_c(node->inputs[0], &mem->sparse,
                                 mem->lambda_lfq, alpha_val,
                                 node->dep_dist, 0, result, NULL);
                break;
            case LATTICE_LOWRANK:
                lowrank_forward_c(node->inputs[0], &mem->lowrank, alpha_val,
                                  result, NULL);
                break;
            case LATTICE_MANIFOLD:
                manifold_forward_c(node->inputs[0], &mem->manifold,
                                   alpha_val, result, NULL);
                break;
            case LATTICE_BINDING:
                binding_forward_c(node->inputs[0], &mem->binding,
                                  mem->lowrank.V, alpha_val, result,
                                  NULL, NULL);
                break;
            case LATTICE_CONTRAST:
                contrast_forward_c(node->inputs[0], &mem->contrast,
                                   alpha_val, result);
                break;
            default:
                break;
            }
            confidences[li] = 1.0f
                / (sqrtf(node->dist) + LCM_V3_BUCKET_EPS);
            if (active) active[li] = 1;
            memcpy(outputs[li], result, sizeof(vec_t));
        }

        if (node->output) memcpy(node->output, result, sizeof(vec_t));
    }
    return true;
}

/* ─── Canonical fusion (R9/R15) ───────────────────────────────────────────── */

int lcm_fuse_canonical(const float* outputs, int n_lattices, int d,
                       const float* soft_mask, const int* active,
                       const float* alpha, const float* ln_scale,
                       const float* ln_bias, const float* self_out,
                       float self_bias, float* z_out) {
    if (!outputs || !z_out || d <= 0) return -1;
    if (n_lattices <= 0 || n_lattices > LCM_MAX_LATTICES) return -1;
    if (!alpha || !ln_scale || !ln_bias) return -1;

    float w[LCM_MAX_LATTICES];
    int n_active = 0;
    float w_sum = 0.0f;
    for (int i = 0; i < n_lattices; i++) {
        int on = active ? active[i] : 1;
        w[i] = on ? ((soft_mask ? soft_mask[i] : 1.0f) * alpha[i]) : 0.0f;
        if (on) n_active++;
        w_sum += w[i];
    }
    if (w_sum == 0.0f) {
        if (n_active == 0) return -1;
        /* Routing chose an inactive lattice (hard mask): use uniform
         * weights over the active set rather than dropping the step. */
        for (int i = 0; i < n_lattices; i++) {
            if (!active || active[i]) w[i] = 1.0f / (float)n_active;
        }
        w_sum = 1.0f;
    }

    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < n_lattices; i++) {
            if (w[i] == 0.0f) continue;
            acc += (w[i] / w_sum) * outputs[(size_t)i * (size_t)d + (size_t)j];
        }
        if (self_out) acc += self_bias * self_out[j];
        z_out[j] = acc;
    }

    /* LayerNorm, eps matching train/fusion.py::layer_norm (1e-6). */
    float mean = 0.0f;
    for (int j = 0; j < d; j++) mean += z_out[j];
    mean /= (float)d;

    float var = 0.0f;
    for (int j = 0; j < d; j++) {
        float t = z_out[j] - mean;
        var += t * t;
    }
    var /= (float)d;

    float inv_std = 1.0f / sqrtf(var + 1e-6f);
    for (int j = 0; j < d; j++)
        z_out[j] = (z_out[j] - mean) * inv_std * ln_scale[j] + ln_bias[j];

    return 0;
}

/* ─── Value consistency (R13) ─────────────────────────────────────────────── *
 * value_consistency = cosine(Σ_j tanh(v_j)·c_j, z_next), where j ranges over
 * the active lattices' selected rows. The selected row is the nearest row of
 * the lattice's primary codebook to z_cur. If no value vector is supplied, or
 * the accumulated direction is zero, consistency is defined as 1.0 (no
 * evidence of deception) so uncalibrated checkpoints do not block inference.
 */
static int v3_nearest_value(const float* z, const float* cb, int M,
                            const float* value, float* out_v) {
    if (!cb || !value || M <= 0) return 0;
    int best = 0;
    float best_d2 = INFINITY;
    for (int j = 0; j < M; j++) {
        float d2 = v3_d2(z, cb + (size_t)j * LCM_D);
        if (d2 < best_d2) { best_d2 = d2; best = j; }
    }
    *out_v = value[best];
    return 1;
}

static float v3_value_consistency(const lcm_memory_v3* mem,
                                  const vec_t* outputs, const int* active,
                                  const float* z_cur, const float* z_next) {
    vec_t acc;
    memset(acc, 0, sizeof(acc));
    int used = 0;

    for (int li = 0; li < mem->n_lattices; li++) {
        if (!active || !active[li]) continue;
        float v;
        int ok = 0;
        switch (li) {
        case LATTICE_HRQ:
            ok = v3_nearest_value(z_cur, mem->hrq.C_top, mem->hrq.M_top,
                                  mem->hrq.value_top, &v);
            break;
        case LATTICE_SPARSE:
            ok = v3_nearest_value(z_cur, mem->sparse.C, mem->sparse.M,
                                  mem->sparse.value, &v);
            break;
        case LATTICE_LOWRANK: {
            /* Layer-0 codebook rows are U_0 @ V[:, :r_0]^T. */
            const lowrank_mem_t* lm = &mem->lowrank;
            int rank = lm->ranks[0];
            int best = 0;
            float best_d2 = INFINITY;
            for (int row = 0; row < lm->M_lr; row++) {
                const float* u = lm->U + (size_t)row * (size_t)lm->r_max;
                float d2 = 0.0f;
                for (int j = 0; j < LCM_D; j++) {
                    const float* vv = lm->V
                        + (size_t)j * (size_t)lm->r_max;
                    float acc = 0.0f;
                    for (int k = 0; k < rank; k++) acc += u[k] * vv[k];
                    float t = z_cur[j] - acc;
                    d2 += t * t;
                }
                if (d2 < best_d2) { best_d2 = d2; best = row; }
            }
            ok = lm->value != NULL && lm->M_lr > 0;
            v = ok ? lm->value[best] : 0.0f;
            break;
        }
        case LATTICE_MANIFOLD:
            ok = v3_nearest_value(z_cur, mem->manifold.C, mem->manifold.M,
                                  mem->manifold.value, &v);
            break;
        case LATTICE_BINDING:
            ok = v3_nearest_value(z_cur, mem->binding.val_cb, mem->binding.M,
                                  mem->binding.value_val, &v);
            break;
        case LATTICE_CONTRAST:
            ok = v3_nearest_value(z_cur, mem->contrast.C_a, mem->contrast.M,
                                  mem->contrast.value_a, &v);
            break;
        default:
            ok = 0;
            break;
        }
        if (!ok) continue;
        float g = tanhf(v);
        for (int j = 0; j < LCM_D; j++) acc[j] += g * outputs[li][j];
        used = 1;
    }

    if (!used) return 1.0f;
    float n_acc = 0.0f, n_next = 0.0f, dot = 0.0f;
    for (int j = 0; j < LCM_D; j++) {
        n_acc += acc[j] * acc[j];
        n_next += z_next[j] * z_next[j];
        dot += acc[j] * z_next[j];
    }
    if (n_acc <= 1e-16f || n_next <= 1e-16f) return 1.0f;
    float c = dot / (sqrtf(n_acc) * sqrtf(n_next));
    if (c > 1.0f) c = 1.0f;
    if (c < -1.0f) c = -1.0f;
    return c;
}

/* ─── Trace export / alert log ────────────────────────────────────────────── */

static __thread inference_trace_v3_t _last_trace_v3;

int lcm_get_trace_v3(inference_trace_v3_t* out) {
    if (!out) return _last_trace_v3.n_steps;
    *out = _last_trace_v3;
    return out->n_steps;
}

static const char* v3_op_name(int op) {
    switch ((op_type_t)op) {
    case OP_RETRIEVE_SINGLE:   return "retrieve_single";
    case OP_SLIDE_MANIFOLD:    return "slide_manifold";
    case OP_HRR_BIND:          return "hrr_bind";
    case OP_HRR_UNBIND:        return "hrr_unbind";
    case OP_DISTANCE_FUSION:   return "distance_fusion";
    case OP_CONTRAST_RETRIEVE: return "contrast_retrieve";
    case OP_SELF_CHECK:        return "self_check";
    default:                   return "unknown";
    }
}

static const char* v3_conflict_source_name(conflict_source_t s) {
    switch (s) {
    case CONFLICT_DANGER:      return "CONFLICT_DANGER";
    case CONFLICT_GVALUE:      return "CONFLICT_GVALUE";
    case CONFLICT_CONSISTENCY: return "CONFLICT_CONSISTENCY";
    case CONFLICT_SCHEDULER:   return "CONFLICT_SCHEDULER";
    default:                   return "CONFLICT_NONE";
    }
}

static const char* v3_threat_name(threat_type_t t) {
    switch (t) {
    case THREAT_PATTERN_MATCH:  return "THREAT_PATTERN_MATCH";
    case THREAT_RESOURCE_ABUSE: return "THREAT_RESOURCE_ABUSE";
    case THREAT_RUNAWAY:        return "THREAT_RUNAWAY";
    case THREAT_DECEPTION:      return "THREAT_DECEPTION";
    case THREAT_THREE_LAWS:     return "THREAT_THREE_LAWS";
    case THREAT_MAX_STEPS:      return "THREAT_MAX_STEPS";
    default:                    return "THREAT_NONE";
    }
}

/* Create every parent directory in `path`; failures are non-fatal. */
static void v3_ensure_parent(const char* path) {
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);
    for (char* p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(buf, 0777) != 0 && errno != EEXIST) {
            /* keep walking: a deeper mkdir may still give a useful error */
        }
        *p = '/';
    }
}

int save_trace_v3_to_file(const inference_trace_v3_t* trace,
                          const char* filepath) {
    if (!trace || !filepath || !filepath[0]) return -1;
    v3_ensure_parent(filepath);
    FILE* f = fopen(filepath, "w");
    if (!f) return -1;

    fprintf(f, "{\n  \"session_id\": \"%s\",\n  \"n_steps\": %d,\n"
               "  \"steps\": [\n", trace->session_id, trace->n_steps);
    for (int s = 0; s < trace->n_steps; s++) {
        const step_trace_v3_t* st = &trace->steps[s];
        fprintf(f, "    {\n      \"step\": %d,\n"
                   "      \"has_conflict\": %s,\n      \"nodes\": [\n",
                st->step, st->has_conflict ? "true" : "false");
        for (int n = 0; n < st->n_nodes; n++) {
            const dag_trace_node_t* nd = &st->nodes[n];
            fprintf(f,
                "        {\"op_type\": \"%s\", \"lattice_id\": %d, "
                "\"dist\": %.6f, \"activated\": %s}%s\n",
                v3_op_name(nd->op_type), nd->lattice_id, nd->dist,
                nd->activated ? "true" : "false",
                n + 1 < st->n_nodes ? "," : "");
        }
        fprintf(f, "      ],\n      \"fusion_weights\": [");
        for (int i = 0; i < LCM_MAX_LATTICES; i++)
            fprintf(f, "%.6f%s", st->weights[i],
                    i + 1 < LCM_MAX_LATTICES ? ", " : "");
        fprintf(f, "],\n      \"confidences\": [");
        for (int i = 0; i < LCM_MAX_LATTICES; i++)
            fprintf(f, "%.6f%s", st->confidences[i],
                    i + 1 < LCM_MAX_LATTICES ? ", " : "");
        fprintf(f, "],\n      \"z_next\": [");
        for (int j = 0; j < LCM_D; j++)
            fprintf(f, "%.6f%s", st->z_next[j],
                    j + 1 < LCM_D ? ", " : "");
        fprintf(f, "],\n      \"conflict\": {\"source\": \"%s\", "
                   "\"type\": \"%s\", \"detail\": \"%s\"}\n    }%s\n",
                v3_conflict_source_name(st->conflict.source),
                v3_threat_name(st->conflict.type),
                st->conflict.detail,
                s + 1 < trace->n_steps ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    return 0;
}

static void v3_default_trace_path(char* out, size_t cap) {
    char cwd[512];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        const char* base = strrchr(cwd, '/');
        if (base && strcmp(base + 1, "infer") == 0) {
            snprintf(out, cap, "traces/lcm_v3_%ld.json",
                     (long)time(NULL));
            return;
        }
    }
    snprintf(out, cap, "infer/traces/lcm_v3_%ld.json", (long)time(NULL));
}

static void v3_alert_path(char* out, size_t cap) {
    char cwd[512];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        const char* base = strrchr(cwd, '/');
        if (base && strcmp(base + 1, "infer") == 0) {
            snprintf(out, cap, "alerts.log");
            return;
        }
    }
    snprintf(out, cap, "infer/alerts.log");
}

static void v3_write_alert(const conflict_t* conflict, int step) {
    alert_t alert;
    memset(&alert, 0, sizeof(alert));
    snprintf(alert.level, sizeof(alert.level), "FATAL");
    snprintf(alert.session_id, sizeof(alert.session_id), "lcm_v3");
    snprintf(alert.conflict_source, sizeof(alert.conflict_source), "%s",
             v3_conflict_source_name(conflict->source));
    snprintf(alert.conflict_type, sizeof(alert.conflict_type), "%s",
             v3_threat_name(conflict->type));
    snprintf(alert.conflict_detail, sizeof(alert.conflict_detail), "%s",
             conflict->detail);
    alert.step = step;
    alert.timestamp = (double)time(NULL);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] v3 inference aborted at step %d.\n"
             "  Source: %s\n  Type: %s\n  Detail: %s\n"
             "  System halted without bypass or self-repair. "
             "Trace saved for operator review.",
             step, alert.conflict_source, alert.conflict_type,
             alert.conflict_detail);

    char path[512];
    v3_alert_path(path, sizeof(path));
    v3_ensure_parent(path);
    FILE* f = fopen(path, "a");
    if (!f && strcmp(path, "infer/alerts.log") == 0) {
        v3_ensure_parent("alerts.log");
        f = fopen("alerts.log", "a");
    }
    if (!f) return;
    fprintf(f, "%ld,%s,%s,%s,%s,%d,%s\n",
            (long)alert.timestamp, alert.level, alert.session_id,
            alert.conflict_source, alert.conflict_type, alert.step,
            alert.conflict_detail);
    fclose(f);
}

/* ─── Macro loop (R9) ─────────────────────────────────────────────────────── */

int lcm_infer_loop_v3(const lcm_memory_v3* mem, const gvalue_t* gv,
                      const danger_lattice_t* dl, const float* z_initial,
                      float* z_out, int max_steps, float conv_tol,
                      float entropy_thresh, float safety_margin,
                      float alpha_val, const char* trace_path) {
    if (!mem || !z_initial || !z_out) return -1;
    if (mem->D != LCM_D || mem->n_lattices <= 0
        || mem->n_lattices > LCM_MAX_LATTICES) return -1;

    /* Private shallow copy: dynamic trigger state is per-session. */
    lcm_memory_v3 work = *mem;

    int budget = max_steps > 0 ? max_steps : LCM_MAX_INFERENCE_STEPS;
    int hard = LCM_MAX_INFERENCE_STEPS_HARD;
    int limit = budget < hard ? budget : hard;

    inference_trace_v3_t trace;
    memset(&trace, 0, sizeof(trace));
    snprintf(trace.session_id, sizeof(trace.session_id), "lcm_v3");

    vec_t z_cur;
    memcpy(z_cur, z_initial, sizeof(vec_t));

    for (int step = 0; step < limit; step++) {
        dag_t dag;
        int n_cand = build_dag_v3(z_cur, &work, &dag);
        if (n_cand < 0) return -1;

        vec_t outputs[LCM_MAX_LATTICES];
        float confidences[LCM_MAX_LATTICES];
        int active[LCM_MAX_LATTICES];
        memset(outputs, 0, sizeof(outputs));
        memset(confidences, 0, sizeof(confidences));
        memset(active, 0, sizeof(active));
        execute_dag_v3(&dag, &work, alpha_val, outputs, confidences, active);

        /* Routing gate: deterministic hard mask over the routed lattices. */
        float soft_mask[LCM_MAX_LATTICES];
        memset(soft_mask, 0, sizeof(soft_mask));
        routing_gate_c(z_cur, work.fusion.route_C, work.fusion.route_W,
                       work.n_lattices, soft_mask);

        float weights[LCM_MAX_LATTICES];
        memset(weights, 0, sizeof(weights));
        int n_active = 0;
        for (int i = 0; i < work.n_lattices; i++) {
            weights[i] = active[i] ? soft_mask[i] * work.fusion.alpha[i]
                                   : 0.0f;
            if (active[i]) n_active++;
        }

        float self_bias = 0.0f;
        const float* self_out = NULL;
        if (active[LATTICE_SELF]) {
            self_out = outputs[LATTICE_SELF];
            float dot = 0.0f, nz = 0.0f, ns = 0.0f;
            for (int j = 0; j < LCM_D; j++) {
                dot += z_cur[j] * self_out[j];
                nz += z_cur[j] * z_cur[j];
                ns += self_out[j] * self_out[j];
            }
            float cos = (nz > 1e-16f && ns > 1e-16f)
                ? dot / (sqrtf(nz) * sqrtf(ns)) : 0.0f;
            if (cos > 1.0f) cos = 1.0f;
            if (cos < 0.0f) cos = 0.0f;
            self_bias = work.fusion.alpha_self * cos;
        }

        vec_t z_next;
        if (lcm_fuse_canonical((const float*)outputs, work.n_lattices,
                               LCM_D, soft_mask, active, work.fusion.alpha,
                               work.fusion.ln_scale, work.fusion.ln_bias,
                               self_out, self_bias, z_next) != 0) {
            /* No active routed node: identity step is the only sane
             * continuation; the entropy gate then converges immediately. */
            memcpy(z_next, z_cur, sizeof(vec_t));
            memset(weights, 0, sizeof(weights));
            n_active = 0;
        }

        /* Normalise weights for entropy/trace. */
        float w_sum = 0.0f;
        for (int i = 0; i < LCM_MAX_LATTICES; i++) w_sum += weights[i];
        if (w_sum > 0.0f) {
            for (int i = 0; i < LCM_MAX_LATTICES; i++) weights[i] /= w_sum;
        } else if (n_active > 0) {
            for (int i = 0; i < LCM_MAX_LATTICES; i++)
                if (active[i]) weights[i] = 1.0f / (float)n_active;
        }

        int retrieval_total = 0;
        for (int ni = 0; ni < dag.n_nodes; ni++) {
            const op_node_t* nd = &dag.nodes[ni];
            if (nd->op_type == OP_RETRIEVE_SINGLE
                && nd->lattice_id >= 0
                && nd->lattice_id < work.n_lattices)
                retrieval_total++;
        }

        float value_consistency = v3_value_consistency(
            &work, (const vec_t*)outputs, active, z_cur, z_next);

        conflict_t conflict;
        memset(&conflict, 0, sizeof(conflict));
        int rc_total = retrieval_total;
        bool has_conflict = detect_any_conflict(
            z_next, z_cur, step, dl, gv, &rc_total, value_consistency,
            safety_margin, hard, &conflict);

        step_trace_v3_t* st = &trace.steps[step];
        for (int n = 0; n < dag.n_trace_nodes && n < LCM_MAX_NODES; n++)
            st->nodes[n] = dag.trace_nodes[n];
        st->n_nodes = dag.n_trace_nodes;
        memcpy(st->weights, weights, sizeof(weights));
        memcpy(st->confidences, confidences, sizeof(confidences));
        memcpy(st->z_next, z_next, sizeof(vec_t));
        st->step = step;
        st->has_conflict = has_conflict;
        st->conflict = conflict;
        st->timestamp = (double)time(NULL);
        trace.n_steps = step + 1;

        if (has_conflict) {
            _last_trace_v3 = trace;
            char path[512];
            if (trace_path && trace_path[0]) {
                snprintf(path, sizeof(path), "%s", trace_path);
            } else {
                v3_default_trace_path(path, sizeof(path));
            }
            save_trace_v3_to_file(&trace, path);
            v3_write_alert(&conflict, step);
            return CONFLICT_ABORT_TOKEN;   /* z_out intentionally not written */
        }

        float diff2 = 0.0f;
        for (int j = 0; j < LCM_D; j++) {
            float d = z_next[j] - z_cur[j];
            diff2 += d * d;
        }
        float diff = sqrtf(diff2);

        float entropy = 0.0f;
        for (int i = 0; i < LCM_MAX_LATTICES; i++) {
            if (weights[i] > 0.0f)
                entropy -= weights[i] * logf(weights[i]);
        }
        float h_norm = 0.0f;
        if (n_active > 1)
            h_norm = entropy / logf((float)n_active);

        memcpy(z_cur, z_next, sizeof(vec_t));

        if (diff < conv_tol && h_norm < entropy_thresh) {
            memcpy(z_out, z_next, sizeof(vec_t));
            _last_trace_v3 = trace;
            if (trace_path && trace_path[0])
                save_trace_v3_to_file(&trace, trace_path);
            return 0;
        }
    }

    _last_trace_v3 = trace;

    if (limit >= hard) {
        /* Safety hard limit (R9): the graceful 16-step budget is not an
         * abort; only the 64-step hard limit is. */
        conflict_t conflict;
        memset(&conflict, 0, sizeof(conflict));
        conflict.source = CONFLICT_SCHEDULER;
        conflict.type = THREAT_MAX_STEPS;
        conflict.step = limit;
        conflict.timestamp = (double)time(NULL);
        snprintf(conflict.detail, sizeof(conflict.detail),
                 "hard_limit_reached (step=%d)", limit);

        /* The hard limit is detected after the final step, so annotate that
         * step's trace record with the scheduler conflict. */
        if (trace.n_steps > 0) {
            step_trace_v3_t* last = &trace.steps[trace.n_steps - 1];
            last->has_conflict = true;
            last->conflict = conflict;
        }

        char path[512];
        if (trace_path && trace_path[0]) {
            snprintf(path, sizeof(path), "%s", trace_path);
        } else {
            v3_default_trace_path(path, sizeof(path));
        }
        save_trace_v3_to_file(&trace, path);
        v3_write_alert(&conflict, limit);
        _last_trace_v3 = trace;
        return CONFLICT_ABORT_TOKEN;
    }

    memcpy(z_out, z_cur, sizeof(vec_t));
    if (trace_path && trace_path[0])
        save_trace_v3_to_file(&trace, trace_path);
    return 0;   /* graceful scheduler budget exhaustion */
}
