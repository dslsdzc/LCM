/* LCM Inference Engine — C API implementation for Python ctypes bridge.
 *
 * Constructs engine internal structs from flat float arrays, runs the
 * cognitive DAG inference, and returns the fused output vector.
 */
#include "lcm_api.h"
#include "lcm.h"
#include <string.h>
#include <stdlib.h>

/* ─── Internal helpers ──────────────────────────────────────────────────── */

/* Populate memory_t from flat codebook arrays. Returns 0 on success. */
static int fill_memory(memory_t* mem, int d,
                       const float* hrq_C, int hrq_M,
                       const float* sparse_C, int sparse_M,
                       const float* lr_C, int lr_M,
                       const float* man_C, int man_M,
                       const float* man_T, int man_t_dim,
                       const float* bind_C, int bind_M,
                       const float* contrast_C, int contrast_M,
                       int n_lattices) {
    if (!mem || d <= 0) return -1;

    memset(mem, 0, sizeof(memory_t));
    mem->n_lattices = n_lattices;
    mem->manifold_T_space = man_T;
    mem->manifold_t_dim = man_t_dim > 0 ? man_t_dim : 4;

    /* Helper macro: set a lattice's codebook pointer */
#define SET_LATTICE(id, ptr, m) do { \
        if ((id) < n_lattices && (ptr)) { \
            mem->lattices[id].C = (vec_t*)(ptr); \
            mem->lattices[id].M = (m); \
            mem->lattices[id].D = d; \
        } \
    } while(0)

    SET_LATTICE(LATTICE_HRQ,      hrq_C,     hrq_M);
    SET_LATTICE(LATTICE_SPARSE,   sparse_C,  sparse_M);
    SET_LATTICE(LATTICE_LOWRANK,  lr_C,      lr_M);
    SET_LATTICE(LATTICE_MANIFOLD, man_C,     man_M);
    SET_LATTICE(LATTICE_BINDING,  bind_C,    bind_M);
    SET_LATTICE(LATTICE_CONTRAST, contrast_C, contrast_M);
#undef SET_LATTICE

    return 0;
}

/* ─── v3 full-parameter memory binding (R10, task 7) ───────────────────── */

static int v3_hrq_valid(const hrq_mem_t* m) {
    if (!m || m->M_top <= 0 || !m->C_top) return 0;
    if (m->n_fine < 0 || m->M_fine < 0) return 0;
    if (m->n_fine > 0 && (m->M_fine <= 0 || !m->C_fine)) return 0;
    return 1;
}

static int v3_lowrank_valid(const lowrank_mem_t* m) {
    if (!m || m->n_layers <= 0 || m->M_lr <= 0 || m->r_max <= 0) return 0;
    if (!m->U || !m->V || !m->ranks) return 0;
    for (int i = 0; i < m->n_layers; i++)
        if (m->ranks[i] <= 0 || m->ranks[i] > m->r_max) return 0;
    return 1;
}

int lcm_bind_memory_v3(lcm_memory_v3* mem, int D, int n_lattices,
                       const hrq_mem_t* hrq,
                       const sparse_mem_t* sparse,
                       const lowrank_mem_t* lowrank,
                       const manifold_mem_t* manifold,
                       const binding_mem_t* binding,
                       const contrast_mem_t* contrast,
                       const fusion_mem_t* fusion) {
    if (!mem) return -1;
    memset(mem, 0, sizeof(*mem));

    /* Dimension and routing count. The .so is compiled for one LCM_D; a
     * mismatch silently corrupts every fixed-size vec_t access. */
    if (D <= 0 || D != LCM_D) return -1;
    if (n_lattices <= 0 || n_lattices > LCM_MAX_LATTICES) return -1;
    if (!hrq || !sparse || !lowrank || !manifold || !binding ||
        !contrast || !fusion) return -1;
    if (fusion->n_lattices != n_lattices) return -1;

    if (!v3_hrq_valid(hrq)) return -1;

    if (sparse->M <= 0 || !sparse->C || !sparse->zero_vec) return -1;

    if (!v3_lowrank_valid(lowrank)) return -1;

    if (manifold->M <= 0 || manifold->t_dim <= 0 ||
        !manifold->C || !manifold->T) return -1;

    if (binding->M <= 0 || binding->n_layers <= 0) return -1;
    if (binding->n_layers > LCM_MAX_BIND_LAYERS) return -1;
    if (binding->r_max <= 0 || binding->r_max != lowrank->r_max) return -1;
    if (!binding->key_cb || !binding->val_cb || !binding->bind_cb
        || !binding->A_k || !binding->A_v) return -1;

    if (contrast->M <= 0 || contrast->n_layers <= 0) return -1;
    if (!contrast->C_a || !contrast->C_b) return -1;

    if (!fusion->route_C || !fusion->route_W || !fusion->alpha
        || !fusion->ln_scale || !fusion->ln_bias || !fusion->self_C)
        return -1;
    if (fusion->M_self <= 0) return -1;

    /* Descriptors are copies of caller-owned structs; every array pointer is
     * borrowed unchanged. Nothing is allocated or written. */
    mem->D = D;
    mem->n_lattices = n_lattices;
    mem->hrq = *hrq;
    mem->sparse = *sparse;
    mem->lowrank = *lowrank;
    mem->manifold = *manifold;
    mem->binding = *binding;
    mem->contrast = *contrast;
    mem->fusion = *fusion;
    mem->tau_route = 0.5f;
    mem->tau_route_fallback = 0.1f;
    mem->tau_self = 1.0f;
    mem->lambda_lfq = 1.5f;
    return 0;
}

/* ─── Single-step inference ─────────────────────────────────────────────── */

int lcm_infer_step(const float* z, int d,
                   const float* hrq_C, int hrq_M,
                   const float* sparse_C, int sparse_M,
                   const float* lr_C, int lr_M,
                   const float* man_C, int man_M,
                   const float* man_T, int man_t_dim,
                   const float* bind_C, int bind_M,
                   const float* contrast_C, int contrast_M,
                   const float* gv_pos, int gv_n,
                   const float* gv_neg,
                   int n_lattices,
                   float* z_out) {
    if (!z || !z_out || d <= 0) return -1;
    (void)gv_pos; (void)gv_n; (void)gv_neg;

    /* Build memory */
    memory_t mem;
    if (fill_memory(&mem, d,
                    hrq_C, hrq_M, sparse_C, sparse_M,
                    lr_C, lr_M, man_C, man_M,
                    man_T, man_t_dim,
                    bind_C, bind_M, contrast_C, contrast_M,
                    n_lattices) != 0) return -1;

    /* Build DAG from z */
    dag_t dag = build_dag(z, &mem, true);

    /* Execute DAG */
    vec_t outputs[LCM_MAX_LATTICES];
    float confidences[LCM_MAX_LATTICES];
    memset(outputs, 0, sizeof(outputs));
    /* Same as dynamic_inference: absent lattices start at zero confidence
     * so zero vectors do not pollute the fusion with full weight. */
    for (int i = 0; i < LCM_MAX_LATTICES; i++) confidences[i] = 0.0f;

    execute_dag(&dag, &mem, outputs, confidences);

    /* Fusion (no gvalue bias in single-step mode) */
    float weights[LCM_MAX_LATTICES];
    distance_weighted_fusion((const vec_t*)outputs, confidences, z_out,
                              weights, NULL, 0.0f);

    return 0;
}

/* ─── Single step with the canonical (JAX) fusion ───────────────────────── */

int lcm_infer_step_v2(const float* z, int d,
                      const float* hrq_C, int hrq_M,
                      const float* sparse_C, int sparse_M,
                      const float* lr_C, int lr_M,
                      const float* man_C, int man_M,
                      const float* man_T, int man_t_dim,
                      const float* bind_C, int bind_M,
                      const float* contrast_C, int contrast_M,
                      const float* hrq_top_C, int hrq_top_M,
                      const float* hrq_fine_C, int hrq_fine_M, int hrq_n_fine,
                      float tau_route_fallback,
                      const float* soft_mask, int n_lattices,
                      const float* alpha, int n_alpha,
                      const float* ln_scale,
                      const float* ln_bias,
                      float* z_out) {
    if (!z || !z_out || d <= 0) return -1;
    if (!soft_mask || !alpha || !ln_scale || !ln_bias) return -1;
    if (n_lattices <= 0 || n_lattices > LCM_MAX_LATTICES) return -1;
    if (n_alpha < n_lattices) return -1;

    memory_t mem;
    if (fill_memory(&mem, d,
                    hrq_C, hrq_M, sparse_C, sparse_M,
                    lr_C, lr_M, man_C, man_M,
                    man_T, man_t_dim,
                    bind_C, bind_M, contrast_C, contrast_M,
                    n_lattices) != 0) return -1;

    dag_t dag = build_dag(z, &mem, false);

    vec_t outputs[LCM_MAX_LATTICES];
    float confidences[LCM_MAX_LATTICES];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < LCM_MAX_LATTICES; i++) confidences[i] = 0.0f;
    execute_dag(&dag, &mem, outputs, confidences);

    /* Faithful HRQ, when its stack is supplied. The DAG retrieves lattice 0 as
     * a flat nearest neighbour, which the ablation measured at 99.7% of the
     * whole JAX<->C gap; every other lattice contributed under 0.05%. */
    if (hrq_top_C != NULL && hrq_top_M > 0) {
        if (hrq_forward_c(z, d, hrq_top_C, hrq_top_M,
                          hrq_fine_C, hrq_fine_M, hrq_n_fine,
                          tau_route_fallback, outputs[LATTICE_HRQ]) != 0)
            return -1;
    }

    /* Canonical fusion, mirroring fuse_lattices_with_aux with gvalue == NULL.
     * fusion_alpha is [n_lattices]; ln_scale / ln_bias are [d]. */
    float w[LCM_MAX_LATTICES];
    float w_sum = 0.0f;
    for (int i = 0; i < n_lattices; i++) {
        w[i] = soft_mask[i] * alpha[i];
        w_sum += w[i];
    }
    if (w_sum == 0.0f) return -1;

    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < n_lattices; i++) {
            acc += (w[i] / w_sum) * outputs[i][j];
        }
        z_out[j] = acc;
    }

    /* LayerNorm, eps matching fusion.py::layer_norm. */
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
    for (int j = 0; j < d; j++) {
        z_out[j] = (z_out[j] - mean) * inv_std * ln_scale[j] + ln_bias[j];
    }

    return 0;
}

/* ─── Canonical fusion only, lattice outputs supplied by the caller ─────── */

int lcm_fuse_only(const float* outputs, int n_lattices, int d,
                  const float* soft_mask,
                  const float* alpha, int n_alpha,
                  const float* ln_scale,
                  const float* ln_bias,
                  float* z_out) {
    if (!outputs || !z_out || d <= 0) return -1;
    if (!soft_mask || !alpha || !ln_scale || !ln_bias) return -1;
    if (n_lattices <= 0 || n_lattices > LCM_MAX_LATTICES) return -1;
    if (n_alpha < n_lattices) return -1;

    float w[LCM_MAX_LATTICES];
    float w_sum = 0.0f;
    for (int i = 0; i < n_lattices; i++) {
        w[i] = soft_mask[i] * alpha[i];
        w_sum += w[i];
    }
    if (w_sum == 0.0f) return -1;

    for (int j = 0; j < d; j++) {
        float acc = 0.0f;
        for (int i = 0; i < n_lattices; i++) {
            acc += (w[i] / w_sum) * outputs[i * d + j];
        }
        z_out[j] = acc;
    }

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
    for (int j = 0; j < d; j++) {
        z_out[j] = (z_out[j] - mean) * inv_std * ln_scale[j] + ln_bias[j];
    }

    return 0;
}

/* ─── Full inference loop (multi-step with convergence, gvalue, danger) ─── */

/* Last-trace storage for visualization.
 *
 * Thread-local (GCC/Clang `__thread` extension — C99 has no standard TLS
 * keyword) so concurrent inference sessions do not corrupt each other's
 * trace. Sized for the engine's max trace (LCM_MAX_TRACE_STEPS=64); the
 * previous 32 truncated traces when Python drove max_steps > 32. */
#define LCM_TRACE_BUF_STEPS 64
static __thread struct {
    float fusion_weights[LCM_TRACE_BUF_STEPS][LCM_MAX_LATTICES];
    float confidences[LCM_TRACE_BUF_STEPS][LCM_MAX_LATTICES];
    float z_next[LCM_TRACE_BUF_STEPS][LCM_D];
    int   step[LCM_TRACE_BUF_STEPS];
    int   has_conflict[LCM_TRACE_BUF_STEPS];
    int   n_steps;
} _last_trace;

int lcm_infer_loop(const float* z, int d,
                   const float* hrq_C, int hrq_M,
                   const float* sparse_C, int sparse_M,
                   const float* lr_C, int lr_M,
                   const float* man_C, int man_M,
                   const float* man_T, int man_t_dim,
                   const float* bind_C, int bind_M,
                   const float* contrast_C, int contrast_M,
                   const float* gv_pos, int gv_n,
                   const float* gv_neg,
                   const float* danger_t, int danger_m,
                   const float* danger_n,
                   int n_lattices,
                   float conv_tol, float entropy_thresh, int max_steps,
                   float* z_out) {
    if (!z || !z_out || d <= 0) return -1;

    /* Build memory */
    memory_t mem;
    if (fill_memory(&mem, d,
                    hrq_C, hrq_M, sparse_C, sparse_M,
                    lr_C, lr_M, man_C, man_M,
                    man_T, man_t_dim,
                    bind_C, bind_M, contrast_C, contrast_M,
                    n_lattices) != 0) return -1;

    /* Build gvalue through gvalue_init so the anchors are projected into the
     * Poincaré ball and hashed.  A raw memcpy here would feed
     * gvalue_check_safety anchors from a different domain (R6). */
    gvalue_t gv;
    memset(&gv, 0, sizeof(gv));
    if (gv_pos && gv_neg && gv_n > 0) {
        float pos_pad[LCM_N_VALUE_PAIRS * LCM_D];
        float neg_pad[LCM_N_VALUE_PAIRS * LCM_D];
        memset(pos_pad, 0, sizeof(pos_pad));
        memset(neg_pad, 0, sizeof(neg_pad));
        int n_copy = gv_n < LCM_N_VALUE_PAIRS ? gv_n : LCM_N_VALUE_PAIRS;
        memcpy(pos_pad, gv_pos, (size_t)n_copy * (size_t)d * sizeof(float));
        memcpy(neg_pad, gv_neg, (size_t)n_copy * (size_t)d * sizeof(float));
        gvalue_init(&gv, pos_pad, neg_pad, d);
    }

    /* Build danger lattice through danger_init (sets the R7 threshold and the
     * shared SHA-256 integrity hash). */
    danger_lattice_t dl;
    memset(&dl, 0, sizeof(dl));
    if (danger_t && danger_n && danger_m > 0) {
        danger_init(&dl, danger_t, danger_n, danger_m, d);
    }

    /* Build inference state */
    inference_state_t state;
    memset(&state, 0, sizeof(state));
    snprintf(state.session_id, sizeof(state.session_id), "lcm_py_bridge");
    state.max_steps     = max_steps > 0 ? max_steps : LCM_MAX_STEPS;
    state.memory        = &mem;
    state.gvalue        = gv.D > 0 ? &gv : NULL;
    state.danger        = dl.M_danger > 0 ? &dl : NULL;
    state.value_consistency[0] = 1.0f;  /* Start clean (no consistency violation) */
    state.conv_tol      = conv_tol;     /* Agency-modulated convergence tolerance */
    state.entropy_thresh = entropy_thresh;  /* Agency-modulated entropy threshold */

    /* Run dynamic inference */
    int result = dynamic_inference(&state, z, NULL);

    /* Copy output */
    memcpy(z_out, state.z_current, d * sizeof(float));

    /* Extract trace for visualization */
    _last_trace.n_steps = 0;
    for (int s = 0; s < state.trace.n_steps && s < LCM_TRACE_BUF_STEPS; s++) {
        const step_trace_t* st = &state.trace.steps[s];
        for (int i = 0; i < LCM_MAX_LATTICES; i++) {
            _last_trace.fusion_weights[s][i] = st->fusion_weights[i];
            _last_trace.confidences[s][i]    = st->confidences[i];
        }
        memcpy(_last_trace.z_next[s], st->z_next, sizeof(vec_t));
        _last_trace.step[s]        = st->step;
        _last_trace.has_conflict[s] = st->has_conflict ? 1 : 0;
        _last_trace.n_steps++;
    }

    return result;
}

/* ─── Trace extraction ─────────────────────────────────────────────────────── */
int lcm_get_trace(float* trace_buf, int buf_capacity_floats) {
    if (!trace_buf) return _last_trace.n_steps;

    int per_step = LCM_MAX_LATTICES + LCM_MAX_LATTICES + LCM_D + 2;
    if (buf_capacity_floats < _last_trace.n_steps * per_step) return -1;

    int pos = 0;
    for (int s = 0; s < _last_trace.n_steps; s++) {
        for (int i = 0; i < LCM_MAX_LATTICES; i++)
            trace_buf[pos++] = _last_trace.fusion_weights[s][i];
        for (int i = 0; i < LCM_MAX_LATTICES; i++)
            trace_buf[pos++] = _last_trace.confidences[s][i];
        memcpy(&trace_buf[pos], _last_trace.z_next[s], LCM_D * sizeof(float));
        pos += LCM_D;
        trace_buf[pos++] = (float)_last_trace.step[s];
        trace_buf[pos++] = (float)_last_trace.has_conflict[s];
    }
    return _last_trace.n_steps;
}

/* ─── Compiled dimension query ────────────────────────────────────────────────
 *
 * Lets the Python bridge verify at runtime that the .so was built with the
 * same LCM_D as the checkpoint (a mismatch silently corrupts the bridge).
 */
int lcm_dim(void) {
    return LCM_D;
}
