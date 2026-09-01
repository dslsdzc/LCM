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

    /* Build gvalue */
    gvalue_t gv;
    memset(&gv, 0, sizeof(gv));
    if (gv_pos && gv_neg && gv_n > 0) {
        int n_copy = gv_n < LCM_N_VALUE_PAIRS ? gv_n : LCM_N_VALUE_PAIRS;
        for (int i = 0; i < n_copy; i++) {
            memcpy(gv.C_pos[i], gv_pos + i * d, d * sizeof(float));
            memcpy(gv.C_neg[i], gv_neg + i * d, d * sizeof(float));
        }
        gv.D = d;
    }

    /* Build danger lattice */
    danger_lattice_t dl;
    memset(&dl, 0, sizeof(dl));
    if (danger_t && danger_n && danger_m > 0) {
        dl.C_threats = (const vec_t*)danger_t;
        dl.C_normal  = (const vec_t*)danger_n;
        dl.M_danger  = danger_m;
        dl.D         = d;
        dl.safety_threshold = 0.005f;
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
