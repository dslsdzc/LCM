/* LCM Inference Engine — C API for Python ctypes bridge
 *
 * Exposes flat-array inference entry point that Python can call via ctypes.
 * No structs, no pointers-to-pointers: everything is float* or int.
 *
 * Usage (Python):
 *   lib = ctypes.CDLL("infer/liblcm.so")
 *   result = lib.lcm_infer(z_ptr, d, ...)
 */
#ifndef LCM_API_H
#define LCM_API_H

#include "lcm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Single-step cognitive inference ──────────────────────────────────────
 *
 * Constructs engine state from flat arrays, runs build_dag → execute_dag →
 * fusion in a single step (no convergence loop), returns fused z_q.
 *
 * All codebook arrays are flat float32 in row-major order: [M * d].
 * Pointers must remain valid for the duration of the call.
 *
 * Returns 0 on success, -1 on error.
 */
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
                   float* z_out);

/* ─── Single step with the canonical (JAX) fusion ─────────────────────────
 *
 * Same lattice retrieval as lcm_infer_step, but the fusion is the one
 * train/fusion.py::fuse_lattices_with_aux performs, instead of
 * distance_weighted_fusion's inverse-distance weighting:
 *
 *     weights_i     = soft_mask_i * alpha_i
 *     weights_norm  = weights / sum(weights)
 *     z_q           = sum_i weights_norm_i * o_i
 *     z_q           = layer_norm(z_q, ln_scale, ln_bias, eps=1e-6)
 *
 * `soft_mask` is an INPUT here rather than computed internally. The routing
 * gate draws Gumbel noise, and reproducing the same draw in C is a separate
 * problem; taking the mask as given keeps a fusion mismatch attributable to the
 * fusion. A later revision can compute it from route_C / route_W / tau.
 *
 * Lattice order matches fill_memory and the JAX side:
 *   [0]=HRQ [1]=SPARSE [2]=LOWRANK [3]=MANIFOLD [4]=BINDING [5]=CONTRAST
 *
 * `alpha` must have at least n_lattices entries; ln_scale / ln_bias at least d.
 * Returns 0 on success, -1 on error.
 */
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
                      float* z_out);

/* ─── Canonical fusion only, with the lattice outputs supplied ────────────
 *
 * Diagnostic scaffolding for the JAX<->C parity work, not a deployment entry.
 *
 * Runs exactly the fusion of lcm_infer_step_v2 over six already-computed
 * lattice outputs, skipping retrieval entirely. Feeding it the JAX outputs
 * answers one question cleanly: is the remaining JAX<->C gap the fusion or the
 * lattice forwards?
 *
 *   ~0 gap  -> fusion is correct, and every remaining percent is retrieval
 *   non-zero -> the fusion itself still differs
 *
 * `outputs` is a flat [n_lattices * d] row-major array in the usual order
 * (HRQ, SPARSE, LOWRANK, MANIFOLD, BINDING, CONTRAST).
 *
 * Returns 0 on success, -1 on invalid arguments. A zero weight sum is an
 * error here (active == NULL, the historical lcm_fuse_only contract). The v3
 * caller passes an active mask and gets a uniform fallback over that set
 * instead (Ruling minor 3).
 */
int lcm_fuse_only(const float* outputs, int n_lattices, int d,
                  const float* soft_mask,
                  const float* alpha, int n_alpha,
                  const float* ln_scale,
                  const float* ln_bias,
                  float* z_out);

/* ─── Full cognitive inference loop (multi-step until convergence) ────────
 *
 * Like lcm_infer_step but runs the full dynamic_inference loop:
 *   build_dag → execute_dag → fusion → detect_any_conflict → converge_check
 *
 * Returns 0 on normal convergence, -1 on conflict or max_steps exceeded.
 * On conflict, z_out is still populated (last fused output before abort).
 */
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
                   float* z_out);

/* ─── Trace extraction for visualization ──────────────────────────────────────
 *
 * After lcm_infer_loop returns, call this to extract per-step trace data.
 * The trace holds one record per inference step (up to max_steps).
 *
 * Buffer layout (per step, in order):
 *   fusion_weights[LCM_MAX_LATTICES]   (7 floats)
 *   confidences[LCM_MAX_LATTICES]       (7 floats)
 *   z_next[LCM_D]                       (d floats)
 *   step (int as float)                 (1 float)
 *   has_conflict (int as float)         (1 float)
 *
 * Returns the number of steps recorded, or 0 if no trace available.
 * The buffer must have space for at least max_steps * (7 + 7 + LCM_D + 2) floats.
 */
int lcm_get_trace(float* trace_buf, int buf_capacity_floats);

/* ─── v3 macro loop, flat ctypes bridge (task 10) ──────────────────────────
 *
 * Python cannot cheaply build `lcm_memory_v3` (seven nested descriptors with
 * ~40 borrowed pointers). This wrapper takes the exact flat arrays
 * train/c_load.py already produces, binds a v3 memory internally, builds the
 * gvalue/danger structs, and calls lcm_infer_loop_v3.
 *
 * All arrays are caller-owned; no JSON and no allocation. Optional value
 * scalar pointers may be NULL. `lr_ranks` is int32. n_lattices is the routed
 * lattice count (self is the separate 7th fusion term). `self_M` excludes the
 * frozen core row, exactly like fusion_mem_t. `trace_path` may be NULL.
 *
 * Returns 0 (converged or graceful budget), -1 (argument error), or
 * CONFLICT_ABORT_TOKEN (-2, safety abort; z_out must not be used).
 * `gv_sha256` / `danger_sha256` are optional lowercase hex digests of the
 * exported raw payloads. When supplied, init must reproduce them or the call
 * fails with -1 (a tampered/truncated file never reaches the loop).
 * `alert_cb` (optional) receives the fully populated FATAL alert exactly once
 * on any abort.
 */
int lcm_infer_loop_v3_flat(
    const float* z, int d,
    const float* hrq_top_C, int hrq_M_top,
    const float* hrq_fine_C, int hrq_M_fine, int hrq_n_fine,
    const float* sparse_C, int sparse_M, const float* sparse_zero,
    const float* lr_U, const float* lr_V, const int* lr_ranks,
    int lr_n_layers, int lr_M, int lr_r_max,
    const float* man_C, int man_M, const float* man_T, int man_t_dim,
    const float* bind_key, const float* bind_val, const float* bind_bind,
    int bind_M, int bind_n_layers,
    const float* bind_A_k, const float* bind_A_v, int bind_r_max,
    const float* contrast_a, const float* contrast_b,
    int contrast_M, int contrast_n_layers,
    const float* route_C, const float* route_W, int n_lattices,
    const float* alpha, const float* ln_scale, const float* ln_bias,
    const float* self_C, int self_M, float alpha_self,
    const float* hrq_value_top, const float* hrq_value_fine,
    const float* sparse_value, const float* lr_value,
    const float* man_value,
    const float* bind_value_key, const float* bind_value_val,
    const float* bind_value_bind,
    const float* contrast_value_a, const float* contrast_value_b,
    const float* gv_pos, int gv_n, const float* gv_neg,
    const float* danger_t, const float* danger_n, int danger_M,
    const char* gv_sha256, const char* danger_sha256,
    int max_steps, float conv_tol, float entropy_thresh,
    float safety_margin, float alpha_val, const char* trace_path,
    void (*alert_cb)(const alert_t*),
    float* z_out);


#ifdef __cplusplus
}
#endif
#endif /* LCM_API_H */
