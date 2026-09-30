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
 * Returns 0 on success, -1 on error.
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

#ifdef __cplusplus
}
#endif
#endif /* LCM_API_H */
