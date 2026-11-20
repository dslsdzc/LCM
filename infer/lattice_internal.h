/* Internal helpers shared by infer/lattice.c and infer/lattice_layers.c.
 * Not part of the public ABI: not exported in lcm.h. */
#ifndef LCM_LATTICE_INTERNAL_H
#define LCM_LATTICE_INTERNAL_H

#include "lcm.h"

/* Fetch one codebook row. `row` is in [0, M); `out` receives LCM_D floats. */
typedef void (*lattice_row_fn)(const void* ctx, int row, float* out);

/* Shared R22 value-biased retrieval for one residual input:
 *
 *   score_j = -||r - C_j||² + alpha_val·tanh(v_j)·mean_j(||r - C_j||²)
 *   best    = argmax(score_j)            (first occurrence on ties)
 *
 * With value == NULL or alpha_val <= 0 it is the plain Euclidean argmin. The
 * row accessor is a callback so materialised codebooks (Contrast/Binding/
 * Manifold) and on-demand rows (LowRank's U_k @ V^T) share exactly one copy of
 * the tanh/mean/tie-break formula. c_out (optional) receives the selected row.
 *
 * Callers must have validated 1 <= M <= LCM_MAX_M.
 */
int lattice_pick_row(const float* r, int M, const float* value,
                     float alpha_val, lattice_row_fn row_fn, const void* ctx,
                     float* c_out);

/* Row accessor for a contiguous [M][LCM_D] codebook passed as ctx. */
void lattice_cb_row(const void* ctx, int row, float* out);

#endif /* LCM_LATTICE_INTERNAL_H */
