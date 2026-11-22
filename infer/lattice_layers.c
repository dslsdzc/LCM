/* Full multi-layer lattice forwards — R10 / b.md §4.3–§4.6.
 *
 * Task 7 froze the per-layer memory layout; this file implements the four
 * layers whose old C versions only reconstructed the first block/layer:
 * LowRank, Contrast and Binding (Manifold lives in lattice.c next to
 * slide_manifold because the two share the tangent-space core).
 *
 * No heap allocation, no JSON, no global mutable state: every temporary is a
 * fixed-size vec_t or a capped stack array. All arrays are caller-owned.
 */
#include "lcm.h"
#include "lattice_internal.h"
#include <assert.h>
#include <math.h>
#include <string.h>

/* ─── Shared residual-VQ pick ───────────────────────────────────────────────
 *
 * One implementation of JAX's value_biased_scores over a residual input:
 *   score_j = -||r - C_j||² + alpha_val·tanh(v_j)·mean_j(||r - C_j||²)
 * with argmax (first occurrence on ties). Without an active value vector this
 * is the plain Euclidean argmin. See lattice_internal.h.
 */
int lattice_pick_row(const float* r, int M, const float* value,
                     float alpha_val, lattice_row_fn row_fn, const void* ctx,
                     float* c_out) {
    assert(r != NULL && row_fn != NULL);
    assert(M > 0 && M <= LCM_MAX_M);

    int best = 0;
    vec_t row;

    if (value != NULL && alpha_val > 0.0f) {
        float d2[LCM_MAX_M];
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            row_fn(ctx, j, row);
            d2[j] = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = r[i] - row[i];
                d2[j] += diff * diff;
            }
            sum += d2[j];
        }
        float avg = sum / (float)M;
        float best_score = -INFINITY;
        for (int j = 0; j < M; j++) {
            float score = -d2[j] + alpha_val * tanhf(value[j]) * avg;
            if (score > best_score) { best_score = score; best = j; }
        }
    } else {
        float best_d2 = INFINITY;
        for (int j = 0; j < M; j++) {
            row_fn(ctx, j, row);
            float d2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = r[i] - row[i];
                d2 += diff * diff;
            }
            if (d2 < best_d2) { best_d2 = d2; best = j; }
        }
    }

    if (c_out) row_fn(ctx, best, c_out);
    return best;
}

void lattice_cb_row(const void* ctx, int row, float* out) {
    const float* cb = (const float*)ctx;    memcpy(out, cb + (size_t)row * LCM_D, sizeof(vec_t));
}

/* ─── LowRank: incremental-rank VQ with one shared basis (b.md §4.3) ─────── */

/* C_k = U_k @ V[:, :rank]^T for one row, computed on demand so the full
 * [M_lr][D] layer codebook never has to be materialised. */
static void lowrank_row(const lowrank_mem_t* m, int layer, int rank, int row,
                        float* out) {
    const float* u = m->U
        + ((size_t)layer * (size_t)m->M_lr + (size_t)row) * (size_t)m->r_max;
    for (int d = 0; d < LCM_D; d++) {
        const float* v = m->V + (size_t)d * (size_t)m->r_max;
        float acc = 0.0f;
        for (int j = 0; j < rank; j++) acc += u[j] * v[j];
        out[d] = acc;
    }
}

typedef struct {
    const lowrank_mem_t* m;
    int layer;
    int rank;
} lowrank_row_ctx_t;

static void lowrank_row_cb(const void* ctx, int row, float* out) {
    const lowrank_row_ctx_t* c = (const lowrank_row_ctx_t*)ctx;
    lowrank_row(c->m, c->layer, c->rank, row, out);
}

void lowrank_forward_c(const float* z, const lowrank_mem_t* m, float alpha_val,
                       float* out, int* idx) {
    if (!out) return;
    if (!z || !m || !m->U || !m->V || !m->ranks || m->n_layers <= 0
        || m->M_lr <= 0 || m->M_lr > LCM_MAX_M
        || m->r_max <= 0 || m->r_max > LCM_MAX_RANK) {
        memset(out, 0, sizeof(vec_t));
        if (idx && m && m->n_layers > 0)
            memset(idx, 0, (size_t)m->n_layers * sizeof(int));
        return;
    }
    for (int l = 0; l < m->n_layers; l++) {
        if (m->ranks[l] <= 0 || m->ranks[l] > m->r_max) {
            memset(out, 0, sizeof(vec_t));
            if (idx) memset(idx, 0, (size_t)m->n_layers * sizeof(int));
            return;
        }
    }

    vec_t r, c, acc;
    memcpy(r, z, sizeof(vec_t));
    memset(acc, 0, sizeof(vec_t));

    for (int l = 0; l < m->n_layers; l++) {
        int rank = m->ranks[l];
        lowrank_row_ctx_t row_ctx = {m, l, rank};
        int best = lattice_pick_row(r, m->M_lr, m->value, alpha_val,
                                    lowrank_row_cb, &row_ctx, c);

        if (idx) idx[l] = best;
        for (int i = 0; i < LCM_D; i++) {
            acc[i] += c[i];
            r[i] -= c[i];
        }
    }

    memcpy(out, acc, sizeof(vec_t));
}

/* ─── Contrast: two independent residual chains (b.md §4.6) ─────────────── */

void contrast_forward_v31(const float* z, const contrast_mem_t* m,
                          float alpha_val, float* out,
                          int* idx_a, int* idx_b) {
    if (!out) return;
    if (!z || !m || !m->C_a || !m->C_b || m->M <= 0
        || m->M > LCM_MAX_M || m->n_layers <= 0) {
        memset(out, 0, sizeof(vec_t));
        if (m && m->n_layers > 0) {
            if (idx_a) memset(idx_a, 0, (size_t)m->n_layers * sizeof(int));
            if (idx_b) memset(idx_b, 0, (size_t)m->n_layers * sizeof(int));
        }
        return;
    }

    vec_t ra, rb, c, oa, ob;
    memcpy(ra, z, sizeof(vec_t));
    memcpy(rb, z, sizeof(vec_t));
    memset(oa, 0, sizeof(vec_t));
    memset(ob, 0, sizeof(vec_t));

    for (int l = 0; l < m->n_layers; l++) {
        const float* cba = m->C_a
            + (size_t)l * (size_t)m->M * LCM_D;
        int a = lattice_pick_row(ra, m->M, m->value_a, alpha_val,
                                 lattice_cb_row, cba, c);
        if (idx_a) idx_a[l] = a;
        for (int i = 0; i < LCM_D; i++) {
            oa[i] += c[i];
            ra[i] -= c[i];
        }

        const float* cbb = m->C_b
            + (size_t)l * (size_t)m->M * LCM_D;
        int b = lattice_pick_row(rb, m->M, m->value_b, alpha_val,
                                 lattice_cb_row, cbb, c);
        if (idx_b) idx_b[l] = b;
        for (int i = 0; i < LCM_D; i++) {
            ob[i] += c[i];
            rb[i] -= c[i];
        }
    }

    for (int i = 0; i < LCM_D; i++)
        out[i] = 0.5f * (oa[i] + ob[i]);
}

void contrast_forward_c(const float* z, const contrast_mem_t* m,
                        float alpha_val, float* out) {
    contrast_forward_v31(z, m, alpha_val, out, NULL, NULL);
}

/* ─── Binding: cross-layer HRR bind + conjugate unbind (b.md §4.5) ──────── */

/* Selection-only direction matching. Binding projections can be far below
 * the codebook row norms, and row norms vary; raw Euclidean NN is then decided
 * by scale, not direction. The picker compares unit query/rows, then returns
 * the raw selected row. */
static void unit_copy(const float* x, float* out) {
    double n2 = 0.0;
    for (int i = 0; i < LCM_D; i++) n2 += (double)x[i] * (double)x[i];
    float n = (float)sqrt(n2);
    float s = n > 1e-12f ? 1.0f / n : 0.0f;
    for (int i = 0; i < LCM_D; i++) out[i] = x[i] * s;
}

static int pick_row_unit(const float* r, int M, const float* value,
                         float alpha_val, const float* cb, float* c_out) {
    vec_t ru, rowu;
    unit_copy(r, ru);
    int best = 0;
    if (value != NULL && alpha_val > 0.0f) {
        float d2[LCM_MAX_M];
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            unit_copy(cb + (size_t)j * LCM_D, rowu);
            d2[j] = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = ru[i] - rowu[i];
                d2[j] += diff * diff;
            }
            sum += d2[j];
        }
        float avg = sum / (float)M;
        float best_score = -INFINITY;
        for (int j = 0; j < M; j++) {
            float score = -d2[j] + alpha_val * tanhf(value[j]) * avg;
            if (score > best_score) { best_score = score; best = j; }
        }
    } else {
        float best_d2 = INFINITY;
        for (int j = 0; j < M; j++) {
            unit_copy(cb + (size_t)j * LCM_D, rowu);
            float d2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = ru[i] - rowu[i];
                d2 += diff * diff;
            }
            if (d2 < best_d2) { best_d2 = d2; best = j; }
        }
    }
    if (c_out) memcpy(c_out, cb + (size_t)best * LCM_D, sizeof(vec_t));
    return best;
}

void binding_forward_c(const float* z, const binding_mem_t* m, const float* V,
                       float alpha_val, float* out, float* b_q, float* k_q,
                       int* val_idx) {
    if (!out) return;
    if (!z || !m || !V || !m->key_cb || !m->val_cb || !m->bind_cb
        || !m->A_k || !m->A_v || m->M <= 0 || m->M > LCM_MAX_M
        || m->n_layers <= 0 || m->n_layers > LCM_MAX_BIND_LAYERS
        || m->r_max <= 0 || m->r_max > LCM_MAX_RANK) {
        memset(out, 0, sizeof(vec_t));
        if (b_q) memset(b_q, 0, sizeof(vec_t));
        if (k_q) memset(k_q, 0, sizeof(vec_t));
        if (val_idx) *val_idx = -1;
        return;
    }

    /* u_k = A_k z; z_k = V u_k = W_k z (b.md §4.5, W_k = V @ A_k).
     * Same for A_v with an independent u_v. This is the projection order the
     * JAX forward uses (z @ W_k^T == W_k z for one query). */
    float u_k[LCM_MAX_RANK], u_v[LCM_MAX_RANK];
    vec_t zk, zv;
    for (int p = 0; p < m->r_max; p++) {
        u_k[p] = 0.0f;
        u_v[p] = 0.0f;
        for (int i = 0; i < LCM_D; i++) {
            u_k[p] += m->A_k[(size_t)p * LCM_D + (size_t)i] * z[i];
            u_v[p] += m->A_v[(size_t)p * LCM_D + (size_t)i] * z[i];
        }
    }
    for (int j = 0; j < LCM_D; j++) {
        zk[j] = 0.0f;
        zv[j] = 0.0f;
        for (int p = 0; p < m->r_max; p++) {
            zk[j] += V[(size_t)j * (size_t)m->r_max + (size_t)p] * u_k[p];
            zv[j] += V[(size_t)j * (size_t)m->r_max + (size_t)p] * u_v[p];
        }
    }

    /* Three-layer (in general n_layers) residual VQ for key and value. */
    float key_rows[LCM_MAX_BIND_LAYERS * LCM_D];
    float val_rows[LCM_MAX_BIND_LAYERS * LCM_D];
    const float* kp[LCM_MAX_BIND_LAYERS];
    const float* vp[LCM_MAX_BIND_LAYERS];

    vec_t r, c, kq;
    memcpy(r, zk, sizeof(vec_t));
    memset(kq, 0, sizeof(vec_t));
    for (int l = 0; l < m->n_layers; l++) {
        const float* cb = m->key_cb
            + (size_t)l * (size_t)m->M * LCM_D;
        pick_row_unit(r, m->M, m->value_key, alpha_val, cb, c);
        kp[l] = key_rows + (size_t)l * LCM_D;
        memcpy(key_rows + (size_t)l * LCM_D, c, sizeof(vec_t));
        for (int i = 0; i < LCM_D; i++) {
            kq[i] += c[i];
            r[i] -= c[i];
        }
    }

    memcpy(r, zv, sizeof(vec_t));
    for (int l = 0; l < m->n_layers; l++) {
        const float* cb = m->val_cb
            + (size_t)l * (size_t)m->M * LCM_D;
        pick_row_unit(r, m->M, m->value_val, alpha_val, cb, c);
        vp[l] = val_rows + (size_t)l * LCM_D;
        memcpy(val_rows + (size_t)l * LCM_D, c, sizeof(vec_t));
        for (int i = 0; i < LCM_D; i++) r[i] -= c[i];
    }

    /* b_raw = Σ_{i,j} IFFT(norm_fft(k^(i)) ⊙ norm_fft(v^(j))). hrr_bind is
     * exactly that pairwise accumulation (n_layers² terms; 9 for 3 layers). */
    vec_t b_raw;
    hrr_bind(kp, m->n_layers, vp, m->n_layers, b_raw, LCM_D);

    /* b_q = residual VQ over the binding codebooks. */
    vec_t bq_acc;
    memcpy(r, b_raw, sizeof(vec_t));
    memset(bq_acc, 0, sizeof(vec_t));
    for (int l = 0; l < m->n_layers; l++) {
        const float* cb = m->bind_cb
            + (size_t)l * (size_t)m->M * LCM_D;
        pick_row_unit(r, m->M, m->value_bind, alpha_val, cb, c);
        for (int i = 0; i < LCM_D; i++) {
            bq_acc[i] += c[i];
            r[i] -= c[i];
        }
    }

    /* Unbind: recovered = IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q)), then
     * nearest neighbour in C_val = every val_cb layer concatenated. */
    vec_t recovered;
    hrr_unbind(bq_acc, kq, recovered, LCM_D);

    int n_val = m->n_layers * m->M;
    int best = pick_row_unit(recovered, n_val, NULL, alpha_val,
                             m->val_cb, NULL);

    memcpy(out, m->val_cb + (size_t)best * LCM_D, sizeof(vec_t));
    if (val_idx) *val_idx = best;
    if (b_q) memcpy(b_q, bq_acc, sizeof(vec_t));
    if (k_q) memcpy(k_q, kq, sizeof(vec_t));
}
