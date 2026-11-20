/* Lattice retrieval operations — inference-only implementations.
 *
 * Each lattice type performs a specific cognitive retrieval operation.
 * All codebooks are read-only (frozen) during inference.
 *
 * Formal contracts:
 *   - All lattice memory pointers must be initialized
 *   - Output vectors must be non-NULL and have space for LCM_D elements
 *   - indices are optional (pass NULL if not needed)
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ─── Single lattice retrieval (通用单格检索) ─────────────────────────────
 *
 * Finds the nearest codebook vector by Euclidean distance.
 * Used by: HRQ (fine layers), Sparse, LowRank, Contrast in inference mode.
 */

/*@
  requires \valid_read(z + (0 .. LCM_D-1));
  requires \valid_read(mem) && \valid_read(mem->C);
  requires \valid(out + (0 .. LCM_D-1));
  requires \separated(out + (0 .. LCM_D-1), z + (0 .. LCM_D-1));
  requires mem->M > 0 && mem->D == LCM_D;
  requires \forall integer j; 0 <= j < mem->M ==> \valid_read(mem->C[j] + (0 .. LCM_D-1));
  requires dist == \null || \valid(dist);
  requires idx == \null || \valid(idx);
  ensures  \valid(out + (0 .. LCM_D-1));
  assigns  out[0 .. LCM_D-1], *dist, *idx;
 */
/* ─── HRQ: hierarchical residual quantization on the Poincaré ball (R1) ────
 *
 * R1 two-stage routing (mirrors train/lattices.py::hrq_forward):
 *   1. geometric top-k by poincare_similarity(z_P, C_top_P);
 *   2. score_j = sim_j - alpha_val·tanh(v_j)·scale inside that top-k, with
 *      scale = mean over all prototypes; argmin picks the hard top-1.
 * The fallback decision is the *relative* gap of the two nearest prototypes
 * and blends only the top-k with softmax(-score).
 *
 * The R6 domain is exp_map(safe_unit(z)) for the query; codebooks are mapped
 * raw (exp_map(C)) exactly like JAX. HRQ's output telescopes through the fine
 * chain, so the routed top prototype is observable through the fine-layer
 * selection — that is what the fallback test exercises.
 *
 * Index insertion keeps equal similarities in first-occurrence order, matching
 * jnp.argsort / jnp.argmin tie-breaking.
 */
static void hrq_topk_insert(int* idx, float* sim, int k, int* n, int m, float s) {
    int p;
    if (*n < k) {
        p = *n;
        while (p > 0 && sim[p - 1] > s) {
            sim[p] = sim[p - 1];
            idx[p] = idx[p - 1];
            p--;
        }
        sim[p] = s;
        idx[p] = m;
        (*n)++;
    } else if (s < sim[k - 1]) {
        p = k - 1;
        while (p > 0 && sim[p - 1] > s) {
            sim[p] = sim[p - 1];
            idx[p] = idx[p - 1];
            p--;
        }
        sim[p] = s;
        idx[p] = m;
    }
}

/* One fine layer: nearest exp_map(C_fine[l][j]) to r, value-rescored when
 * value_fine is present and alpha_val > 0 (R22: the vector is shared by all
 * fine layers). Writes the selected ball prototype to c_p. */
static void hrq_fine_pick(const hrq_mem_t* m, int l, const float* r,
                          float alpha_val, float* c_p) {
    const float* layer = m->C_fine + (size_t)l * (size_t)m->M_fine * LCM_D;
    int use_value = (m->value_fine != NULL && alpha_val > 0.0f);
    float scale = 0.0f;

    if (use_value) {
        float sum = 0.0f;
        for (int j = 0; j < m->M_fine; j++) {
            exp_map_c(layer + (size_t)j * LCM_D, c_p, LCM_D, 1.0f);
            sum += poincare_similarity_c(r, c_p, LCM_D, 1.0f);
        }
        scale = sum / (float)m->M_fine + 1e-6f;
    }

    float best = INFINITY;
    int best_j = 0;
    for (int j = 0; j < m->M_fine; j++) {
        exp_map_c(layer + (size_t)j * LCM_D, c_p, LCM_D, 1.0f);
        float s = poincare_similarity_c(r, c_p, LCM_D, 1.0f);
        float score = s;
        if (use_value)
            score = s - alpha_val * tanhf(m->value_fine[j]) * scale;
        if (score < best) { best = score; best_j = j; }
    }
    exp_map_c(layer + (size_t)best_j * LCM_D, c_p, LCM_D, 1.0f);
}

/* out = log_map(top ⊕ r) where r starts at z_P ⊕ (-top) and each fine layer
 * subtracts its selected prototype (Möbius residual chain, b.md §4.1). */
static void hrq_fine_chain(const hrq_mem_t* m, const float* z_P,
                           const float* top, float alpha_val, float* out) {
    vec_t r, t2, c_p, neg;

    for (int i = 0; i < LCM_D; i++) neg[i] = -top[i];
    mobius_add_c(z_P, neg, r, LCM_D, 1.0f);

    for (int l = 0; l < m->n_fine; l++) {
        hrq_fine_pick(m, l, r, alpha_val, c_p);
        for (int i = 0; i < LCM_D; i++) neg[i] = -c_p[i];
        mobius_add_c(r, neg, t2, LCM_D, 1.0f);
        memcpy(r, t2, sizeof(vec_t));
    }

    mobius_add_c(top, r, t2, LCM_D, 1.0f);
    log_map_c(t2, out, LCM_D, 1.0f);
}

int hrq_forward_v31(const float* z, const hrq_mem_t* m, float alpha_val,
                    int value_topk, float tau_fallback,
                    float* out, int* idx, float* top_sim) {
    if (!z || !m || !out || !m->C_top || m->M_top <= 0) return -1;
    if (m->n_fine > 0 && (!m->C_fine || m->M_fine <= 0)) return -1;
    if (value_topk <= 0) return -1;

    int k = value_topk < m->M_top ? value_topk : m->M_top;
    if (k > LCM_HRQ_MAX_TOPK) k = LCM_HRQ_MAX_TOPK;

    vec_t z_u, z_P, c_p, c_top, c_top_w, acc;
    safe_unit_c(z, z_u, LCM_D, 1e-8f);
    exp_map_c(z_u, z_P, LCM_D, 1.0f);

    /* Stage 1: geometric top-k, the two smallest sims for the relative gap,
     * and the mean similarity used as the value term's scale. */
    int k_idx[LCM_HRQ_MAX_TOPK];
    float k_sim[LCM_HRQ_MAX_TOPK];
    float k_score[LCM_HRQ_MAX_TOPK];
    int n_top = 0;
    float min1 = INFINITY, min2 = INFINITY, sum_sim = 0.0f;

    for (int j = 0; j < m->M_top; j++) {
        exp_map_c(m->C_top + (size_t)j * LCM_D, c_p, LCM_D, 1.0f);
        float s = poincare_similarity_c(z_P, c_p, LCM_D, 1.0f);
        sum_sim += s;
        if (s < min1) { min2 = min1; min1 = s; }
        else if (s < min2) { min2 = s; }
        hrq_topk_insert(k_idx, k_sim, k, &n_top, j, s);
    }
    assert(n_top == k);

    /* Stage 2: value rescoring inside the top-k only. */
    float scale = sum_sim / (float)m->M_top + 1e-6f;
    int best = 0;
    for (int i = 0; i < k; i++) {
        float v = m->value_top ? m->value_top[k_idx[i]] : 0.0f;
        k_score[i] = k_sim[i] - alpha_val * tanhf(v) * scale;
        if (k_score[i] < k_score[best]) best = i;
    }
    int top_i = k_idx[best];
    float top_s = k_sim[best];
    exp_map_c(m->C_top + (size_t)top_i * LCM_D, c_top, LCM_D, 1.0f);

    /* R1.4 relative fallback: scale-free, so initialisation-scale codebooks
     * do not make the hard route unreachable. */
    float gap = (m->M_top > 1) ? (min2 - min1) / (min1 + 1e-8f) : 0.0f;
    int use_fallback = gap < tau_fallback;

    if (use_fallback) {
        float min_score = k_score[0];
        for (int i = 1; i < k; i++)
            if (k_score[i] < min_score) min_score = k_score[i];

        float den = 0.0f;
        memset(acc, 0, sizeof(vec_t));
        for (int i = 0; i < k; i++) {
            float e = expf(min_score - k_score[i]);   /* softmax(-score) */
            exp_map_c(m->C_top + (size_t)k_idx[i] * LCM_D, c_p, LCM_D, 1.0f);
            den += e;
            for (int d = 0; d < LCM_D; d++) acc[d] += e * c_p[d];
        }
        if (!(den > 0.0f)) return -1;
        for (int d = 0; d < LCM_D; d++) c_top_w[d] = acc[d] / den;
        hrq_fine_chain(m, z_P, c_top_w, alpha_val, out);
    } else {
        hrq_fine_chain(m, z_P, c_top, alpha_val, out);
    }

    if (idx) *idx = top_i;
    if (top_sim) *top_sim = top_s;
    return 0;
}

/* Thin wrapper for the v2 bridge: D must be LCM_D, v = 0, k = 4 (the
 * hrq_value_topk default) or M_top when smaller. */
int hrq_forward_c(const float* z, int D,
                  const float* top_C, int M_top,
                  const float* fine_C, int M_fine, int n_fine,
                  float tau_fallback, float* out) {
    if (D != LCM_D) return -1;
    hrq_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_top = top_C;
    m.M_top = M_top;
    m.C_fine = fine_C;
    m.M_fine = M_fine;
    m.n_fine = n_fine;
    int k = M_top < 4 ? M_top : 4;
    return hrq_forward_v31(z, &m, 0.0f, k, tau_fallback, out, NULL, NULL);
}

/* ─── Sparse: training/inference split + LFQ (R2) ────────────────────────── */

int sparse_forward_c(const float* z, const sparse_mem_t* m, float lambda_lfq,
                     float alpha_val, float d_top, int training,
                     float* out, int* idx) {
    if (!z || !m || !m->C || m->M <= 0 || !out) return -1;
    if (!training && !m->zero_vec) return -1;

    /* Retrieval (b.md §4.2). Inference prepends zero_vec, so its index space
     * is 0 = zero_vec, j+1 = C[j]; training is C-only. With a value vector
     * and alpha_val > 0 this is JAX's value-biased argmax instead. */
    int n_search = training ? m->M : m->M + 1;
    int offset = training ? 0 : 1;
    int best = 0;

    if (alpha_val > 0.0f && m->value != NULL) {
        float sum_d2 = 0.0f;
        for (int j = 0; j < n_search; j++) {
            const float* row = (j == 0 && offset == 1)
                                   ? m->zero_vec
                                   : m->C + (size_t)(j - offset) * LCM_D;
            for (int i = 0; i < LCM_D; i++) {
                float diff = z[i] - row[i];
                sum_d2 += diff * diff;
            }
        }
        float avg_d2 = sum_d2 / (float)n_search;

        /* score = -dist² + alpha·tanh(v)·avg_dist²; zero_vec has v = 0. */
        float best_score = -INFINITY;
        for (int j = 0; j < n_search; j++) {
            const float* row = (j == 0 && offset == 1)
                                   ? m->zero_vec
                                   : m->C + (size_t)(j - offset) * LCM_D;
            float d2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = z[i] - row[i];
                d2 += diff * diff;
            }
            float v = (j == 0 && offset == 1) ? 0.0f : m->value[j - offset];
            float score = -d2 + alpha_val * tanhf(v) * avg_d2;
            if (score > best_score) { best_score = score; best = j; }
        }
    } else {
        float best_d2 = INFINITY;
        for (int j = 0; j < n_search; j++) {
            const float* row = (j == 0 && offset == 1)
                                   ? m->zero_vec
                                   : m->C + (size_t)(j - offset) * LCM_D;
            float d2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = z[i] - row[i];
                d2 += diff * diff;
            }
            if (d2 < best_d2) { best_d2 = d2; best = j; }
        }
    }

    /* LFQ (inference only, R2/C-5). d_top <= 0 means HRQ did not activate,
     * so the threshold has no reference and binarisation is disabled. */
    if (!training && d_top > 0.0f) {
        vec_t z_u, z_P, c_p, c_clamped;
        safe_unit_c(z, z_u, LCM_D, 1e-8f);
        exp_map_c(z_u, z_P, LCM_D, 1.0f);

        float d_min = INFINITY;
        for (int j = 0; j < m->M; j++) {
            const float* row = m->C + (size_t)j * LCM_D;
            float norm2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) norm2 += row[i] * row[i];
            float norm = sqrtf(norm2);
            float scale = 1.0f;
            if (norm > 0.99f) scale = 0.99f / norm;
            for (int i = 0; i < LCM_D; i++) c_clamped[i] = row[i] * scale;
            exp_map_c(c_clamped, c_p, LCM_D, 1.0f);
            float s = poincare_similarity_c(z_P, c_p, LCM_D, 1.0f);
            if (s < d_min) d_min = s;
        }

        if (d_min > lambda_lfq * d_top) best = 0;
    }

    if (best == 0 && offset == 1)
        memcpy(out, m->zero_vec, sizeof(vec_t));
    else
        memcpy(out, m->C + (size_t)(best - offset) * LCM_D, sizeof(vec_t));
    if (idx) *idx = best;
    return 0;
}

/* ─── Routing gate: deterministic one-hot argmax (R19) ───────────────────── */

void routing_gate_c(const float* z, const float* C_route, const float* W_route,
                    int n_lattices, float* soft_mask) {
    if (!soft_mask) return;
    if (!z || !C_route || !W_route || n_lattices <= 0
        || n_lattices > LCM_MAX_LATTICES) {
        if (n_lattices > 0) {
            int n = n_lattices < LCM_MAX_LATTICES
                        ? n_lattices : LCM_MAX_LATTICES;
            memset(soft_mask, 0, (size_t)n * sizeof(float));
        }
        return;
    }

    int best_c = 0;
    float best_d2 = INFINITY;
    for (int j = 0; j < n_lattices; j++) {
        const float* row = C_route + (size_t)j * LCM_D;
        float d2 = 0.0f;
        for (int i = 0; i < LCM_D; i++) {
            float diff = z[i] - row[i];
            d2 += diff * diff;
        }
        if (d2 < best_d2) { best_d2 = d2; best_c = j; }
    }

    /* logits = W_route^T z_route, W_route is [D][n_lattices]. */
    const float* z_route = C_route + (size_t)best_c * LCM_D;
    int best_l = 0;
    float best_logit = -INFINITY;
    for (int l = 0; l < n_lattices; l++) {
        float logit = 0.0f;
        for (int i = 0; i < LCM_D; i++)
            logit += z_route[i] * W_route[(size_t)i * n_lattices + l];
        if (logit > best_logit) { best_logit = logit; best_l = l; }
    }

    memset(soft_mask, 0, (size_t)n_lattices * sizeof(float));
    soft_mask[best_l] = 1.0f;
}

void retrieve_single(const float* z, const lattice_memory_t* mem,
                     float* out, float* dist, int* idx) {
    assert(mem != NULL && mem->C != NULL && out != NULL);
    assert(mem->M > 0 && mem->D == LCM_D);

    float best_dist = 1e20f;
    int   best_idx = 0;

    /*@
      loop invariant 0 <= j <= mem->M;
      loop invariant best_idx == 0 || (best_dist >= 0.0f);
      loop invariant 0 <= best_idx < mem->M;
      loop assigns j, best_dist, best_idx;
      loop variant mem->M - j;
     */
    for (int j = 0; j < mem->M; j++) {
        float d = 0.0f;
        /*@
          loop invariant 0 <= i <= LCM_D;
          loop invariant d >= 0.0f;
          loop assigns i, d;
          loop variant LCM_D - i;
         */
        for (int i = 0; i < LCM_D; i++) {
            float diff = z[i] - mem->C[j][i];
            d += diff * diff;
        }
        if (d < best_dist) {
            best_dist = d;
            best_idx = j;
        }
    }

    if (dist) *dist = best_dist;
    if (idx)  *idx  = best_idx;
    /*@ assert best_idx >= 0 && best_idx < mem->M; */
    memcpy(out, mem->C[best_idx], sizeof(vec_t));
}

/* ─── Tangent space slide (流形格切空间滑动) ───────────────────────────────
 *
 * Sliding on the Poincaré ball via local tangent space projection.
 * Used by: Manifold lattice.
 *
 * T_space: [M_man x D x t_dim] — local tangent bases (flattened: M_man * D * t_dim)
 */

/*@
  requires \valid_read(z + (0 .. LCM_D-1));
  requires \valid_read(z_P + (0 .. LCM_D-1));
  requires \valid_read(C + (0 .. M * LCM_D - 1));
  requires \valid_read(T_space + (0 .. M * LCM_D * t_dim - 1));
  requires \valid(out + (0 .. LCM_D-1));
  requires \separated(out, z) && \separated(out, z_P) && \separated(out, T_space);
  requires M > 0 && t_dim > 0;
  requires value == \null || \valid_read(value + (0 .. M-1));
  requires idx_out == \null || \valid(idx_out);
  ensures  \valid(out + (0 .. LCM_D-1));
  assigns  out[0 .. LCM_D-1], *idx_out;
 */
static void manifold_slide_core(const float* z, const float* z_P,
                                const float* C, int M,
                                const float* T_space, int t_dim,
                                const float* value, float alpha_val,
                                float* out, int* idx_out) {
    int best = 0;

    if (value != NULL && alpha_val > 0.0f) {
        /* JAX's value_biased_retrieve(z, C, ..., C_out=C_P): score in the
         * tangent-space coordinates, then exp_map the selected row. */
        float d2[M];
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            d2[j] = 0.0f;
            const float* row = C + (size_t)j * LCM_D;
            for (int i = 0; i < LCM_D; i++) {
                float diff = z[i] - row[i];
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
        /* Poincaré nearest prototype in the R6 domain. */
        float best_sim = INFINITY;
        for (int j = 0; j < M; j++) {
            vec_t c_P;
            exp_map_c(C + (size_t)j * LCM_D, c_P, LCM_D, 1.0f);
            float sim = poincare_similarity_c(z_P, c_P, LCM_D, 1.0f);
            if (sim < best_sim) { best_sim = sim; best = j; }
        }
    }

    /* Tangent space projection at the selected prototype. */
    vec_t c_P;
    exp_map_c(C + (size_t)best * LCM_D, c_P, LCM_D, 1.0f);

    /* r = z_P - c_idx (in tangent space) */
    vec_t r, proj;
    for (int i = 0; i < LCM_D; i++) r[i] = z_P[i] - c_P[i];

    /* proj = T @ T.T @ r, with T stored row-major (D, t_dim). */
    const float* T_j = T_space + (size_t)best * LCM_D * t_dim;
    float t_proj[t_dim];
    memset(t_proj, 0, (size_t)t_dim * sizeof(float));
    for (int k = 0; k < t_dim; k++) {
        for (int i = 0; i < LCM_D; i++)
            t_proj[k] += T_j[i * t_dim + k] * r[i];
    }
    for (int i = 0; i < LCM_D; i++) {
        proj[i] = 0.0f;
        for (int k = 0; k < t_dim; k++)
            proj[i] += T_j[i * t_dim + k] * t_proj[k];
    }

    /* c_idx + proj → log_map back to Euclidean */
    vec_t combined;
    for (int i = 0; i < LCM_D; i++) combined[i] = c_P[i] + proj[i];
    log_map_c(combined, out, LCM_D, 1.0f);

    if (idx_out) *idx_out = best;
}

void slide_manifold(const float* z, const lattice_memory_t* mem,
                     const float* T_space, int t_dim, float* out) {
    assert(mem != NULL && T_space != NULL && out != NULL);
    assert(mem->M > 0 && t_dim > 0);

    /* R6 domain: every hyperbolic comparison uses exp_map(safe_unit(z)). */
    vec_t z_u, z_P;
    safe_unit_c(z, z_u, LCM_D, 1e-8f);
    exp_map_c(z_u, z_P, LCM_D, 1.0f);

    manifold_slide_core(z, z_P, (const float*)mem->C, mem->M,
                        T_space, t_dim, NULL, 0.0f, out, NULL);
}

void manifold_forward_c(const float* z, const manifold_mem_t* m,
                        float alpha_val, float* out, int* idx) {
    if (!out) return;
    if (!z || !m || !m->C || !m->T || m->M <= 0 || m->t_dim <= 0) {
        memset(out, 0, sizeof(vec_t));
        if (idx) *idx = 0;
        return;
    }

    vec_t z_u, z_P;
    safe_unit_c(z, z_u, LCM_D, 1e-8f);
    exp_map_c(z_u, z_P, LCM_D, 1.0f);

    manifold_slide_core(z, z_P, m->C, m->M, m->T, m->t_dim,
                        m->value, alpha_val, out, idx);
}

/* ─── Cooley-Tukey radix-2 FFT (iterative, in-place) ────────────────────────
 *
 * O(N log N) — replaces O(N²) DFT for HRR bind/unbind.
 * N must be a power of 2.
 * sign = -1 for forward FFT, +1 for inverse (output scaled by 1/N).
 */

/*@
  requires \valid(re + (0 .. n-1));
  requires \valid(im + (0 .. n-1));
  requires n > 0 && (n & (n - 1)) == 0;
  assigns  re[0 .. n-1], im[0 .. n-1];
 */
static void fft_c2c(float* re, float* im, int n, int sign) {
    /* Bit-reversal permutation */
    int bits = 0;
    while ((1 << bits) < n) bits++;
    /*@
      loop invariant 0 <= i <= n;
      loop assigns i, re[0 .. n-1], im[0 .. n-1];
      loop variant n - i;
     */
    for (int i = 0; i < n; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++) {
            if (i & (1 << b)) rev |= (1 << (bits - 1 - b));
        }
        if (rev < i) {
            float tr = re[i]; re[i] = re[rev]; re[rev] = tr;
            float ti = im[i]; im[i] = im[rev]; im[rev] = ti;
        }
    }

    /* Butterfly stages */
    /*@
      loop invariant 2 <= len || len == n;
      loop assigns len, re[0 .. n-1], im[0 .. n-1];
      loop variant n - len;
     */
    for (int len = 2; len <= n; len <<= 1) {
        float w_re = cosf(sign * (float)M_PI / ((float)len / 2.0f));
        float w_im = sinf(sign * (float)M_PI / ((float)len / 2.0f));
        /*@
          loop invariant 0 <= i <= n;
          loop assigns i, re[0 .. n-1], im[0 .. n-1];
          loop variant n - i;
         */
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            /*@
              loop invariant 0 <= j <= len/2;
              loop assigns j, re[0 .. n-1], im[0 .. n-1];
              loop variant len/2 - j;
             */
            for (int j = 0; j < len / 2; j++) {
                float u_re = re[i + j];
                float u_im = im[i + j];
                float v_re = re[i + j + len / 2] * cur_re
                           - im[i + j + len / 2] * cur_im;
                float v_im = re[i + j + len / 2] * cur_im
                           + im[i + j + len / 2] * cur_re;
                re[i + j]            = u_re + v_re;
                im[i + j]            = u_im + v_im;
                re[i + j + len / 2]  = u_re - v_re;
                im[i + j + len / 2]  = u_im - v_im;
                float nw_re = cur_re * w_re - cur_im * w_im;
                float nw_im = cur_re * w_im + cur_im * w_re;
                cur_re = nw_re;
                cur_im = nw_im;
            }
        }
    }

    /* Scale for inverse */
    if (sign > 0) {
        /*@
          loop invariant 0 <= i <= n;
          loop assigns i, re[0 .. n-1], im[0 .. n-1];
          loop variant n - i;
         */
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
}

/* ─── HRR Bind (绑定) — O(D log D) FFT-based convolution ────────────────────
 *
 * b = Σ IFFT(FFT_norm(k^(i)) ⊙ FFT_norm(v^(j)))
 *
 * Uses iterative Cooley-Tukey radix-2 FFT (D must be power of 2).
 */

/*@
  requires n_keys > 0;
  requires n_vals > 0;
  requires \valid_read(keys + (0 .. n_keys-1));
  requires \valid_read(vals + (0 .. n_vals-1));
  requires \valid(out + (0 .. D-1));
  requires \separated(out, keys) && \separated(out, vals);
  requires \forall integer ki; 0 <= ki < n_keys ==> \valid_read(keys[ki] + (0 .. D-1));
  requires \forall integer vi; 0 <= vi < n_vals ==> \valid_read(vals[vi] + (0 .. D-1));
  requires D > 0 && (D & (D - 1)) == 0;
  ensures  \valid(out + (0 .. D-1));
  assigns  out[0 .. D-1];
 */
void hrr_bind(const float** keys, int n_keys, const float** vals, int n_vals,
              float* out, int D) {
    assert(keys != NULL && vals != NULL && out != NULL);
    assert((D & (D - 1)) == 0); /* D must be power of 2 for FFT */

    float re[D], im[D];
    float k_re[D], k_im[D];
    float accum_re[D], accum_im[D];
    memset(accum_re, 0, sizeof(accum_re));
    memset(accum_im, 0, sizeof(accum_im));

    /*@
      loop invariant 0 <= ki <= n_keys;
      loop assigns ki, re[0 .. D-1], im[0 .. D-1], k_re[0 .. D-1], k_im[0 .. D-1], accum_re[0 .. D-1], accum_im[0 .. D-1];
      loop variant n_keys - ki;
     */
    for (int ki = 0; ki < n_keys; ki++) {
        /* FFT of key[n] → K[f] */
        /*@
          loop invariant 0 <= i <= D;
          loop assigns i, re[0 .. D-1], im[0 .. D-1];
          loop variant D - i;
         */
        for (int i = 0; i < D; i++) { re[i] = keys[ki][i]; im[i] = 0.0f; }
        fft_c2c(re, im, D, -1);

        /* Normalize to unit magnitude per bin */
        /*@
          loop invariant 0 <= f <= D;
          loop assigns f, k_re[0 .. D-1], k_im[0 .. D-1];
          loop variant D - f;
         */
        for (int f = 0; f < D; f++) {
            float mag = sqrtf(re[f] * re[f] + im[f] * im[f]) + 1e-8f;
            /*@ assert mag > 0.0f; */
            k_re[f] = re[f] / mag;
            k_im[f] = im[f] / mag;
        }

        /*@
          loop invariant 0 <= vi <= n_vals;
          loop assigns vi, re[0 .. D-1], im[0 .. D-1], accum_re[0 .. D-1], accum_im[0 .. D-1];
          loop variant n_vals - vi;
         */
        for (int vi = 0; vi < n_vals; vi++) {
            /* FFT of val[n] → V[f] */
            /*@
              loop invariant 0 <= i <= D;
              loop assigns i, re[0 .. D-1], im[0 .. D-1];
              loop variant D - i;
             */
            for (int i = 0; i < D; i++) { re[i] = vals[vi][i]; im[i] = 0.0f; }
            fft_c2c(re, im, D, -1);

            /* Normalize */
            /*@
              loop invariant 0 <= f <= D;
              loop assigns f, re[0 .. D-1], im[0 .. D-1];
              loop variant D - f;
             */
            for (int f = 0; f < D; f++) {
                float mag = sqrtf(re[f] * re[f] + im[f] * im[f]) + 1e-8f;
                /*@ assert mag > 0.0f; */
                re[f] /= mag;
                im[f] /= mag;
            }

            /* Accumulate K[f] * V[f] (elementwise complex multiply) */
            /*@
              loop invariant 0 <= f <= D;
              loop assigns f, accum_re[0 .. D-1], accum_im[0 .. D-1];
              loop variant D - f;
             */
            for (int f = 0; f < D; f++) {
                float r = k_re[f] * re[f] - k_im[f] * im[f];
                float i_val = k_re[f] * im[f] + k_im[f] * re[f];
                accum_re[f] += r;
                accum_im[f] += i_val;
            }
        }
    }

    /* IFFT back to time domain */
    memcpy(re, accum_re, sizeof(accum_re));
    memcpy(im, accum_im, sizeof(accum_im));
    fft_c2c(re, im, D, 1);

    /* Real output (imaginary part should be ~0 for Hermitian input) */
    /*@
      loop invariant 0 <= i <= D;
      loop assigns i, out[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) out[i] = re[i];
}

/* ─── HRR Unbind (解绑) — O(D log D) FFT-based ─────────────────────────────
 *
 * v_approx = IFFT(conj(FFT_norm(k)) ⊙ FFT_norm(b))
 */

/*@
  requires \valid_read(b + (0 .. D-1));
  requires \valid_read(k + (0 .. D-1));
  requires \valid(out + (0 .. D-1));
  requires \separated(out, b) && \separated(out, k);
  requires \separated(out + (0 .. D-1), b + (0 .. D-1));
  requires \separated(out + (0 .. D-1), k + (0 .. D-1));
  requires D > 0 && (D & (D - 1)) == 0;
  ensures  \valid(out + (0 .. D-1));
  assigns  out[0 .. D-1];
 */
void hrr_unbind(const float* b, const float* k, float* out, int D) {
    assert(b != NULL && k != NULL && out != NULL);
    assert((D & (D - 1)) == 0); /* D must be power of 2 for FFT */

    float re[D], im[D];

    /* FFT of k (query) — normalize */
    /*@
      loop invariant 0 <= i <= D;
      loop assigns i, re[0 .. D-1], im[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) { re[i] = k[i]; im[i] = 0.0f; }
    fft_c2c(re, im, D, -1);
    /*@
      loop invariant 0 <= f <= D;
      loop assigns f, re[0 .. D-1], im[0 .. D-1];
      loop variant D - f;
     */
    for (int f = 0; f < D; f++) {
        float mag = sqrtf(re[f] * re[f] + im[f] * im[f]) + 1e-8f;
        /*@ assert mag > 0.0f; */
        re[f] /= mag; im[f] /= mag;
    }
    float k_re[D], k_im[D];
    memcpy(k_re, re, sizeof(k_re));
    memcpy(k_im, im, sizeof(k_im));

    /* FFT of b — normalize */
    /*@
      loop invariant 0 <= i <= D;
      loop assigns i, re[0 .. D-1], im[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) { re[i] = b[i]; im[i] = 0.0f; }
    fft_c2c(re, im, D, -1);
    /*@
      loop invariant 0 <= f <= D;
      loop assigns f, re[0 .. D-1], im[0 .. D-1];
      loop variant D - f;
     */
    for (int f = 0; f < D; f++) {
        float mag = sqrtf(re[f] * re[f] + im[f] * im[f]) + 1e-8f;
        /*@ assert mag > 0.0f; */
        re[f] /= mag; im[f] /= mag;
    }

    /* conj(k) ⊙ b (elementwise complex multiply with conjugate) */
    /*@
      loop invariant 0 <= f <= D;
      loop assigns f, re[0 .. D-1], im[0 .. D-1];
      loop variant D - f;
     */
    for (int f = 0; f < D; f++) {
        float r = k_re[f] * re[f] + k_im[f] * im[f];  /* conj(k) * b */
        float i_val = k_re[f] * im[f] - k_im[f] * re[f];
        re[f] = r; im[f] = i_val;
    }

    /* IFFT back to time domain */
    fft_c2c(re, im, D, 1);

    /* Real output */
    /*@
      loop invariant 0 <= i <= D;
      loop assigns i, out[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) out[i] = re[i];
}
