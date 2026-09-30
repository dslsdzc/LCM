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
/* ─── HRQ: hierarchical residual quantization on the Poincaré ball ─────────
 *
 * Mirrors train/lattices.py::hrq_forward in the configuration the parity
 * harness runs: value_scalars are all zero (their init is jnp.zeros), so
 * `value_biased_retrieve`'s score reduces to `-dist2` and the top-level
 * selection is a *Euclidean* argmin over the tangent-space codebook, retrieving
 * the ball prototype. When alpha_val > 0 the score is genuinely value-biased
 * and that branch would need value_scalars plumbed in here too.
 *
 * Note the two different metrics, which is deliberate and easy to get wrong:
 *   - top-level SELECTION is Euclidean (over top_C)
 *   - the fallback DECISION and the fine-layer selection are Poincaré
 *     (poincare_similarity, distance-monotone: nearest = argmin)
 *
 * `top_C` is (M_top, D) raw; `fine_C` is (n_fine * M_fine, D) raw. Both are
 * mapped into the ball internally, layer by layer, rather than materialised up
 * front — a (512, 256) ball copy is 512 KB and does not belong on the stack.
 *
 * Returns 0 on success, -1 on bad arguments.
 */
int hrq_forward_c(const float* z, int D,
                  const float* top_C, int M_top,
                  const float* fine_C, int M_fine, int n_fine,
                  float tau_fallback, float* out) {
    if (!z || !top_C || !out || D <= 0 || M_top <= 0) return -1;
    if (n_fine > 0 && (!fine_C || M_fine <= 0)) return -1;

    const float c = 1.0f;
    vec_t z_P, c_p, r, t1, t2, t3, acc, neg;

    exp_map_c(z, z_P, D, c);

    /* Pass 1: the two smallest similarities (ascending; closest first), the
     * Poincaré argmin, and separately the Euclidean argmin over top_C. */
    float min1 = 1e20f, min2 = 1e20f;
    int euclid_idx = 0;
    float euclid_best = 1e20f;

    for (int m = 0; m < M_top; m++) {
        const float* row = top_C + (size_t)m * D;
        exp_map_c(row, c_p, D, c);
        float s = poincare_similarity_c(z_P, c_p, D, c);
        if (s < min1) { min2 = min1; min1 = s; }
        else if (s < min2) { min2 = s; }

        float d2 = 0.0f;
        for (int i = 0; i < D; i++) {
            float diff = z[i] - row[i];
            d2 += diff * diff;
        }
        if (d2 < euclid_best) { euclid_best = d2; euclid_idx = m; }
    }

    /* Top prototype in the ball: the Euclidean winner, mapped. */
    exp_map_c(top_C + (size_t)euclid_idx * D, c_p, D, c);
    memcpy(t3, c_p, sizeof(float) * (size_t)D);   /* c_top */

    /* Fallback when the top-1 / top-2 similarity gap is small. */
    int use_fallback = (min2 - min1) < tau_fallback;

    if (use_fallback) {
        /* c_top_w = sum_m softmax(-sims)_m * C_top_P[m], computed in one pass
         * (numerator vector and denominator accumulated together) so the
         * ball-mapped rows need not be stored. */
        float den = 0.0f;
        memset(acc, 0, sizeof(vec_t));
        for (int m = 0; m < M_top; m++) {
            exp_map_c(top_C + (size_t)m * D, c_p, D, c);
            float s = poincare_similarity_c(z_P, c_p, D, c);
            float e = expf(-s + min1);          /* max(-sims) == -min1 */
            den += e;
            for (int i = 0; i < D; i++) acc[i] += e * c_p[i];
        }
        if (den <= 0.0f) return -1;
        for (int i = 0; i < D; i++) t3[i] = acc[i] / den;   /* c_top_w */
    }

    /* t1 = z_P (-) c_top. mobius_add takes v directly, so pass -c_top. */
    for (int i = 0; i < D; i++) neg[i] = -t3[i];
    mobius_add_c(z_P, neg, t1, D, c);

    /* fine_residual: r = t1; per layer r = r (+) (-C_fb_P[argmin sim]) */
    memcpy(r, t1, sizeof(float) * (size_t)D);
    for (int l = 0; l < n_fine; l++) {
        const float* layer = fine_C + (size_t)l * M_fine * D;
        float best = 1e20f;
        int best_m = 0;
        for (int m = 0; m < M_fine; m++) {
            exp_map_c(layer + (size_t)m * D, c_p, D, c);
            float s = poincare_similarity_c(r, c_p, D, c);
            if (s < best) { best = s; best_m = m; }
        }
        exp_map_c(layer + (size_t)best_m * D, c_p, D, c);
        for (int i = 0; i < D; i++) neg[i] = -c_p[i];
        mobius_add_c(r, neg, t2, D, c);
        memcpy(r, t2, sizeof(float) * (size_t)D);
    }

    /* out = log_map( c_top (+) fine_residual ) */
    mobius_add_c(t3, r, t2, D, c);
    log_map_c(t2, out, D, c);
    return 0;
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
  requires \valid_read(mem) && \valid_read(mem->C);
  requires \valid_read(T_space + (0 .. mem->M * LCM_D * t_dim - 1));
  requires \valid(out + (0 .. LCM_D-1));
  requires \separated(out, z) && \separated(out, T_space);
  requires mem->M > 0 && t_dim > 0;
  requires \forall integer j; 0 <= j < mem->M ==> \valid_read(mem->C[j] + (0 .. LCM_D-1));
  ensures  \valid(out + (0 .. LCM_D-1));
  assigns  out[0 .. LCM_D-1];
 */
void slide_manifold(const float* z, const lattice_memory_t* mem,
                     const float* T_space, int t_dim, float* out) {
    assert(mem != NULL && T_space != NULL && out != NULL);
    assert(mem->M > 0 && t_dim > 0);

    /* Step 1: exp_map to Poincaré ball */
    vec_t z_P;
    exp_map_c(z, z_P, LCM_D, 1.0f);

    /* Step 2: Find nearest codebook entry using Poincaré similarity */
    float best_sim = 1e20f;
    int   best_idx = 0;
    /*@
      loop invariant 0 <= j <= mem->M;
      loop invariant 0 <= best_idx < mem->M || best_idx == 0;
      loop assigns j, best_sim, best_idx;
      loop variant mem->M - j;
     */
    for (int j = 0; j < mem->M; j++) {
        vec_t c_P;
        exp_map_c(mem->C[j], c_P, LCM_D, 1.0f);
        float sim = poincare_similarity_c(z_P, c_P, LCM_D, 1.0f);
        if (sim < best_sim) {
            best_sim = sim;
            best_idx = j;
        }
    }

    /* Step 3: Tangent space projection */
    vec_t c_P;
    exp_map_c(mem->C[best_idx], c_P, LCM_D, 1.0f);

    /* r = z_P - c_idx (in tangent space) */
    vec_t r, proj;
    /*@
      loop invariant 0 <= i <= LCM_D;
      loop assigns i, r[0 .. LCM_D-1];
      loop variant LCM_D - i;
     */
    for (int i = 0; i < LCM_D; i++) r[i] = z_P[i] - c_P[i];

    /* proj = T @ T.T @ r */
    const float* T_j = T_space + best_idx * LCM_D * t_dim;
    float t_proj[t_dim];
    memset(t_proj, 0, t_dim * sizeof(float));
    /*@
      loop invariant 0 <= k <= t_dim;
      loop assigns k, t_proj[0 .. t_dim-1];
      loop variant t_dim - k;
     */
    for (int k = 0; k < t_dim; k++) {
        /*@
          loop invariant 0 <= i <= LCM_D;
          loop assigns i;
          loop variant LCM_D - i;
         */
        for (int i = 0; i < LCM_D; i++) {
            /* T_space is (M_man, D, t_dim) row-major (matches training
             * layout), so entry (i, k) is at flat offset i * t_dim + k. */
            t_proj[k] += T_j[i * t_dim + k] * r[i];
        }
    }
    /*@
      loop invariant 0 <= i <= LCM_D;
      loop assigns i, proj[0 .. LCM_D-1];
      loop variant LCM_D - i;
     */
    for (int i = 0; i < LCM_D; i++) {
        proj[i] = 0;
        /*@
          loop invariant 0 <= k <= t_dim;
          loop assigns k;
          loop variant t_dim - k;
         */
        for (int k = 0; k < t_dim; k++) {
            /* Same (D, t_dim) row-major indexing as the projection above. */
            proj[i] += T_j[i * t_dim + k] * t_proj[k];
        }
    }

    /* Step 4: c_idx + proj → log_map back to Euclidean */
    vec_t combined;
    /*@
      loop invariant 0 <= i <= LCM_D;
      loop assigns i, combined[0 .. LCM_D-1];
      loop variant LCM_D - i;
     */
    for (int i = 0; i < LCM_D; i++) combined[i] = c_P[i] + proj[i];
    log_map_c(combined, out, LCM_D, 1.0f);
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
