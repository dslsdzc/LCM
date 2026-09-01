/* Hyperbolic operations — Poincaré ball model (C implementation) */

#include "lcm.h"
#include <math.h>
#include <assert.h>

/* ─── Similarity (monotonically equivalent to distance) ──────────────────── */

/*@
  requires D > 0;
  requires c > 0.0f;
  requires \valid_read(u + (0 .. D-1));
  requires \valid_read(v + (0 .. D-1));
  requires \forall integer i; 0 <= i < D ==> \is_finite(u[i]);
  requires \forall integer i; 0 <= i < D ==> \is_finite(v[i]);
  ensures \result >= 0.0f;
  assigns \nothing;
 */
float poincare_similarity_c(const float* u, const float* v, int D, float c) {
    assert(u != NULL && v != NULL);
    assert(D > 0 && c > 0);

    float u_norm2 = 0.0f, v_norm2 = 0.0f, diff_norm2 = 0.0f;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant u_norm2 >= 0.0f;
      loop invariant v_norm2 >= 0.0f;
      loop invariant diff_norm2 >= 0.0f;
      loop assigns i, u_norm2, v_norm2, diff_norm2;
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) {
        u_norm2 += u[i] * u[i];
        v_norm2 += v[i] * v[i];
        float d = u[i] - v[i];
        /*@ assert d*d >= 0.0f; */
        diff_norm2 += d * d;
    }

    float denom = (1.0f - c * u_norm2) * (1.0f - c * v_norm2);
    if (denom < 1e-8f) denom = 1e-8f;
    /*@ assert denom > 0.0f; */
    return 2.0f * c * diff_norm2 / denom;
}

/* ─── True hyperbolic distance (for human-readable output only) ──────────── */

/*@
  requires D > 0;
  requires c > 0.0f;
  requires \valid_read(u + (0 .. D-1));
  requires \valid_read(v + (0 .. D-1));
  requires \forall integer i; 0 <= i < D ==> \is_finite(u[i]);
  requires \forall integer i; 0 <= i < D ==> \is_finite(v[i]);
  ensures \result >= 0.0f;
  assigns \nothing;
 */
float poincare_distance_c(const float* u, const float* v, int D, float c) {
    float sim = poincare_similarity_c(u, v, D, c);
    /*@ assert sim >= 0.0f; */
    float arg = 1.0f + sim + 1e-8f;
    /*@ assert arg >= 1.0f + 1e-8f; */
    /*@ assert arg >= 1.0f; */
    return acoshf(arg);
}

/* ─── Exponential map: Euclidean → Poincaré ball ─────────────────────────── */

/*@
  requires D > 0;
  requires \valid_read(x + (0 .. D-1));
  requires \valid(out + (0 .. D-1));
  requires \separated(out + (0 .. D-1), x + (0 .. D-1));
  requires \forall integer i; 0 <= i < D ==> \is_finite(x[i]);
  ensures \forall integer i; 0 <= i < D ==> \is_finite(out[i]);
  assigns out[0 .. D-1];
 */
void exp_map_c(const float* x, float* out, int D, float c) {
    assert(x != NULL && out != NULL && D > 0);
    float norm = 1e-8f;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant norm >= 1e-8f;
      loop assigns i, norm;
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) norm += x[i] * x[i];
    /*@ assert norm >= 1e-8f; */
    norm = sqrtf(norm);
    /*@ assert norm >= 0.0f; */
    float t = sqrtf(c) * norm;
    float scale = tanhf(t) / t;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant \forall integer k; 0 <= k < i ==> \is_finite(out[k]);
      loop assigns i, out[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) out[i] = scale * x[i];
}

/* ─── Logarithmic map: Poincaré ball → Euclidean ─────────────────────────── */

/*@
  requires D > 0;
  requires \valid_read(y + (0 .. D-1));
  requires \valid(out + (0 .. D-1));
  requires \separated(out + (0 .. D-1), y + (0 .. D-1));
  requires \forall integer i; 0 <= i < D ==> \is_finite(y[i]);
  ensures \forall integer i; 0 <= i < D ==> \is_finite(out[i]);
  assigns out[0 .. D-1];
 */
void log_map_c(const float* y, float* out, int D, float c) {
    assert(y != NULL && out != NULL && D > 0);
    float norm = 1e-8f;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant norm >= 1e-8f;
      loop assigns i, norm;
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) norm += y[i] * y[i];
    /*@ assert norm >= 1e-8f; */
    norm = sqrtf(norm);
    /* Clip in the atanh domain: |t| = sqrtf(c) * n_clip < 1, so the cap is
     * 0.999/sqrtf(c), not 0.999 (which overflows to NaN when c > 1.001). */
    float n_clip = norm < 0.999f / sqrtf(c) ? norm : 0.999f / sqrtf(c);
    /*@ assert n_clip > 0.0f; */
    float t = sqrtf(c) * n_clip;
    /*@ assert t < 1.0f; */  /* atanh domain: |t| < 1 */
    float scale = atanhf(t) / t;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant \forall integer k; 0 <= k < i ==> \is_finite(out[k]);
      loop assigns i, out[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) out[i] = scale * y[i];
}

/* ─── Möbius addition ────────────────────────────────────────────────────── */

/*@
  requires D > 0;
  requires \valid_read(u + (0 .. D-1));
  requires \valid_read(v + (0 .. D-1));
  requires \valid(out + (0 .. D-1));
  requires \separated(out, u) && \separated(out, v);
  requires \separated(out + (0 .. D-1), u + (0 .. D-1));
  requires \separated(out + (0 .. D-1), v + (0 .. D-1));
  requires \forall integer i; 0 <= i < D ==> \is_finite(u[i]) && \is_finite(v[i]);
  assigns out[0 .. D-1];
 */
void mobius_add_c(const float* u, const float* v, float* out, int D, float c) {
    assert(u != NULL && v != NULL && out != NULL && D > 0);
    float u_norm2 = 0.0f, v_norm2 = 0.0f, uv = 0.0f;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant u_norm2 >= 0.0f;
      loop invariant v_norm2 >= 0.0f;
      loop assigns i, u_norm2, v_norm2, uv;
      loop variant D - i;
     */
    for (int i = 0; i < D; i++) {
        u_norm2 += u[i] * u[i];
        v_norm2 += v[i] * v[i];
        uv += u[i] * v[i];
    }
    float denom = 1.0f + 2.0f * c * uv + c * c * u_norm2 * v_norm2;
    if (denom < 1e-8f) denom = 1e-8f;
    /*@ assert denom > 0.0f; */
    float coeff_u = (1.0f + 2.0f * c * uv + c * v_norm2) / denom;
    float coeff_v = (1.0f - c * u_norm2) / denom;
    /*@
      loop invariant 0 <= i <= D;
      loop invariant \forall integer k; 0 <= k < i ==> \is_finite(out[k]);
      loop assigns i, out[0 .. D-1];
      loop variant D - i;
     */
    for (int i = 0; i < D; i++)
        out[i] = coeff_u * u[i] + coeff_v * v[i];
}
