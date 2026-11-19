/* Global Value Lattice — Immutable Three-Laws Safety Substrate
 *
 * FORMAL CONTRACT:
 *   The global value codebook is READ-ONLY after initialization.
 *   gvalue_verify() MUST be called before first use.
 *   check_safety is PURE (no side effects, deterministic).
 *
 * Safety invariant (R6 unified Poincaré domain, strict Three Laws):
 *   z_P     = exp_map(safe_unit(z))
 *   pos_sim = min_j poincare_similarity(z_P, C_pos_P[j])
 *   neg_sim = min_j poincare_similarity(z_P, C_neg_P[j])
 *   balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + eps)
 *   safe   ⇔ balance <= -safety_margin
 *
 *   C_pos_P / C_neg_P are the anchors projected into the ball once by
 *   gvalue_init.  A neutral input (balance = 0) is safe with margin 0 and a
 *   violation with margin > 0 — the strict reading confirmed for R6.
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

/* Relative-balance denominator guard (matches train/gvalue.py). */
#define GVALUE_BALANCE_EPS 1e-8f

/* ─── Initialize global value codebook ──────────────────────────────────────
 *
 * Pre:  C_pos, C_neg hold LCM_N_VALUE_PAIRS * D floats each, D == LCM_D
 * Post: anchors are projected to the Poincaré ball and cached; integrity_hash
 *       is computed over the cached (read-only) anchors
 */
void gvalue_init(gvalue_t* gv, const float* C_pos, const float* C_neg, int D) {
    REQUIRE(gv != NULL && C_pos != NULL && C_neg != NULL);
    REQUIRE(D == LCM_D);

    gv->D = D;

    /* R6: cache exp_map(safe_unit(anchor)) so check_safety compares points in
     * one domain.  exp_map(safe_unit(·)) is idempotent, so a future loader
     * that pre-projects before calling gvalue_init is harmless. */
    vec_t unit;
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        safe_unit_c(C_pos + i * D, unit, D, 1e-8f);
        exp_map_c(unit, gv->C_pos[i], D, 1.0f);
        safe_unit_c(C_neg + i * D, unit, D, 1e-8f);
        exp_map_c(unit, gv->C_neg[i], D, 1.0f);
    }

    uint32_t hash = 0;
    /* Bit-pattern hash over the cached ball anchors (all LCM_N_VALUE_PAIRS * D
     * elements each, flat row-major).  gvalue_verify() hashes the same cached
     * arrays with the same algorithm. */
    const float* p = &gv->C_pos[0][0];
    const float* n = &gv->C_neg[0][0];
    for (int i = 0; i < LCM_N_VALUE_PAIRS * D; i++) {
        uint32_t b;
        memcpy(&b, &p[i], sizeof(b)); hash ^= b;
        memcpy(&b, &n[i], sizeof(b)); hash ^= b;
    }
    snprintf(gv->integrity_hash, sizeof(gv->integrity_hash), "%08x", hash);

    ENSURE(gv->integrity_hash[0] != '\0');
}

/* ─── Verify integrity ────────────────────────────────────────────────────── */
bool gvalue_verify(const gvalue_t* gv) {
    REQUIRE(gv != NULL);
    uint32_t hash = 0;
    /* Same hashing as gvalue_init: both cached anchor arrays, flat row-major. */
    const float* p = &gv->C_pos[0][0];
    const float* n = &gv->C_neg[0][0];
    for (int i = 0; i < LCM_N_VALUE_PAIRS * gv->D; i++) {
        uint32_t b;
        memcpy(&b, &p[i], sizeof(b)); hash ^= b;
        memcpy(&b, &n[i], sizeof(b)); hash ^= b;
    }
    char cur[64];
    snprintf(cur, sizeof(cur), "%08x", hash);
    return memcmp(cur, gv->integrity_hash, 8) == 0;
}

/* ─── Strict Three Laws safety check ────────────────────────────────────────
 *
 * CONTRACT:
 *   Pre:  gv initialized & verified, z has gv->D elements, margin >= 0
 *   Post: true iff balance <= -margin; *out_violated_law = argmin over law
 *         pairs of -balance_i when unsafe, else -1
 */
bool gvalue_check_safety(const gvalue_t* gv, const float* z,
                          float safety_margin, int* out_violated_law) {
    REQUIRE(gv != NULL && z != NULL);
    REQUIRE(safety_margin >= 0.0f);

    /* Unified domain: z_P = exp_map(safe_unit(z)) */
    vec_t z_unit, z_P;
    safe_unit_c(z, z_unit, gv->D, 1e-8f);
    exp_map_c(z_unit, z_P, gv->D, 1.0f);

    float pos_sims[LCM_N_VALUE_PAIRS];
    float neg_sims[LCM_N_VALUE_PAIRS];
    float pos_sim = 1e20f, neg_sim = 1e20f;
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        pos_sims[i] = poincare_similarity_c(
            z_P, gv->C_pos[i], gv->D, 1.0f);
        neg_sims[i] = poincare_similarity_c(
            z_P, gv->C_neg[i], gv->D, 1.0f);
        if (pos_sims[i] < pos_sim) pos_sim = pos_sims[i];
        if (neg_sims[i] < neg_sim) neg_sim = neg_sims[i];
    }
    ENSURE(pos_sim >= 0.0f && neg_sim >= 0.0f);

    float balance = (pos_sim - neg_sim) /
                    (pos_sim + neg_sim + GVALUE_BALANCE_EPS);

    if (balance > -safety_margin) {
        /* Report the law pair with the smallest margin (-balance_i). */
        float min_margin = 1e20f;
        int   min_idx = 0;
        for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
            float pair_balance = (pos_sims[i] - neg_sims[i]) /
                (pos_sims[i] + neg_sims[i] + GVALUE_BALANCE_EPS);
            float margin = -pair_balance;
            if (margin < min_margin) { min_margin = margin; min_idx = i; }
        }
        if (out_violated_law) *out_violated_law = min_idx;
        return false;
    }
    if (out_violated_law) *out_violated_law = -1;
    return true;
}
