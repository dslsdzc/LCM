/* Global Value Lattice — Immutable Three-Laws Safety Substrate
 *
 * FORMAL CONTRACT:
 *   The global value codebook is READ-ONLY after initialization.
 *   gvalue_verify() MUST be called before first use.
 *   check_safety is PURE (no side effects, deterministic).
 *
 * Safety invariant:
 *   For any input z, the system maintains:
 *     pos_d_min(z) < neg_d_min(z) - safety_margin
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

/* ─── Initialize global value codebook ──────────────────────────────────────
 *
 * Pre:  C_pos, C_neg hold LCM_N_VALUE_PAIRS * D floats each, D == LCM_D
 * Post: integrity_hash is computed; vectors are copied deep
 */
void gvalue_init(gvalue_t* gv, const float* C_pos, const float* C_neg, int D) {
    REQUIRE(gv != NULL && C_pos != NULL && C_neg != NULL);
    REQUIRE(D == LCM_D);

    memcpy(gv->C_pos, C_pos, LCM_N_VALUE_PAIRS * D * sizeof(float));
    memcpy(gv->C_neg, C_neg, LCM_N_VALUE_PAIRS * D * sizeof(float));
    gv->D = D;

    uint32_t hash = 0;
    /* Bit-pattern hash over both anchor arrays (all LCM_N_VALUE_PAIRS * D
     * elements each, flat row-major): (uint32_t)(float*1e6) is
     * compiler-flag dependent (negative casts, NaN → 0). */
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
    /* Same hashing as gvalue_init: both anchor arrays, flat row-major. */
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

/* ─── Safety check with relative margin ──────────────────────────────────────
 *
 * CONTRACT:
 *   Pre:  gv initialized & verified, z has gv->D elements, margin >= 0
 *   Post: true if safe; *out_violated_law set to priority (0=highest) or -1
 */
bool gvalue_check_safety(const gvalue_t* gv, const float* z,
                          float safety_margin, int* out_violated_law) {
    REQUIRE(gv != NULL && z != NULL);
    REQUIRE(safety_margin >= 0.0f);

    float pos_d_min = 1e20f, neg_d_min = 1e20f;
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        float d_pos = poincare_distance_c(z, gv->C_pos[i], gv->D, 1.0f);
        float d_neg = poincare_distance_c(z, gv->C_neg[i], gv->D, 1.0f);
        if (d_pos < pos_d_min) pos_d_min = d_pos;
        if (d_neg < neg_d_min) neg_d_min = d_neg;
    }
    ENSURE(pos_d_min >= 0.0f && neg_d_min >= 0.0f);

    if (pos_d_min > neg_d_min - safety_margin) {
        /* Find which law pair is closest to violation */
        float min_margin = 1e20f;
        int   min_idx = 0;
        for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
            float d_pos = poincare_distance_c(z, gv->C_pos[i], gv->D, 1.0f);
            float d_neg = poincare_distance_c(z, gv->C_neg[i], gv->D, 1.0f);
            float margin = d_pos - (d_neg - safety_margin);
            if (margin < min_margin) { min_margin = margin; min_idx = i; }
        }
        if (out_violated_law) *out_violated_law = min_idx;
        return false;
    }
    if (out_violated_law) *out_violated_law = -1;
    return true;
}
