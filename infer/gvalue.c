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

/* R13: one shared SHA-256 over both codebooks.  Each array is hashed as its
 * raw float32 bytes, then the two 32-byte digests are hashed together. */
static void hash_anchors(const float* pos, const float* neg, int n_floats,
                         char out[65]) {
    unsigned char h_pos[32], h_neg[32], both[64];
    lcm_sha256(pos, (size_t)n_floats * sizeof(float), h_pos);
    lcm_sha256(neg, (size_t)n_floats * sizeof(float), h_neg);
    memcpy(both, h_pos, sizeof(h_pos));
    memcpy(both + sizeof(h_pos), h_neg, sizeof(h_neg));
    lcm_sha256_hex(both, sizeof(both), out);
}

/* Saved-file digest: SHA-256 over the raw C_pos/C_neg payload. This is the
 * same byte stream the Python exporter appends to the file and records in
 * manifest.json, so the bridge can compare it against the export digest
 * instead of trusting self-consistency alone. */
static void hash_source(const float* pos, const float* neg, int n_floats,
                        char out[65]) {
    float raw[2 * LCM_N_VALUE_PAIRS * LCM_D];
    memcpy(raw, pos, (size_t)n_floats * sizeof(float));
    memcpy(raw + n_floats, neg, (size_t)n_floats * sizeof(float));
    lcm_sha256_hex(raw, (size_t)(2 * n_floats) * sizeof(float), out);
}

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

    /* R13: SHA-256 over the cached ball anchors (all LCM_N_VALUE_PAIRS * D
     * elements each, flat row-major).  gvalue_verify() hashes the same cached
     * arrays with the same algorithm. */
    hash_anchors(&gv->C_pos[0][0], &gv->C_neg[0][0],
                 LCM_N_VALUE_PAIRS * D, gv->integrity_hash);
    hash_source(C_pos, C_neg, LCM_N_VALUE_PAIRS * D, gv->source_hash);

    ENSURE(gv->integrity_hash[0] != '\0');
    ENSURE(gv->source_hash[0] != '\0');
}

/* ─── Verify integrity ────────────────────────────────────────────────────── */
bool gvalue_verify(const gvalue_t* gv) {
    REQUIRE(gv != NULL);
    /* Same hashing as gvalue_init: both cached anchor arrays, flat row-major. */
    char cur[65];
    hash_anchors(&gv->C_pos[0][0], &gv->C_neg[0][0],
                 LCM_N_VALUE_PAIRS * gv->D, cur);
    return strcmp(cur, gv->integrity_hash) == 0
        && gv->source_hash[0] != '\0';
}

/* Compare the digest captured from the raw file payload at init with the
 * exporter's saved digest. NULL means "no saved digest supplied"; the
 * bridge then relies on gvalue_verify's self-consistency plus the Python
 * file/manifest checks. */
bool gvalue_verify_digest(const gvalue_t* gv, const char* expected_hex) {
    REQUIRE(gv != NULL);
    if (expected_hex == NULL) return true;
    return strlen(expected_hex) == 64
        && strcmp(gv->source_hash, expected_hex) == 0;
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
