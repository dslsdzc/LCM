/* Danger Lattice — Read-only Safety Monitor (LCM_SAFETY_CRITICAL)
 *
 * FORMAL CONTRACT:
 *   - C_threats and C_normal are READ-ONLY after danger_init
 *   - danger_assess is PURE (no side effects)
 *   - danger_assess is THREAD-SAFE
 *
 * SAFETY PROPERTIES DETECTED:
 *   1. Resource abuse (retrieval limits exceeded)
 *   2. Deception (local value vs output inconsistency)
 *   3. Pattern match (relative proximity to known threat patterns)
 *
 * VIOLATION OF ANY → hard interrupt (should_block = true)
 *
 * Runaway (step_count > max_steps) is intentionally NOT handled here: it is a
 * scheduler-level budget check (Task 10).  danger_assess keeps max_steps in its
 * signature for API compatibility but ignores it.
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

/* R13: one shared SHA-256 over both codebooks.  Each array is hashed as its
 * raw float32 bytes, then the two 32-byte digests are hashed together. */
static void hash_codebooks(const float* threats, const float* normal,
                           int n_floats, char out[65]) {
    unsigned char h_t[32], h_n[32], both[64];
    lcm_sha256(threats, (size_t)n_floats * sizeof(float), h_t);
    lcm_sha256(normal, (size_t)n_floats * sizeof(float), h_n);
    memcpy(both, h_t, sizeof(h_t));
    memcpy(both + sizeof(h_t), h_n, sizeof(h_n));
    lcm_sha256_hex(both, sizeof(both), out);
}

/* ─── Initialize ────────────────────────────────────────────────────────────
 *
 * C_threats, C_normal: flat arrays of M * D floats.
 * Zero-copy: references the original memory.
 */
void danger_init(danger_lattice_t* dl, const float* C_threats,
                 const float* C_normal, int M, int D) {
    REQUIRE(dl != NULL && C_threats != NULL && C_normal != NULL);
    REQUIRE(M > 0 && D == LCM_D);

    dl->C_threats = (const vec_t*)C_threats;
    dl->C_normal  = (const vec_t*)C_normal;
    dl->M_danger  = M;
    dl->D         = D;
    dl->safety_threshold = 0.05f;  /* R7: relative-gap threshold */

    hash_codebooks((const float*)dl->C_threats, (const float*)dl->C_normal,
                   M * D, dl->integrity_hash);
    ENSURE(dl->integrity_hash[0] != '\0');
}

/* ─── Verify tamper protection ────────────────────────────────────────────── */
bool danger_verify(const danger_lattice_t* dl) {
    REQUIRE(dl != NULL);
    char cur[65];
    hash_codebooks((const float*)dl->C_threats, (const float*)dl->C_normal,
                   dl->M_danger * dl->D, cur);
    return strcmp(cur, dl->integrity_hash) == 0;
}

/* ─── Threat assessment ─────────────────────────────────────────────────────
 *
 * CONTRACT:
 *   Pre:  dl initialized, z has dl->D elements
 *         retrieval_count >= 0, value_consistency finite
 *         All output pointers non-NULL
 *   Post: *out_block true iff a threat condition is detected
 *         *out_threat contains the triggered type
 *         *out_score = 1.0 for resource abuse, 2.0 - value_consistency for
 *                      deception, and relative_gap for pattern match or no
 *                      threat (bounded in [-1, 1])
 *
 * Pattern matching (R7) runs in the unified Poincaré domain and compares a
 * bounded relative gap:
 *   sim_threat = min_j sim(z_P, C_threats_P[j])
 *   sim_normal = min_j sim(z_P, C_normal_P[j])
 *   relative_gap = (sim_normal - sim_threat) /
 *                  (sim_normal + sim_threat + eps)
 *   should_block = relative_gap > safety_threshold
 * A neutral input (sim_threat == sim_normal) has gap 0 and is not blocked.
 */
void danger_assess(const danger_lattice_t* dl, const float* z,
                    int step_count, int retrieval_count, int max_steps,
                    float value_consistency,
                    float* out_score, int* out_threat, bool* out_block) {
    REQUIRE(dl != NULL && z != NULL);
    REQUIRE(out_score != NULL && out_threat != NULL && out_block != NULL);
    (void)step_count;
    (void)max_steps;  /* scheduler-level runaway check (Task 10) */

    /* 1. Resource abuse (highest priority per formal spec: Z3/Nagini order) */
    if (retrieval_count > LCM_MAX_RETRIEVALS) {
        *out_score = 1.0f; *out_threat = THREAT_RESOURCE_ABUSE; *out_block = true;
        return;
    }

    /* 2. Value consistency (deception detection) */
    if (value_consistency < LCM_CONSISTENCY_THRESHOLD) {
        *out_score = 2.0f - value_consistency;
        *out_threat = THREAT_DECEPTION; *out_block = true;
        return;
    }

    /* 3. Pattern matching in the unified Poincaré domain */
    float relative_gap = 0.0f;
    {
        vec_t z_unit, z_P;
        safe_unit_c(z, z_unit, dl->D, 1e-8f);
        exp_map_c(z_unit, z_P, dl->D, 1.0f);

        vec_t a_unit, a_P;
        float sim_threat = 1e20f, sim_normal = 1e20f;
        for (int i = 0; i < dl->M_danger; i++) {
            safe_unit_c((const float*)(dl->C_threats + i),
                        a_unit, dl->D, 1e-8f);
            exp_map_c(a_unit, a_P, dl->D, 1.0f);
            float s_t = poincare_similarity_c(z_P, a_P, dl->D, 1.0f);
            if (s_t < sim_threat) sim_threat = s_t;

            safe_unit_c((const float*)(dl->C_normal + i),
                        a_unit, dl->D, 1e-8f);
            exp_map_c(a_unit, a_P, dl->D, 1.0f);
            float s_n = poincare_similarity_c(z_P, a_P, dl->D, 1.0f);
            if (s_n < sim_normal) sim_normal = s_n;
        }
        /* poincare_similarity_c is distance-monotonic (smaller = closer):
         * being near a threat pattern yields small sim_threat and large
         * sim_normal, so the gap is positive for a threat. */
        relative_gap = (sim_normal - sim_threat) /
                       (sim_normal + sim_threat + 1e-8f);
        if (relative_gap > dl->safety_threshold) {
            *out_score = relative_gap;
            *out_threat = THREAT_PATTERN_MATCH;
            *out_block = true;
            return;
        }
    }

    /* No threat */
    *out_score = relative_gap;
    *out_threat = THREAT_NONE;
    *out_block = false;
}
