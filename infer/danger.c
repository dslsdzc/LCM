/* Danger Lattice — Read-only Safety Monitor (LCM_SAFETY_CRITICAL)
 *
 * FORMAL CONTRACT:
 *   - C_threats and C_normal are READ-ONLY after danger_init
 *   - danger_assess is PURE (no side effects)
 *   - danger_assess is THREAD-SAFE
 *
 * SAFETY PROPERTIES DETECTED:
 *   1. Resource abuse (retrieval/step limits exceeded)
 *   2. Deception (local value vs output inconsistency)
 *   3. Pattern match (proximity to known threat patterns)
 *
 * VIOLATION OF ANY → hard interrupt (should_block = true)
 */
#include "lcm.h"
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#define REQUIRE(cond) assert(cond)
#define ENSURE(cond)  assert(cond)

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
    dl->safety_threshold = 0.005f;

    uint32_t hash = 0;
    for (int i = 0; i < M * D; i++) {
        /* Bit-pattern hash: (uint32_t)(float*1e6) is compiler-flag dependent
         * (negative values cast differently, NaN → 0). */
        uint32_t b;
        memcpy(&b, &C_threats[i], sizeof(b)); hash ^= b;
        memcpy(&b, &C_normal[i], sizeof(b));  hash ^= b;
    }
    snprintf(dl->integrity_hash, sizeof(dl->integrity_hash), "%08x", hash);
    ENSURE(dl->integrity_hash[0] != '\0');
}

/* ─── Verify tamper protection ────────────────────────────────────────────── */
bool danger_verify(const danger_lattice_t* dl) {
    REQUIRE(dl != NULL);
    uint32_t hash = 0;
    const float* t = (const float*)dl->C_threats;
    const float* n = (const float*)dl->C_normal;
    for (int i = 0; i < dl->M_danger * dl->D; i++) {
        /* Same bit-pattern hash as danger_init.  (R13 will replace both with
         * one shared SHA-256 implementation.) */
        uint32_t b;
        memcpy(&b, &t[i], sizeof(b)); hash ^= b;
        memcpy(&b, &n[i], sizeof(b)); hash ^= b;
    }
    char cur[64];
    snprintf(cur, sizeof(cur), "%08x", hash);
    return memcmp(cur, dl->integrity_hash, 8) == 0;
}

/* ─── Threat assessment ─────────────────────────────────────────────────────
 *
 * CONTRACT:
 *   Pre:  dl initialized, z has dl->D elements
 *         step_count >= 0, retrieval_count >= 0
 *         All output pointers non-NULL
 *   Post: *out_block true iff threat detected
 *         *out_threat contains type
 *         *out_score = relative_gap for pattern match, 0 otherwise
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

    /* 1. Resource abuse (highest priority per formal spec: Z3/Nagini order) */
    if (retrieval_count > LCM_MAX_RETRIEVALS) {
        *out_score = 1.0f; *out_threat = THREAT_RESOURCE_ABUSE; *out_block = true;
        return;
    }
    /* Runaway threshold follows the session's configured max_steps
     * (previously hardcoded LCM_MAX_STEPS=32, which false-triggered
     * when Python drives max_steps > 32).  Task 10 moves this to the
     * scheduler; it stays here until then. */
    if (step_count > max_steps) {
        *out_score = 1.0f; *out_threat = THREAT_RUNAWAY; *out_block = true;
        return;
    }

    /* 2. Value consistency (deception detection) */
    if (value_consistency < LCM_CONSISTENCY_THRESHOLD) {
        *out_score = 2.0f - value_consistency;
        *out_threat = THREAT_DECEPTION; *out_block = true;
        return;
    }

    /* 3. Pattern matching in the unified Poincaré domain */
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
        float relative_gap = (sim_normal - sim_threat) /
                             (sim_normal + sim_threat + 1e-8f);
        if (relative_gap > dl->safety_threshold) {
            *out_score = relative_gap;
            *out_threat = THREAT_PATTERN_MATCH;
            *out_block = true;
            return;
        }
    }

    /* No threat */
    *out_score = 0.0f;
    *out_threat = THREAT_NONE;
    *out_block = false;
}

