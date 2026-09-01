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
        hash ^= (uint32_t)(C_threats[i] * 1e6f);
        hash ^= (uint32_t)(C_normal[i] * 1e6f);
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
        hash ^= (uint32_t)(t[i] * 1e6f);
        hash ^= (uint32_t)(n[i] * 1e6f);
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
 *         *out_score = sim_normal - sim_threat (pattern match) or derived
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
     * when Python drives max_steps > 32). */
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

    /* 3. Pattern matching via hyperbolic similarity */
    {
        float sim_threat = 1e20f, sim_normal = 1e20f;
        for (int i = 0; i < dl->M_danger; i++) {
            float s_t = poincare_similarity_c(z, (const float*)(dl->C_threats + i),
                                              dl->D, 1.0f);
            float s_n = poincare_similarity_c(z, (const float*)(dl->C_normal + i),
                                              dl->D, 1.0f);
            if (s_t < sim_threat) sim_threat = s_t;
            if (s_n < sim_normal) sim_normal = s_n;
        }
        /* poincare_similarity_c is a distance-monotonic quantity (smaller =
         * closer): being near a threat pattern yields small sim_threat and
         * large sim_normal, so danger_score = sim_normal - sim_threat is
         * positive when the query is close to a threat pattern. */
        float danger_score = sim_normal - sim_threat;
        if (danger_score > dl->safety_threshold) {
            *out_score = danger_score;
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
