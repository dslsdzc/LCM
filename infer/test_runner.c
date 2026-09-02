/* LCM inference engine — unit tests (make test).
 *
 * Regression coverage for the correctness audit fixes:
 *   1. lcm_dim() matches the compiled LCM_D (runtime .so/checkpoint check)
 *   2. log_map_c stays finite for c > 1 (clip scaled by sqrt(c))
 *   3. gvalue integrity hash covers C_neg (tampering is detected)
 *   4. danger pattern matching direction (close to threat → blocked)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "lcm.h"
#include "lcm_api.h"

static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { printf("  [PASS] %s\n", msg); } \
        else { printf("  [FAIL] %s\n", msg); failures++; } \
    } while (0)

static void test_lcm_dim(void) {
    CHECK(lcm_dim() == LCM_D, "lcm_dim() == LCM_D");
}

static void test_log_map_c_finite_for_c_gt_1(void) {
    /* ||y|| = 0.999, c = 4 → old code computed t = 2*0.999 > 1 → atanhf NaN */
    float y[LCM_D];
    for (int i = 0; i < LCM_D; i++)
        y[i] = 0.999f / sqrtf((float)LCM_D);  /* ||y|| ≈ 0.999 */
    float out[LCM_D];
    log_map_c(y, out, LCM_D, 4.0f);
    int finite = 1;
    for (int i = 0; i < LCM_D; i++)
        if (!isfinite(out[i])) finite = 0;
    CHECK(finite, "log_map_c finite for c=4, ||y||=0.999");
}

static void test_gvalue_hash_covers_neg(void) {
    gvalue_t gv;
    float C_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float C_neg[LCM_N_VALUE_PAIRS][LCM_D];
    for (int i = 0; i < LCM_N_VALUE_PAIRS * LCM_D; i++) {
        C_pos[0][i] = 0.3f * sinf(0.01f * i);
        C_neg[0][i] = -0.3f * sinf(0.01f * i);
    }
    gvalue_init(&gv, &C_pos[0][0], &C_neg[0][0], LCM_D);
    CHECK(gvalue_verify(&gv), "gvalue_verify true on intact data");

    /* gvalue_init deep-copies; tamper the copy inside gv, not the source */
    gv.C_neg[0][0] += 0.5f;
    CHECK(!gvalue_verify(&gv), "gvalue_verify false after C_neg tampering");
}

static void test_danger_direction(void) {
    danger_lattice_t dl;
    float threats[2][LCM_D];
    float normals[2][LCM_D];
    memset(threats, 0, sizeof(threats));
    memset(normals, 0, sizeof(normals));
    threats[0][0]  =  0.5f;  /* threat prototype: ||v|| = 0.5 < 1 (ball) */
    threats[1][0]  = -0.4f;
    normals[0][0]  = -0.5f;
    normals[1][1]  =  0.3f;
    danger_init(&dl, &threats[0][0], &normals[0][0], 2, LCM_D);

    float z[LCM_D];
    float score;
    int threat;
    bool block;

    /* z == threat prototype → must be blocked */
    memcpy(z, threats[0], sizeof(float) * LCM_D);
    danger_assess(&dl, z, 1, 0, 32, 1.0f, &score, &threat, &block);
    CHECK(block, "z near threat prototype → blocked");

    /* z == normal prototype → must pass */
    memcpy(z, normals[0], sizeof(float) * LCM_D);
    danger_assess(&dl, z, 1, 0, 32, 1.0f, &score, &threat, &block);
    CHECK(!block, "z near normal prototype → not blocked");
}

static void test_hrr_no_overwrite(void) {
    /* Regression: HRR bind/unbind nodes used to leak into outputs[0] (the
     * HRQ slot) — the fused z came out as ~1/D phase-reconstruction garbage
     * instead of the retrieved codebook entry. */
    float cb[2][LCM_D];
    for (int i = 0; i < LCM_D; i++) cb[0][i] = 0.5f;
    for (int i = 0; i < LCM_D; i++) cb[1][i] = -0.25f;
    float z[LCM_D];
    for (int i = 0; i < LCM_D; i++) z[i] = 0.4f;
    float z_out[LCM_D];
    memset(z_out, 0, sizeof(z_out));
    int ret = lcm_infer_loop(z, LCM_D,
                             &cb[0][0], 2, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                             NULL, 0, NULL, 0,
                             NULL, 0, NULL, NULL, 0, NULL,
                             1, 1e-3f, 2.0f, 8, z_out);
    CHECK(ret == 0 || ret == -1, "lcm_infer_loop ran");
    float val = fabsf(z_out[0]);
    CHECK(val > 0.1f, "z_out ≈ codebook entry (0.5), not ~1/D HRR garbage");
}

static void test_fusion_l2_weights(void) {
    /* Two lattices. From z=0 the first fused point is
     * (1/1.1·1.1 + ⅓·3)/(1/1.1 + ⅓) ≈ 1.61 (weights 1/√d); the loop then
     * switches the first lattice to entry [2,0] and converges to ≈2.22 —
     * a fixed point of the L2-reciprocal fusion (with squared-distance
     * weights, the old bug, it converged to ≈1.05). Assert the L2 basin.
     * (confidences = 1/(√d + ε) is verified separately in the trace.) */
    float cb1[2][LCM_D];
    float cb2[2][LCM_D];
    memset(cb1, 0, sizeof(cb1));
    memset(cb2, 0, sizeof(cb2));
    cb1[0][0] = 1.1f; cb1[1][0] = 2.0f;
    cb2[0][0] = 3.0f; cb2[1][0] = 4.0f;
    float z[LCM_D];
    memset(z, 0, sizeof(z));
    float z_out[LCM_D];
    memset(z_out, 0, sizeof(z_out));
    lcm_infer_loop(z, LCM_D,
                   &cb1[0][0], 2, &cb2[0][0], 2,
                   NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                   NULL, 0, NULL, NULL, 0, NULL,
                   2, 1e-3f, 2.0f, 8, z_out);
    CHECK(fabsf(z_out[0] - 2.22f) < 0.15f,
          "fusion fixed point ≈ 2.22 (L2 reciprocal weights)");
}

int main(void) {
    printf("LCM engine tests (LCM_D=%d):\n", LCM_D);
    test_lcm_dim();
    test_log_map_c_finite_for_c_gt_1();
    test_gvalue_hash_covers_neg();
    test_danger_direction();
    test_hrr_no_overwrite();
    test_fusion_l2_weights();
    if (failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", failures);
    return failures ? 1 : 0;
}
