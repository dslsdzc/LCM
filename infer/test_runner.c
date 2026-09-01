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

int main(void) {
    printf("LCM engine tests (LCM_D=%d):\n", LCM_D);
    test_lcm_dim();
    test_log_map_c_finite_for_c_gt_1();
    test_gvalue_hash_covers_neg();
    test_danger_direction();
    if (failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", failures);
    return failures ? 1 : 0;
}
