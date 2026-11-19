/* LCM inference engine — unit tests (make test).
 *
 * Regression coverage for the correctness audit fixes:
 *   1. lcm_dim() matches the compiled LCM_D (runtime .so/checkpoint check)
 *   2. log_map_c stays finite for c > 1 (clip scaled by sqrt(c))
 *   3. SHA-256 known vectors + gvalue/danger tamper detection
 *   4. danger pattern matching direction (close to threat → blocked)
 *   5. strict Three Laws (neutral safe at margin 0, unsafe at margin > 0)
 *   6. danger bounded relative gap (neutral not blocked; threshold 0.05)
 *   7. gvalue anchors projected to ||z|| = tanh(1)
 *   8. lcm_infer_loop bridge calls gvalue_init (projected anchors)
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

static void test_sha256_known_vector(void) {
    char hex[65];
    lcm_sha256_hex("abc", 3, hex);
    CHECK(strcmp(hex,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        == 0, "SHA-256(\"abc\") known vector");

    lcm_sha256_hex("", 0, hex);
    CHECK(strcmp(hex,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        == 0, "SHA-256(\"\") known vector");

    /* 56-byte FIPS 180-2 vector exercises the two-block padding path. */
    const char* msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    lcm_sha256_hex(msg, strlen(msg), hex);
    CHECK(strcmp(hex,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
        == 0, "SHA-256(56-byte FIPS vector) known vector");
}

static void test_gvalue_anchors_projected(void) {
    gvalue_t gv;
    float C_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float C_neg[LCM_N_VALUE_PAIRS][LCM_D];
    memset(C_pos, 0, sizeof(C_pos));
    memset(C_neg, 0, sizeof(C_neg));
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        C_pos[i][i] =  4.0f;  /* deliberately outside the ball */
        C_neg[i][i] = -4.0f;
    }
    gvalue_init(&gv, &C_pos[0][0], &C_neg[0][0], LCM_D);

    float expected = tanhf(1.0f);
    int ok = 1;
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        float n2p = 0.0f, n2n = 0.0f;
        for (int j = 0; j < LCM_D; j++) {
            n2p += gv.C_pos[i][j] * gv.C_pos[i][j];
            n2n += gv.C_neg[i][j] * gv.C_neg[i][j];
        }
        if (fabsf(sqrtf(n2p) - expected) > 1e-4f) ok = 0;
        if (fabsf(sqrtf(n2n) - expected) > 1e-4f) ok = 0;
    }
    CHECK(ok, "gvalue_init projects anchors to ||z|| = tanh(1)");
}

static void test_gvalue_strict_neutral(void) {
    gvalue_t gv;
    float C_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float C_neg[LCM_N_VALUE_PAIRS][LCM_D];
    memset(C_pos, 0, sizeof(C_pos));
    memset(C_neg, 0, sizeof(C_neg));
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        C_pos[i][i] =  0.9f;
        C_neg[i][i] = -0.9f;
    }
    gvalue_init(&gv, &C_pos[0][0], &C_neg[0][0], LCM_D);
    CHECK(gvalue_verify(&gv), "gvalue_verify true after init");

    float z[LCM_D] = {0};
    int law = -2;
    /* z=0 is neutral: pos_sim == neg_sim ⇒ balance = 0.
     * Strict Three Laws with margin 0.5: 0 > -0.5 ⇒ violation. */
    bool safe = gvalue_check_safety(&gv, z, 0.5f, &law);
    CHECK(!safe, "gvalue strict neutral (z=0, margin=0.5) → unsafe");
    CHECK(law >= 0 && law < LCM_N_VALUE_PAIRS,
          "gvalue strict neutral reports a law pair");
}

static void test_gvalue_margin_zero(void) {
    gvalue_t gv;
    float C_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float C_neg[LCM_N_VALUE_PAIRS][LCM_D];
    memset(C_pos, 0, sizeof(C_pos));
    memset(C_neg, 0, sizeof(C_neg));
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        C_pos[i][i] =  0.9f;
        C_neg[i][i] = -0.9f;
    }
    gvalue_init(&gv, &C_pos[0][0], &C_neg[0][0], LCM_D);

    float z[LCM_D] = {0};
    int law = -2;
    /* margin=0: violated ⇔ balance > 0; neutral balance=0 is safe. */
    bool safe = gvalue_check_safety(&gv, z, 0.0f, &law);
    CHECK(safe, "gvalue margin=0 neutral (z=0) → safe");
    CHECK(law == -1, "gvalue margin=0 neutral reports law=-1");
}

static void test_gvalue_positive_anchor_safe(void) {
    gvalue_t gv;
    float C_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float C_neg[LCM_N_VALUE_PAIRS][LCM_D];
    memset(C_pos, 0, sizeof(C_pos));
    memset(C_neg, 0, sizeof(C_neg));
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        C_pos[i][i] =  0.9f;
        C_neg[i][i] = -0.9f;
    }
    gvalue_init(&gv, &C_pos[0][0], &C_neg[0][0], LCM_D);

    int law = -2;
    bool safe = gvalue_check_safety(&gv, C_pos[0], 0.5f, &law);
    CHECK(safe, "gvalue positive anchor → safe even with margin=0.5");
}

static void test_danger_neutral_no_block(void) {
    danger_lattice_t dl;
    float threats[2][LCM_D];
    float normals[2][LCM_D];
    memset(threats, 0, sizeof(threats));
    memset(normals, 0, sizeof(normals));
    threats[0][0] =  0.5f;
    threats[1][1] =  0.4f;
    normals[0][0] = -0.5f;
    normals[1][1] = -0.4f;
    danger_init(&dl, &threats[0][0], &normals[0][0], 2, LCM_D);
    CHECK(danger_verify(&dl), "danger_verify true on intact data");

    /* z=0 is equidistant (after ball projection) from +v and -v anchors:
     * sim_threat == sim_normal ⇒ relative_gap = 0 ⇒ not blocked. */
    float z[LCM_D] = {0};
    float score;
    int threat;
    bool block;
    danger_assess(&dl, z, 1, 0, 32, 1.0f, &score, &threat, &block);
    CHECK(!block, "danger neutral (threat≈normal) → not blocked");
    CHECK(threat == THREAT_NONE, "danger neutral → THREAT_NONE");
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

static void test_danger_threshold_relative_gap(void) {
    danger_lattice_t dl;
    float threats[1][LCM_D];
    float normals[1][LCM_D];
    memset(threats, 0, sizeof(threats));
    memset(normals, 0, sizeof(normals));

    /* Construct relative_gap ≈ 0.02: threat slightly closer than normal.
     * After ball projection both anchors have the same radius, so
     * gap = (s_n - s_t)/(s_n + s_t) = ((1-cos θ_n) - (1-cos θ_t)) /
     *                                  ((1-cos θ_n) + (1-cos θ_t)).
     * Choose θ_t = 0.2 and ratio (1+g)/(1-g) for θ_n. */
    const float target_gap = 0.02f;
    const float theta_t = 0.2f;
    const float ratio = (1.0f + target_gap) / (1.0f - target_gap);
    float one_minus_cos_t = 1.0f - cosf(theta_t);
    float one_minus_cos_n = ratio * one_minus_cos_t;
    float cos_n = 1.0f - one_minus_cos_n;
    float sin_n = sqrtf(fmaxf(0.0f, 1.0f - cos_n * cos_n));

    threats[0][0] = 0.5f * cosf(theta_t);
    threats[0][1] = 0.5f * sinf(theta_t);
    normals[0][0] = 0.5f * cos_n;
    normals[0][1] = 0.5f * sin_n;
    danger_init(&dl, &threats[0][0], &normals[0][0], 1, LCM_D);

    float z[LCM_D] = {0};
    z[0] = 0.5f;
    float score;
    int threat;
    bool block;

    /* R7 threshold is 0.05: gap 0.02 must not block. */
    CHECK(dl.safety_threshold > 0.049f && dl.safety_threshold < 0.051f,
          "danger_init threshold = 0.05 (R7)");
    danger_assess(&dl, z, 0, 0, 32, 1.0f, &score, &threat, &block);
    CHECK(!block, "danger relative_gap≈0.02 < 0.05 → not blocked");
    CHECK(fabsf(score - target_gap) < 0.002f,
          "danger no-threat score = relative_gap ≈ 0.02");

    /* Discriminating check: the same input at the old 0.005 threshold blocks. */
    dl.safety_threshold = 0.005f;
    danger_assess(&dl, z, 0, 0, 32, 1.0f, &score, &threat, &block);
    CHECK(block, "danger relative_gap≈0.02 > 0.005 → blocked (discriminating)");
}

static void test_danger_hash_tamper(void) {
    danger_lattice_t dl;
    float threats[1][LCM_D];
    float normals[1][LCM_D];
    memset(threats, 0, sizeof(threats));
    memset(normals, 0, sizeof(normals));
    threats[0][0] =  0.5f;
    normals[0][0] = -0.5f;
    danger_init(&dl, &threats[0][0], &normals[0][0], 1, LCM_D);
    CHECK(danger_verify(&dl), "danger_verify true after init");

    threats[0][0] += 0.25f;
    CHECK(!danger_verify(&dl), "danger_verify false after C_threats tampering");

    threats[0][0] -= 0.25f;
    normals[0][0] += 0.25f;
    CHECK(!danger_verify(&dl), "danger_verify false after C_normal tampering");
}

/* has_conflict flag for step 0 of the most recent lcm_infer_loop call. */
static int last_loop_first_conflict(void) {
    int n = lcm_get_trace(NULL, 0);
    int per_step = 2 * LCM_MAX_LATTICES + LCM_D + 2;
    if (n < 1 || n > 4) return -1;
    float buf[4 * (2 * LCM_MAX_LATTICES + LCM_D + 2)];
    if (lcm_get_trace(buf, n * per_step) != n) return -1;
    return (int)buf[2 * LCM_MAX_LATTICES + LCM_D + 1];
}

static void test_bridge_uses_projected_anchors(void) {
    /* Two HRQ entries: +0.5 e0 and -0.5 e0.  Passing z near one of them
     * selects that entry as z_next. */
    float hrq[2][LCM_D];
    memset(hrq, 0, sizeof(hrq));
    hrq[0][0] =  0.5f;
    hrq[1][0] = -0.5f;

    /* Anchors deliberately outside the ball (norm 4).  The bridge must call
     * gvalue_init, which projects them; a raw memcpy would make the positive
     * input violate the strict margin (raw balance ≈ -0.25 > -0.5). */
    float gv_pos[LCM_N_VALUE_PAIRS][LCM_D];
    float gv_neg[LCM_N_VALUE_PAIRS][LCM_D];
    memset(gv_pos, 0, sizeof(gv_pos));
    memset(gv_neg, 0, sizeof(gv_neg));
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        gv_pos[i][i] =  4.0f;
        gv_neg[i][i] = -4.0f;
    }

    float z_safe[LCM_D] = {0};
    float z_bad[LCM_D] = {0};
    z_safe[0] =  0.5f;
    z_bad[0]  = -0.5f;
    float z_out[LCM_D];

    int ret_safe = lcm_infer_loop(
        z_safe, LCM_D,
        &hrq[0][0], 2, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        &gv_pos[0][0], LCM_N_VALUE_PAIRS, &gv_neg[0][0],
        NULL, 0, NULL,
        1, 1e9f, 1e9f, 2, z_out);
    int safe_conflict = last_loop_first_conflict();

    int ret_bad = lcm_infer_loop(
        z_bad, LCM_D,
        &hrq[0][0], 2, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        &gv_pos[0][0], LCM_N_VALUE_PAIRS, &gv_neg[0][0],
        NULL, 0, NULL,
        1, 1e9f, 1e9f, 2, z_out);
    int bad_conflict = last_loop_first_conflict();

    CHECK(safe_conflict == 0,
          "bridge: positive input → no gvalue conflict (projected anchors)");
    CHECK(ret_safe == 0, "bridge: positive input converges");
    CHECK(bad_conflict == 1,
          "bridge: negative input → gvalue conflict");
    CHECK(ret_bad == -1, "bridge: negative input aborts");
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
    test_sha256_known_vector();
    test_gvalue_hash_covers_neg();
    test_gvalue_anchors_projected();
    test_gvalue_strict_neutral();
    test_gvalue_margin_zero();
    test_gvalue_positive_anchor_safe();
    test_danger_neutral_no_block();
    test_danger_direction();
    test_danger_threshold_relative_gap();
    test_danger_hash_tamper();
    test_bridge_uses_projected_anchors();
    test_hrr_no_overwrite();
    test_fusion_l2_weights();
    if (failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", failures);
    return failures ? 1 : 0;
}
