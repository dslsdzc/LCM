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

/* ── v3 full-parameter memory layout (task 7) ───────────────────────────── */

enum {
    T7_M = 3,          /* rows per codebook */
    T7_NL = 2,         /* layers for lowrank/binding/contrast */
    T7_NF = 2,         /* HRQ fine layers */
    T7_R = 4,          /* r_max */
    T7_NLAT = 6        /* fusion/route lattice count */
};

static float t7_hrq_top[T7_M * LCM_D];
static float t7_hrq_fine[T7_NF * T7_M * LCM_D];
static float t7_sparse_C[T7_M * LCM_D];
static float t7_sparse_zero[LCM_D];
static float t7_lowrank_U[T7_NL * T7_M * T7_R];
static float t7_lowrank_V[LCM_D * T7_R];
static float t7_man_C[T7_M * LCM_D];
static float t7_man_T[T7_M * LCM_D * 2];
static float t7_bind_key[T7_NL * T7_M * LCM_D];
static float t7_bind_val[T7_NL * T7_M * LCM_D];
static float t7_bind_bind[T7_NL * T7_M * LCM_D];
static float t7_A_k[T7_R * LCM_D];
static float t7_A_v[T7_R * LCM_D];
static float t7_contrast_a[T7_NL * T7_M * LCM_D];
static float t7_contrast_b[T7_NL * T7_M * LCM_D];
static float t7_route_C[T7_NLAT * LCM_D];
static float t7_route_W[LCM_D * T7_NLAT];
static float t7_alpha[T7_NLAT];
static float t7_ln_scale[LCM_D];
static float t7_ln_bias[LCM_D];
static float t7_self_C[(1 + 2) * LCM_D];
static int   t7_ranks[T7_NL] = {2, 4};

/* Plant a distinguishable first element in every layer. */
static void t7_mark_layers(float* p, int n_layers, int stride, float base) {
    for (int l = 0; l < n_layers; l++) p[l * stride] = base + (float)l;
}

static int t7_bind_valid(lcm_memory_v3* out) {
    hrq_mem_t hrq;
    sparse_mem_t sparse;
    lowrank_mem_t lowrank;
    manifold_mem_t manifold;
    binding_mem_t binding;
    contrast_mem_t contrast;
    fusion_mem_t fusion;

    memset(&hrq, 0, sizeof(hrq));
    hrq.C_top = t7_hrq_top; hrq.M_top = T7_M;
    hrq.C_fine = t7_hrq_fine; hrq.M_fine = T7_M; hrq.n_fine = T7_NF;
    hrq.value_top = NULL; hrq.value_fine = NULL;

    memset(&sparse, 0, sizeof(sparse));
    sparse.C = t7_sparse_C; sparse.M = T7_M;
    sparse.zero_vec = t7_sparse_zero;
    sparse.value = NULL; sparse.N = NULL; sparse.m = NULL; sparse.last_used = NULL;

    memset(&lowrank, 0, sizeof(lowrank));
    lowrank.U = t7_lowrank_U; lowrank.V = t7_lowrank_V;
    lowrank.ranks = t7_ranks; lowrank.n_layers = T7_NL;
    lowrank.M_lr = T7_M; lowrank.r_max = T7_R; lowrank.value = NULL;

    memset(&manifold, 0, sizeof(manifold));
    manifold.C = t7_man_C; manifold.M = T7_M;
    manifold.T = t7_man_T; manifold.t_dim = 2; manifold.value = NULL;

    memset(&binding, 0, sizeof(binding));
    binding.key_cb = t7_bind_key; binding.val_cb = t7_bind_val;
    binding.bind_cb = t7_bind_bind;
    binding.M = T7_M; binding.n_layers = T7_NL;
    binding.A_k = t7_A_k; binding.A_v = t7_A_v;
    binding.value_key = NULL; binding.value_val = NULL;
    binding.value_bind = NULL;

    memset(&contrast, 0, sizeof(contrast));
    contrast.C_a = t7_contrast_a; contrast.C_b = t7_contrast_b;
    contrast.M = T7_M; contrast.n_layers = T7_NL;
    contrast.value_a = NULL; contrast.value_b = NULL;

    memset(&fusion, 0, sizeof(fusion));
    fusion.route_C = t7_route_C; fusion.route_W = t7_route_W;
    fusion.n_lattices = T7_NLAT;
    fusion.alpha = t7_alpha; fusion.ln_scale = t7_ln_scale;
    fusion.ln_bias = t7_ln_bias;
    fusion.self_C = t7_self_C; fusion.M_self = 2;
    fusion.alpha_self = 0.05f;

    return lcm_bind_memory_v3(out, LCM_D, T7_NLAT, &hrq, &sparse, &lowrank,
                              &manifold, &binding, &contrast, &fusion);
}

static void test_memory_v3_layout(void) {
    static lcm_memory_v3 mem;

    memset(t7_hrq_top, 0, sizeof(t7_hrq_top));
    memset(t7_hrq_fine, 0, sizeof(t7_hrq_fine));
    memset(t7_sparse_C, 0, sizeof(t7_sparse_C));
    memset(t7_sparse_zero, 0, sizeof(t7_sparse_zero));
    memset(t7_lowrank_U, 0, sizeof(t7_lowrank_U));
    memset(t7_lowrank_V, 0, sizeof(t7_lowrank_V));
    memset(t7_man_C, 0, sizeof(t7_man_C));
    memset(t7_man_T, 0, sizeof(t7_man_T));
    memset(t7_bind_key, 0, sizeof(t7_bind_key));
    memset(t7_bind_val, 0, sizeof(t7_bind_val));
    memset(t7_bind_bind, 0, sizeof(t7_bind_bind));
    memset(t7_A_k, 0, sizeof(t7_A_k));
    memset(t7_A_v, 0, sizeof(t7_A_v));
    memset(t7_contrast_a, 0, sizeof(t7_contrast_a));
    memset(t7_contrast_b, 0, sizeof(t7_contrast_b));
    memset(t7_route_C, 0, sizeof(t7_route_C));
    memset(t7_route_W, 0, sizeof(t7_route_W));
    memset(t7_alpha, 0, sizeof(t7_alpha));
    memset(t7_ln_scale, 0, sizeof(t7_ln_scale));
    memset(t7_ln_bias, 0, sizeof(t7_ln_bias));
    memset(t7_self_C, 0, sizeof(t7_self_C));

    t7_hrq_top[1] = 11.0f;
    t7_mark_layers(t7_hrq_fine, T7_NF, T7_M * LCM_D, 21.0f);
    t7_mark_layers(t7_sparse_C, 1, T7_M * LCM_D, 31.0f);
    t7_sparse_zero[0] = 41.0f;
    t7_mark_layers(t7_lowrank_U, T7_NL, T7_M * T7_R, 51.0f);
    t7_lowrank_V[0] = 61.0f;
    t7_mark_layers(t7_man_C, 1, T7_M * LCM_D, 71.0f);
    t7_mark_layers(t7_man_T, T7_M, LCM_D * 2, 81.0f);
    t7_mark_layers(t7_bind_key, T7_NL, T7_M * LCM_D, 91.0f);
    t7_mark_layers(t7_bind_val, T7_NL, T7_M * LCM_D, 101.0f);
    t7_mark_layers(t7_bind_bind, T7_NL, T7_M * LCM_D, 111.0f);
    t7_A_k[0] = 121.0f;
    t7_A_v[0] = 131.0f;
    t7_mark_layers(t7_contrast_a, T7_NL, T7_M * LCM_D, 141.0f);
    t7_mark_layers(t7_contrast_b, T7_NL, T7_M * LCM_D, 151.0f);
    t7_mark_layers(t7_route_C, T7_NLAT, LCM_D, 161.0f);
    t7_route_W[0] = 171.0f;
    t7_alpha[T7_NLAT - 1] = 181.0f;
    t7_ln_scale[0] = 191.0f;
    t7_ln_bias[0] = 201.0f;
    t7_self_C[0] = 211.0f;

    int rc = t7_bind_valid(&mem);
    CHECK(rc == 0, "lcm_bind_memory_v3 accepts a valid descriptor");
    CHECK(mem.D == LCM_D && mem.n_lattices == T7_NLAT,
          "bound memory carries D and n_lattices");
    CHECK(mem.hrq.C_top == t7_hrq_top && mem.hrq.M_top == T7_M,
          "HRQ top pointer/rows preserved");
    CHECK(mem.hrq.C_fine == t7_hrq_fine && mem.hrq.n_fine == T7_NF,
          "HRQ fine pointer/layers preserved");
    CHECK(mem.sparse.zero_vec == t7_sparse_zero,
          "sparse zero_vec pointer preserved");
    CHECK(mem.lowrank.U == t7_lowrank_U && mem.lowrank.ranks == t7_ranks,
          "lowrank U/ranks pointers preserved");
    CHECK(mem.manifold.T == t7_man_T && mem.manifold.t_dim == 2,
          "manifold T pointer/t_dim preserved");
    CHECK(mem.binding.A_k == t7_A_k && mem.binding.A_v == t7_A_v,
          "binding projections preserved");
    CHECK(mem.fusion.route_C == t7_route_C && mem.fusion.alpha_self == 0.05f,
          "fusion route/alpha_self preserved");
    CHECK(mem.fusion.self_C == t7_self_C && mem.fusion.M_self == 2,
          "fusion self codebook preserved");

    /* Every layer starts at layer * M * D (or layer * M * r_max for U). */
    int layers_ok = 1;
    for (int l = 0; l < T7_NL; l++) {
        if (mem.lowrank.U[l * T7_M * T7_R] != 51.0f + (float)l) layers_ok = 0;
        if (mem.binding.key_cb[l * T7_M * LCM_D] != 91.0f + (float)l) layers_ok = 0;
        if (mem.binding.val_cb[l * T7_M * LCM_D] != 101.0f + (float)l) layers_ok = 0;
        if (mem.binding.bind_cb[l * T7_M * LCM_D] != 111.0f + (float)l) layers_ok = 0;
        if (mem.contrast.C_a[l * T7_M * LCM_D] != 141.0f + (float)l) layers_ok = 0;
        if (mem.contrast.C_b[l * T7_M * LCM_D] != 151.0f + (float)l) layers_ok = 0;
    }
    for (int l = 0; l < T7_NF; l++)
        if (mem.hrq.C_fine[l * T7_M * LCM_D] != 21.0f + (float)l) layers_ok = 0;
    for (int l = 0; l < T7_NLAT; l++)
        if (mem.fusion.route_C[l * LCM_D] != 161.0f + (float)l) layers_ok = 0;
    CHECK(layers_ok, "v3 layer offset = layer*M*D (and layer*M*r_max for U)");
    CHECK(mem.lowrank.ranks[1] == 4 && mem.lowrank.r_max == T7_R,
          "lowrank ranks kept per layer");
    CHECK(mem.tau_route == 0.5f && mem.tau_route_fallback == 0.1f
              && mem.lambda_lfq == 1.5f,
          "bind sets the documented scalar defaults");

    /* D / layer-count / pointer validation. */
    CHECK(t7_bind_valid(&mem) == 0, "re-binding an already bound struct works");
    CHECK(lcm_bind_memory_v3(&mem, LCM_D - 1, T7_NLAT,
                             &(hrq_mem_t){0}, &(sparse_mem_t){0},
                             &(lowrank_mem_t){0}, &(manifold_mem_t){0},
                             &(binding_mem_t){0}, &(contrast_mem_t){0},
                             &(fusion_mem_t){0}) == -1,
          "rejects D != LCM_D");
    CHECK(mem.D == 0, "a rejected bind leaves mem zeroed");

    int bad_ranks[2] = {1, T7_R + 1};
    lowrank_mem_t bad_lr;
    binding_mem_t bad_bind;
    fusion_mem_t  bad_fu;
    hrq_mem_t hrq; sparse_mem_t sp; manifold_mem_t man;
    contrast_mem_t ct;

    memset(&hrq, 0, sizeof(hrq));
    hrq.C_top = t7_hrq_top; hrq.M_top = T7_M;
    hrq.C_fine = t7_hrq_fine; hrq.M_fine = T7_M; hrq.n_fine = T7_NF;
    memset(&sp, 0, sizeof(sp));
    sp.C = t7_sparse_C; sp.M = T7_M; sp.zero_vec = t7_sparse_zero;
    memset(&man, 0, sizeof(man));
    man.C = t7_man_C; man.M = T7_M; man.T = t7_man_T; man.t_dim = 2;
    memset(&ct, 0, sizeof(ct));
    ct.C_a = t7_contrast_a; ct.C_b = t7_contrast_b; ct.M = T7_M;
    ct.n_layers = T7_NL;

    bad_lr = (lowrank_mem_t){0};
    bad_lr.U = t7_lowrank_U; bad_lr.V = t7_lowrank_V;
    bad_lr.ranks = bad_ranks; bad_lr.n_layers = T7_NL;
    bad_lr.M_lr = T7_M; bad_lr.r_max = T7_R;
    bad_bind = (binding_mem_t){0};
    bad_bind.key_cb = t7_bind_key; bad_bind.val_cb = t7_bind_val;
    bad_bind.bind_cb = t7_bind_bind;
    bad_bind.M = T7_M; bad_bind.n_layers = 0;
    bad_bind.A_k = t7_A_k; bad_bind.A_v = t7_A_v;
    bad_fu = (fusion_mem_t){0};
    bad_fu.route_C = t7_route_C; bad_fu.route_W = t7_route_W;
    bad_fu.n_lattices = T7_NLAT;
    bad_fu.alpha = NULL;  /* required */
    bad_fu.ln_scale = t7_ln_scale; bad_fu.ln_bias = t7_ln_bias;
    bad_fu.self_C = t7_self_C; bad_fu.M_self = 2;

    CHECK(lcm_bind_memory_v3(&mem, LCM_D, T7_NLAT, &hrq, &sp, &bad_lr,
                             &man, &bad_bind, &ct, &bad_fu) == -1,
          "rejects rank > r_max / zero binding layers / NULL fusion alpha");
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
    test_memory_v3_layout();
    test_hrr_no_overwrite();
    test_fusion_l2_weights();
    if (failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", failures);
    return failures ? 1 : 0;
}
