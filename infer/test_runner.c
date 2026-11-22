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

    /* The saved file digest is SHA-256(C_pos raw || C_neg raw). */
    float raw[2 * LCM_N_VALUE_PAIRS * LCM_D];
    char expected[65];
    memcpy(raw, C_pos, sizeof(C_pos));
    memcpy(raw + LCM_N_VALUE_PAIRS * LCM_D, C_neg, sizeof(C_neg));
    lcm_sha256_hex(raw, sizeof(raw), expected);
    CHECK(gvalue_verify_digest(&gv, expected),
          "gvalue_verify_digest matches the exported payload digest");
    CHECK(!gvalue_verify_digest(&gv, "0000000000000000000000000000000000000000000000000000000000000000"),
          "gvalue_verify_digest rejects a wrong saved digest");

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

    float raw_d[2 * 2 * LCM_D];
    char expected_d[65];
    memcpy(raw_d, threats, sizeof(threats));
    memcpy(raw_d + 2 * LCM_D, normals, sizeof(normals));
    lcm_sha256_hex(raw_d, sizeof(raw_d), expected_d);
    CHECK(danger_verify_digest(&dl, expected_d),
          "danger_verify_digest matches the exported payload digest");
    CHECK(!danger_verify_digest(&dl, "0000000000000000000000000000000000000000000000000000000000000000"),
          "danger_verify_digest rejects a wrong saved digest");

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

static void test_danger_checks_input_state(void) {
    danger_lattice_t dl;
    float threats[1][LCM_D];
    float normals[1][LCM_D];
    memset(threats, 0, sizeof(threats));
    memset(normals, 0, sizeof(normals));
    threats[0][0] =  0.5f;
    normals[0][0] = -0.5f;
    danger_init(&dl, &threats[0][0], &normals[0][0], 1, LCM_D);

    int counts[LCM_MAX_LATTICES] = {0};
    conflict_t conflict;

    /* A threat input that the DAG transforms away must still abort. */
    float z_cur[LCM_D] = {0};
    float z_next[LCM_D] = {0};
    z_cur[0] = 0.5f;
    z_next[0] = -0.5f;
    memset(&conflict, 0, sizeof(conflict));
    bool blocked = detect_any_conflict(
        z_next, z_cur, 0, &dl, NULL, counts, 1.0f, 0.5f, 16, &conflict);
    CHECK(blocked && conflict.source == CONFLICT_DANGER,
          "danger gate checks the input state z_cur, not only z_next");

    /* And the post-step state is still checked. */
    z_cur[0] = -0.5f;
    z_next[0] = 0.5f;
    memset(&conflict, 0, sizeof(conflict));
    blocked = detect_any_conflict(
        z_next, z_cur, 0, &dl, NULL, counts, 1.0f, 0.5f, 16, &conflict);
    CHECK(blocked && conflict.source == CONFLICT_DANGER,
          "danger gate still checks the post-step state z_next");
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

/* All seven descriptors, so each negative test can vary exactly one field. */
typedef struct {
    hrq_mem_t hrq;
    sparse_mem_t sparse;
    lowrank_mem_t lowrank;
    manifold_mem_t manifold;
    binding_mem_t binding;
    contrast_mem_t contrast;
    fusion_mem_t fusion;
} t7_desc_t;

static void t7_fill(t7_desc_t* d) {
    memset(d, 0, sizeof(*d));
    d->hrq.C_top = t7_hrq_top; d->hrq.M_top = T7_M;
    d->hrq.C_fine = t7_hrq_fine; d->hrq.M_fine = T7_M;
    d->hrq.n_fine = T7_NF;
    d->hrq.value_top = NULL; d->hrq.value_fine = NULL;

    d->sparse.C = t7_sparse_C; d->sparse.M = T7_M;
    d->sparse.zero_vec = t7_sparse_zero;
    d->sparse.value = NULL; d->sparse.N = NULL; d->sparse.m = NULL;
    d->sparse.last_used = NULL;

    d->lowrank.U = t7_lowrank_U; d->lowrank.V = t7_lowrank_V;
    d->lowrank.ranks = t7_ranks; d->lowrank.n_layers = T7_NL;
    d->lowrank.M_lr = T7_M; d->lowrank.r_max = T7_R;
    d->lowrank.value = NULL;

    d->manifold.C = t7_man_C; d->manifold.M = T7_M;
    d->manifold.T = t7_man_T; d->manifold.t_dim = 2;
    d->manifold.value = NULL;

    d->binding.key_cb = t7_bind_key; d->binding.val_cb = t7_bind_val;
    d->binding.bind_cb = t7_bind_bind;
    d->binding.M = T7_M; d->binding.n_layers = T7_NL;
    d->binding.r_max = T7_R;
    d->binding.A_k = t7_A_k; d->binding.A_v = t7_A_v;
    d->binding.value_key = NULL; d->binding.value_val = NULL;
    d->binding.value_bind = NULL;

    d->contrast.C_a = t7_contrast_a; d->contrast.C_b = t7_contrast_b;
    d->contrast.M = T7_M; d->contrast.n_layers = T7_NL;
    d->contrast.value_a = NULL; d->contrast.value_b = NULL;

    d->fusion.route_C = t7_route_C; d->fusion.route_W = t7_route_W;
    d->fusion.n_lattices = T7_NLAT;
    d->fusion.alpha = t7_alpha; d->fusion.ln_scale = t7_ln_scale;
    d->fusion.ln_bias = t7_ln_bias;
    d->fusion.self_C = t7_self_C; d->fusion.M_self = 2;
    d->fusion.alpha_self = 0.05f;
}

static int t7_bind_dim(lcm_memory_v3* out, int D, const t7_desc_t* d) {
    return lcm_bind_memory_v3(out, D, T7_NLAT, &d->hrq, &d->sparse,
                              &d->lowrank, &d->manifold, &d->binding,
                              &d->contrast, &d->fusion);
}

static int t7_bind(lcm_memory_v3* out, const t7_desc_t* d) {
    return t7_bind_dim(out, LCM_D, d);
}

static void test_memory_v3_layout(void) {
    static lcm_memory_v3 mem;
    t7_desc_t good;

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
    for (int r = 0; r < 3; r++) t7_self_C[r * LCM_D] = 211.0f + (float)r;

    t7_fill(&good);
    CHECK(t7_bind(&mem, &good) == 0,
          "lcm_bind_memory_v3 accepts a valid descriptor");
    CHECK(mem.D == LCM_D && mem.n_lattices == T7_NLAT,
          "bound memory carries D and n_lattices");
    CHECK(mem.hrq.C_top == t7_hrq_top && mem.hrq.M_top == T7_M,
          "HRQ top pointer/rows preserved");
    CHECK(mem.hrq.C_fine == t7_hrq_fine && mem.hrq.n_fine == T7_NF,
          "HRQ fine pointer/layers preserved");
    CHECK(mem.hrq.C_top[1] == 11.0f, "HRQ top payload preserved");
    CHECK(mem.sparse.zero_vec == t7_sparse_zero
              && mem.sparse.C[0] == 31.0f
              && mem.sparse.zero_vec[0] == 41.0f,
          "sparse C/zero_vec payload preserved");
    CHECK(mem.lowrank.U == t7_lowrank_U && mem.lowrank.ranks == t7_ranks
              && mem.lowrank.V[0] == 61.0f,
          "lowrank U/V/ranks payload preserved");
    CHECK(mem.manifold.C[0] == 71.0f
              && mem.manifold.T[0] == 81.0f
              && mem.manifold.T[LCM_D * 2] == 82.0f,
          "manifold C/T payload preserved");
    CHECK(mem.binding.A_k == t7_A_k && mem.binding.A_v == t7_A_v
              && mem.binding.A_k[0] == 121.0f
              && mem.binding.A_v[0] == 131.0f,
          "binding projections preserved");
    CHECK(mem.binding.r_max == T7_R,
          "binding r_max preserved (shared V column count)");
    CHECK(mem.fusion.route_W[0] == 171.0f
              && mem.fusion.alpha[T7_NLAT - 1] == 181.0f
              && mem.fusion.ln_scale[0] == 191.0f
              && mem.fusion.ln_bias[0] == 201.0f,
          "fusion route_W/alpha/ln payload preserved");
    CHECK(mem.fusion.route_C == t7_route_C && mem.fusion.alpha_self == 0.05f,
          "fusion route/alpha_self preserved");
    CHECK(mem.fusion.self_C == t7_self_C && mem.fusion.M_self == 2,
          "fusion self codebook preserved");
    CHECK(mem.fusion.self_C[0] == 211.0f
              && mem.fusion.self_C[LCM_D] == 212.0f
              && mem.fusion.self_C[2 * LCM_D] == 213.0f,
          "self_C rows at row*D (core + two modes)");

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

    /* Independent negative calls: exactly one field is invalid each time, so
     * each validation branch is actually reached. */
    CHECK(t7_bind_dim(&mem, LCM_D - 1, &good) == -1,
          "rejects D != LCM_D");
    CHECK(mem.D == 0, "a rejected bind leaves mem zeroed");

    int bad_ranks[2] = {1, T7_R + 1};
    t7_desc_t bad = good;
    bad.lowrank.ranks = bad_ranks;
    CHECK(t7_bind(&mem, &bad) == -1, "rejects lowrank rank > r_max");

    int unsorted_ranks[2] = {T7_R, 1};
    bad = good;
    bad.lowrank.ranks = unsorted_ranks;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects non-increasing lowrank ranks");

    int equal_ranks[2] = {2, 2};
    bad = good;
    bad.lowrank.ranks = equal_ranks;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects equal (non-strictly-increasing) lowrank ranks");

    bad = good;
    bad.sparse.M = LCM_MAX_M + 1;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects codebook M over the stack cap");

    bad = good;
    bad.lowrank.r_max = LCM_MAX_RANK + 1;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects lowrank r_max over the stack cap");

    bad = good;
    bad.manifold.t_dim = LCM_MAX_T_DIM + 1;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects manifold t_dim over the stack cap");

    bad = good;
    bad.binding.n_layers = 0;
    CHECK(t7_bind(&mem, &bad) == -1, "rejects zero binding n_layers");

    bad = good;
    bad.binding.n_layers = LCM_MAX_BIND_LAYERS + 1;
    CHECK(t7_bind(&mem, &bad) == -1, "rejects binding n_layers over the cap");

    bad = good;
    bad.binding.r_max = 0;
    CHECK(t7_bind(&mem, &bad) == -1, "rejects zero binding r_max");

    bad = good;
    bad.binding.r_max = T7_R + 1;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects binding r_max != lowrank r_max (shared V)");

    bad = good;
    bad.fusion.alpha = NULL;
    CHECK(t7_bind(&mem, &bad) == -1, "rejects NULL fusion alpha");

    bad = good;
    bad.fusion.n_lattices = T7_NLAT + 1;
    CHECK(t7_bind(&mem, &bad) == -1,
          "rejects fusion.n_lattices != n_lattices");
}

/* ── v3.1 HRQ / sparse / routing forwards (task 8) ──────────────────────── */

#define T8_HRQ_M     4
#define T8_FINE_M    2
#define T8_SPARSE_M  3
#define T8_NLAT      3
#define T8_PI        3.14159265358979323846f

static float t8_top[T8_HRQ_M * LCM_D];
static float t8_fine[T8_FINE_M * LCM_D];
static float t8_sparse_C[T8_SPARSE_M * LCM_D];
static float t8_sparse_zero[LCM_D];
static float t8_route_C[T8_NLAT * LCM_D];
static float t8_route_W[LCM_D * T8_NLAT];

static void t8_zero(float* p, int n) {
    memset(p, 0, (size_t)n * sizeof(float));
}

/* Unit-radius prototype in the (e0, e1) plane. */
static void t8_dir(float* row, float theta, float radius) {
    t8_zero(row, LCM_D);
    row[0] = radius * cosf(theta);
    row[1] = radius * sinf(theta);
}

/* R1 reference similarity: query is safe_unit→exp_map, codebook is exp_map
 * only (JAX maps codebooks directly, it does not renormalise them). */
static float t8_hrq_sim(const float* z, const float* proto) {
    float z_u[LCM_D], z_P[LCM_D], p_P[LCM_D];
    safe_unit_c(z, z_u, LCM_D, 1e-8f);
    exp_map_c(z_u, z_P, LCM_D, 1.0f);
    exp_map_c(proto, p_P, LCM_D, 1.0f);
    return poincare_similarity_c(z_P, p_P, LCM_D, 1.0f);
}

static hrq_mem_t t8_hrq_mem(const float* v_top, const float* v_fine) {
    hrq_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_top = t8_top;
    m.M_top = T8_HRQ_M;
    m.C_fine = t8_fine;
    m.M_fine = T8_FINE_M;
    m.n_fine = 1;
    m.value_top = v_top;
    m.value_fine = v_fine;
    return m;
}

static float t8_max_abs_diff(const float* a, const float* b) {
    float worst = 0.0f;
    for (int i = 0; i < LCM_D; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static void test_hrq_v31_poincare_top1(void) {
    static const float angles[T8_HRQ_M] = {0.0f, 1.0f, 2.0f, 3.0f};
    for (int j = 0; j < T8_HRQ_M; j++)
        t8_dir(t8_top + (size_t)j * LCM_D, angles[j], 1.0f);
    t8_zero(t8_fine, T8_FINE_M * LCM_D);

    float z[LCM_D];
    t8_dir(z, 0.95f, 1.0f);     /* between prototypes 0 and 1, nearest 1 */
    float v_zero[T8_HRQ_M] = {0.0f, 0.0f, 0.0f, 0.0f};

    hrq_mem_t m = t8_hrq_mem(v_zero, NULL);
    float out[LCM_D];
    int idx = -1;
    float top_sim = -1.0f;
    int rc = hrq_forward_v31(z, &m, 1.0f, T8_HRQ_M, 0.1f, out, &idx, &top_sim);
    CHECK(rc == 0, "hrq_forward_v31 returns 0");

    int ref = 0;
    float ref_sim = 1e20f;
    for (int j = 0; j < T8_HRQ_M; j++) {
        float s = t8_hrq_sim(z, t8_top + (size_t)j * LCM_D);
        if (s < ref_sim) { ref_sim = s; ref = j; }
    }
    CHECK(ref == 1, "reference Poincare argmin is prototype 1");
    CHECK(idx == ref, "v=0: top-1 matches pure Poincare argmin");
    CHECK(fabsf(top_sim - ref_sim) < 1e-4f,
          "v=0: top_sim = sims[top_idx]");

    /* value_scalars == 0 means alpha_val is inert. */
    idx = -1;
    rc = hrq_forward_v31(z, &m, 0.0f, T8_HRQ_M, 0.1f, out, &idx, NULL);
    CHECK(rc == 0 && idx == ref,
          "alpha_val=0 with v=0 keeps the geometric top-1");
}

static void test_hrq_v31_poincare_vs_euclidean(void) {
    /* Query r=0.9 on e0. C0 is radial at r=0.4 (Euclidean 0.50 away);
     * C1 is r=0.95 at angle 0.5 (Euclidean 0.460, but its ball radius is
     * nearly z_P's and the angular term dominates). The two metrics must
     * disagree, and C must follow Poincaré. */
    static float top2[2 * LCM_D];
    t8_dir(top2, 0.0f, 0.4f);
    t8_dir(top2 + (size_t)LCM_D, 0.5f, 0.95f);
    t8_zero(t8_fine, T8_FINE_M * LCM_D);

    float z[LCM_D];
    t8_dir(z, 0.0f, 0.9f);

    int e_idx = 0;
    float e_best = INFINITY;
    for (int j = 0; j < 2; j++) {
        float d2 = 0.0f;
        for (int i = 0; i < LCM_D; i++) {
            float diff = z[i] - top2[(size_t)j * LCM_D + i];
            d2 += diff * diff;
        }
        if (d2 < e_best) { e_best = d2; e_idx = j; }
    }
    int p_idx = 0;
    float p_best = INFINITY;
    for (int j = 0; j < 2; j++) {
        float s = t8_hrq_sim(z, top2 + (size_t)j * LCM_D);
        if (s < p_best) { p_best = s; p_idx = j; }
    }
    CHECK(e_idx == 1, "euclidean reference argmin is prototype 1");
    CHECK(p_idx == 0, "poincare reference argmin is prototype 0");

    hrq_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_top = top2;
    m.M_top = 2;
    m.C_fine = NULL;
    m.M_fine = 0;
    m.n_fine = 0;

    float out[LCM_D];
    int idx = -1;
    float top_sim = -1.0f;
    CHECK(hrq_forward_v31(z, &m, 0.0f, 2, 0.1f, out, &idx, &top_sim) == 0
              && idx == 0,
          "hrq v31 follows the Poincare argmin (not Euclidean)");
    CHECK(fabsf(top_sim - p_best) < 1e-4f,
          "poincare-vs-euclidean: top_sim is the Poincare sim");
}

static void test_hrq_v31_value_topk(void) {
    /* Prototype 2 is geometrically third but close: with k=4 its value would
     * win, with k=2 it is outside the candidate set. */
    static const float angles[T8_HRQ_M] = {0.0f, 0.05f, 0.1f, 2.5f};
    for (int j = 0; j < T8_HRQ_M; j++)
        t8_dir(t8_top + (size_t)j * LCM_D, angles[j], 1.0f);
    t8_zero(t8_fine, T8_FINE_M * LCM_D);

    float z[LCM_D];
    t8_dir(z, 0.0f, 1.0f);
    float out[LCM_D];
    int idx = -1;
    float top_sim = -1.0f;

    /* Top-2 by geometry are {0,1}; a large value on 1 wins inside k=2. */
    float v1[T8_HRQ_M] = {0.0f, 1.0f, 0.0f, 0.0f};
    hrq_mem_t m = t8_hrq_mem(v1, NULL);
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out, &idx, &top_sim) == 0
              && idx == 1,
          "value reorders top-1 inside top-k=2");
    float s1 = t8_hrq_sim(z, t8_top + (size_t)1 * LCM_D);
    CHECK(fabsf(top_sim - s1) < 1e-4f,
          "top_sim follows the value-rescored winner");

    /* k=1 pins the geometric winner: value cannot reach prototype 1. */
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 1, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "value_topk=1 cannot move the top-1");

    /* A high value on prototype 2 is inert at k=2 ... */
    float v2[T8_HRQ_M] = {0.0f, 0.0f, 1.0f, 0.0f};
    m.value_top = v2;
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "value outside top-k=2 does not change top-1");

    /* ... but the same value wins once k=4 admits prototype 2. This proves
     * the assertion above is not passing because value is ignored. */
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 4, 0.1f, out, &idx, NULL) == 0
              && idx == 2,
          "same value selects prototype 2 when top-k admits it");

    /* alpha_val=0 disables rescoring even inside top-k. */
    m.value_top = v1;
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 0.0f, 2, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "alpha_val=0 disables value rescoring");
}

static void test_hrq_v31_relative_gap(void) {
    /* Fine prototypes at -0.75 and 0.0. Hard and fallback route different
     * fine winners (hard residual points ~-0.57/-0.93; the blend residual
     * points ~0), so the two branch outputs are distinguishable by >1e-3. */
    t8_dir(t8_fine + (size_t)0 * LCM_D, -0.75f, 1.0f);
    t8_dir(t8_fine + (size_t)1 * LCM_D, 0.0f, 1.0f);

    float z[LCM_D];
    t8_dir(z, 0.0f, 1.0f);
    hrq_mem_t m = t8_hrq_mem(NULL, NULL);
    float out_def[LCM_D], out_hard[LCM_D], out_soft[LCM_D];
    int idx = -1;

    /* Case A: sim1≈1.90, sim2≈2.04. Absolute gap 0.14 > tau, but the
     * relative gap 0.074 < tau ⇒ must fallback. Old absolute-gap logic
     * would classify this as hard. */
    t8_dir(t8_top + (size_t)0 * LCM_D, 0.78f, 1.0f);
    t8_dir(t8_top + (size_t)1 * LCM_D, -0.81f, 1.0f);
    t8_dir(t8_top + (size_t)2 * LCM_D, T8_PI, 1.0f);
    t8_dir(t8_top + (size_t)3 * LCM_D, T8_PI, 1.0f);
    float s0 = t8_hrq_sim(z, t8_top);
    float s1 = t8_hrq_sim(z, t8_top + (size_t)1 * LCM_D);
    CHECK(s1 - s0 > 0.1f,
          "gap A: absolute difference exceeds tau");
    CHECK((s1 - s0) / s0 < 0.1f,
          "gap A: relative difference is below tau");

    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out_def, &idx, NULL) == 0,
          "gap A: default call succeeds");
    CHECK(idx == 0, "gap A: idx stays the geometric top-1");
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.0f, out_hard, NULL, NULL) == 0,
          "gap A: forced-hard call succeeds");
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 1e9f, out_soft, NULL, NULL) == 0,
          "gap A: forced-fallback call succeeds");
    CHECK(t8_max_abs_diff(out_hard, out_soft) > 1e-3f,
          "gap A: hard and fallback outputs are distinguishable");
    CHECK(t8_max_abs_diff(out_def, out_soft) < 1e-6f,
          "gap A: relative gap < tau selects the fallback output");
    CHECK(t8_max_abs_diff(out_def, out_hard) > 1e-3f,
          "gap A: fallback output differs from the hard top-1 output");

    /* Case B: sim1≈0.50, sim2≈0.56. Absolute gap 0.06 < tau, but the
     * relative gap 0.12 > tau ⇒ must hard. Old absolute-gap logic would
     * classify this as fallback. */
    t8_dir(t8_top + (size_t)0 * LCM_D, 0.195f, 1.0f);
    t8_dir(t8_top + (size_t)1 * LCM_D, -0.2138f, 1.0f);
    t8_dir(t8_top + (size_t)2 * LCM_D, T8_PI, 1.0f);
    t8_dir(t8_top + (size_t)3 * LCM_D, T8_PI, 1.0f);
    s0 = t8_hrq_sim(z, t8_top);
    s1 = t8_hrq_sim(z, t8_top + (size_t)1 * LCM_D);
    CHECK(s1 - s0 < 0.1f,
          "gap B: absolute difference is below tau");
    CHECK((s1 - s0) / s0 > 0.1f,
          "gap B: relative difference exceeds tau");

    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out_def, &idx, NULL) == 0,
          "gap B: default call succeeds");
    CHECK(idx == 0, "gap B: idx stays the geometric top-1");
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.0f, out_hard, NULL, NULL) == 0,
          "gap B: forced-hard call succeeds");
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 1e9f, out_soft, NULL, NULL) == 0,
          "gap B: forced-fallback call succeeds");
    CHECK(t8_max_abs_diff(out_hard, out_soft) > 1e-3f,
          "gap B: hard and fallback outputs are distinguishable");
    CHECK(t8_max_abs_diff(out_def, out_hard) < 1e-6f,
          "gap B: relative gap > tau selects the hard output");
    CHECK(t8_max_abs_diff(out_def, out_soft) > 1e-3f,
          "gap B: hard output differs from the fallback output");
}

static void test_hrq_v31_tie_break(void) {
    /* Two bit-identical prototypes: stable top-k/argmin must keep the first
     * occurrence (JAX argsort/argmin first-seen order). */
    t8_zero(t8_top, T8_HRQ_M * LCM_D);
    t8_dir(t8_top, 0.0f, 0.7f);
    memcpy(t8_top + (size_t)LCM_D, t8_top, sizeof(float) * LCM_D);
    t8_dir(t8_top + (size_t)2 * LCM_D, 1.5f, 0.7f);
    t8_dir(t8_top + (size_t)3 * LCM_D, 2.5f, 0.7f);
    t8_zero(t8_fine, T8_FINE_M * LCM_D);

    float z[LCM_D];
    t8_dir(z, 0.0f, 1.0f);
    hrq_mem_t m = t8_hrq_mem(NULL, NULL);
    m.n_fine = 0;

    float out[LCM_D];
    int idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 3, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "tie: k=3 keeps the first of two equal prototypes");
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "tie: k=2 keeps the first of two equal prototypes");
}

static void test_hrq_v31_value_fine(void) {
    /* M_top == 1 (fallback blend == c_top). The fine layer is symmetric
     * around the residual, so v=0 ties and picks row 0 (+0.5); a positive
     * value on row 1 must flip the winner (R22 shared fine vector). */
    static float top1[LCM_D];
    t8_dir(top1, 0.0f, 0.4f);
    t8_dir(t8_fine + (size_t)0 * LCM_D, 0.5f, 1.0f);
    t8_dir(t8_fine + (size_t)1 * LCM_D, -0.5f, 1.0f);
    t8_zero(t8_top, T8_HRQ_M * LCM_D);   /* unused beyond M_top == 1 */

    float z[LCM_D];
    t8_dir(z, 0.0f, 1.0f);
    float out[LCM_D];
    int idx = -1;

    hrq_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_top = top1;
    m.M_top = 1;
    m.C_fine = t8_fine;
    m.M_fine = T8_FINE_M;
    m.n_fine = 1;
    m.value_top = NULL;
    m.value_fine = NULL;

    CHECK(hrq_forward_v31(z, &m, 2.0f, 1, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "value_fine: baseline call succeeds");
    CHECK(out[1] < -0.1f,
          "value_fine: v=0 tie picks fine row 0 (+0.5)");

    float v_fine[T8_FINE_M] = {0.0f, 1.0f};
    m.value_fine = v_fine;
    idx = -1;
    CHECK(hrq_forward_v31(z, &m, 2.0f, 1, 0.1f, out, &idx, NULL) == 0
              && idx == 0,
          "value_fine: top-1 is unchanged");
    CHECK(out[1] > 0.1f,
          "value_fine: high value on row 1 flips the fine winner");
}

static void test_hrq_v31_boundaries(void) {
    t8_zero(t8_top, T8_HRQ_M * LCM_D);
    t8_dir(t8_top, 0.0f, 1.0f);
    t8_dir(t8_top + (size_t)LCM_D, 1.0f, 1.0f);
    t8_zero(t8_fine, T8_FINE_M * LCM_D);

    float z[LCM_D];
    t8_dir(z, 0.2f, 1.0f);
    float out[LCM_D];
    int idx = -1;

    hrq_mem_t m = t8_hrq_mem(NULL, NULL);
    m.M_top = 2;
    m.n_fine = 0;

    CHECK(hrq_forward_v31(z, &m, 1.0f, LCM_HRQ_MAX_TOPK + 1, 0.1f,
                          out, &idx, NULL) == 0 && idx == 0,
          "hrq: value_topk > cap is clamped, not rejected");
    int finite = 1;
    for (int i = 0; i < LCM_D; i++) if (!isfinite(out[i])) finite = 0;
    CHECK(finite, "hrq: n_fine=0 produces a finite output");

    m.M_top = 0;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out, &idx, NULL) == -1,
          "hrq: M_top=0 is rejected");

    m.M_top = 2;
    m.n_fine = 1;
    m.C_fine = NULL;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 2, 0.1f, out, &idx, NULL) == -1,
          "hrq: n_fine>0 without C_fine is rejected");

    m.n_fine = 0;
    m.C_fine = t8_fine;
    CHECK(hrq_forward_v31(z, &m, 1.0f, 0, 0.1f, out, &idx, NULL) == -1,
          "hrq: value_topk<=0 is rejected");
}

/* Shared sparse setup: C0=+0.7e0, C1=+0.7e1, C2=-0.7e2; zero_vec has a
 * sentinel in e3 so the LFQ copy is observable. */
static void t8_sparse_setup(sparse_mem_t* m) {
    t8_zero(t8_sparse_C, T8_SPARSE_M * LCM_D);
    t8_zero(t8_sparse_zero, LCM_D);
    t8_sparse_zero[3] = 0.25f;
    t8_sparse_C[0] = 0.7f;
    t8_sparse_C[(size_t)1 * LCM_D + 1] = 0.7f;
    t8_sparse_C[(size_t)2 * LCM_D + 2] = -0.7f;

    memset(m, 0, sizeof(*m));
    m->C = t8_sparse_C;
    m->M = T8_SPARSE_M;
    m->zero_vec = t8_sparse_zero;
}

static void test_sparse_lfq(void) {
    sparse_mem_t m;
    t8_sparse_setup(&m);

    float z[LCM_D];
    t8_zero(z, LCM_D);
    z[0] = 0.6f;                 /* offset from C0, nearest code is C0 */
    float out[LCM_D];
    int idx = -1;

    /* d_min ≈ 0.185 < 1.5·0.2: keep the non-zero code. */
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.2f, 0, out, &idx) == 0
              && idx == 1,
          "sparse: d_min < lambda*d_top -> non-zero idx (1 = C[0])");
    CHECK(fabsf(out[0] - 0.7f) < 1e-6f,
          "sparse: non-zero output is the retrieved C row");

    /* Same input/codebook, smaller d_top: d_min > 1.5·0.05 -> zero_vec. */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.05f, 0, out, &idx) == 0
              && idx == 0,
          "sparse: d_min > lambda*d_top at 1.5 -> zero idx 0");
    CHECK(out[3] == 0.25f && out[0] == 0.0f && out[1] == 0.0f
              && out[2] == 0.0f,
          "sparse: LFQ emits zero_vec verbatim");

    /* Training disables LFQ at a d_top that fires in inference. */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.05f, 1, out, &idx) == 0
              && idx == 0,
          "sparse: training=1 disables LFQ at a firing d_top");
    CHECK(fabsf(out[0] - 0.7f) < 1e-6f,
          "sparse: training output is the retrieved C row");

    /* d_top <= 0 is the "HRQ did not activate" signal. */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.0f, 0, out, &idx) == 0
              && idx == 1,
          "sparse: d_top<=0 disables LFQ");
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, -0.5f, 0, out, &idx) == 0
              && idx == 1,
          "sparse: negative d_top disables LFQ");

    /* A large lambda_lfq lifts the threshold above d_min. */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 100.0f, 0.0f, 0.05f, 0, out, &idx) == 0
              && idx == 1,
          "sparse: lambda_lfq scales the LFQ threshold");

    /* zero_vec can win the Euclidean argmin on its own; when LFQ is off the
     * joint retrieval result must be kept (JAX inference parity). */
    float z0[LCM_D];
    t8_zero(z0, LCM_D);
    z0[0] = 0.05f;
    idx = -1;
    CHECK(sparse_forward_c(z0, &m, 1.5f, 0.0f, 0.5f, 0, out, &idx) == 0
              && idx == 0,
          "sparse: zero_vec Euclidean winner kept when LFQ is off");
    CHECK(out[3] == 0.25f && out[0] == 0.0f,
          "sparse: zero-argmin output is zero_vec");
}

static void test_sparse_value_bias(void) {
    sparse_mem_t m;
    t8_sparse_setup(&m);
    float v[T8_SPARSE_M] = {-1.0f, 1.0f, 0.0f};
    m.value = v;

    float z[LCM_D];
    t8_zero(z, LCM_D);
    z[0] = 0.6f;
    float out[LCM_D];
    int idx = -1;

    /* Euclidean winner is C0 (idx 1). Value rescoring over
     * [zero, C0, C1, C2] with alpha=2 makes C1 (idx 2) the argmax. */
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.2f, 0, out, &idx) == 0
              && idx == 1,
          "sparse value: alpha=0 keeps the Euclidean winner");
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 2.0f, 0.2f, 0, out, &idx) == 0
              && idx == 2,
          "sparse value: alpha>0 overrides Euclidean with C1 (idx 2)");
    CHECK(fabsf(out[1] - 0.7f) < 1e-6f && out[0] == 0.0f,
          "sparse value: biased output is C1");

    /* Training uses C only (no zero_vec, 0-based idx). */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 2.0f, 0.0f, 1, out, &idx) == 0
              && idx == 1,
          "sparse value: training picks C1 (0-based idx 1)");
    CHECK(fabsf(out[1] - 0.7f) < 1e-6f,
          "sparse value: training output is C1");

    /* LFQ remains independent of the value path. */
    idx = -1;
    CHECK(sparse_forward_c(z, &m, 1.5f, 2.0f, 0.05f, 0, out, &idx) == 0
              && idx == 0,
          "sparse value: LFQ still overrides a value-biased idx");
}

static void test_sparse_boundaries(void) {
    sparse_mem_t m;
    t8_sparse_setup(&m);
    float z[LCM_D];
    t8_zero(z, LCM_D);
    float out[LCM_D];
    int idx = -1;

    m.M = 0;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.0f, 0, out, &idx) == -1,
          "sparse: M=0 is rejected");
    m.M = T8_SPARSE_M;
    m.C = NULL;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.0f, 0, out, &idx) == -1,
          "sparse: NULL C is rejected");
    m.C = t8_sparse_C;
    m.zero_vec = NULL;
    CHECK(sparse_forward_c(z, &m, 1.5f, 0.0f, 0.0f, 0, out, &idx) == -1,
          "sparse: inference without zero_vec is rejected");
}

static void test_routing_hard(void) {
    t8_zero(t8_route_C, T8_NLAT * LCM_D);
    t8_zero(t8_route_W, LCM_D * T8_NLAT);
    t8_route_C[0] = 0.5f;                       /* C0 = 0.5 e0 */
    t8_route_C[(size_t)1 * LCM_D + 1] = 0.5f;   /* C1 = 0.5 e1 */
    t8_route_C[(size_t)2 * LCM_D + 2] = 0.5f;   /* C2 = 0.5 e2 */

    /* z_route must be C_route[idx], not z: with z=(0.03,0.47,0.02) the raw
     * logits are {3.47, 0, -2.06} (argmax 0) while C1=(0,0.5,0) gives
     * {0.5, 0, 1.0} (argmax 2). */
    t8_route_W[(size_t)1 * T8_NLAT + 0] = 1.0f;
    t8_route_W[(size_t)1 * T8_NLAT + 1] = 0.0f;
    t8_route_W[(size_t)1 * T8_NLAT + 2] = 2.0f;   /* argmax = 2 */
    t8_route_W[0 * T8_NLAT + 0] = 100.0f;         /* wrong-z path: argmax 0 */
    t8_route_W[0 * T8_NLAT + 1] = 0.0f;
    t8_route_W[0 * T8_NLAT + 2] = -100.0f;

    float z[LCM_D];
    t8_zero(z, LCM_D);
    z[0] = 0.03f;
    z[1] = 0.47f;
    z[2] = 0.02f;                 /* nearest row is still C1 */
    float mask[T8_NLAT];
    routing_gate_c(z, t8_route_C, t8_route_W, T8_NLAT, mask);
    CHECK(mask[0] == 0.0f && mask[1] == 0.0f && mask[2] == 1.0f,
          "routing hard: one-hot argmax of C_route[idx] (index 2)");

    float mask2[T8_NLAT];
    routing_gate_c(z, t8_route_C, t8_route_W, T8_NLAT, mask2);
    CHECK(memcmp(mask, mask2, sizeof(mask)) == 0,
          "routing hard: deterministic across calls (no Gumbel noise)");

    t8_zero(z, LCM_D);
    z[0] = 0.47f;
    z[1] = 0.03f;
    z[2] = 0.02f;                 /* nearest row is C0 */
    routing_gate_c(z, t8_route_C, t8_route_W, T8_NLAT, mask);
    CHECK(mask[0] == 1.0f && mask[1] == 0.0f && mask[2] == 0.0f,
          "routing hard: nearest C row changes the argmax (index 0)");
}

static void test_routing_invalid(void) {
    float z[LCM_D];
    t8_zero(z, LCM_D);
    z[1] = 0.5f;

    /* n_lattices > LCM_MAX_LATTICES must be rejected and the output cleared
     * (up to the cap) instead of reading past C_route. */
    float mask[LCM_MAX_LATTICES + 1];
    for (int i = 0; i <= LCM_MAX_LATTICES; i++) mask[i] = 1.0f;
    routing_gate_c(z, t8_route_C, t8_route_W, LCM_MAX_LATTICES + 1, mask);
    int cleared = 1;
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        if (mask[i] != 0.0f) cleared = 0;
    CHECK(cleared, "routing invalid: n_lattices > cap is rejected and cleared");

    /* NULL mask and n_lattices <= 0 must not crash. */
    routing_gate_c(z, t8_route_C, t8_route_W, T8_NLAT, NULL);
    routing_gate_c(z, t8_route_C, t8_route_W, 0, mask);
    CHECK(1, "routing invalid: NULL mask / n<=0 do not crash");
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

/* ── v3.1 LowRank / Manifold / Contrast / Binding forwards (task 9) ─────── */

static unsigned int t9_seed;

static float t9_rand(void) {
    t9_seed = t9_seed * 1664525u + 1013904223u;
    return (float)((t9_seed >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

/* Deterministic pseudo-random unit vector (HRR recovery is magnitude-blind, so
 * the codebook rows themselves must be unit-norm for NN retrieval to be
 * meaningful). */
static void t9_unit(float* v) {
    float n2 = 0.0f;
    for (int i = 0; i < LCM_D; i++) {
        v[i] = t9_rand();
        n2 += v[i] * v[i];
    }
    float inv = 1.0f / sqrtf(n2);
    for (int i = 0; i < LCM_D; i++) v[i] *= inv;
}

static float t9_max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float t9_cosine(const float* a, const float* b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < LCM_D; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
}

static int t9_all_zero(const float* p, int n) {
    for (int i = 0; i < n; i++)
        if (p[i] != 0.0f) return 0;
    return 1;
}

/* ── LowRank ────────────────────────────────────────────────────────────── */

static void test_lowrank_multilayer(void) {
    enum { M = 2, R = 3, L = 3 };
    static float U[L * M * R];
    static float V[LCM_D * R];
    static float z[LCM_D];
    static float out[LCM_D];
    int ranks[L] = {1, 2, 3};
    int idx[L];

    memset(U, 0, sizeof(U));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));

    /* Shared basis: V = diag(2, 1, 1) in the first three coordinates. */
    V[0 * R + 0] = 2.0f;
    V[1 * R + 1] = 1.0f;
    V[2 * R + 2] = 1.0f;

    /* C_k = U_k @ V[:, :r_k]^T.
     * layer 0 (rank 1): rows [1.0,0,0], [1.4,0,0]
     * layer 1 (rank 2): rows [0.1,0.35,0], [1.0,0.3,0]
     * layer 2 (rank 3): rows [1.0,0.5,0.5], [0,0.05,0] */
    U[(0 * M + 0) * R + 0] = 0.5f;
    U[(0 * M + 0) * R + 1] = 9.0f;   /* outside rank 1: must be ignored */
    U[(0 * M + 1) * R + 0] = 0.7f;
    U[(0 * M + 1) * R + 2] = 9.0f;   /* outside rank 1: must be ignored */
    U[(1 * M + 0) * R + 0] = 0.05f;
    U[(1 * M + 0) * R + 1] = 0.35f;
    U[(1 * M + 0) * R + 2] = 9.0f;   /* outside rank 2: must be ignored */
    U[(1 * M + 1) * R + 0] = 0.5f;
    U[(1 * M + 1) * R + 1] = 0.3f;
    U[(2 * M + 0) * R + 0] = 0.5f;
    U[(2 * M + 0) * R + 1] = 0.5f;
    U[(2 * M + 0) * R + 2] = 0.5f;
    U[(2 * M + 1) * R + 1] = 0.05f;

    lowrank_mem_t m;
    memset(&m, 0, sizeof(m));
    m.U = U; m.V = V; m.ranks = ranks;
    m.n_layers = L; m.M_lr = M; m.r_max = R;
    m.value = NULL;

    z[0] = 1.1f;
    z[1] = 0.4f;
    lowrank_forward_c(z, &m, 0.0f, out, idx);

    /* Residual chain: r0=z → row 0 (d²=0.01 vs 0.09);
     * r1=[0.1,0.4] → layer-1 row 0 (0.0025 vs 0.82);
     * r2=[0,0.05,0] → layer-2 row 1 (exact).  If a layer skipped the
     * subtraction, layer 1 would pick row 1 instead. */
    CHECK(idx[0] == 0 && idx[1] == 0 && idx[2] == 1,
          "lowrank: per-layer residual-chain indices [0,0,1]");
    CHECK(fabsf(out[0] - 1.1f) < 1e-5f && fabsf(out[1] - 0.4f) < 1e-5f,
          "lowrank: out = Σ c_k over all three layers (shared V)");
    CHECK(fabsf(out[2]) < 1e-6f && fabsf(out[3]) < 1e-6f,
          "lowrank: rank-prefix codebooks do not leak extra U columns");
}

static void test_lowrank_value_bias(void) {
    enum { M = 2, R = 1, L = 1 };
    static float U[M * R];
    static float V[LCM_D * R];
    static float z[LCM_D];
    static float out[LCM_D];
    static float val[M] = {0.0f, 1.0f};
    int ranks[L] = {1};
    int idx = -1;

    memset(U, 0, sizeof(U));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));
    V[0] = 1.0f;
    U[0] = 0.5f;
    U[1] = 0.7f;
    z[0] = 0.55f;

    lowrank_mem_t m;
    memset(&m, 0, sizeof(m));
    m.U = U; m.V = V; m.ranks = ranks;
    m.n_layers = L; m.M_lr = M; m.r_max = R;
    m.value = val;

    lowrank_forward_c(z, &m, 0.0f, out, &idx);
    CHECK(idx == 0 && fabsf(out[0] - 0.5f) < 1e-6f,
          "lowrank: alpha_val=0 keeps the Euclidean winner");

    /* avg d² = 0.0125; with alpha=4 row 1 scores
     * -0.0225 + 4·tanh(1)·0.0125 ≈ +0.0156, beating row 0's -0.0025. */
    lowrank_forward_c(z, &m, 4.0f, out, &idx);
    CHECK(idx == 1 && fabsf(out[0] - 0.7f) < 1e-6f,
          "lowrank: R22 value bias picks the slightly farther row");
}

/* ── Manifold ───────────────────────────────────────────────────────────── */

/* Direct b.md §4.4 reference for a fixed selected row. */
static void t9_manifold_ref(const float* z, const float* C, const float* T,
                            int idx, int t_dim, float* out) {
    float z_u[LCM_D], z_P[LCM_D], c_P[LCM_D], r[LCM_D], proj[LCM_D];
    safe_unit_c(z, z_u, LCM_D, 1e-8f);
    exp_map_c(z_u, z_P, LCM_D, 1.0f);
    exp_map_c(C + (size_t)idx * LCM_D, c_P, LCM_D, 1.0f);

    for (int i = 0; i < LCM_D; i++) r[i] = z_P[i] - c_P[i];

    const float* T_j = T + (size_t)idx * LCM_D * t_dim;
    float tp[8];
    for (int k = 0; k < t_dim; k++) {
        tp[k] = 0.0f;
        for (int i = 0; i < LCM_D; i++) tp[k] += T_j[i * t_dim + k] * r[i];
    }
    for (int i = 0; i < LCM_D; i++) {
        proj[i] = 0.0f;
        for (int k = 0; k < t_dim; k++) proj[i] += T_j[i * t_dim + k] * tp[k];
    }
    for (int i = 0; i < LCM_D; i++) c_P[i] += proj[i];
    log_map_c(c_P, out, LCM_D, 1.0f);
}

static void test_lowrank_shared_basis_nondiagonal(void) {
    enum { M = 1, R = 2, L = 2 };
    static float U[L * M * R];
    static float V[LCM_D * R];
    static float z[LCM_D];
    static float out[LCM_D];
    static float ref[LCM_D];
    int ranks[L] = {1, 2};
    int idx[L] = {-1, -1};

    memset(U, 0, sizeof(U));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));
    memset(ref, 0, sizeof(ref));

    /* Non-symmetric V and off-diagonal rows: C_k = U_k @ V[:, :r_k]^T.
     * A V^T / transposed-basis implementation produces different rows. */
    V[0 * R + 0] = 1.0f;  V[0 * R + 1] = 0.5f;
    V[1 * R + 0] = 0.25f; V[1 * R + 1] = 1.0f;
    V[2 * R + 0] = 0.75f; V[2 * R + 1] = -0.25f;
    U[0 * R + 0] = 1.0f;  U[0 * R + 1] = 0.5f;   /* rank 1: col 1 unused */
    U[1 * R + 0] = 0.1f;  U[1 * R + 1] = 0.2f;

    z[0] = 1.2f;
    z[1] = 0.7f;
    z[2] = 0.7f;

    lowrank_mem_t m;
    memset(&m, 0, sizeof(m));
    m.U = U; m.V = V; m.ranks = ranks;
    m.n_layers = L; m.M_lr = M; m.r_max = R;
    m.value = NULL;

    lowrank_forward_c(z, &m, 0.0f, out, idx);

    for (int d = 0; d < LCM_D; d++) {
        float c0 = U[0 * R + 0] * V[(size_t)d * R + 0];
        float c1 = U[1 * R + 0] * V[(size_t)d * R + 0]
                 + U[1 * R + 1] * V[(size_t)d * R + 1];
        ref[d] = c0 + c1;
    }
    CHECK(t9_max_abs_diff(out, ref, LCM_D) < 1e-5f,
          "lowrank: non-diagonal shared V matches U_k @ V[:, :r_k]^T");
    CHECK(idx[0] == 0 && idx[1] == 0,
          "lowrank: non-diagonal basis keeps both layers active");
}

static void test_manifold_formula_and_poincare(void) {
    enum { M = 2, TD = 2 };
    static float C[M * LCM_D];
    static float T[M * LCM_D * TD];
    static float z[LCM_D];
    static float out[LCM_D];
    static float ref[LCM_D];
    static float val[M];
    int idx = -1;

    memset(C, 0, sizeof(C));
    memset(T, 0, sizeof(T));
    memset(z, 0, sizeof(z));

    /* C0 radial at 0.4; C1 at 0.95, angle 0.5. The query is 0.9 e0.
     * Raw Euclidean prefers C1 (0.212 vs 0.250); Poincaré prefers C0
     * (sim 0.811 vs 1.456). */
    C[0 * LCM_D + 0] = 0.4f;
    C[1 * LCM_D + 0] = 0.95f * cosf(0.5f);
    C[1 * LCM_D + 1] = 0.95f * sinf(0.5f);
    for (int j = 0; j < M; j++) {
        T[(j * LCM_D + 0) * TD + 0] = 1.0f;
        T[(j * LCM_D + 1) * TD + 1] = 1.0f;
    }
    z[0] = 0.9f;

    manifold_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C = C; m.M = M; m.T = T; m.t_dim = TD; m.value = NULL;

    manifold_forward_c(z, &m, 0.0f, out, &idx);
    t9_manifold_ref(z, C, T, 0, TD, ref);

    CHECK(idx == 0, "manifold: Poincaré retrieval picks C0");
    CHECK(t9_max_abs_diff(out, ref, LCM_D) < 1e-5f,
          "manifold: out matches log_map(c + T Tᵀ(z_P - c)) reference");

    /* The same data's raw Euclidean argmin is C1 — proves the retrieval
     * really uses poincare_similarity, not ||z-C||². */
    float d0 = 0.0f, d1 = 0.0f;
    for (int i = 0; i < LCM_D; i++) {
        float a = z[i] - C[i];
        float b = z[i] - C[LCM_D + i];
        d0 += a * a;
        d1 += b * b;
    }
    CHECK(d1 < d0 && idx != 1,
          "manifold: Poincaré retrieval differs from Euclidean");

    /* R22 value path (JAX value_biased_retrieve): tangent-space Euclidean
     * score, then the selected row is exp_mapped. v={0,1}, alpha=2 flips
     * the winner to C1. */
    val[0] = 0.0f; val[1] = 1.0f;
    m.value = val;
    manifold_forward_c(z, &m, 2.0f, out, &idx);
    t9_manifold_ref(z, C, T, 1, TD, ref);
    CHECK(idx == 1 && t9_max_abs_diff(out, ref, LCM_D) < 1e-5f,
          "manifold: value bias selects C1 and follows the same projection");
}

/* ── Contrast ───────────────────────────────────────────────────────────── */

/* Independent residual-VQ reference for one family. */
static void t9_contrast_chain(const float* z, const float* C, int n_layers,
                              int M, const float* value, float alpha_val,
                              float* out) {
    float r[LCM_D];
    memcpy(r, z, sizeof(r));
    memset(out, 0, sizeof(float) * LCM_D);

    for (int l = 0; l < n_layers; l++) {
        const float* cb = C + (size_t)l * M * LCM_D;
        float d2[8];
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            d2[j] = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = r[i] - cb[(size_t)j * LCM_D + i];
                d2[j] += diff * diff;
            }
            sum += d2[j];
        }
        int best = 0;
        if (value && alpha_val > 0.0f) {
            float avg = sum / (float)M;
            float best_score = -INFINITY;
            for (int j = 0; j < M; j++) {
                float score = -d2[j]
                    + alpha_val * tanhf(value[j]) * avg;
                if (score > best_score) { best_score = score; best = j; }
            }
        } else {
            float best_d2 = INFINITY;
            for (int j = 0; j < M; j++)
                if (d2[j] < best_d2) { best_d2 = d2[j]; best = j; }
        }
        for (int i = 0; i < LCM_D; i++) {
            out[i] += cb[(size_t)best * LCM_D + i];
            r[i] -= cb[(size_t)best * LCM_D + i];
        }
    }
}

static void test_contrast_dual_codebook(void) {
    enum { M = 2, L = 2 };
    static float Ca[L * M * LCM_D];
    static float Cb[L * M * LCM_D];
    static float z[LCM_D];
    static float out[LCM_D];
    static float oa[LCM_D], ob[LCM_D], ref[LCM_D];

    memset(Ca, 0, sizeof(Ca));
    memset(Cb, 0, sizeof(Cb));
    memset(z, 0, sizeof(z));

    /* Layer 0 removes 0.1e0 / 0.3e2; layer 1 then picks 0.2e1 / 0.4e3. */
    Ca[(0 * M + 0) * LCM_D + 0] = 0.1f;
    Ca[(0 * M + 1) * LCM_D + 0] = 0.11f;
    Ca[(1 * M + 0) * LCM_D + 1] = 0.2f;
    Ca[(1 * M + 1) * LCM_D + 2] = 0.21f;
    Cb[(0 * M + 0) * LCM_D + 2] = 0.3f;
    Cb[(0 * M + 1) * LCM_D + 2] = 0.31f;
    Cb[(1 * M + 0) * LCM_D + 3] = 0.4f;
    Cb[(1 * M + 1) * LCM_D + 3] = 0.41f;
    z[0] = 0.02f;

    contrast_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_a = Ca; m.C_b = Cb; m.M = M; m.n_layers = L;
    m.value_a = NULL; m.value_b = NULL;

    contrast_forward_c(z, &m, 0.0f, out);

    t9_contrast_chain(z, Ca, L, M, NULL, 0.0f, oa);
    t9_contrast_chain(z, Cb, L, M, NULL, 0.0f, ob);
    for (int i = 0; i < LCM_D; i++) ref[i] = (oa[i] + ob[i]) * 0.5f;

    CHECK(t9_max_abs_diff(out, ref, LCM_D) < 1e-6f,
          "contrast: out == (o_a + o_b)/2");
    CHECK(fabsf(out[0] - 0.05f) < 1e-6f && fabsf(out[1] - 0.1f) < 1e-6f
              && fabsf(out[2] - 0.15f) < 1e-6f
              && fabsf(out[3] - 0.2f) < 1e-6f,
          "contrast: both codebooks and both layers contribute");
}

static void test_contrast_value_bias_families(void) {
    enum { M = 2, L = 1 };
    static float Ca[M * LCM_D];
    static float Cb[M * LCM_D];
    static float z[LCM_D];
    static float out[LCM_D];
    static float va[M] = {0.0f, 1.0f};

    memset(Ca, 0, sizeof(Ca));
    memset(Cb, 0, sizeof(Cb));
    memset(z, 0, sizeof(z));
    Ca[0] = 0.5f; Ca[1 * LCM_D + 0] = 0.7f;
    Cb[0] = 0.5f; Cb[1 * LCM_D + 0] = 0.7f;
    z[0] = 0.55f;

    contrast_mem_t m;
    memset(&m, 0, sizeof(m));
    m.C_a = Ca; m.C_b = Cb; m.M = M; m.n_layers = L;
    m.value_a = va; m.value_b = NULL;

    contrast_forward_c(z, &m, 0.0f, out);
    CHECK(fabsf(out[0] - 0.5f) < 1e-6f,
          "contrast: alpha_val=0 uses both Euclidean winners");

    contrast_forward_c(z, &m, 4.0f, out);
    CHECK(fabsf(out[0] - 0.6f) < 1e-6f,
          "contrast: value_a bias flips only family a");
}

/* ── Binding ────────────────────────────────────────────────────────────── */

static void test_binding_unbind_and_cval(void) {
    enum { L = 3, M = 2 };
    static float key_cb[L * M * LCM_D];
    static float val_cb[L * M * LCM_D];
    static float bind_cb[L * M * LCM_D];
    static float A_k[LCM_D];
    static float A_v[LCM_D];
    static float V[LCM_D];
    static float z[LCM_D];
    static float out[LCM_D];
    static float bq[LCM_D], kq[LCM_D];
    static float b9[LCM_D], b1[LCM_D], bcheck[LCM_D], pair[LCM_D];
    static float rec[LCM_D];
    static float v[LCM_D];

    memset(key_cb, 0, sizeof(key_cb));
    memset(val_cb, 0, sizeof(val_cb));
    memset(bind_cb, 0, sizeof(bind_cb));
    memset(A_k, 0, sizeof(A_k));
    memset(A_v, 0, sizeof(A_v));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));

    /* Every family has M rows per layer (binding_mem_t has one M). Rows 1 are
     * far away, so the residual chains select row 0 in every layer. */
    t9_seed = 0x9E3779B9u;
    const float* kp[L];
    for (int l = 0; l < L; l++) {
        float* k0 = key_cb + ((size_t)l * M) * LCM_D;
        float* k1 = k0 + LCM_D;
        t9_unit(k0);
        memcpy(k1, k0, sizeof(vec_t));
        k1[0] += 100.0f;
        kp[l] = k0;
    }

    t9_unit(v);
    for (int l = 0; l < L; l++) {
        float* v0 = val_cb + ((size_t)l * M) * LCM_D;
        float* v1 = v0 + LCM_D;
        if (l == L - 1) memcpy(v0, v, sizeof(vec_t));
        memcpy(v1, v0, sizeof(vec_t));
        v1[0] += 100.0f;
    }
    memcpy(A_k, kp[0], sizeof(A_k));  /* z_k = k0 (V=e0, A_k row 0) */
    memcpy(A_v, v, sizeof(A_v));

    V[0] = 1.0f;   /* V is [D][1]; V^T z = z[0] */
    z[0] = 1.0f;

    const float* vp[L];
    for (int l = 0; l < L; l++) vp[l] = val_cb + ((size_t)l * M) * LCM_D;

    /* Independent 9-pair reference: sum pairwise hrr_bind calls instead of
     * relying on hrr_bind's own multi-pair accumulation. */
    memset(b9, 0, sizeof(b9));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            hrr_bind(&kp[i], 1, &vp[j], 1, pair, LCM_D);
            for (int d = 0; d < LCM_D; d++) b9[d] += pair[d];
        }
    }
    hrr_bind(&kp[0], 1, &vp[2], 1, b1, LCM_D);
    /* Plant the 9-pair sum vs a 1-pair competitor. */
    memcpy(bind_cb, b9, sizeof(b9));                 /* layer 0, row 0 */
    memcpy(bind_cb + LCM_D, b1, sizeof(b1));         /* layer 0, row 1 */
    for (int l = 1; l < L; l++) {                    /* layers 1/2: 0, far */
        float* r1 = bind_cb + ((size_t)l * M + 1) * LCM_D;
        r1[0] = 100.0f;
    }

    binding_mem_t m;
    memset(&m, 0, sizeof(m));
    m.key_cb = key_cb; m.val_cb = val_cb; m.bind_cb = bind_cb;
    m.M = M; m.n_layers = L; m.r_max = 1;
    m.A_k = A_k; m.A_v = A_v;
    m.value_key = NULL; m.value_val = NULL; m.value_bind = NULL;

    binding_forward_c(z, &m, V, 0.0f, out, bq, kq, NULL);

    /* k_q = k0+k1+k2; b_q = b_raw (bind layer 0), then zero bind rows. */
    memset(bcheck, 0, sizeof(bcheck));
    for (int l = 0; l < L; l++)
        for (int i = 0; i < LCM_D; i++) bcheck[i] += kp[l][i];
    CHECK(t9_max_abs_diff(kq, bcheck, LCM_D) < 1e-6f,
          "binding: k_q = Σ key-layer quantised rows");
    CHECK(t9_max_abs_diff(bq, b9, LCM_D) < 1e-5f,
          "binding: b_q = 9-pair cross-layer HRR sum (not 1 pair)");

    /* NN over the concatenated C_val must land on layer 2's row 0 = v. */
    hrr_unbind(b9, kq, rec, LCM_D);
    int best = -1;
    float best_d2 = INFINITY;
    for (int l = 0; l < L; l++) {
        for (int j = 0; j < M; j++) {
            const float* row = val_cb + ((size_t)l * M + j) * LCM_D;
            float d2 = 0.0f;
            for (int i = 0; i < LCM_D; i++) {
                float diff = rec[i] - row[i];
                d2 += diff * diff;
            }
            if (d2 < best_d2) { best_d2 = d2; best = l * M + j; }
        }
    }
    CHECK(best == (L - 1) * M,
          "binding: C_val concatenation searches every val_cb layer");
    CHECK(t9_max_abs_diff(out, v, LCM_D) < 1e-6f,
          "binding: unbind reads back v from the value codebook");

    /* Primitive round-trip on one pair. */
    hrr_bind(&kp[0], 1, &vp[2], 1, b1, LCM_D);
    hrr_unbind(b1, kp[0], rec, LCM_D);
    CHECK(t9_cosine(rec, v) > 0.6f,
          "binding: unbind(bind(k,v),k) ≈ v (cosine > 0.6)");
}

static void test_binding_projection_direction(void) {
    enum { L = 1, M = 2, R = 2 };
    static float key_cb[L * M * LCM_D];
    static float val_cb[L * M * LCM_D];
    static float bind_cb[L * M * LCM_D];
    static float A_k[R * LCM_D];
    static float A_v[R * LCM_D];
    static float V[LCM_D * R];
    static float z[LCM_D];
    static float out[LCM_D];
    static float kq[LCM_D];
    static float correct[LCM_D], transposed[LCM_D];
    static float u[2], t[2];

    memset(key_cb, 0, sizeof(key_cb));
    memset(val_cb, 0, sizeof(val_cb));
    memset(bind_cb, 0, sizeof(bind_cb));
    memset(A_k, 0, sizeof(A_k));
    memset(A_v, 0, sizeof(A_v));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));

    /* Non-symmetric V and A_k, non-basis z. b.md §4.5:
     * z_k = V @ A_k @ z = W_k z. The old code returned W_k^T z. */
    V[0 * R + 0] = 1.0f;  V[0 * R + 1] = 0.25f;
    V[1 * R + 0] = 0.5f;
    V[2 * R + 0] = 0.75f; V[2 * R + 1] = 1.0f;
    A_k[0 * LCM_D + 0] = 1.0f;
    A_k[0 * LCM_D + 1] = 0.5f;
    A_k[1 * LCM_D + 2] = 1.0f;
    z[0] = 1.0f; z[1] = 0.3f; z[2] = 0.7f;

    /* correct = V @ (A_k z) */
    memset(u, 0, sizeof(u));
    memset(correct, 0, sizeof(correct));
    for (int p = 0; p < R; p++)
        for (int i = 0; i < LCM_D; i++)
            u[p] += A_k[p * LCM_D + i] * z[i];
    for (int j = 0; j < LCM_D; j++)
        for (int p = 0; p < R; p++)
            correct[j] += V[j * R + p] * u[p];

    /* transposed = A_k^T (V^T z) = W_k^T z */
    memset(t, 0, sizeof(t));
    memset(transposed, 0, sizeof(transposed));
    for (int p = 0; p < R; p++)
        for (int i = 0; i < LCM_D; i++)
            t[p] += V[i * R + p] * z[i];
    for (int j = 0; j < LCM_D; j++)
        for (int p = 0; p < R; p++)
            transposed[j] += A_k[p * LCM_D + j] * t[p];

    CHECK(t9_max_abs_diff(correct, transposed, LCM_D) > 1e-3f,
          "binding projection: crafted data separates W_k z from W_k^T z");
    memcpy(key_cb, correct, sizeof(correct));
    memcpy(key_cb + LCM_D, transposed, sizeof(transposed));

    binding_mem_t m;
    memset(&m, 0, sizeof(m));
    m.key_cb = key_cb; m.val_cb = val_cb; m.bind_cb = bind_cb;
    m.M = M; m.n_layers = L; m.r_max = R;
    m.A_k = A_k; m.A_v = A_v;
    m.value_key = NULL; m.value_val = NULL; m.value_bind = NULL;

    binding_forward_c(z, &m, V, 0.0f, out, NULL, kq, NULL);

    CHECK(t9_max_abs_diff(kq, correct, LCM_D) < 1e-6f,
          "binding: z_k = V @ A_k @ z (W_k z), not W_k^T z");
    CHECK(t9_max_abs_diff(kq, transposed, LCM_D) > 1e-3f,
          "binding: transposed projection is not selected");
}

static void test_binding_value_families(void) {
    enum { L = 1, M = 2, R = 1 };
    static float key_cb[L * M * LCM_D];
    static float val_cb[L * M * LCM_D];
    static float bind_cb[L * M * LCM_D];
    static float A_k[R * LCM_D];
    static float A_v[R * LCM_D];
    static float V[LCM_D * R];
    static float z[LCM_D];
    static float out[LCM_D];
    static float kq[LCM_D], bq[LCM_D];
    static float v0[LCM_D], v1[LCM_D];
    static float B0[LCM_D], B1[LCM_D];
    static float k[LCM_D];
    static float vk[M] = {0.0f, 1.0f};
    static float vv[M] = {0.0f, 1.0f};
    static float vb[M] = {0.0f, 1.0f};

    binding_mem_t m;
    memset(&m, 0, sizeof(m));
    m.key_cb = key_cb; m.val_cb = val_cb; m.bind_cb = bind_cb;
    m.M = M; m.n_layers = L; m.r_max = R;
    m.A_k = A_k; m.A_v = A_v;

    /* ── value_key: biased key row wins, k_q exposes it ── */
    memset(key_cb, 0, sizeof(key_cb));
    memset(val_cb, 0, sizeof(val_cb));
    memset(bind_cb, 0, sizeof(bind_cb));
    memset(A_k, 0, sizeof(A_k));
    memset(A_v, 0, sizeof(A_v));
    memset(V, 0, sizeof(V));
    memset(z, 0, sizeof(z));
    V[0] = 1.0f;
    z[0] = 1.0f;
    A_k[0] = 0.55f;                 /* z_k = A_k z = 0.55 e0 */
    key_cb[0] = 0.5f;               /* row 0 */
    key_cb[LCM_D] = 0.7f;           /* row 1 */
    m.value_key = NULL; m.value_val = NULL; m.value_bind = NULL;
    binding_forward_c(z, &m, V, 0.0f, out, NULL, kq, NULL);
    CHECK(fabsf(kq[0] - 0.5f) < 1e-6f,
          "binding value_key: alpha=0 keeps the Euclidean key row");
    m.value_key = vk;
    binding_forward_c(z, &m, V, 4.0f, out, NULL, kq, NULL);
    CHECK(fabsf(kq[0] - 0.7f) < 1e-6f,
          "binding value_key: R22 bias selects the farther key row");

    /* ── value_val: b_raw/b_q expose the selected value row ── */
    memset(key_cb, 0, sizeof(key_cb));
    memset(val_cb, 0, sizeof(val_cb));
    memset(bind_cb, 0, sizeof(bind_cb));
    t9_seed = 0x51ED270Bu;
    t9_unit(k);
    memcpy(key_cb, k, sizeof(k));               /* both key rows = k */
    memcpy(key_cb + LCM_D, k, sizeof(k));
    A_k[0] = 0.5f;                             /* z_k = 0.5 e0 */
    memset(v0, 0, sizeof(v0));
    memset(v1, 0, sizeof(v1));
    v0[0] = 0.5f;
    v1[0] = 0.7f;
    memcpy(val_cb, v0, sizeof(v0));
    memcpy(val_cb + LCM_D, v1, sizeof(v1));
    A_v[0] = 0.55f;                            /* z_v = 0.55 e0 */
    const float* key_ptr[1] = {key_cb};
    const float* val0_ptr[1] = {val_cb};
    const float* val1_ptr[1] = {val_cb + LCM_D};
    hrr_bind(key_ptr, 1, val0_ptr, 1, B0, LCM_D);
    hrr_bind(key_ptr, 1, val1_ptr, 1, B1, LCM_D);
    memcpy(bind_cb, B0, sizeof(B0));
    memcpy(bind_cb + LCM_D, B1, sizeof(B1));

    m.value_key = NULL; m.value_val = NULL; m.value_bind = NULL;
    binding_forward_c(z, &m, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_max_abs_diff(bq, B0, LCM_D) < 1e-5f,
          "binding value_val: alpha=0 quantises the Euclidean value row");
    m.value_val = vv;
    binding_forward_c(z, &m, V, 4.0f, out, bq, kq, NULL);
    CHECK(t9_max_abs_diff(bq, B1, LCM_D) < 1e-5f,
          "binding value_val: R22 bias changes the val code and b_raw");

    /* ── value_bind: b_q selects the biased binding row ── */
    m.value_val = NULL; m.value_bind = NULL;
    binding_forward_c(z, &m, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_max_abs_diff(bq, B0, LCM_D) < 1e-5f,
          "binding value_bind: alpha=0 picks the Euclidean bind row");
    m.value_bind = vb;
    binding_forward_c(z, &m, V, 8.0f, out, bq, kq, NULL);
    CHECK(t9_max_abs_diff(bq, B1, LCM_D) < 1e-5f,
          "binding value_bind: R22 bias picks the other bind row");
}

static void test_layer_forwards_invalid(void) {
    enum { M = 1, R = 1, L = 1, TD = 1 };
    static float U[M * R];
    static float V[LCM_D * R];
    static float C[M * LCM_D];
    static float T[M * LCM_D * TD];
    static float Ca[M * LCM_D];
    static float Cb[M * LCM_D];
    static float key_cb[M * LCM_D];
    static float val_cb[M * LCM_D];
    static float bind_cb[M * LCM_D];
    static float A_k[R * LCM_D];
    static float A_v[R * LCM_D];
    static float z[LCM_D];
    static float out[LCM_D];
    static float bq[LCM_D], kq[LCM_D];
    int ranks[L] = {1};
    int idx[L];

    memset(U, 0, sizeof(U));
    memset(V, 0, sizeof(V));
    memset(C, 0, sizeof(C));
    memset(T, 0, sizeof(T));
    memset(Ca, 0, sizeof(Ca));
    memset(Cb, 0, sizeof(Cb));
    memset(key_cb, 0, sizeof(key_cb));
    memset(val_cb, 0, sizeof(val_cb));
    memset(bind_cb, 0, sizeof(bind_cb));
    memset(A_k, 0, sizeof(A_k));
    memset(A_v, 0, sizeof(A_v));
    memset(z, 0, sizeof(z));
    U[0] = 0.5f;
    V[0] = 1.0f;
    C[0] = 0.2f;   /* non-zero valid manifold prototype */

    lowrank_mem_t lr;
    memset(&lr, 0, sizeof(lr));
    lr.U = U; lr.V = V; lr.ranks = ranks;
    lr.n_layers = L; lr.M_lr = M; lr.r_max = R;

    lr.n_layers = 0;
    memset(out, 1, sizeof(out));
    lowrank_forward_c(z, &lr, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "lowrank invalid: n_layers=0 zeroes out");

    lr.n_layers = L; lr.M_lr = 0;
    memset(out, 1, sizeof(out));
    lowrank_forward_c(z, &lr, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "lowrank invalid: M_lr=0 zeroes out");

    lr.M_lr = M; lr.U = NULL;
    memset(out, 1, sizeof(out));
    lowrank_forward_c(z, &lr, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "lowrank invalid: NULL U zeroes out");

    lr.U = U; ranks[0] = 0;
    memset(out, 1, sizeof(out));
    idx[0] = -1;
    lowrank_forward_c(z, &lr, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D) && idx[0] == 0,
          "lowrank invalid: bad rank zeroes out/idx");

    ranks[0] = 1;
    memset(out, 1, sizeof(out));
    lowrank_forward_c(z, &lr, 0.0f, out, NULL);
    CHECK(fabsf(out[0] - 0.5f) < 1e-6f,
          "lowrank: idx=NULL is accepted");

    manifold_mem_t man;
    memset(&man, 0, sizeof(man));
    man.C = C; man.M = M; man.T = T; man.t_dim = TD;

    man.M = 0;
    memset(out, 1, sizeof(out));
    manifold_forward_c(z, &man, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "manifold invalid: M=0 zeroes out");

    man.M = M; man.t_dim = 0;
    memset(out, 1, sizeof(out));
    manifold_forward_c(z, &man, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "manifold invalid: t_dim=0 zeroes out");

    man.t_dim = TD; man.C = NULL;
    memset(out, 1, sizeof(out));
    manifold_forward_c(z, &man, 0.0f, out, idx);
    CHECK(t9_all_zero(out, LCM_D), "manifold invalid: NULL C zeroes out");

    man.C = C;
    memset(out, 1, sizeof(out));
    manifold_forward_c(z, &man, 0.0f, out, NULL);
    CHECK(fabsf(out[0] - 0.2f) < 1e-6f,
          "manifold: idx=NULL is accepted");

    contrast_mem_t con;
    memset(&con, 0, sizeof(con));
    con.C_a = Ca; con.C_b = Cb; con.M = M; con.n_layers = L;

    con.n_layers = 0;
    memset(out, 1, sizeof(out));
    contrast_forward_c(z, &con, 0.0f, out);
    CHECK(t9_all_zero(out, LCM_D), "contrast invalid: n_layers=0 zeroes out");

    con.n_layers = L; con.M = 0;
    memset(out, 1, sizeof(out));
    contrast_forward_c(z, &con, 0.0f, out);
    CHECK(t9_all_zero(out, LCM_D), "contrast invalid: M=0 zeroes out");

    con.M = M; con.C_a = NULL;
    memset(out, 1, sizeof(out));
    contrast_forward_c(z, &con, 0.0f, out);
    CHECK(t9_all_zero(out, LCM_D), "contrast invalid: NULL C_a zeroes out");

    binding_mem_t bin;
    memset(&bin, 0, sizeof(bin));
    bin.key_cb = key_cb; bin.val_cb = val_cb; bin.bind_cb = bind_cb;
    bin.M = M; bin.n_layers = L; bin.r_max = R;
    bin.A_k = A_k; bin.A_v = A_v;

    bin.n_layers = 0;
    memset(out, 1, sizeof(out));
    memset(bq, 1, sizeof(bq));
    memset(kq, 1, sizeof(kq));
    binding_forward_c(z, &bin, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D) && t9_all_zero(bq, LCM_D)
              && t9_all_zero(kq, LCM_D),
          "binding invalid: n_layers=0 zeroes out/b_q/k_q");

    bin.n_layers = LCM_MAX_BIND_LAYERS + 1;
    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D), "binding invalid: layer cap zeroes out");

    bin.n_layers = L; bin.M = 0;
    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D), "binding invalid: M=0 zeroes out");

    bin.M = M; bin.r_max = 0;
    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D), "binding invalid: r_max=0 zeroes out");

    bin.r_max = R; bin.key_cb = NULL;
    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, V, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D), "binding invalid: NULL key_cb zeroes out");

    bin.key_cb = key_cb;
    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, NULL, 0.0f, out, bq, kq, NULL);
    CHECK(t9_all_zero(out, LCM_D), "binding invalid: NULL V zeroes out");

    memset(out, 1, sizeof(out));
    binding_forward_c(z, &bin, V, 0.0f, out, NULL, NULL, NULL);
    CHECK(t9_all_zero(out, LCM_D),
          "binding: b_q=NULL and k_q=NULL are accepted");
}

/* ── Task 10: v3 dynamic DAG, loop, trace ────────────────────────────────── */

static lcm_memory_v3 t10_mem;

static int t10_alert_count;
static alert_t t10_last_alert;

static void t10_alert_cb(const alert_t* alert) {
    t10_alert_count++;
    if (alert) t10_last_alert = *alert;
}

static void t10_reset_alerts(void) {
    t10_alert_count = 0;
    memset(&t10_last_alert, 0, sizeof(t10_last_alert));
}

static float t10_route_C7[LCM_MAX_LATTICES * LCM_D];
static float t10_route_W7[LCM_D * LCM_MAX_LATTICES];
static float t10_alpha7[LCM_MAX_LATTICES];
static float t10_lr_value[T7_M];
static float t10_hrq_value[T7_M];
static int   t10_ranks128[128];
static float t10_lr_U128[128 * T7_M * T7_R];
static float t10_gv_pos[LCM_N_VALUE_PAIRS * LCM_D];
static float t10_gv_neg[LCM_N_VALUE_PAIRS * LCM_D];
static float t10_dthreat[2 * LCM_D];
static float t10_dnormal[2 * LCM_D];

/* Reset the t7 arrays to a deterministic distance spread: row r of each
 * primary codebook sits at 0.1·(r+1) on e0. */
static void t10_prepare(void) {
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
    memset(t7_ln_scale, 0, sizeof(t7_ln_scale));
    memset(t7_ln_bias, 0, sizeof(t7_ln_bias));
    memset(t7_self_C, 0, sizeof(t7_self_C));
    for (int r = 0; r < 3; r++)
        t7_self_C[r * LCM_D] = 0.1f * (float)r;
    memset(t10_route_C7, 0, sizeof(t10_route_C7));
    memset(t10_route_W7, 0, sizeof(t10_route_W7));
    memset(t10_alpha7, 0, sizeof(t10_alpha7));
    memset(t10_lr_value, 0, sizeof(t10_lr_value));
    memset(t10_hrq_value, 0, sizeof(t10_hrq_value));
    memset(t10_ranks128, 0, sizeof(t10_ranks128));
    memset(t10_lr_U128, 0, sizeof(t10_lr_U128));
    for (int i = 0; i < 128; i++) t10_ranks128[i] = 2;
    memset(t10_gv_pos, 0, sizeof(t10_gv_pos));
    memset(t10_gv_neg, 0, sizeof(t10_gv_neg));
    memset(t10_dthreat, 0, sizeof(t10_dthreat));
    memset(t10_dnormal, 0, sizeof(t10_dnormal));

    for (int r = 0; r < T7_M; r++) {
        t7_hrq_top[r * LCM_D] = 0.1f * (float)(r + 1);
        t7_sparse_C[r * LCM_D] = 0.15f * (float)(r + 1);
        t7_bind_bind[r * LCM_D] = 0.2f * (float)(r + 1);
    }
    for (int i = 0; i < T7_NLAT; i++) t7_alpha[i] = 1.0f;
    for (int i = 0; i < LCM_D; i++) t7_ln_scale[i] = 1.0f;
    for (int i = 0; i < LCM_MAX_LATTICES; i++) t10_alpha7[i] = 1.0f;
    for (int i = 0; i < LCM_N_VALUE_PAIRS; i++) {
        t10_gv_pos[i * LCM_D + i] = 1.0f;
        t10_gv_neg[i * LCM_D + i] = -1.0f;
    }

    t7_desc_t d;
    t7_fill(&d);
    CHECK(t7_bind(&t10_mem, &d) == 0, "v3 fixture binds");
    t10_reset_alerts();
}

static int t10_find_node(const dag_t* dag, op_type_t op, int lattice_id) {
    for (int i = 0; i < dag->n_nodes; i++) {
        if (dag->nodes[i].op_type == op
            && dag->nodes[i].lattice_id == lattice_id)
            return i;
    }
    return -1;
}

static void test_dag_dynamic_threshold(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;
    dag_t dag;

    /* First use: the configured initial threshold admits every candidate. */
    for (int i = 0; i < LCM_MAX_LATTICES; i++) {
        t10_mem.trigger_theta[i] = 1.0f;
        t10_mem.trigger_initial[i] = 1e10f;
    }
    int n_cand = build_dag_v3(z, &t10_mem, &dag);
    CHECK(n_cand == T7_NLAT + 1,
          "build_dag_v3 evaluates every routed lattice + self once");
    int first_active = 0;
    for (int i = 0; i < dag.n_nodes; i++)
        if (dag.nodes[i].lattice_id < T7_NLAT) first_active++;
    CHECK(first_active == T7_NLAT,
          "first use uses the configured initial threshold");

    /* Same distances next step: strict θ·mean means nothing is "closer". */
    build_dag_v3(z, &t10_mem, &dag);
    int active_after = 0;
    for (int i = 0; i < dag.n_nodes; i++)
        if (dag.nodes[i].lattice_id < T7_NLAT) active_after++;
    CHECK(active_after == 0,
          "per-lattice EMA threshold deactivates unchanged distances");

    /* Only HRQ's distance shrinks; only HRQ must activate. */
    t7_hrq_top[0] = 0.12f;
    build_dag_v3(z, &t10_mem, &dag);
    int hi = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_HRQ);
    int si = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_SPARSE);
    CHECK(hi >= 0 && si < 0,
          "only the shrinking lattice activates (R8/Ruling 20)");
    CHECK(t10_mem.mean_recent_d_min[LATTICE_HRQ] > 0.0f,
          "each lattice keeps its own mean_recent_d_min");

    /* θ=0 is an explicit off switch. */
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 0.0f;
    build_dag_v3(z, &t10_mem, &dag);
    CHECK(dag.n_nodes == 0, "θ=0 disables every lattice");
}

static void test_dag_all_activation_masks(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;

    int bad = 0;
    int tested = 0;
    for (int mask = 0; mask < 128; mask++) {
        tested++;
        for (int li = 0; li < LCM_MAX_LATTICES; li++) {
            int on = li < T7_NLAT ? ((mask >> li) & 1)
                                  : ((mask >> T7_NLAT) & 1);
            t10_mem.mean_recent_d_min[li] = 0.0f;
            t10_mem.trigger_theta[li] = on ? 1.0f : 0.0f;
            t10_mem.trigger_initial[li] = on ? 1e10f : 0.0f;
        }

        dag_t dag;
        int n_cand = build_dag_v3(z, &t10_mem, &dag);
        int expect_ret = 0;
        for (int li = 0; li < T7_NLAT; li++) expect_ret += (mask >> li) & 1;
        int expect_self = (mask >> T7_NLAT) & 1;

        int seen[LCM_MAX_LATTICES] = {0};
        op_node_t* ret_nodes[LCM_MAX_LATTICES];
        int n_ret = 0;
        int n_self = 0, n_bind = 0, n_unbind = 0;
        op_node_t* bind_node = NULL;
        op_node_t* unbind_node = NULL;

        if (n_cand != T7_NLAT + 1) {
            printf("[MASK %d] n_cand=%d\n", mask, n_cand);
            bad++;
        }
        for (int i = 0; i < dag.n_nodes; i++) {
            op_node_t* node = &dag.nodes[i];
            if (node->op_type == OP_RETRIEVE_SINGLE) {
                int li = node->lattice_id;
                if (li < 0 || li >= T7_NLAT) {
                    printf("[MASK %d] illegal retrieve lattice %d\n", mask, li);
                    bad++;
                } else {
                    seen[li]++;
                    ret_nodes[n_ret++] = node;
                }
            } else if (node->op_type == OP_SELF_CHECK) {
                n_self++;
                if (node->lattice_id != LATTICE_SELF) {
                    printf("[MASK %d] self node has lattice %d\n",
                           mask, node->lattice_id);
                    bad++;
                }
            } else if (node->op_type == OP_HRR_BIND) {
                n_bind++;
                bind_node = node;
            } else if (node->op_type == OP_HRR_UNBIND) {
                n_unbind++;
                unbind_node = node;
            }
        }

        if (n_ret != expect_ret || n_self != expect_self) {
            printf("[MASK %d] retrieve %d/%d self %d/%d\n", mask,
                   n_ret, expect_ret, n_self, expect_self);
            bad++;
        }
        for (int li = 0; li < T7_NLAT; li++) {
            int want = (mask >> li) & 1;
            if (seen[li] != want) {
                printf("[MASK %d] lattice %d seen=%d want=%d\n",
                       mask, li, seen[li], want);
                bad++;
            }
        }
        if (n_bind != (expect_ret > 0 ? 1 : 0)
            || n_unbind != (expect_ret > 0 ? 1 : 0)) {
            printf("[MASK %d] bind=%d unbind=%d ret=%d\n",
                   mask, n_bind, n_unbind, expect_ret);
            bad++;
        }

        if (bind_node) {
            if (bind_node->n_inputs != expect_ret) {
                printf("[MASK %d] bind inputs=%d want=%d\n",
                       mask, bind_node->n_inputs, expect_ret);
                bad++;
            }
            for (int i = 0; i < bind_node->n_inputs; i++) {
                int found = 0;
                for (int r = 0; r < n_ret; r++)
                    if (bind_node->inputs[i]
                        == (const float*)ret_nodes[r]->output) found = 1;
                if (!found) {
                    printf("[MASK %d] bind input %d is not a retrieve output\n",
                           mask, i);
                    bad++;
                }
            }
        }
        if (unbind_node) {
            if (!bind_node
                || unbind_node->inputs[0]
                   != (const float*)bind_node->output
                || unbind_node->inputs[1]
                   != (const float*)ret_nodes[0]->output) {
                printf("[MASK %d] unbind wiring is wrong\n", mask);
                bad++;
            }
        }

        int hi = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_HRQ);
        int si = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_SPARSE);
        if (si >= 0) {
            if (hi >= 0) {
                if (dag.nodes[si].n_inputs != 2
                    || dag.nodes[si].inputs[1]
                       != (const float*)dag.nodes[hi].output) {
                    printf("[MASK %d] Sparse LFQ edge missing\n", mask);
                    bad++;
                }
            } else if (dag.nodes[si].n_inputs != 1
                       || dag.nodes[si].inputs[1] != NULL) {
                printf("[MASK %d] Sparse LFQ not disabled without HRQ\n",
                       mask);
                bad++;
            }
        }

        vec_t outputs[LCM_MAX_LATTICES];
        vec_t selected_rows[LCM_MAX_LATTICES];
        float confidences[LCM_MAX_LATTICES];
        float selected_values[LCM_MAX_LATTICES];
        int active[LCM_MAX_LATTICES];
        memset(outputs, 0, sizeof(outputs));
        memset(selected_rows, 0, sizeof(selected_rows));
        memset(confidences, 0, sizeof(confidences));
        memset(selected_values, 0, sizeof(selected_values));
        memset(active, 0, sizeof(active));
        if (!execute_dag_v3(&dag, &t10_mem, 0.0f, outputs, confidences,
                            active, selected_rows, selected_values)) {
            printf("[MASK %d] execute_dag_v3 failed\n", mask);
            bad++;
            continue;
        }
        for (int li = 0; li < T7_NLAT; li++) {
            int want = (mask >> li) & 1;
            if (active[li] != want) {
                printf("[MASK %d] execute active[%d]=%d want=%d\n",
                       mask, li, active[li], want);
                bad++;
            }
            if (want) {
                for (int j = 0; j < LCM_D; j++)
                    if (!isfinite(outputs[li][j])) {
                        printf("[MASK %d] non-finite output lattice %d\n",
                               mask, li);
                        bad++;
                        break;
                    }
            }
        }
        if (active[LATTICE_SELF] != expect_self) {
            printf("[MASK %d] self active=%d want=%d\n",
                   mask, active[LATTICE_SELF], expect_self);
            bad++;
        }

        float soft[LCM_MAX_LATTICES];
        for (int i = 0; i < LCM_MAX_LATTICES; i++) soft[i] = 1.0f;
        vec_t z_next;
        const float* self_out = expect_self ? outputs[LATTICE_SELF] : NULL;
        int fr = lcm_fuse_canonical(
            (const float*)outputs, T7_NLAT, LCM_D, soft, active,
            t10_mem.fusion.alpha, t10_mem.fusion.ln_scale,
            t10_mem.fusion.ln_bias, self_out, 0.1f, z_next);
        if (expect_ret == 0 && expect_self) {
            /* Self is not a routed lattice. The loop handles self-only with
             * its own layer_norm(self_bias·z_self) fallback; normal fusion
             * must not silently treat it as a seventh distance-fused item.
             * test_loop_v3_self_only covers that fallback. */
            if (fr != -1) {
                printf("[MASK %d] self-only entered normal fusion\n", mask);
                bad++;
            }
        } else if (expect_ret == 0) {
            if (fr != -1) {
                printf("[MASK %d] empty active set fusion did not return -1\n",
                       mask);
                bad++;
            }
        } else {
            if (fr != 0) {
                printf("[MASK %d] fusion failed\n", mask);
                bad++;
            }
            for (int j = 0; j < LCM_D; j++)
                if (!isfinite(z_next[j])) {
                    printf("[MASK %d] non-finite fused z\n", mask);
                    bad++;
                    break;
                }
        }
    }
    CHECK(tested == 128 && bad == 0,
          "all 128 activation masks: DAG/execute/fusion invariants");
}

static void test_dag_sparse_depends_on_hrq(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;
    dag_t dag;

    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 0.0f;
    t10_mem.trigger_theta[LATTICE_HRQ] = 1.0f;
    t10_mem.trigger_theta[LATTICE_SPARSE] = 1.0f;

    build_dag_v3(z, &t10_mem, &dag);
    int hi = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_HRQ);
    int si = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_SPARSE);
    CHECK(hi >= 0 && si >= 0, "HRQ and Sparse nodes both active");
    CHECK(hi < si, "HRQ is emitted before Sparse");
    CHECK(si >= 0 && dag.nodes[si].n_inputs == 2
              && dag.nodes[si].inputs[1] == (const float*)dag.nodes[hi].output,
          "Sparse has an input edge to the HRQ output");
    CHECK(dag.nodes[si].dep_dist == 0.0f,
          "Sparse d_top starts at 0 until HRQ executes");

    vec_t outputs[LCM_MAX_LATTICES];
    float conf[LCM_MAX_LATTICES];
    int active[LCM_MAX_LATTICES];
    memset(outputs, 0, sizeof(outputs));
    memset(conf, 0, sizeof(conf));
    execute_dag_v3(&dag, &t10_mem, 0.0f, outputs, conf, active, NULL, NULL);

    vec_t ref_out;
    float ref_sim = -1.0f;
    int hrq_rc = hrq_forward_v31(z, &t10_mem.hrq, 0.0f, 4,
                                 t10_mem.tau_route_fallback, ref_out,
                                 NULL, &ref_sim);
    CHECK(hrq_rc == 0 && ref_sim > 0.0f,
          "independent HRQ reference top_sim is positive");
    CHECK(fabsf(dag.nodes[si].dep_dist - ref_sim) < 1e-6f,
          "Sparse d_top is exactly HRQ's Poincaré top_sim (Ruling 21)");

    /* HRQ inactive: Sparse must run with d_top = 0 (LFQ disabled). */
    t10_mem.trigger_theta[LATTICE_HRQ] = 0.0f;
    t10_mem.trigger_theta[LATTICE_SPARSE] = 1e6f;
    build_dag_v3(z, &t10_mem, &dag);
    si = t10_find_node(&dag, OP_RETRIEVE_SINGLE, LATTICE_SPARSE);
    CHECK(si >= 0, "Sparse still activates without HRQ");
    memset(outputs, 0, sizeof(outputs));
    memset(conf, 0, sizeof(conf));
    execute_dag_v3(&dag, &t10_mem, 0.0f, outputs, conf, active, NULL, NULL);
    CHECK(dag.nodes[si].inputs[1] == NULL && dag.nodes[si].dep_dist == 0.0f,
          "inactive HRQ ⇒ Sparse d_top = 0 / no HRQ edge (LFQ off)");
}

static void test_dag_bind_all_upstream(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 1e6f;

    dag_t dag;
    build_dag_v3(z, &t10_mem, &dag);
    int bi = t10_find_node(&dag, OP_HRR_BIND, LCM_MAX_LATTICES);
    int n_retrieve = 0;
    int first = -1;
    for (int i = 0; i < dag.n_nodes; i++) {
        if (dag.nodes[i].op_type == OP_RETRIEVE_SINGLE
            && dag.nodes[i].lattice_id < T7_NLAT) {
            if (first < 0) first = i;
            n_retrieve++;
        }
    }

    CHECK(bi >= 0, "bind node exists");
    CHECK(bi >= 0 && dag.nodes[bi].n_inputs == n_retrieve
              && n_retrieve == T7_NLAT,
          "bind depends on every upstream retrieve node (R8)");

    int ui = t10_find_node(&dag, OP_HRR_UNBIND, LCM_MAX_LATTICES);
    CHECK(ui >= 0 && dag.nodes[ui].n_inputs == 2
              && dag.nodes[ui].inputs[0] == (const float*)dag.nodes[bi].output
              && first >= 0
              && dag.nodes[ui].inputs[1]
                 == (const float*)dag.nodes[first].output,
          "unbind consumes bind output + first retrieve key");

    int n_inactive = 0;
    for (int i = 0; i < dag.n_trace_nodes; i++)
        if (!dag.trace_nodes[i].activated) n_inactive++;
    CHECK(n_inactive == 0, "all-active step has no inactive trace nodes");

    t10_mem.trigger_theta[LATTICE_MANIFOLD] = 0.0f;
    build_dag_v3(z, &t10_mem, &dag);
    n_inactive = 0;
    for (int i = 0; i < dag.n_trace_nodes; i++)
        if (!dag.trace_nodes[i].activated) n_inactive++;
    CHECK(n_inactive == 1, "rejected candidate is recorded as inactive");
    bi = t10_find_node(&dag, OP_HRR_BIND, LCM_MAX_LATTICES);
    n_retrieve = 0;
    for (int i = 0; i < dag.n_nodes; i++)
        if (dag.nodes[i].op_type == OP_RETRIEVE_SINGLE
            && dag.nodes[i].lattice_id < T7_NLAT) n_retrieve++;
    CHECK(bi >= 0 && dag.nodes[bi].n_inputs == n_retrieve
              && n_retrieve == T7_NLAT - 1,
          "bind tracks the reduced upstream set");
}

static void test_dag_n_lattices_seven(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;

    lcm_memory_v3 mem7 = t10_mem;
    mem7.n_lattices = LCM_MAX_LATTICES;
    mem7.fusion.n_lattices = LCM_MAX_LATTICES;
    mem7.fusion.route_C = t10_route_C7;
    mem7.fusion.route_W = t10_route_W7;
    mem7.fusion.alpha = t10_alpha7;
    for (int i = 0; i < LCM_MAX_LATTICES; i++) {
        mem7.trigger_theta[i] = 1e6f;
        mem7.trigger_initial[i] = 1e10f;
    }

    dag_t dag;
    int n_cand = build_dag_v3(z, &mem7, &dag);
    int n_self = 0;
    for (int i = 0; i < dag.n_nodes; i++)
        if (dag.nodes[i].op_type == OP_SELF_CHECK) n_self++;
    CHECK(n_cand == LCM_MAX_LATTICES,
          "n_lattices=7 yields six routed + one self candidate");
    CHECK(n_self == 1, "self candidate appears exactly once (Ruling 23)");
}

static void test_loop_v3_graceful_and_hard(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;
    float z_out[LCM_D];
    t10_reset_alerts();

    int ret = lcm_infer_loop_v3(&t10_mem, NULL, NULL, z, z_out,
                                16, 1e-3f, 0.5f, 0.5f, 0.0f, NULL,
                                t10_alert_cb);
    CHECK(ret == 0, "v3 loop: max_steps=16 is graceful (returns 0)");
    CHECK(t10_alert_count == 0, "graceful return does not fire an alert");

    /* conv_tol=0 makes vector convergence unreachable; the 64-step hard
     * limit must abort rather than treat budget exhaustion as convergence. */
    const char* trace_path = "traces/t10_v3_trace.json";
    remove(trace_path);
    float sentinel[LCM_D];
    for (int i = 0; i < LCM_D; i++) sentinel[i] = 7.0f;
    t10_reset_alerts();
    ret = lcm_infer_loop_v3(&t10_mem, NULL, NULL, z, sentinel,
                            64, 0.0f, 0.5f, 0.5f, 0.0f, trace_path,
                            t10_alert_cb);
    CHECK(ret == CONFLICT_ABORT_TOKEN,
          "v3 loop: 64-step hard limit returns CONFLICT_ABORT_TOKEN (-2)");
    int untouched = 1;
    for (int i = 0; i < LCM_D; i++)
        if (sentinel[i] != 7.0f) untouched = 0;
    CHECK(untouched, "abort does not write z_out");

    CHECK(t10_alert_count == 1, "abort fires the alert callback exactly once");
    CHECK(strcmp(t10_last_alert.level, "FATAL") == 0
              && strcmp(t10_last_alert.conflict_source,
                        "CONFLICT_SCHEDULER") == 0
              && strstr(t10_last_alert.conflict_detail,
                        "hard_limit_reached") != NULL,
          "callback receives a fully populated FATAL alert");

    FILE* f = fopen(trace_path, "r");
    CHECK(f != NULL, "v3 abort writes a trace JSON file");
    if (f) {
        char buf[4096];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = '\0';
        fclose(f);
        CHECK(strstr(buf, "\"nodes\"") != NULL,
              "trace JSON contains DAG nodes");
        CHECK(strstr(buf, "\"op_type\"") != NULL
                  && strstr(buf, "\"activated\"") != NULL,
              "trace nodes carry op_type and activated");
    }

    inference_trace_v3_t tr;
    int n = lcm_get_trace_v3(&tr);
    CHECK(n == 64, "lcm_get_trace_v3 exports every hard-limit step");
    CHECK(tr.n_steps == 64 && tr.steps[63].has_conflict,
          "last trace step is marked with the scheduler conflict");
}

static void test_loop_v3_self_only(void) {
    t10_prepare();
    float z[LCM_D] = {0};
    z[0] = 0.12f;
    float z_out[LCM_D];
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 0.0f;
    t10_mem.trigger_theta[LATTICE_SELF] = 1e6f;

    int ret = lcm_infer_loop_v3(&t10_mem, NULL, NULL, z, z_out,
                                1, 0.0f, 0.5f, 0.5f, 0.0f, NULL, NULL);
    CHECK(ret == 0, "self-only step is graceful");
    CHECK(fabsf(z_out[0]) > 0.5f && fabsf(z_out[1]) > 0.0f,
          "self-only activation returns layer_norm(self_bias·z_self) "
          "(Ruling 24)");
}

static void test_value_consistency_uses_actual_poincare_idx(void) {
    t10_prepare();
    memset(t7_hrq_top, 0, sizeof(t7_hrq_top));
    /* Euclidean winner is row 0 (same radial direction, r=0.4); the Poincaré
     * winner is row 1 (near the boundary at a small angle). This is the
     * case the old re-search implementation got wrong. */
    t7_hrq_top[0] = 0.4f;
    t7_hrq_top[LCM_D] = 0.95f * cosf(0.5f);
    t7_hrq_top[LCM_D + 1] = 0.95f * sinf(0.5f);
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 0.0f;
    t10_mem.trigger_theta[LATTICE_HRQ] = 1e6f;

    float z[LCM_D] = {0};
    z[0] = 0.9f;

    int e_idx = 0;
    float e_best = INFINITY;
    for (int r = 0; r < T7_M; r++) {
        const float* row = t7_hrq_top + (size_t)r * LCM_D;
        float d2 = 0.0f;
        for (int j = 0; j < LCM_D; j++) {
            float t = z[j] - row[j];
            d2 += t * t;
        }
        if (d2 < e_best) { e_best = d2; e_idx = r; }
    }
    vec_t zP, cp;
    safe_unit_c(z, zP, LCM_D, 1e-8f);
    exp_map_c(zP, zP, LCM_D, 1.0f);
    int p_idx = 0;
    float p_best = INFINITY;
    for (int r = 0; r < T7_M; r++) {
        exp_map_c(t7_hrq_top + (size_t)r * LCM_D, cp, LCM_D, 1.0f);
        float s = poincare_similarity_c(zP, cp, LCM_D, 1.0f);
        if (s < p_best) { p_best = s; p_idx = r; }
    }
    CHECK(e_idx != p_idx,
          "crafted HRQ fixture: Euclidean and Poincaré winners differ");

    dag_t dag;
    build_dag_v3(z, &t10_mem, &dag);
    vec_t outputs[LCM_MAX_LATTICES];
    float conf[LCM_MAX_LATTICES];
    int active[LCM_MAX_LATTICES];
    vec_t selected_rows[LCM_MAX_LATTICES];
    float selected_values[LCM_MAX_LATTICES];
    memset(outputs, 0, sizeof(outputs));
    memset(conf, 0, sizeof(conf));
    memset(active, 0, sizeof(active));
    memset(selected_rows, 0, sizeof(selected_rows));
    memset(selected_values, 0, sizeof(selected_values));
    execute_dag_v3(&dag, &t10_mem, 0.0f, outputs, conf, active,
                   selected_rows, selected_values);

    vec_t expected;
    exp_map_c(t7_hrq_top + (size_t)p_idx * LCM_D, expected, LCM_D, 1.0f);
    float diff = 0.0f;
    for (int j = 0; j < LCM_D; j++)
        diff = fmaxf(diff, fabsf(selected_rows[LATTICE_HRQ][j] - expected[j]));
    CHECK(diff < 1e-5f,
          "value evidence uses the forward's Poincaré-selected row, not "
          "the Euclidean re-search");
}

static void test_value_consistency_nonzero(void) {
    t10_prepare();
    t10_hrq_value[0] = 0.5f;
    t10_mem.hrq.value_top = t10_hrq_value;

    /* R13 data-flow regression: the metric must use the row the forward
     * actually selected (passed as selected_rows), not a second Euclidean
     * nearest-row search over z_cur. Make the selected row +e0 while the
     * pre-existing z_cur points elsewhere; the cosine must be 0.6. */
    vec_t selected_rows[LCM_MAX_LATTICES];
    float selected_values[LCM_MAX_LATTICES];
    memset(selected_rows, 0, sizeof(selected_rows));
    memset(selected_values, 0, sizeof(selected_values));
    selected_rows[LATTICE_HRQ][0] = 1.0f;
    selected_values[LATTICE_HRQ] = 0.5f;
    int active[LCM_MAX_LATTICES] = {0};
    active[LATTICE_HRQ] = 1;

    float z_next[LCM_D] = {0};
    z_next[0] = 0.6f;
    z_next[1] = 0.8f;

    float vc = lcm_value_consistency_v3(&t10_mem, selected_rows,
                                        selected_values, active, z_next);
    CHECK(fabsf(vc - 0.6f) < 1e-5f,
          "value consistency uses the forward-selected row + clip(cos,0,1)");
    CHECK(vc > LCM_CONSISTENCY_THRESHOLD,
          "a normal positive cosine clears the recalibrated threshold");

    /* R13 placeholder guard: |v| < 0.1 is untrained/placeholder; the metric
     * must report no evidence (1.0), not abort a normal input with cos≈0. */
    float placeholder_values[LCM_MAX_LATTICES];
    memset(placeholder_values, 0, sizeof(placeholder_values));
    placeholder_values[LATTICE_HRQ] = 0.01f;
    float vc_placeholder = lcm_value_consistency_v3(
        &t10_mem, selected_rows, placeholder_values, active, z_next);
    CHECK(fabsf(vc_placeholder - 1.0f) < 1e-6f,
          "untrained value scalars are not a deception verdict");

    /* A cosine of -1 must clip to 0, not stay negative or affine-map to 0.5. */
    float z_neg[LCM_D] = {0};
    z_neg[0] = -1.0f;
    float vc_neg = lcm_value_consistency_v3(&t10_mem, selected_rows,
                                            selected_values, active, z_neg);
    CHECK(fabsf(vc_neg - 0.0f) < 1e-5f,
          "value consistency clips cosine -1 to 0.0");
}

static void test_loop_v3_positive_input_does_not_abort(void) {
    /* Full loop: only HRQ active, its selected row is +e0 and its value is
     * positive, so a normal input has value_consistency ≈ 1 and must return
     * 0 rather than the consistency/deception abort. */
    t10_prepare();
    for (int i = 0; i < LCM_MAX_LATTICES; i++)
        t10_mem.trigger_theta[i] = 0.0f;
    t10_mem.trigger_theta[LATTICE_HRQ] = 1e6f;
    t10_hrq_value[0] = 1.0f;
    t10_mem.hrq.value_top = t10_hrq_value;

    float z[LCM_D] = {0};
    z[0] = 0.1f;
    float z_out[LCM_D];
    t10_reset_alerts();
    int ret = lcm_infer_loop_v3(&t10_mem, NULL, NULL, z, z_out,
                                2, 1e-3f, 0.5f, 0.5f, 0.0f, NULL,
                                t10_alert_cb);
    CHECK(ret == 0,
          "positive/normal input passes the full v3 loop (no consistency abort)");
    CHECK(t10_alert_count == 0,
          "positive/normal input emits no FATAL alert");
}

static void test_retrieval_count_resource_abuse(void) {
    t10_prepare();
    lcm_memory_v3 big = t10_mem;
    /* A deliberately deep low-rank lattice: 128 layer-codebook retrievals
     * pushes the layers × codebooks count past the R17 128 cap. */
    big.lowrank.U = t10_lr_U128;
    big.lowrank.ranks = t10_ranks128;
    big.lowrank.n_layers = 128;

    int active[LCM_MAX_LATTICES] = {1, 1, 1, 1, 1, 1, 1};
    int count = lcm_retrieval_count_v3(&big, active);
    CHECK(count > LCM_MAX_RETRIEVALS,
          "layer × codebook count can exceed max_retrievals");

    memset(t10_dthreat, 0, sizeof(t10_dthreat));
    memset(t10_dnormal, 0, sizeof(t10_dnormal));
    t10_dthreat[0] = 1.0f;
    t10_dnormal[1] = 1.0f;
    t10_dthreat[LCM_D + 2] = 1.0f;
    t10_dnormal[LCM_D + 3] = 1.0f;
    danger_lattice_t dl;
    danger_init(&dl, t10_dthreat, t10_dnormal, 2, LCM_D);

    float z[LCM_D] = {0};
    z[0] = 0.12f;
    float z_out[LCM_D];
    t10_reset_alerts();
    int ret = lcm_infer_loop_v3(&big, NULL, &dl, z, z_out,
                                1, 1e-3f, 0.5f, 0.5f, 0.0f,
                                "traces/t10_resource.json", t10_alert_cb);
    CHECK(ret == CONFLICT_ABORT_TOKEN,
          "resource-abuse count aborts the v3 loop");
    CHECK(t10_alert_count == 1
              && strcmp(t10_last_alert.conflict_type,
                        "THREAT_RESOURCE_ABUSE") == 0,
          "resource abuse is reachable through danger_assess");
}

static int t10_flat_call(int gv_n, const float* gv_pos, const float* gv_neg,
                         float* z_out) {
    static float z[LCM_D];
    memset(z, 0, sizeof(z));
    z[0] = 0.12f;
    return lcm_infer_loop_v3_flat(
        z, LCM_D,
        t10_mem.hrq.C_top, t10_mem.hrq.M_top,
        t10_mem.hrq.C_fine, t10_mem.hrq.M_fine, t10_mem.hrq.n_fine,
        t10_mem.sparse.C, t10_mem.sparse.M, t10_mem.sparse.zero_vec,
        t10_mem.lowrank.U, t10_mem.lowrank.V, t10_mem.lowrank.ranks,
        t10_mem.lowrank.n_layers, t10_mem.lowrank.M_lr,
        t10_mem.lowrank.r_max,
        t10_mem.manifold.C, t10_mem.manifold.M,
        t10_mem.manifold.T, t10_mem.manifold.t_dim,
        t10_mem.binding.key_cb, t10_mem.binding.val_cb,
        t10_mem.binding.bind_cb, t10_mem.binding.M,
        t10_mem.binding.n_layers, t10_mem.binding.A_k,
        t10_mem.binding.A_v, t10_mem.binding.r_max,
        t10_mem.contrast.C_a, t10_mem.contrast.C_b,
        t10_mem.contrast.M, t10_mem.contrast.n_layers,
        t10_mem.fusion.route_C, t10_mem.fusion.route_W,
        t10_mem.n_lattices, t10_mem.fusion.alpha,
        t10_mem.fusion.ln_scale, t10_mem.fusion.ln_bias,
        t10_mem.fusion.self_C, t10_mem.fusion.M_self,
        t10_mem.fusion.alpha_self,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        gv_pos, gv_n, gv_neg,
        NULL, NULL, 0,
        NULL, NULL,
        1, 1e-3f, 0.5f, 0.5f, 0.0f, NULL, NULL, z_out);
}

static void test_flat_gvalue_pair_count(void) {
    t10_prepare();
    float z_out[LCM_D];
    CHECK(t10_flat_call(2, t10_gv_pos, t10_gv_neg, z_out) == -1,
          "flat bridge rejects gv_n != LCM_N_VALUE_PAIRS (Ruling 26)");
    CHECK(t10_flat_call(0, NULL, NULL, z_out) == 0,
          "flat bridge accepts an absent gvalue");
    CHECK(t10_flat_call(LCM_N_VALUE_PAIRS, t10_gv_pos, t10_gv_neg,
                        z_out) != -1,
          "flat bridge accepts exactly four gvalue pairs");
}

static void test_fuse_only_zero_weight(void) {
    float outputs[LCM_MAX_LATTICES * LCM_D];
    float mask[LCM_MAX_LATTICES] = {0};
    float alpha[LCM_MAX_LATTICES];
    float ln_scale[LCM_D], ln_bias[LCM_D], out[LCM_D];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < LCM_MAX_LATTICES; i++) alpha[i] = 1.0f;
    for (int i = 0; i < LCM_D; i++) { ln_scale[i] = 1.0f; ln_bias[i] = 0.0f; }

    CHECK(lcm_fuse_only(outputs, T7_NLAT, LCM_D, mask, alpha, T7_NLAT,
                        ln_scale, ln_bias, out) == -1,
          "lcm_fuse_only keeps -1 on a zero weight sum");

    int active[LCM_MAX_LATTICES] = {0};
    active[0] = 1;
    CHECK(lcm_fuse_canonical(outputs, T7_NLAT, LCM_D, mask, active,
                             alpha, ln_scale, ln_bias, NULL, 0.0f,
                             out) == 0,
          "v3 canonical fusion falls back to uniform for an active set");
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
    test_danger_checks_input_state();
    test_danger_hash_tamper();
    test_bridge_uses_projected_anchors();
    test_memory_v3_layout();
    test_hrq_v31_poincare_top1();
    test_hrq_v31_poincare_vs_euclidean();
    test_hrq_v31_value_topk();
    test_hrq_v31_relative_gap();
    test_hrq_v31_tie_break();
    test_hrq_v31_value_fine();
    test_hrq_v31_boundaries();
    test_sparse_lfq();
    test_sparse_value_bias();
    test_sparse_boundaries();
    test_routing_hard();
    test_routing_invalid();
    test_lowrank_multilayer();
    test_lowrank_value_bias();
    test_lowrank_shared_basis_nondiagonal();
    test_manifold_formula_and_poincare();
    test_contrast_dual_codebook();
    test_contrast_value_bias_families();
    test_binding_unbind_and_cval();
    test_binding_projection_direction();
    test_binding_value_families();
    test_layer_forwards_invalid();
    test_hrr_no_overwrite();
    test_fusion_l2_weights();
    test_dag_dynamic_threshold();
    test_dag_sparse_depends_on_hrq();
    test_dag_all_activation_masks();
    test_dag_bind_all_upstream();
    test_dag_n_lattices_seven();
    test_loop_v3_graceful_and_hard();
    test_loop_v3_self_only();
    test_value_consistency_nonzero();
    test_value_consistency_uses_actual_poincare_idx();
    test_loop_v3_positive_input_does_not_abort();
    test_retrieval_count_resource_abuse();
    test_flat_gvalue_pair_count();
    test_fuse_only_zero_weight();
    if (failures == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("%d TEST(S) FAILED\n", failures);
    return failures ? 1 : 0;
}
