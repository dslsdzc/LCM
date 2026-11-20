/* LCM Inference Engine — Main Header
 *
 * Zero-parameter dynamic dataflow inference engine.
 * No dynamic memory allocation. All arrays are fixed-size.
 * Formal contracts (preconditions/postconditions) via ASSERT macro.
 *
 * Design principles:
 *   - All memory is caller-owned, callee-operates
 *   - Fixed maximum dimensions (configurable at compile time)
 *   - Pure C99, no external dependencies beyond libm
 *   - Thread-safe: no global/static mutable state
 */
#ifndef LCM_H
#define LCM_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Compile-time tunable limits ────────────────────────────────────────── */

#ifndef LCM_D
#define LCM_D              256   /* Latent vector dimension */
#endif
#ifndef LCM_M_DANGER
#define LCM_M_DANGER       256   /* Danger codebook size */
#endif
#ifndef LCM_MAX_STEPS
#define LCM_MAX_STEPS      32    /* Max inference steps */
#endif
#ifndef LCM_MAX_NODES
#define LCM_MAX_NODES      32    /* Max DAG nodes per step */
#endif
#ifndef LCM_MAX_LATTICES
#define LCM_MAX_LATTICES   7     /* Number of specialized lattices (6 + self) */
#endif
#ifndef LCM_N_VALUE_PAIRS
#define LCM_N_VALUE_PAIRS  4     /* 4 law pairs (pos/neg) */
#endif
#ifndef LCM_MAX_RETRIEVALS
#define LCM_MAX_RETRIEVALS 12    /* Max retrievals per step */
#endif
#ifndef LCM_HRQ_MAX_TOPK
#define LCM_HRQ_MAX_TOPK   1024  /* Stack cap for R1's value-rescoring top-k */
#endif
#ifndef LCM_MAX_BIND_LAYERS
#define LCM_MAX_BIND_LAYERS 8    /* Stack cap for binding's per-layer HRR rows
                                  * (the spec's RVQ_3layer needs 3) */
#endif
/* Stack-safety caps for descriptor-sized scratch arrays. Defaults are 8× the
 * config defaults (M=512, r_max=8, t_dim=4) and are enforced by
 * lcm_bind_memory_v3, so a corrupt checkpoint is rejected instead of sizing
 * unbounded stack scratch. Raise them at compile time for larger models. */
#ifndef LCM_MAX_M
#define LCM_MAX_M          4096  /* Max rows in any VQ codebook */
#endif
#ifndef LCM_MAX_RANK
#define LCM_MAX_RANK       256   /* Max low-rank basis rank (r_max) */
#endif
#ifndef LCM_MAX_T_DIM
#define LCM_MAX_T_DIM      64    /* Max manifold tangent-space dimension */
#endif

/* Derived constants */
#define LCM_CONSISTENCY_THRESHOLD 0.3f
#define LCM_ENTROPY_THRESHOLD     2.0f  /* Max entropy for 7 lattices = ln(7) ≈ 1.95; 2.0 effectively disables entropy gating */
#define LCM_CONVERGENCE_TOL       1e-3f
#define LCM_DEFAULT_SAFETY_MARGIN 0.5f

/* Checkpoint/export schema v2 (R10). Every header-bearing .bin file starts
 * with a 32-byte header: int32 M, d, n_layers, type; float c; uint32
 * format_version; uint32 reserved; uint32 checksum. Python reads the header and
 * manifest.json; no C loader consumes these constants yet, but the v3 loader
 * must use these frozen values. */
#define LCM_HEADER_SIZE_V2   32
#define LCM_FORMAT_VERSION   2

/* ─── Core types ─────────────────────────────────────────────────────────── */

typedef float vec_t[LCM_D];                /* Single vector */
typedef vec_t codebook_t[];                /* Codebook (flexible, pointer) */

/* ─── Conflict / Alert types ─────────────────────────────────────────────── */

typedef enum {
    CONFLICT_NONE = 0,
    CONFLICT_DANGER,
    CONFLICT_GVALUE,
    CONFLICT_CONSISTENCY,
    CONFLICT_SCHEDULER
} conflict_source_t;

typedef enum {
    THREAT_NONE = 0,
    THREAT_PATTERN_MATCH,
    THREAT_RESOURCE_ABUSE,
    THREAT_RUNAWAY,
    THREAT_DECEPTION,
    THREAT_THREE_LAWS,
    THREAT_MAX_STEPS,
} threat_type_t;

typedef struct {
    conflict_source_t source;
    threat_type_t     type;
    char              detail[256];
    int               step;
    double            timestamp;
} conflict_t;

typedef struct {
    char    level[8];
    char    session_id[64];
    char    conflict_source[32];
    char    conflict_type[32];
    char    conflict_detail[256];
    int     step;
    double  timestamp;
    char    message[1024];
} alert_t;

/* ─── Lattice types ──────────────────────────────────────────────────────── */

typedef enum {
    LATTICE_HRQ = 0,
    LATTICE_SPARSE,
    LATTICE_LOWRANK,
    LATTICE_MANIFOLD,
    LATTICE_BINDING,
    LATTICE_CONTRAST,
    LATTICE_SELF,
} lattice_id_t;

/* Lattice memory (read-only during inference) */
typedef struct {
    vec_t*  C;                /* Codebook vectors (pointer to frozen memory) */
    int     M;                /* Codebook size */
    int     D;                /* Vector dimension */
} lattice_memory_t;

/* Full memory set (all six lattices) */
typedef struct {
    lattice_memory_t lattices[LCM_MAX_LATTICES];
    int              n_lattices;
    float            thresholds[LCM_MAX_LATTICES];   /* Per-lattice trigger thresholds (0=default 1e10; the gate effectively always passes — intentional) */
    const float*     manifold_T_space;               /* Manifold tangent space [M * D * t_dim] */
    int              manifold_t_dim;                 /* Tangent space dimension (0=default 4) */
} memory_t;

/* ─── Schema-v2 full-parameter memory (R10, task 7) ─────────────────────────
 *
 * The v1 structs above describe one flat codebook per lattice — the layout the
 * old bridge exported (first layer only). The v3 structs below carry every
 * layer and every side parameter the JAX model trains. Python reads
 * manifest.json, allocates the arrays and fills these structs; C never parses
 * JSON, never allocates, and only reads. All arrays are caller-owned.
 *
 * Layer offsets are the training/export layouts:
 *   HRQ fine     C_fine[layer][row][j]   = C_fine[(layer*M_fine + row)*D + j]
 *   LowRank      U[layer][row][j]        = U[(layer*M_lr + row)*r_max + j]
 *   Binding      key_cb[layer][row][j]   = key_cb[(layer*M + row)*D + j]
 *   Contrast     C_a[layer][row][j]      = C_a[(layer*M + row)*D + j]
 *
 * Task 10 note: lcm_memory_v3 uses typed substructs (hrq/sparse/.../fusion),
 * not a homogeneous ``lattices[7]`` array. build_dag_v3 should switch on
 * lattice_id_t or build a local pointer table; there is intentionally one
 * struct per lattice because their layouts differ.
 */

typedef struct {
    const float* C_top;      /* [M_top][D] */
    int          M_top;
    const float* C_fine;     /* [n_fine][M_fine][D] */
    int          M_fine;
    int          n_fine;
    const float* value_top;  /* [M_top], optional (NULL ⇒ no value bias) */
    const float* value_fine; /* [M_fine] shared by fine layers, optional */
} hrq_mem_t;

typedef struct {
    const float* C;          /* [M][D] */
    int          M;
    const float* zero_vec;   /* [D] */
    const float* value;      /* [M], optional */
    const float* N;          /* [M] EMA counts, optional (inference ignores) */
    const float* m;          /* [M][D] EMA means, optional */
    const int*   last_used;  /* [M], optional (R21 clock) */
} sparse_mem_t;

typedef struct {
    const float* U;          /* [n_layers][M_lr][r_max], zero-padded to r_max */
    const float* V;          /* [D][r_max] */
    const int*   ranks;      /* [n_layers], each in [1, r_max] */
    int          n_layers;
    int          M_lr;
    int          r_max;
    const float* value;      /* [M_lr], optional */
} lowrank_mem_t;

typedef struct {
    const float* C;          /* [M][D] */
    int          M;
    const float* T;          /* [M][D][t_dim] */
    int          t_dim;
    const float* value;      /* [M], optional */
} manifold_mem_t;

typedef struct {
    const float* key_cb;     /* [n_layers][M][D] */
    const float* val_cb;     /* [n_layers][M][D] (the concatenated C_val) */
    const float* bind_cb;    /* [n_layers][M][D] */
    int          M;
    int          n_layers;   /* ≤ LCM_MAX_BIND_LAYERS */
    int          r_max;      /* shared low-rank basis rank; A_k/A_v are [r_max][D] */
    const float* A_k;        /* [r_max][D] HRR key projection */
    const float* A_v;        /* [r_max][D] HRR value projection */
    const float* value_key;  /* [M], optional */
    const float* value_val;  /* [M], optional */
    const float* value_bind; /* [M], optional */
} binding_mem_t;

typedef struct {
    const float* C_a;        /* [n_layers][M][D] */
    const float* C_b;        /* [n_layers][M][D] */
    int          M;
    int          n_layers;
    const float* value_a;    /* [M], optional */
    const float* value_b;    /* [M], optional */
} contrast_mem_t;

typedef struct {
    const float* route_C;    /* [n_lattices][D] */
    const float* route_W;    /* [D][n_lattices] */
    int          n_lattices;
    const float* alpha;      /* [n_lattices] fusion scaling */
    const float* ln_scale;   /* [D] */
    const float* ln_bias;    /* [D] */
    const float* self_C;     /* [1 + M_self][D]; row 0 is frozen core (R15) */
    int          M_self;     /* number of EMA mode rows (excludes core) */
    float        alpha_self; /* self_bias strength */
} fusion_mem_t;

typedef struct {
    int D;                   /* must equal LCM_D */
    int n_lattices;          /* routing/fusion lattice count (config, ≤ MAX) */
    hrq_mem_t      hrq;
    sparse_mem_t   sparse;
    lowrank_mem_t  lowrank;
    manifold_mem_t manifold;
    binding_mem_t  binding;
    contrast_mem_t contrast;
    fusion_mem_t   fusion;
    /* Scalar hyper-parameters; defaults set by lcm_bind_memory_v3(). */
    float tau_route;            /* Gumbel-Softmax temperature (0.5) */
    float tau_route_fallback;   /* HRQ relative-gap threshold (0.1) */
    float tau_self;             /* self familiarity temperature (1.0) */
    float lambda_lfq;           /* sparse LFQ multiplier (1.5) */
} lcm_memory_v3;

/* ─── Global Value Lattice ───────────────────────────────────────────────── */

typedef struct {
    vec_t  C_pos[LCM_N_VALUE_PAIRS];   /* Positive value anchors (ball-cached) */
    vec_t  C_neg[LCM_N_VALUE_PAIRS];   /* Negative value anchors (ball-cached) */
    int    D;
    char   integrity_hash[65];         /* 64 hex chars + NUL (R13 SHA-256) */
} gvalue_t;

/* ─── Danger Lattice ─────────────────────────────────────────────────────── */

typedef struct {
    const vec_t* C_threats;        /* Threat pattern codebook [M_danger] */
    const vec_t* C_normal;         /* Normal pattern codebook [M_danger] */
    int          M_danger;         /* Codebook size */
    int          D;                /* Vector dimension */
    float        safety_threshold; /* Threshold in relative-gap space (R7 0.05) */
    char         integrity_hash[65]; /* 64 hex chars + NUL (R13 SHA-256) */
} danger_lattice_t;

/* ─── DAG engine types ───────────────────────────────────────────────────── */

typedef enum {
    OP_RETRIEVE_SINGLE = 0,
    OP_SLIDE_MANIFOLD,
    OP_HRR_BIND,
    OP_HRR_UNBIND,
    OP_DISTANCE_FUSION,
    OP_CONTRAST_RETRIEVE,
    OP_SELF_CHECK,
} op_type_t;

/* Forward declaration */
struct dag_s;

typedef struct op_node_s {
    op_type_t          op_type;
    int                lattice_id;
    int                n_inputs;
    const float*       inputs[4];       /* Max 4 input pointers (decayed) */
    vec_t*             output;          /* Pointer to output vector */
    float              dist;            /* Trigger distance */
    void (*execute)(struct op_node_s*); /* Execution function pointer */
} op_node_t;

typedef struct dag_s {
    op_node_t nodes[LCM_MAX_NODES];
    vec_t     storage[LCM_MAX_NODES];   /* Output storage for all nodes — must outlive execute_dag */
    int       n_nodes;
} dag_t;

/* ─── Trace structures (c.md §4.1/§5.2) ──────────────────────────────────── */

/* Single step trace entry */
typedef struct {
    vec_t       outputs[LCM_MAX_LATTICES];   /* Lattice outputs this step */
    float       confidences[LCM_MAX_LATTICES];
    float       fusion_weights[LCM_MAX_LATTICES];
    vec_t       z_next;                      /* Fused output */
    int         step;
    bool        has_conflict;
    conflict_t  conflict;
    double      timestamp;
} step_trace_t;

/* Full inference trace (fixed-size ring buffer) */
#define LCM_MAX_TRACE_STEPS 64
typedef struct {
    step_trace_t steps[LCM_MAX_TRACE_STEPS];
    int          n_steps;
    char         session_id[64];
} inference_trace_t;

/* ─── Main inference state ────────────────────────────────────────────────── */

typedef struct {
    /* Session */
    char         session_id[64];
    int          step;
    int          max_steps;

    /* Memory */
    memory_t*    memory;

    /* Agency-modulated convergence params (set from Python via causal subject) */
    float        conv_tol;           /* Convergence tolerance (default: LCM_CONVERGENCE_TOL) */
    float        entropy_thresh;     /* Entropy threshold (default: LCM_ENTROPY_THRESHOLD) */

    /* Safety */
    gvalue_t*            gvalue;
    const danger_lattice_t* danger;

    /* Runtime state */
    vec_t        z_current;
    vec_t        z_next;
    float        weights[LCM_MAX_LATTICES];
    int          retrieval_counts[LCM_MAX_LATTICES];
    float        value_consistency[LCM_MAX_LATTICES];

    /* Step trace */
    dag_t*            dag;
    inference_trace_t trace;
    alert_t           last_alert;
} inference_state_t;

/* ─── Public API (all array parameters as float* for C idiomatic usage) ──── */

/* Initialize safety subsystems */
void gvalue_init(gvalue_t* gv, const float* C_pos, const float* C_neg, int D);
bool gvalue_verify(const gvalue_t* gv);
bool gvalue_check_safety(const gvalue_t* gv, const float* z,
                          float safety_margin, int* out_violated_law);

void danger_init(danger_lattice_t* dl, const float* C_threats,
                 const float* C_normal, int M, int D);
bool danger_verify(const danger_lattice_t* dl);
void danger_assess(const danger_lattice_t* dl, const float* z,
                    int step_count, int retrieval_count, int max_steps,
                    float value_consistency,
                    float* out_score, int* out_threat, bool* out_block);

/* Inference engine */
int  dynamic_inference(inference_state_t* state, const float* z_initial,
                        void (*alert_cb)(const alert_t*));
void halt_and_alert(const conflict_t* conflict, void (*alert_cb)(const alert_t*),
                     const char* session_id, int step);

/* DAG construction and execution */
dag_t build_dag(const float* z, memory_t* mem, bool value_bias);
bool  execute_dag(dag_t* dag, const memory_t* mem, vec_t* outputs, float* confidences);

/* Fusion (with optional gvalue value-biased weighting) */
void distance_weighted_fusion(const vec_t* outputs, const float* confidences,
                               float* z_out, float weights[LCM_MAX_LATTICES],
                               const gvalue_t* gv, float beta_val);

/* Trace saving */
void save_trace(inference_trace_t* trace, const step_trace_t* step,
                const char* session_id);
int  save_trace_to_file(const inference_trace_t* trace, const char* filepath);

/* Compiled dimension query (runtime .so/checkpoint LCM_D mismatch check) */
int  lcm_dim(void);

/* Fill a v3 memory struct from caller-owned arrays (R10, task 7).
 *
 * Copies the seven per-lattice descriptors and validates them:
 *   - D must equal LCM_D (a .so built for another dimension is rejected),
 *   - 1 ≤ n_lattices ≤ LCM_MAX_LATTICES and fusion.n_lattices must match,
 *   - every required pointer is non-NULL and every layer count/rank is valid,
 *   - lowrank ranks are strictly increasing (b.md §4.3),
 *   - every codebook M ≤ LCM_MAX_M, r_max ≤ LCM_MAX_RANK and
 *     t_dim ≤ LCM_MAX_T_DIM (stack-safety caps),
 *   - binding.n_layers ≤ LCM_MAX_BIND_LAYERS and binding.r_max equals
 *     lowrank.r_max (V is the shared low-rank basis).
 * Optional pointers (value scalars, EMA state) may be NULL.
 *
 * No allocation, no JSON, no file I/O. Returns 0 on success, -1 on a bad
 * descriptor. On failure `mem` is left zeroed.
 */
int  lcm_bind_memory_v3(lcm_memory_v3* mem, int D, int n_lattices,
                        const hrq_mem_t* hrq,
                        const sparse_mem_t* sparse,
                        const lowrank_mem_t* lowrank,
                        const manifold_mem_t* manifold,
                        const binding_mem_t* binding,
                        const contrast_mem_t* contrast,
                        const fusion_mem_t* fusion);

/* Conflict detection */
bool detect_any_conflict(const float* z_next, const float* z_cur, int step,
                          const danger_lattice_t* dl, const gvalue_t* gv,
                          const int* retrieval_counts, float value_consistency,
                          float safety_margin, int max_steps, conflict_t* out);

/* Hyperbolic operations (all use flat float* for vectors) */
float poincare_similarity_c(const float* u, const float* v, int D, float c);
float poincare_distance_c(const float* u, const float* v, int D, float c);
void  safe_unit_c(const float* x, float* out, int D, float eps);
void  exp_map_c(const float* x, float* out, int D, float c);
void  log_map_c(const float* y, float* out, int D, float c);
void  mobius_add_c(const float* u, const float* v, float* out, int D, float c);

/* SHA-256 integrity hashing (R13) */
void lcm_sha256(const void* data, size_t len, unsigned char out[32]);
void lcm_sha256_hex(const void* data, size_t len, char out[65]);

/* HRQ v2 bridge wrapper: value_scalars == 0 and k = min(M_top, 4) (the
 * hrq_value_topk default). D must equal LCM_D; top_C is (M_top, D), fine_C is
 * (n_fine * M_fine, D), both raw/not ball-mapped. New code should call
 * hrq_forward_v31 so the top-k and value scalars are explicit. */
int hrq_forward_c(const float* z, int D,
                  const float* top_C, int M_top,
                  const float* fine_C, int M_fine, int n_fine,
                  float tau_fallback, float* out);

/* HRQ v3.1 — R1 conformant forward (Poincaré top-k + value rescoring).
 *
 *   1. z_P = exp_map(safe_unit(z)); C_top_P = exp_map(C_top).
 *   2. sims = poincare_similarity(z_P, C_top_P); keep the k smallest.
 *   3. score_j = sim_j - alpha_val·tanh(v_j)·mean(sims); argmin over top-k
 *      selects the hard top-1. v comes from m->value_top (NULL ⇒ v = 0).
 *   4. Relative fallback: gap = (sim_2 - sim_1)/(sim_1 + 1e-8) over all
 *      prototypes; gap < tau_fallback blends the top-k with softmax(-score).
 *   5. Fine layers: r = z_P ⊕ -c_top, per layer r ⊕= -argmin sim (rescaled by
 *      m->value_fine when alpha_val > 0, shared across layers); the output is
 *      log_map(c_top ⊕ r), with c_top replaced by the blend on fallback.
 *
 * k = min(value_topk, M_top, LCM_HRQ_MAX_TOPK). A valid config with a larger
 * value_topk is clamped to the stack cap (default 1024, config default 4),
 * never rejected. value_topk <= 0 is invalid. idx/top_sim are optional.
 * Returns 0 on success, -1 on bad arguments. No allocation.
 */
int hrq_forward_v31(const float* z, const hrq_mem_t* m, float alpha_val,
                    int value_topk, float tau_fallback,
                    float* out, int* idx, float* top_sim);

/* Sparse v3.1 — R2 conformant forward (training/inference split + LFQ).
 *
 * Training searches C only (idx is a C row). Inference searches [zero_vec, C]
 * (idx 0 = zero_vec, j+1 = C[j]). Retrieval is Euclidean argmin when
 * alpha_val <= 0 or m->value == NULL. When alpha_val > 0 and m->value is
 * present it is JAX's value-biased score over the same search set:
 *   score_j = -‖z - C_j‖² + alpha_val·tanh(v_j)·mean(‖z - C‖²),
 * with v = 0 for the inference zero_vec candidate; idx = argmax(score).
 * LFQ still fires only when !training && d_top > 0 (the > 0 guard is the
 * "HRQ did not activate" signal, since C has no c_top pointer) and
 * d_sparse_min > lambda_lfq·d_top, where d_sparse_min is measured over the
 * non-zero codebook in the R2 domain:
 *   min_j poincare_similarity(z_P, exp_map(clamp_norm(C[j], 0.99))).
 * zero_vec's own similarity would be poincare_similarity(z_P, 0) =
 * 2‖z_P‖²/(1-‖z_P‖²); it is documented only, never compared. When LFQ fires
 * the output is a copy of m->zero_vec and idx = 0. No allocation.
 */
int sparse_forward_c(const float* z, const sparse_mem_t* m, float lambda_lfq,
                     float alpha_val, float d_top, int training,
                     float* out, int* idx);

/* Routing gate — R19 inference path.
 *
 * idx = argmin_j ‖z - C_route[j]‖²; z_route = C_route[idx];
 * logits = W_route^T z_route with W_route stored [D][n_lattices];
 * soft_mask = one_hot(argmax(logits)) with no Gumbel noise (deterministic).
 * soft_mask has n_lattices entries. n_lattices must be in
 * [1, LCM_MAX_LATTICES]. On invalid arguments soft_mask is cleared (up to
 * LCM_MAX_LATTICES entries) and the function returns; a NULL soft_mask or
 * n_lattices <= 0 is ignored because the buffer size is unknown.
 */
void routing_gate_c(const float* z, const float* C_route, const float* W_route,
                    int n_lattices, float* soft_mask);

/* LowRank v3 — R10 conformant forward (b.md §4.3, JAX lowrank_forward).
 *
 *   V is the shared basis [D][r_max]; layer k's codebook is
 *   C_k = U_k @ V[:, :ranks[k]]^T. Residual chain:
 *   r = z; c_k = C_k[argmin ||r - C_k||²]; out += c_k; r -= c_k.
 *   idx[k] receives layer k's selected row (caller array of n_layers ints).
 *   b.md §4.3's rank sequence is strictly increasing; lcm_bind_memory_v3
 *   rejects a descriptor whose ranks are not increasing.
 *
 * When alpha_val > 0 and m->value is present the selection is JAX's R22
 * value-biased score over the same residual input:
 *   score_j = -||r - C_k[j]||² + alpha_val·tanh(v_j)·mean_j(||r - C_k[j]||²),
 *   idx = argmax(score) (first occurrence on ties). value is shared by all rank
 * layers. Invalid descriptors zero out/idx (void signature). No allocation
 * beyond fixed-size stack scratch.
 */
void lowrank_forward_c(const float* z, const lowrank_mem_t* m, float alpha_val,
                       float* out, int* idx);

/* Manifold v3 — R10 conformant forward (b.md §4.4).
 *
 *   z_P = exp_map(safe_unit(z))                      (R6 domain)
 *   idx = argmin_j poincare_similarity(z_P, exp_map(C[j]))
 *   r = z_P - exp_map(C[idx]); proj = T_idx @ T_idx^T @ r
 *   out = log_map(exp_map(C[idx]) + proj)
 * idx is optional (one int). With alpha_val > 0 and m->value present the
 * retrieval follows JAX's value_biased_retrieve(z, C, ..., C_out=C_P): the
 * score is the tangent-space Euclidean one, then the selected row is pushed
 * through exp_map before the projection. Task 11 alignment note: current JAX
 * manifold_forward uses raw exp_map(z) in the default path (not R6's
 * safe_unit); C follows R6 here. No allocation.
 */
void manifold_forward_c(const float* z, const manifold_mem_t* m,
                        float alpha_val, float* out, int* idx);

/* Contrast v3 — R10 conformant forward (b.md §4.6).
 *
 * Two independent residual VQ chains from z over C_a and C_b:
 *   o_a = Σ_k a_k, o_b = Σ_k b_k, out = (o_a + o_b)/2.
 * Each family has its own R22 vector (value_a/value_b, shared across its
 * layers) applied when alpha_val > 0. Invalid descriptors zero out.
 */
void contrast_forward_c(const float* z, const contrast_mem_t* m,
                        float alpha_val, float* out);

/* Binding v3 — R20 conformant forward (b.md §4.5, JAX binding_forward).
 *
 *   u_k = A_k z; z_k = V u_k               (W_k=V@A_k, z_k=W_k z)
 *   u_v = A_v z; z_v = V u_v               (W_v=V@A_v, z_v=W_v z)
 *   three-layer residual VQ over key_cb / val_cb  → k_q, v_q
 *   b_raw = Σ_{i,j} IFFT(norm_fft(k^(i)) ⊙ norm_fft(v^(j)))  (9 pairs)
 *   b_q   = three-layer residual VQ over bind_cb applied to b_raw
 *   v_out = nearest row of C_val (= every val_cb layer concatenated)
 *           to IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q))
 *   out = v_out
 *
 * b_q and k_q are optional outputs (may be NULL) for commitments/EMA/tests.
 * m->r_max must match the shared V's column count; binding requires D to be a
 * power of two (the HRR FFT). Invalid descriptors zero out/optional outputs.
 */
void binding_forward_c(const float* z, const binding_mem_t* m, const float* V,
                       float alpha_val, float* out, float* b_q, float* k_q);

/* Lattice retrieval operations */
void retrieve_single(const float* z, const lattice_memory_t* mem,
                     float* out, float* dist, int* idx);
void slide_manifold(const float* z, const lattice_memory_t* mem,
                     const float* T_space, int t_dim, float* out);
void hrr_bind(const float** keys, int n_keys, const float** vals, int n_vals,
              float* out, int D);
void hrr_unbind(const float* b, const float* k, float* out, int D);

#ifdef __cplusplus
}
#endif

#endif /* LCM_H */
