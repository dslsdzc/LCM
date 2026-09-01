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

/* Derived constants */
#define LCM_CONSISTENCY_THRESHOLD 0.3f
#define LCM_ENTROPY_THRESHOLD     2.0f  /* Max entropy for 7 lattices = ln(7) ≈ 1.95; 2.0 effectively disables entropy gating */
#define LCM_CONVERGENCE_TOL       1e-3f
#define LCM_DEFAULT_SAFETY_MARGIN 0.5f

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

/* ─── Global Value Lattice ───────────────────────────────────────────────── */

typedef struct {
    vec_t  C_pos[LCM_N_VALUE_PAIRS];   /* Positive value anchors */
    vec_t  C_neg[LCM_N_VALUE_PAIRS];   /* Negative value anchors */
    int    D;
    char   integrity_hash[64];
} gvalue_t;

/* ─── Danger Lattice ─────────────────────────────────────────────────────── */

typedef struct {
    const vec_t* C_threats;        /* Threat pattern codebook [M_danger] */
    const vec_t* C_normal;         /* Normal pattern codebook [M_danger] */
    int          M_danger;         /* Codebook size */
    int          D;                /* Vector dimension */
    float        safety_threshold; /* Threshold in similarity space */
    char         integrity_hash[64];
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

/* Conflict detection */
bool detect_any_conflict(const float* z_next, const float* z_cur, int step,
                          const danger_lattice_t* dl, const gvalue_t* gv,
                          const int* retrieval_counts, float value_consistency,
                          float safety_margin, int max_steps, conflict_t* out);

/* Hyperbolic operations (all use flat float* for vectors) */
float poincare_similarity_c(const float* u, const float* v, int D, float c);
float poincare_distance_c(const float* u, const float* v, int D, float c);
void  exp_map_c(const float* x, float* out, int D, float c);
void  log_map_c(const float* y, float* out, int D, float c);
void  mobius_add_c(const float* u, const float* v, float* out, int D, float c);

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
