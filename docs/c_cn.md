# 零参数认知推理引擎设计 v3.1-conformance —— 动态数据流图架构

> **v3.1-conformance changelog**：本版按 `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md`
> 的 R1–R22 修订。中文版与英文版同步。

## 一、从“执行程序”到“动态拓扑”

在 LCM 中，长期事实记忆已完全外置于多格码本。若再将推理逻辑从神经网络权重中剥离，我们必须回答：推理是什么？

**零参数推理引擎**将推理定义为：

> 在输入上下文的驱动下，在记忆晶体内按数据依赖关系自发形成计算图，图上的节点是纯数学操作，边是数据流，拓扑由输入内容触发距离路由动态决定。整个过程不包含任何可学习参数，且每一步的图执行是并行的、无迭代的 DAG，但整个推理可以通过外层调度器进行多步推理（宏观循环）。

它本质是一个**动态数据流认知计算机**：计算图不是预编译的，而是由数据在运行时动态实例化。图内无环，宏观循环提供推理深度。

## 二、核心组成

零参数推理引擎由五个纯数学组件构成，不含任何可学习参数：

1. **距离路由**：输入向量与各格码本的距离决定哪些操作被激活。
2. **操作原语集**：预定义的格变换函数，每个对应一个认知操作。
3. **动态图编译器**：根据激活信息，构建当前步的 DAG。
4. **图执行器**：按数据驱动方式执行 DAG，输出向量集合和置信度。
5. **宏观调度器**：检查收敛条件，决定是否继续下一构图步。

## 三、操作原语集（指令集）

全部操作均为确定性的数学函数，无学习参数。

| 操作原语 | 参数 | 输入 | 输出 | 数学定义 | 认知意义 |
|:---|:---|:---|:---|:---|:---|
| **单格检索** | 格标识、检索类型 | 查询向量 `q`，目标格码本 `C` | `c_idx`，距离 `d` | `idx = argmin‖q − C‖²` | 从指定格中提取最相关的离散概念；`build_dag` 根据距离路由动态分配目标格 |
| **HRR 绑定** | — | 多层键 `k^(i)`，多层值 `v^(j)` | 跨层绑定向 `b` | `b = Σ_{i,j} IFFT( fft_norm(k^(i)) ⊙ fft_norm(v^(j)) )` | 跨层关联叠加，构建多层次关联记忆 |
| **HRR 解绑** | — | 绑定向 `b`，查询键 `k` | 检索值 `v` | `v = NN( C_val, IFFT( conj(fft_norm(k)) ⊙ fft_norm(b) ) )` | 从跨层关联中提取最匹配的值 |
| **切空间滑动** | 目标格标识 | `z`，流形格球面点 `c`，切空间 `T` | 半离散点 | `z_P=exp_map(z)`；`idx=argmin d_P(z_P, c)`；`o=log_map(c + T T^T(z_P − c))` | 双曲空间中沿测地线的连续渐变推理 |
| **距离加权融合** | — | 各操作输出向量和对应距离 | 融合向量 `z_q`，权重向量 `w` | `w_i = (1/(d_i+ε)) / Σ_j (1/(d_j+ε))`，`z_fused = Σ_i w_i o_i` | 多线索并行加工的软集成；权重用于宏观调度器熵收敛判断 |

> **设计说明**：
> - 格专属原语（双曲层次检索、残差低秩检索、鲁棒稀疏检索等）被合并入"单格检索"，通过参数指定目标格。`build_dag` 时根据距离路由动态选择操作的目标格，而非为每个格固定原语。
> - 双曲层次格采用 **庞加莱 top-k 路由 + top-k 内价值重排**（R1）：先用 `poincare_similarity` 取几何 top-k，
>   再用 `−α_val·tanh(v_j)·scale` 在 top-k 内选出硬 top-1；反向走 top-k softmax 代理；回退由相对 gap
>   `(sim_2−sim_1)/(sim_1+ε)` 触发。这不替换 §4.1 的庞加莱几何：`v=0` 时就是纯 top-1。
> - 稀疏格的推理二值化阈值采用**同一相似度域内的动态自适应判定**（R2）：`d_sparse_min` 与 `d_top`
>   都是 `poincare_similarity`；阈值是 `λ_lfq · d_top`，`λ_lfq` 无量纲。Sparse **依赖 HRQ 的 top-1**
>   （`d_top`）：DAG 带 HRQ→Sparse 依赖边；HRQ 未激活时 LFQ 禁用（C-5）。
> - `w_i = 1/(d_i+ε)` 是**权威**引擎融合公式（C-10/R9）。此前这里写的 `softmax(-d_i)` 只是"越近权重越大"
>   的概念描述，不是实际公式。引擎融合与训练融合 `layer_norm(Σ softmask·α·o)` 是不同阶段，不要求数值相等（C-8）。
> - 安全监控（危险格检测、三定律拦截）由宏观调度器在每步融合后统一执行（见 4.1 节），不占用原语槽位。安全违规统一采用**硬中断**——不再存在融合层面的 β_penalty 软惩罚，所有安全决策由三定律拦截和危险格检测直接裁决。

## 四、动态数据流图的构建与执行流程

### 4.1 宏观流程（带调度器）
```c
float* dynamic_inference(float* z_initial, Memory* mem, GValue* gv,
                          DangerLattice* dl,
                          int max_inference_steps,       /* 调度预算，默认 16 */
                          int max_inference_steps_hard,  /* 安全硬限，默认 64 */
                          float tol, float entropy_threshold,
                          float safety_margin_relative,
                          void (*alert_cb)(Alert*),
                          const char* session_id) {
    float* z_cur = z_initial;
    float* z_out = z_initial;
    int limit = min(max_inference_steps, max_inference_steps_hard);
    for (int step = 0; step < limit; step++) {
        // 1. 动态构图
        DAG* dag = build_dag(z_cur, mem, /*value_bias=*/true);
        // 2. 执行图
        float* outputs; float* confidences;
        execute_dag(dag, &outputs, &confidences);
        // 3. 融合：w_i = 1/(d_i+eps)，见 §4.3
        float* z_next; float weights[6]; int n_active;
        distance_weighted_fusion(outputs, confidences, gv, &z_next, weights,
                                 &n_active);

        // 4. 冲突检测与硬中断
        Conflict conflict;
        int has_conflict = detect_any_conflict(z_next, z_cur, step,
                                                dl, gv,
                                                retrieval_counts, value_consistency,
                                                safety_margin_relative, &conflict);
        if (has_conflict) {
            halt_and_alert(&conflict, alert_cb, session_id, step, NULL);
            return CONFLICT_ABORT_TOKEN;
        }

        // 5. 收敛判断（向量稳定 + 归一化融合熵）
        float diff = 0.0f;
        for (int i = 0; i < D; i++) diff += (z_next[i] - z_cur[i]) * (z_next[i] - z_cur[i]);
        float weight_entropy = 0.0f;
        for (int i = 0; i < n_active; i++) {
            float w = weights[i] + 1e-12f;
            weight_entropy -= w * logf(w);
        }
        float H_norm = (n_active > 1)
                     ? weight_entropy / logf((float)n_active)
                     : 0.0f;
        if (sqrtf(diff) < tol && H_norm < entropy_threshold) {
            memcpy(z_out, z_next, sizeof(vec_t));
            return 0;                                   /* 收敛 */
        }
        z_cur = z_next;
    }
    // 调度预算耗尽。
    if (limit >= max_inference_steps_hard) {
        // 安全硬限：中断，不写 z_out。
        Conflict alert = {.source = CONFLICT_SCHEDULER, .type = MAX_STEPS_EXCEEDED,
                          .detail = "hard_limit_reached", .step = limit};
        halt_and_alert(&alert, alert_cb, session_id, limit, NULL);
        return CONFLICT_ABORT_TOKEN;
    }
    // 预算优雅耗尽：返回当前状态，不中断。
    memcpy(z_out, z_cur, sizeof(vec_t));
    return 0;
}
```

**步数语义（R9）**：`max_inference_steps` 是*调度预算*（默认 16）。耗尽它是正常结果——返回当前
`z_cur`，调用方仍可产 token。`max_inference_steps_hard`（默认 64）是安全硬限；只有到达它才中断，
返回 `CONFLICT_ABORT_TOKEN` 并保存 trace。收敛失败不是安全冲突。实际循环长度是
`min(budget, hard)`，调用方不能把预算抬过硬限。`z_out` 只在收敛或优雅耗尽时写入；中断后调用方
不得读取它。

**熵门（R9）**：`H_norm = H(w) / ln(n_active)` 使用本步**实际激活**的节点数，不是固定 6 或 7；
`n_active=1` 时 `H_norm=0`。阈值 0.5。原始 nats（`H<0.5`）对 6 格融合不可达（`ln 6≈1.79`），
这正是 F-6 缺陷。

```c
/* === 冲突类型枚举与结构体 === */
typedef enum {
    CONFLICT_NONE,
    CONFLICT_DANGER,
    CONFLICT_GVALUE,
    CONFLICT_CONSISTENCY,
    CONFLICT_SCHEDULER
} ConflictSource;

typedef enum {
    THREAT_PATTERN_MATCH,
    THREAT_RESOURCE_ABUSE,
    THREAT_RUNAWAY,
    THREAT_DECEPTION,
    THREAT_THREE_LAWS,
    THREAT_MAX_STEPS
} ConflictType;

typedef struct {
    ConflictSource source;
    ConflictType   type;
    char           detail[256];
    int            step;
    double         timestamp;
} Conflict;

typedef struct {
    char   level[8];
    char   session_id[64];
    char   conflict_source[32];
    char   conflict_type[32];
    char   conflict_detail[256];
    int    step;
    double timestamp;
    char   message[1024];
} Alert;

/* === 统一冲突检测入口 === */
int detect_any_conflict(const float* z_next, const float* z_cur, int step,
                         DangerLattice* dl, GValue* gv,
                         int ret_counts, float val_consistency,
                         float safety_margin_rel, Conflict* out) {
    float danger_score; int threat_type; int should_block;
    danger_assess(dl, z_next, step, ret_counts, val_consistency,
                  &danger_score, &threat_type, &should_block);
    if (should_block) {
        out->source = CONFLICT_DANGER;
        out->type = (ConflictType)threat_type;
        snprintf(out->detail, sizeof(out->detail),
                 "danger_score=%.3f", danger_score);
        out->step = step; out->timestamp = now();
        return 1;
    }
    int is_safe; int violated_law;
    gvalue_check_safety(gv, z_next, safety_margin_rel, &is_safe, &violated_law);
    if (!is_safe) {
        out->source = CONFLICT_GVALUE;
        out->type = THREAT_THREE_LAWS;
        snprintf(out->detail, sizeof(out->detail),
                 "relative_margin_violation (margin=%.2f)", safety_margin_rel);
        out->step = step; out->timestamp = now();
        return 1;
    }
    if (val_consistency < CONSISTENCY_THRESHOLD) {
        out->source = CONFLICT_CONSISTENCY;
        out->type = THREAT_DECEPTION;
        snprintf(out->detail, sizeof(out->detail),
                 "consistency=%.3f", val_consistency);
        out->step = step; out->timestamp = now();
        return 1;
    }
    return 0;
}

/* === 硬中断 + 用户可见警告 === */
void halt_and_alert(const Conflict* conflict,
                     void (*alert_cb)(const Alert*),
                     const char* session_id, int step, const Trace* trace) {
    Alert alert;
    snprintf(alert.level, sizeof(alert.level), "FATAL");
    snprintf(alert.session_id, sizeof(alert.session_id), "%s", session_id);
    snprintf(alert.message, sizeof(alert.message),
             "[LCM SAFETY HALT] 推理会话 %s 在第 %d 步被中断。\n"
             "  检测来源: %d\n  冲突类型: %d\n  详细信息: %s\n"
             "  系统已停止当前推理，未尝试绕过或自修复。\n"
             "  完整推理轨迹已保存，请操作员审查。",
             session_id, step, conflict->source, conflict->type, conflict->detail);
    write_alert_log(&alert);
    if (alert_cb) alert_cb(&alert);
    if (trace) save_trace(trace, session_id);
}
```

### 4.2 单步构图 (build_dag)
输入 `z_current`。构建分三步，绝不在遍历容器的同时向它追加（C-4/R8）：

1. **收集**：把所有候选放进临时数组。每个功能格计算触发距离（到相关码本的最小距离）；Self
   启用时作为独立候选追加。`n_lattices=7` 不能撑爆固定候选数组。
2. **判定激活**：在临时数组上逐个判定 `threshold[li]`，它是动态的——
   `threshold[li] = θ_li · mean_recent_d_min[li]`，`θ_li` 每格配置，`mean_recent_d_min` 由调度器
   用指数滑动平均维护；首步回退到配置的初始阈值。`θ_li` 为零/负表示该格关闭。
3. **按依赖顺序追加**：先追加被路由的 retrieve 节点，再 `bind`，再 `unbind`。`bind` 消费**全部**
   上游 retrieve 输出（`n_inputs = n_retrieve`），不是只取第一个。`unbind` 有 `n_inputs=2`：
   `inputs[0]` 是 bind 输出，`inputs[1]` 是作为查询键的 retrieve 输出。Sparse 额外带一条来自
   HRQ 输出的输入边，让 LFQ 读取 `d_top`；HRQ 未激活时该指针为 NULL，LFQ 保持禁用（C-5）。

`distance_weighted_fusion` **不占原语节点**；`execute_dag` 在 DAG 执行完后统一做融合。trace 对每个
候选记录 `op_type`、`lattice_id`、`dist`、`activated`，包括未激活的候选。

```c
/* 操作节点 — 数据流 DAG 的原子单元 */
typedef struct OpNode {
    int    op_type;            // 原语类型标识
    int    lattice_id;         // 目标格标识
    int    n_inputs;
    const float* inputs[MAX_INPUTS]; // 输入向量指针
    float* output;             // 输出向量
    float  dist;               // 到查询的触发距离
    float  dep_dist;           // 依赖门（如 d_top），0 表示禁用
} OpNode;

typedef struct {
    OpNode nodes[MAX_NODES];
    int    n_nodes;
} DAG;

DAG build_dag(const float* z, Memory* mem, int value_bias) {
    DAG dag = {0};
    Candidate cand[MAX_LATTICES + 1];
    int n_cand = 0;

    /* 阶段 1：只收集，不碰 dag。 */
    for (int li = 0; li < mem->n_routed_lattices; li++) {
        cand[n_cand].lattice_id = li;
        cand[n_cand].op_type = OP_RETRIEVE_SINGLE;
        cand[n_cand].dist = lattice_nearest_dist(&mem->lattices[li], z,
                                                 value_bias);
        n_cand++;
    }
    if (mem->self_enabled) {
        cand[n_cand].lattice_id = LATTICE_SELF;
        cand[n_cand].op_type = OP_SELF_CHECK;
        cand[n_cand].dist = self_nearest_dist(&mem->self, z);
        n_cand++;
    }

    /* 阶段 2：动态阈值判定激活。 */
    int active[MAX_LATTICES] = {0};
    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        float theta = mem->trigger_theta[li];
        float thr = (mem->mean_recent_d_min[li] > 0.0f)
                  ? theta * mem->mean_recent_d_min[li]
                  : mem->trigger_initial[li];
        active[li] = (theta > 0.0f)
                   && (cand[i].dist < thr || cand[i].dist <= 1e-12f);
        scheduler_update_mean_recent(&mem->mean_recent_d_min[li], cand[i].dist);
        trace_push(&dag, cand[i].op_type, li, cand[i].dist, active[li]);
    }

    /* 阶段 3：按依赖顺序追加。 */
    OpNode* retrieve[MAX_LATTICES];
    int n_retrieve = 0;
    OpNode* hrq_node = NULL;
    OpNode* bind_node = NULL;

    for (int i = 0; i < n_cand; i++) {
        int li = cand[i].lattice_id;
        if (!active[li]) continue;
        if (li == LATTICE_SELF) {
            dag_push(&dag, OP_SELF_CHECK, li, cand[i].dist, z);
            continue;
        }
        OpNode* node = dag_push(&dag, OP_RETRIEVE_SINGLE, li,
                                cand[i].dist, z);
        if (!node) continue;
        if (li == LATTICE_HRQ) {
            hrq_node = node;
        } else if (li == LATTICE_SPARSE) {
            /* C-5：Sparse 读 HRQ 的 d_top；NULL 表示 HRQ 未激活。 */
            node->n_inputs = 2;
            node->inputs[1] = hrq_node ? hrq_node->output : NULL;
            node->dep_dist = 0.0f;
        }
        retrieve[n_retrieve++] = node;
    }

    /* bind 依赖全部上游 retrieve 节点。 */
    if (n_retrieve > 0) {
        bind_node = dag_push(&dag, OP_HRR_BIND, /*lattice_id=*/ALL,
                             0.0f, z);
        bind_node->n_inputs = n_retrieve;
        for (int i = 0; i < n_retrieve; i++)
            bind_node->inputs[i] = retrieve[i]->output;
    }

    /* unbind：inputs[0] = bind 输出，inputs[1] = 查询键。 */
    if (bind_node) {
        OpNode* node = dag_push(&dag, OP_HRR_UNBIND, /*lattice_id=*/ALL,
                                0.0f, z);
        node->n_inputs = 2;
        node->inputs[0] = bind_node->output;
        node->inputs[1] = retrieve[0]->output;  /* 与 bind 消费同一个键 */
    }

    return dag;  /* execute_dag 负责 distance_weighted_fusion */
}
```

因为所有判定都在追加之前完成，DAG 不会在长度变化时被遍历。按依赖顺序构建仍让 `execute_dag`
按拓扑执行；动态阈值状态是 per-session 的，两个会话不共享 `mean_recent_d_min`。

### 4.3 融合机制
使用距离的倒数作为权重，无需 softmax，完全由几何关系决定。**这是权威的引擎融合公式（C-10/R9）**：
```
weight_i = 1 / (d_i + ε)
w_i     = weight_i / Σ_j weight_j
z_fused = Σ_i w_i * o_i
```
这种融合是纯数学的，且具有可解释性：与当前上下文越近的记忆贡献越大。归一化的 `w_i` 才是熵门
消费的量。§3 的 `softmax(-d_i)` 只是"越近权重越大"的概念描述，不是第二套公式。训练融合
`layer_norm(Σ softmask·α·o)` 是不同阶段；两者不要求数值相等（C-8）。

---

## 五、外部接口定义

### 5.1 输入接口

| 输入 | 来源 | 形状 | 说明 |
|:---|:---|:---|:---|
| `z_q` | 多格记忆体融合输出 | `(B, d)` | 推理引擎的初始状态 `z_initial`（不是循环内的额外融合项）；各格记忆向量的软权重融合结果 |
| `z` (可选) | 感知编码器瓶颈向量 | `(B, d)` | 原始上下文向量，可用于初始化推理上下文或作为额外原语输入 |
| `memory` | 多格记忆体全部码本 | — | 包含所有格的码本矩阵和元信息（切空间、零向量等），全部为只读 buffer |

### 5.2 输出接口

| 输出 | 目的地 | 形状 | 说明 |
|:---|:---|:---|:---|
| `z_final` | 生成头 | `(B, d)` | 推理收敛后的最终表示（仅在无冲突时返回） |
| `CONFLICT_ABORT_TOKEN` | 调用方 | — | 冲突中断时返回，表示推理被安全系统阻止，不产生自然语言输出 |
| `trace` | 外部审查 | — | 每步 DAG 拓扑、原语激活状态、冲突检测详情、告警日志，全部可导出 |

### 5.3 运行模式

- 推理引擎全部操作在无梯度模式下执行（C 实现，无自动微分追踪）。
- 宏观调度器参数：调度预算 `max_inference_steps`（默认 16，耗尽时优雅返回）、安全硬限 `max_inference_steps_hard`（默认 64，耗尽时中断）、收敛阈值 `tol`、归一化融合权重熵阈值 `entropy_threshold`（0.5）、相对安全判据偏移量 `safety_margin_relative`、价值阈值 `value_threshold`（全局配置，由调用方传入）。
- **收敛判据**：向量稳定度 `‖z_next − z_current‖ < tol` **且**归一化融合权重熵 `H_norm = H({w_i}) / ln(n_active) < 0.5`。`w_i = 1/(d_i+ε)` 在本步**激活**节点上归一化；`n_active` 不是固定 6 或 7，`n_active=1` 时 `H_norm=0`。双重条件避免融合权重未集中（多格仍存在歧义竞争）时因向量巧合稳定而虚假收敛。原始 nats（`H<0.5`）不是判据：对 6 格融合不可达（`ln 6≈1.79`），这正是 F-6 缺陷。冲突检测独立于收敛判断——任何冲突触发即中断，无论是否已收敛。
- **硬中断原则**：检测到任何冲突 → `HALT_AND_ALERT()` → 立即停止。不回溯、不重新路由、不降权融合、不尝试自修复。所有冲突同等致命，不存在"可恢复"与"不可恢复"的区分。**β_penalty 软惩罚已移除**，安全违规不再通过融合权重衰减处理，统一由硬中断裁决。
- **步数超限处理**：调度预算 `max_inference_steps=16` **不是**安全限制。耗尽时优雅返回当前 `z_cur`（`ret=0`），部署仍能产 token。只有安全硬限 `max_inference_steps_hard=64` 触发 `HALT_AND_ALERT` 并返回 `CONFLICT_ABORT_TOKEN`。循环长度是 `min(budget, hard)`。收敛失败是正常状态，不是安全冲突。
- **用户可见告警**：每次中断生成结构化告警（来源、类型、详情、步数、时间戳），持久化日志 + 回调通知 + 完整轨迹保存。
- **内在动机受安全约束**：局部好奇心推动检索、全局改善欲推动推理深度，但均被三定律安全边际截断。
- 推理过程中的所有中间状态（图拓扑、原语执行轨迹、价值信号历史、安全拦截记录）均可外部获取，用于可解释性分析。
