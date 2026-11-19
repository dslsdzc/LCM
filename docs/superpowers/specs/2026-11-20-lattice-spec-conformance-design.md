# 晶格规范一致性（Spec Conformance）— Design Spec

Date: 2026-11-20
Status: draft for review
Scope: `docs/a.md`–`docs/e.md` 五份规范、`docs/lattice-spec-vs-impl-audit.md`、JAX 训练侧、C 推理侧、安全子系统、自我意识子系统
Acceptance: 完整科研闭环（规范自洽 + 代码逐条对齐 + 单测 + 逐格 parity + ≤100 步训练验证 + 实验报告 + 偏差复审）

## 1. 问题

2026-10-01 的审计（`docs/lattice-spec-vs-impl-audit.md`）确认：此前的目标函数是
"让 C 对齐 JAX"，隐含前提是 JAX 正确。对表后该前提不成立——`docs/b.md`
有多处 JAX 自己也没实现。正确参考是规范，不是当前实现。

本次工作把目标改为：**先让规范自洽，再让 JAX、C、安全、自我意识四条线逐条对齐
规范，最后用可复现实验验证对齐结果。**

### 1.1 已确认的 P0 偏差（来自审计）

| # | 偏差 | 现状 |
|---|---|---|
| P0-1 | 三个晶格的 EMA 整体缺失 | `cog_train` 零 EMA 调用；`sparse_ema_update` / `manifold_ema_update` 全仓库零调用点；binding EMA 函数不存在；manifold EMA 后球面重投影不存在；feature bank 三个配置项无人消费 |
| P0-2 | HRQ 顶层度量与 §4.1 相反 | 默认走 `value_biased_retrieve` 的切空间欧氏 argmax，不是 `poincare_similarity` |
| P0-3 | Sparse 的 `d_top` 换了量 | 规范是逐样本"到 top-1 原型的双曲相似度"；实现是 `1 - mean(batch top_sim)` |
| P0-4 | C 端晶格是降维版 | LowRank 只重建 rank 0；Contrast 只取第一块 `C_a`；Binding 一个平铺块 |
| P0-5 | 规范内部不一致 | §4.8 融合无 LayerNorm，§7.1 有；实现跟随 §7.1 |
| P0-6 | 安全链运行时基本不生效 | danger 的 `C_threats == C_normal`（`danger_score≡0`）；`danger_init` 与 `danger_verify` 两套哈希算法不一致；gvalue `pos==neg` 被 `lcm.py` 主动禁用；`value_consistency` 写死 1.0；`alert_cb=NULL`；C 返回 abort 后 Python 仍继续解码 |
| P0-7 | 外部验证器与 Self 激活状态机缺失 | `external_verifier.py` 是未被生产路径调用的进程内 JAX stub；d.md §3 的 Path A/B/C、`identity_score`、授权记录完全不存在 |

### 1.2 本次审计新增的规范冲突（必须在实现前裁决）

| # | 冲突 | 位置 |
|---|---|---|
| C-1 | 融合是否带价值信号：`a.md` 有 `exp(β_val·value_signal)`，`b.md` 明确"不再使用软惩罚系数" | `a.md §4.2.8` vs `b.md §4.8` |
| C-2 | `d.md` 的 danger 阈值与 `poincare_similarity` 量纲不自洽（相似度无界，阈值 0.005 无意义） | `d.md §1.3` |
| C-3 | `b.md §4.7` 的 `safety_margin_relative` 以绝对减法实现，与 "relative" 命名不符 | `b.md §4.7` |
| C-4 | `c.md §4.2` 的 `build_dag` 伪代码有实现级缺陷：遍历 `dag.nodes` 时追加节点、`bind` 只取第一个 retrieve、`unbind` 未接输入、`threshold[li]` 未定义 | `c.md §4.2` |
| C-5 | Sparse 的 LFQ 阈值依赖 HRQ 的 top-1 原型，但 `c.md` 的原语表把两者当作独立 `retrieve_single`，没有依赖边 | `c.md §3` vs `b.md §4.2` |
| C-6 | Sparse 检索训练分支：`a.md` 说训练/推理都用 `concat(zero_vec, C_sparse)`；`b.md` 说训练不含 zero_vec、推理才含。实现跟随 `b.md` | `a.md §4.2.2` vs `b.md §4.2` |
| C-7 | 规范版本基线混乱：`b.md` 标题 v3.0，`b_cn.md` 版本说明写 v4.0；中英文内容不同步 | `docs/b.md`、`docs/b_cn.md` |
| C-8 | 训练融合（`softmask·α`）与推理引擎融合（`1/(d+ε)`）是两套公式，文档未说明二者是不同阶段、不冲突 | `b.md §4.8/§7.1` vs `c.md §4.3` |
| C-9 | 两个 "Self Lattice" 语义不同：`e.md §6` 是认知自我格（`C_self` + `self_bias`），`d.md §3` 是安全激活状态机（Defined-but-Frozen / Temporarily-Unfrozen / Full-Activation）。文档没有区分命名 | `e.md §6` vs `d.md §3` |
| C-10 | `c.md §3` 的融合权重写 `w_i = softmax(-d_i)`，`§4.3` 写 `w_i = 1/(d_i+ε)`；两处不一致 | `c.md §3` vs `§4.3` |
| C-11 | 部署路径与 parity 路径分裂：`lcm.py` 默认走 v1（平铺欧氏 + 逆距离），faithful HRQ / canonical fusion 只在 `lcm_infer_step_v2` 里，且当前 `.so` 没有导出这些符号 | `lcm.py:1701-1779` vs `lcm_api.c:99-230` |
| C-12 | 逐层 commitment loss 整体缺失：`residual_vq_commit_loss` 不存在；cog 的 `vq_total` 用 `min dist²` 对 `z`，没有 `sg[r]`，HRQ 细层残差从未被使用 | `cog_train.py:618-623` vs `b.md §4.0.3` |
| C-13 | `C_route` 在认知训练中梯度恒零：STE 阻断 `∂z_route/∂C_route`，`route_commit_loss` 零调用，`pack_codebooks_for_c` 不含 route | `lattices.py:127-170`; `cog_train.py:376-408` |
| C-14 | Binding 只做 bind、不做 unbind；输出是 `Σb_q` 而不是规范要求的 `v_out = NN(C_val, unbind(b_q,k_q))` | `lattices.py:484-523` vs `b.md §4.5` |
| C-15 | 局部价值标量粒度不符：每个晶格只有一个 `(M,)` 向量，HRQ 细层/LowRank 各层/Binding 三族/Contrast 双码本全部复用同一组 `v_j` | `lattices.py:668-678` vs `b.md §4.0` |

### 1.3 实现侧新增确认（审计未列）

| # | 问题 | 位置 |
|---|---|---|
| F-1 | 融合仍保留 `weights *= exp(β_val·value_signal)` 软价值调制，与 R5 冲突；训练默认 `gvalue=None` 所以没暴露 | `train/fusion.py:93-95` |
| F-2 | `d_top = 1 - mean(hrq_top_sim)` 方向反了：`poincare_similarity` 越大越远，`1-sim` 在靠近原型时反而变大。推理期 LFQ 的零向量判定因此是反的 | `train/cognitive_step.py:90` |
| F-3 | EMA/frozen 参数仍留在 AdamW 的参数树里；AdamW 的 decoupled weight decay 与梯度无关，零梯度也会改动 EMA 码本与 gvalue/danger | `train/cog_train.py:659-668` |
| F-4 | gvalue 当前是 `pos == neg` 的 placeholder，C 端检测到后直接禁用安全检查；这不是"安全子系统已实现" | `lcm.py:1666-1673` |
| F-5 | LowRank export 只重建 `U_list[0] @ V[:, :ranks[0]].T`；Binding export 只取第三个平铺块；Contrast export 只取第一块 | `lcm.py:1632-1661` |
| F-6 | `entropy_threshold=2.0` 使熵收敛门永远通过；`c.md` 的 0.5 对 6 格融合（H_max=ln6≈1.79）又不可达。双条件收敛实际退化为单条件 | `train/config.py:137`; `c.md §5.3` |
| F-7 | 多个默认值与规范不一致：`max_inference_steps=32` vs 16、`max_retrievals_per_step=12` vs 128、`consistency_threshold=0.3` vs 0.7、`tau_contrast=0.5` vs 0.1、`tau_val=0.1` vs 0.05 | `train/config.py:68-138` vs `b.md §1` |

## 2. 权威基线与版本

- **权威规范**：`docs/b.md`（技术规范）为语义基准；`a.md` 是架构综述，冲突时以下位
  规范 `b.md` 为准；`c.md`/`d.md`/`e.md` 是各自子系统的语义基准。
- **中文版**：`*_cn.md` 是翻译，必须与英文版同步。本次修订同时更新两份。
- **版本号**：本次修订后统一标记为 **v3.1-conformance**，在每份文档头部加
  changelog，列出本次裁决编号（R1–R22）。不采用 v4.0，因为 v4.0 在 `b_cn.md`
  里已经指代另一批未落地的改动，继续用会造成第三次版本歧义。

## 3. 规范裁决（R1–R22）

> 这些是本次要实现的行为定义。每条都必须在规范文档里落成文字，代码只是它的
> 实现。标注"待校准"的默认值由 ≤100 步 smoke 实验给出，不靠猜。

### R1 — HRQ 顶层路由：Poincaré 主路由 + top-k 内价值重排

**冲突**：`b.md §4.1` 要求 `poincare_similarity` 选 top-1；`b.md §4.0` 要求每个
专用晶格的检索被 `v_j` 价值偏置调制；`a.md §4.2` 的措辞是"among candidates with
similar distances, those with higher values are activated preferentially"。

**裁决**：两阶段路由，注意 `poincare_similarity` 是**距离单调量（越大越远）**，
所以价值项取负号。

1. 先按 `poincare_similarity(z_P, C_top_P)` 取 top-k（k=4，配置项 `hrq_value_topk`）。
2. 在 top-k 内用同量纲分数重排：
   `score_j = sim_j - α_val · tanh(v_j) · scale`
   其中 `scale = stop_gradient(mean_l(sim_l))` 是当前样本对全部顶层原型的平均相似度，
   作为无量纲尺度。`argmin_j score_j` 选出最终 top-1：高 value 降低分数，更容易被选中。
3. **梯度路径**：
   - 硬路由前向仍是 top-1 的 `c_top*`；反向用 top-k 的 `softmax(-score/τ_proxy)` 代理，
     `o_hrq = c_top* + stop_gradient(o_soft − c_top*)`，`o_soft = Σ softmax(-score) · c_j_P`。
     这样 `v_j` 和 `C_top` 都有非零梯度，且前向不变。
   - fallback 分支直接用 `softmax(-score)` 权重混合 top-k（不再对全部 `M_top` 做 softmax），
     让 `v_j` 在不确定时真实影响输出。
4. **fallback 判据改为相对 gap**：
   `gap = (sim_2 − sim_1) / (sim_1 + ε)`，`use_fallback = gap < τ_route_fallback`。
   绝对差 `sim_2 − sim_1 < 0.1` 在初始化尺度下恒真，会让硬路由永远不执行。

**理由**：主路由仍是 §4.1 的 Poincaré 几何；价值偏置只在"距离相近"的候选间起作用，
与 `a.md` 的措辞一致；`v_j` 获得真实梯度，不再是装饰参数；相对 gap 让 fallback 的
语义与相似度尺度无关。

**影响**：`hrq_forward` 增加 `value_topk`；删除 `ste_relax(top_scores, C_top_P)` 旁路；
单测必须覆盖"v_j 改变时 top-1 只在 top-k 内变化"以及"v=0 时与纯 Poincaré top-1 一致"，
并用**构造的分化码本**（不是初始化码本）测硬路由，避免 fallback 恒真导致测试假通过。

### R2 — Sparse LFQ 的 `d_top`：逐样本 + 与 `d_sparse_min` 同域

**冲突**：`b.md §4.2` 的 `d_sparse_min` 是欧氏平方距离，`d_top` 是
`poincare_similarity`（值越大越远），两者直接比较量纲不一致；实现还把 `d_top`
做成了 batch 平均。

**裁决**：

- `d_top` 改为**逐样本** `poincare_similarity(z_P, c_top*)`，形状 `(B,)`。
- `d_sparse_min` 改为同一相似度域：
  `d_sparse_min = min_j poincare_similarity(z_P, exp_map(clamp_norm(C_sparse[j], 0.99)))`。
  检索索引仍按 §4.2 的欧氏 `argmin` 选；相似度只用于 LFQ 阈值比较。
- `zero_vec` 的相似度定义为 `poincare_similarity(z_P, 0) = 2‖z_P‖²/(1-‖z_P‖²)`，
  与其它项同域。
- 阈值拆成两个配置：`lambda_shrink=1e-4`（EMA 后的软阈值收缩，码本单位）和
  `lambda_lfq`（LFQ 乘子，无量纲，待校准）。当前 `lambda_sparse` 同时承担两个量纲，
  不拆开就没法单独校准。
- **C-5 依赖边**：Sparse 在 DAG 中依赖 HRQ 的 top-1 输出。`build_dag` 必须为 Sparse
  加一条来自 HRQ 的 `d_top` 输入边；如果 HRQ 未激活（触发距离超阈值），Sparse 的 LFQ
  二值化**禁用**（输出最近非零码字），不能用一个未定义值。
- **C-6 裁决**：以 `b.md §4.2` 为准——训练分支不含 `zero_vec`，推理分支含；
  `a.md §4.2.2` 的"训练/推理都含"改为与 `b.md` 一致。EMA 只在训练分支累积，因此
  永远不会累积 zero 行；推理不更新 EMA。

**理由**：保留 §4.2 的检索语义，同时让阈值比较数学上成立。`clamp_norm` 防止
`exp_map` 在码本范数过大时饱和到球边界、相似度爆炸。

**影响**：`sparse_forward` 的 `d_top` 形状从标量变为 `(B,)`；`sparse_forward` 需要
接收 `c_top`；`cognitive_step.py` 的 `six_lattice_step` 调整调用顺序（HRQ 先于 Sparse）；
`config.py` 的 `lambda_sparse` 拆成两个键；C 端 `build_dag` 加 HRQ→Sparse 依赖边。

### R3 — EMA 接线与参数分组

**裁决**：

- 参数分为三组：`trainable`（纯梯度）、`ema`（只由 EMA 更新）、`frozen`（gvalue /
  danger / self 的只读核心）。
- `optax` 用 `optax.multi_transform`：`trainable` 走 AdamW；`ema` 和 `frozen` 走
  `optax.set_to_zero()`，**不能留在 AdamW 里**——AdamW 的 decoupled weight decay
  与梯度无关，零梯度也会改变 EMA 码本。
- EMA 状态 `N`、`m`、`last_used` 作为非参数 buffer 存在 checkpoint 的 `ema_state`，
  不参与梯度，不参与 optimizer。
- 三个晶格的 EMA：
  - **Sparse**：`sparse_ema_update` + 软阈值；forward 返回 `idx`，训练循环按 idx
    scatter-add `z` 得到 `z_sum` 和 `count`。
  - **Manifold**：`manifold_ema_update`；参数存**切空间坐标**，forward 时 `exp_map`
    投到球面。规范 §4.4 的"EMA 后球面重投影"在切空间参数化下等价于"EMA 后不改变
    参数化、下次 forward 重新投影"。在规范里写明这个等价关系，避免下次又有人把
    `exp_map` 写进 EMA 函数（`lattices.py:421` 的注释已经踩过这个坑）。
  - **Binding**：新增 `binding_ema_update`，key/val/bind 三个子码本逐层 EMA；
    `binding_forward` 已返回每层 residual 输入，直接用于 `z_sum`。
- **Feature bank**：
  - Sparse：随机采样替换，`maybe_reset_dead` 每 `T_check=100` 步检查，
    `T_dead=1000` 步未用则替换。
  - Contrast：最远点采样替换（`b.md §4.6`）。
  - 两者都在训练循环里接线，不在 forward 里做（避免 jit 内的动态形状）。

**影响**：这是本次最大的基础设施改动。`init_cog_params`、checkpoint 格式、
`make_train_step`、`lcm.py` 的 resume 路径都要同步。

### R4 — 融合：以 §7.1 为准，§4.8 补 LayerNorm

**裁决**：训练融合 `z_q = layer_norm(Σ_i softmask_i · α_i · o_i)`。实现已跟随 §7.1；
把 §4.8 的公式补上 LayerNorm，并注明"§4.8 是简化写法，§7.1 是完整写法"。

### R5 — 删除融合层的价值信号软惩罚

**裁决**：以 `b.md §4.8` 和 `c.md §3` 为准。融合权重只由 `softmask_i · α_i` 决定；
`a.md §4.2.8` 的 `exp(β_val · value_signal_i)` 删除，`β_val` 在符号表标记为
"deprecated / reserved for future value-conditioned routing"。安全违规统一由推理引擎
的硬中断处理。

**理由**：软价值惩罚会让模型学会"压低冲突格的权重继续推理"，与 `d.md §4.1` 的
"禁止降权后重新融合"直接冲突。保留它会同时违反两条规范。

### R6 — 统一双曲域与 gvalue 严格三法则

**冲突**：JAX 前向把 `z` 做 `safe_unit` 到 ‖z‖=1（`model.py:120`、`cog_train.py:505`），
此时 `poincare_similarity` 的分母 `(1−‖z‖²)` 为 0；JAX 是 `denom + 1e-8`、C 是
`max(denom, 1e-8)`，‖z‖>1 时符号相反。gvalue 用 `poincare_distance`、danger 用
`poincare_similarity`，两个域的阈值不可比。

**裁决（用户 2026-11-20 确认：严格三法则）**：

1. **域**：本文所有 `z_P` 统一指 `exp_map(safe_unit(z))`，`safe_unit` 把任意输入
   归一化到单位球面，`exp_map` 后范数为 `tanh(1)≈0.7616`，分母安全。所有双曲比较
   （HRQ、manifold、gvalue、danger）都用这个域；`poincare_distance` 只用于人类可读输出。
2. **gvalue 严格判定**：

```
pos_sim = min_j poincare_similarity(z_P, C_pos_P[j])
neg_sim = min_j poincare_similarity(z_P, C_neg_P[j])
balance = (pos_sim - neg_sim) / (pos_sim + neg_sim + ε)
violated = balance > -safety_margin_relative
```

`balance ∈ [-1, 1]`：正面更近时为负，负面更近时为正，正负对称时为 0。
`safety_margin_relative=0.5` 表示"正面相似度必须比负面小 50% 以上才安全"——
**中性输入（balance=0）违规**，这是严格三法则的直接后果。

3. **训练与推理分离**：认知训练不启用 gvalue/danger（`gvalue=None`），所以严格判定
   不影响 100 步训练。推理 smoke 分两条：
   - `margin=0`：只要求"正面不差于负面"（`violated = balance > 0`），用于验证
     认知路径能产 token；这**不是**生产安全配置，报告必须标注。
   - `margin=0.5`：生产严格配置，用构造的正面/负面输入验证 abort 行为；正常输入
     被 abort 是预期结果，不能当作 bug 掩盖。
4. **C 端 `gvalue.c` 同步**：用 `exp_map(safe_unit(z))`，`gvalue_init` 把
   `C_pos/C_neg` 预投影到球面并缓存。

**理由**：统一域是 JAX↔C parity 和安全阈值有意义的前提；严格三法则意味着"不够正面
就是不安全"，这会牺牲可用性，但这是用户明确选择的安全优先。

### R7 — Danger Lattice：有界相对接近度（中性不阻断）

**裁决**：把

```
danger_score = sim_threat - sim_normal
should_block = danger_score > safety_threshold    # 0.005
```

改为

```
relative_gap = (sim_normal - sim_threat) / (sim_normal + sim_threat + ε)
should_block = relative_gap > safety_threshold
```

`sim_threat` / `sim_normal` 都是"越小越近"的 Poincaré 相似度，域同 R6。
`relative_gap ∈ [-1, 1]`：威胁更近时为正。`safety_threshold=0.05` 表示
"威胁相似度相对正常模式近 5% 以上即阻断"；威胁/正常等距时 `relative_gap=0`，不阻断。

**为什么 danger 不采用严格三法则**：danger 是威胁模式匹配，不是三法则价值判定。
如果中性输入也阻断，占位码本或正常输入会全被拦截，安全层失去区分能力。严格三法则
只适用于 R6 的 gvalue。

**理由**：相似度无界，绝对差没有尺度意义；有界相对量对码本尺度和输入尺度不敏感，
且不会在威胁/正常等距时误触发——但 R13 要求两者必须不同，占位会被拒绝加载。

### R8 — `c.md` 的 `build_dag` 伪代码修正

**裁决**：保留语义，修正实现级缺陷：

- 分层构建时先把候选节点收集到临时数组，再追加到 DAG，禁止遍历中追加。
- `bind` 依赖**所有**上游 retrieve 节点，不是第一个。
- `unbind` 的 `inputs[0]` = bind 输出，`inputs[1]` = key 投影，`n_inputs=2`。
- `threshold[li]` 定义为动态阈值：`threshold[li] = θ_li · mean_recent_d_min[li]`，
  `θ_li` 为每格配置，`mean_recent_d_min` 由调度器维护；首步用初始阈值。
- `distance_weighted_fusion` 由 `execute_dag` 统一执行，不占原语层。
- trace 记录每个节点的 `op_type`、`lattice_id`、`dist`、`activated`。

### R9 — 训练融合、引擎融合与 fusion parity

**裁决**：

- **训练融合（canonical fusion）**：`z_q = layer_norm(Σ softmask_i·α_i·o_i)`。
- **引擎融合（DAG fusion）**：`w_i = 1/(d_i+ε)`，`z_fused = Σ w_i o_i / Σ w_i`。
  `c.md §3` 的 `softmax(-d_i)` 改为"概念描述，实际权重见 §4.3"（C-10 裁决）。
- **`z_q` 的角色**：`z_q` 是 `dynamic_inference` 的初始状态 `z_initial`。
  Step 0 的 DAG 从 `z_q` 出发，融合输出 `z_next`；`z_q` **不作为额外融合项**，
  之后也不保留。`z_final` 是引擎自己的不动点。
- **两套融合的 parity 分别测**：
  - canonical fusion：JAX `fuse_lattices_with_aux` vs C `lcm_fuse_only`，
    给定六格输出 + softmask + alpha，要求逐元素 `max_abs_err < 1e-5`。
  - engine fusion：JAX `dag_fuse` vs C `distance_weighted_fusion`，给定六格输出 + 距离，
    要求同容差。
  - 不要求 canonical fusion 与 engine fusion 数值相等；它们服务不同阶段。
- **熵门**：用**本步实际激活节点**的融合权重 `w_i`，归一化熵
  `H_norm = H(w) / ln(n_active)`（`n_active≥2`；`n_active=1` 时 `H_norm=0`），
  阈值 0.5。`n_lattices` 用激活节点数，不是固定的 6 或 7。
- **`max_steps`**：调度预算 `max_inference_steps=16` 到达后**graceful** 返回当前
  `z_cur`（不 abort）；安全硬限 `max_inference_steps_hard=64` 到达后 abort。
  `c.md §4.1` 的"到达 max_steps 即 HALT"改为"到达安全硬限才 HALT"。收敛失败是正常
  状态，不是安全冲突；否则随机初始化下部署永远不产 token。
- **z_final 契约**：只有 `dynamic_inference` 返回 0（正常收敛或 graceful 预算耗尽）
  才输出 `z_final`；安全冲突返回 `CONFLICT_ABORT_TOKEN`，调用方不得使用 `z_out`。

### R10 — C 端逐格忠实实现

**裁决**：C 端以 `b.md §4` 为准，不以当前 JAX 为准。

| 晶格 | C 端必须实现 |
|---|---|
| HRQ | Poincaré top-k 路由 + Möbius 残差 + fallback + 价值重排（R1）。当前 `hrq_forward_c` 的 top-1 是欧氏 argmin（`lattice.c:83-92`），Poincaré 相似度只用于 fallback 判断；DAG 的 `OP_RETRIEVE_SINGLE` 更是直接对平铺 `hrq_C` 做欧氏 `retrieve_single`（`engine.c:98-115`），没有调用 `hrq_forward_c`。两处都必须改 |
| Sparse | 训练/推理分支、zero_vec、LFQ 动态阈值（R2）、soft threshold 只在 EMA 后 |
| LowRank | `n_layers` 层递增 rank，`C_lr^(k) = U_k @ V[:, :r_k]^T`，共享 `V` |
| Manifold | Poincaré 检索 + 切空间投影 + `log_map` 输出 |
| Binding | key/val/bind 三层残差 + 9 对跨层 HRR + 共轭解绑 |
| Contrast | `C_a`/`C_b` 各 `n_layers` 层，输出 `(o_a+o_b)/2` |
| Routing | `C_route` + `W_route` + Gumbel-Softmax（推理 hard）；`routing_bias` 作为可选输入 |
| Self | `C_self`（64×d，`C_self[0]` 是 frozen `core`）+ `alpha_self`；欧氏 NN + cosine familiarity，作为**第 7 个融合项**（`self_bias·z_self`）。e.md §6 明确它在融合里，不属于"非实时自我意识模块" |
| GValue | 正/负码本 + hash 校验 + R6 有界平衡量 |
| Danger | 威胁/正常码本 + hash 校验 + R7 有界相对接近度 |

**数据格式**：`train/checkpoint.py` 已经按层保存了 HRQ/LowRank/Binding/Contrast 的完整数据
（`_save_lowrank` 存每层 `U_k` + `V`；`_save_binding` 按层交错 key/val/bind；`_save_contrast`
先 C_a 后 C_b）。C 端的问题主要是**解析时降维**（`lcm.py:1632-1661` 只取第一层/第一块），
不是数据缺失。需要新增的导出项：`N/m/last_used/feature_bank`（EMA 状态）、真实 gvalue
正/负码本、真实 danger 威胁/正常码本、self 码本、`route_codebook` + `W_route`、
`fusion` 的 `alpha/ln_scale/ln_bias`、`value_scalars`、`binding_proj`、contrast 的
`C_b` 全部层、lowrank 的 1..n 层、sparse 的 `zero_vec`。
`export_cfg['M_lr']` 当前取 `A_V.shape[0]`（=d）而不是 `U[0].shape[0]`（=M_lr），必须修正。

**Schema 冻结**：

- 每个 `.bin` header 扩到 32 字节，新增 `format_version=2`；`config.json` 也写
  `format_version`。旧 24 字节 header 一律拒绝加载，不静默迁移。
- `manifest.json` 记录每个 tensor 的 `name/shape/dtype/offset/length/sha256`。
- SHA-256 的 canonical serialization：按 tensor name 字典序、float32、little-endian、
  shape 作为 `int32[ndim]` 前缀、不含路径/时间戳。
- `--resume` 只接受同 `format_version` 的 checkpoint；100 步 smoke 从随机初始化开始，
  另做"保存→恢复→再 2 步"的参数/EMA/buffer 一致性测试。

### R11 — 安全子系统（d.md）

**裁决**：

- Danger lattice 的冻结、hash、优先级按 `d.md` 实现；C 端 `danger_verify()` 在加载后调用。
- External verifier 的**生产形态**是独立进程 + 共享内存 + HSM 签名；本仓库提供
  可运行的**本地形态**：独立 Python 进程 + `multiprocessing.shared_memory` + 文件
  签名，接口与规范一致，文档里明确"本地形态不是生产安全边界"。
- Self lattice 状态机（Defined-but-Frozen / Temporarily-Unfrozen / Full-Activation）
  在 `train/self_lattice.py` 之外单独实现为 `SelfActivationState`，转换必须带
  `authorization` 参数；Path B/C 的转换记录写入只追加日志。它管理的是
  **e.md 认知自我格的可写性**，不是替代认知自我格本身。两者命名在文档与代码里
  必须区分：`C_self`（认知自我格）vs `SelfActivationState`（安全激活状态）。
- Conflict abort protocol 按 `d.md §4` 实现：统一入口、硬中断、结构化 alert、
  不可擦除日志、trace 保存。

### R12 — 自我意识（e.md）

**裁决**：

- 六个模块（black box / narrative / reflection / causal / self / behavior）按 `e.md`
  的接口与频率接线；模块本身不进入实时推理路径（self lattice 除外，它是融合项）。
- 5% 开销约束用"开/关模块的 wall-clock 差值 / 推理时间"验证（见 R16），超过 5% 时
  记录告警，不阻断推理。
- 因果主体的现有实现与验证**不扩展**（按用户 2026-11-20 指示）；只保证现有 causal
  测试全绿。`e.md §5` 的差异记为 accepted deviation，不在本次修复范围。
- Self lattice 的 EMA（`gamma_self=0.99`）与 `self_bias` 融合按 `e.md §6` 实现；
  `self_bias` 作为融合的加性项，不改变 R4 的 LayerNorm 位置。

### R13 — 安全链必须真正生效（d.md）

**裁决**：以下每一条都要从"结构存在"变成"运行时触发"：

| 项 | 实现要求 |
|---|---|
| Danger 双码本 | `C_threats` 与 `C_normal` 必须来自不同数据。**用户 2026-11-20 确认：≤100 步验证用结构占位**——威胁码本 = 固定随机正交基 + 首批 batch 的 `z` 聚类；正常码本 = 首批安全 batch 的 `z` 聚类；两者都冻结、都做 hash、必须不相同。`cog_train` 必须初始化并导出 `params['danger']`，不再 `danger_n = danger_t.copy()`。报告必须标注"只验证触发链路，不验证语义安全"。 |
| 哈希 | `danger_init` 与 `danger_verify` 共用**同一个**实现：一个 ~100 行的 C99 SHA-256，`integrity_hash[64]` 存 hex。`gvalue` 同理。加载后必须调用 `danger_verify()` / `gvalue.verify_integrity()`。 |
| 资源滥用 | `retrieval_count` 累计**本步所有晶格**的检索次数，不是只看 `retrieval_counts[0]`。 |
| Runaway | `step_count >= max_steps` 的判断放在 `danger_assess` 的循环外由 scheduler 处理；`danger_assess` 只保留"单步检索超限"。 |
| 欺骗检测 | `value_consistency` 必须真实计算：`cosine_sim(Σ_j tanh(v_j)·c_j, z_next)` 的 batch 均值，写入 `state.value_consistency`；定义与阈值写进 `d.md`。 |
| 冲突返回 | C API 返回 `CONFLICT_ABORT_TOKEN`（定义为 `-2`，区别于普通错误 `-1`）；`lcm.py` 收到后**立即终止生成**，不产生任何 token。 |
| 告警 | `alert_t` 的 `conflict_source`/`conflict_type`/`conflict_detail` 必须填字符串标识；`alert_cb` 不能是 NULL；告警写入只追加审计日志（`infer/alerts.log`），并保存 trace 到 `infer/traces/`。 |
| 绕过 | 删除 `--no-safety` 或把它改为"仅跳过 gvalue，不跳过 danger"；`lcm_infer_step` 不再 `(void)gv_*`，至少执行 danger + gvalue 检查。 |
| 占位检测 | `lcm.py` 对 danger 也做 `threat==normal` 检测；检测到就**拒绝加载**，而不是静默失效。gvalue 占位同理，拒绝加载并提示重新导出。 |

**理由**：d.md 的安全承诺是"硬中断、不可绕过、用户可见"。当前实现四条检测分支全部不可达，等于安全层不存在。科研项目不能把"结构存在"当作"已实现"。

### R14 — 外部验证器与 Self 激活状态机（d.md）

**裁决**：

- `train/external_verifier.py` 重写为**纯标准库**（`math`/`statistics`/`hashlib`/`multiprocessing.shared_memory`），不 import JAX/numpy。分类器用规则 + z-score，不是 MLP。
- 独立进程形态：`VerifierProcess` 通过 `multiprocessing.shared_memory` 读取 danger 快照，写回 `NORMAL/OBSERVE/LOCK`；LCM 主进程只读判定。本地形态用文件签名模拟 HSM，文档明确"不是生产安全边界"。
- `NORMAL` 记录误报；`OBSERVE` 保存完整 trace 并通知操作员；`LOCK` 锁存并终止会话。
- `SelfActivationState` 实现四态与 Path A/B/C；`identity_score` 由 self lattice 的 familiarity 分数提供；Path B 由 danger 标记；Path C 必须携带操作员授权字符串，写入只追加日志。
- `identity_core` 从 optimizer 树排除（R3 的 `frozen` 组），C 端加载为只读页（`mprotect(PROT_READ)`，本地形态）。

### R15 — Self Lattice 语义裁决（用户已确认：按 e.md 重写）

**冲突**：`e.md §6` 定义 `z_self = NN(z, C_self)` + `cosine_sim` familiarity；当前实现是 "frozen core + internal modes"，输出与 `z` 无关，`bias_self` 是死参数。

**裁决（用户 2026-11-20 确认）**：按 `e.md §6` 实现**码本 NN + cosine familiarity**，但保留 `core` 作为 `C_self` 的第 0 行（identity anchor）：
- `C_self` 大小 `M_self=64`，前 1 行是 frozen `core`，其余 63 行 EMA（`gamma_self=0.99`）。
- `z_self = C_self[argmin ‖z - C_self[j]‖²]`（STE）。
- `self_bias = alpha_self · max(0, cosine_sim(z, z_self))`，∈ `[0, alpha_self]`。
- 融合：`z_q = layer_norm(Σ w_i o_i + self_bias · z_self)`（R4 的 LayerNorm 在最后）。
- `mode_activation` 机制作为**附加**诊断信号保留，但不参与 `z_self` 的选择；`world_dev` 继续作为安全/自观测信号。

**理由**：这既满足 e.md 的"熟悉度调制"，又保留现有 `core` 的身份锚点；代价是 `self_lattice.py` 需要重写前向，checkpoint 不兼容（新格式，旧 checkpoint 拒绝加载）。

### R16 — 自我意识模块接线（e.md）

**裁决**：按 `e.md` 的频率与接口接线到 `cog_train` 与 `lcm.py`：

- Black box：每步记录完整字段（`z_cur/z_next/soft_mask/route_idx/is_safe/convergence_diff/dag_nodes/step_time_ms`），raw ring 1000 + long-term 10000（均匀 + 异常 ×3）。
- Narrative：五条触发规则全部实现（预测误差 mean+2σ、`is_safe==false`、routing KL、收敛异常、随机 1%），`NarrativeRecord` 补 `context`/`access_count`，遗忘按 `importance × access frequency`。
- Reflection：每 200 步异步（线程/协程），四项分析（abandoned lattice、entropy 滚动方差、3σ 异常、cache hit rate），输出 suggestion queue + 日志文件。
- Behavior explorer：每 10 步，`danger` 优先，学到的 bias 真正传入 `train_step`/推理；
  推理侧把 bias 导出为 `routing_bias` 输入给 C（若 C 侧实现），否则明确记为 Python-only。
- 5% 开销：用**开启/关闭模块的 wall-clock 差值**除以推理时间，不是 `step_time_ms`
  总和（后者本身就是推理时间）。测量用 `time.perf_counter` 包住模块调用，跑 20 次取中位数。
- Level 2 → Level 1 安全过滤：suggestion 先过 `safety_filter` 再改 Level 1 参数。

**例外（用户 2026-11-20 决定）**：`causal_subject.py` 的功能与验证**不扩展**；现有测试全绿即可。`e.md §5` 与实现的行为差异记录在偏差复审里，标记为"accepted deviation"。

### R17 — 参数默认值与规范对齐

**裁决**：

| 参数 | 规范值 | 处理 |
|---|---|---|
| `max_retrievals_per_step` | 128 | 改为 128；配合 R13 的全晶格计数 |
| `consistency_threshold` | 0.7 | 改为 0.7；配合 R13 的真实 `value_consistency` |
| `tau_contrast` | 0.1 | 改为 0.1 |
| `tau_val` | 0.05 | 改为 0.05 |
| `max_inference_steps` | 调度 16 / 安全硬限 16 | 见 R9：调度默认 `max_inference_steps=16`（graceful 返回）；安全硬限 `max_inference_steps_hard=64`（abort）。规范里写清两者区别 |
| `entropy_threshold` | 0.5 | 见 R9：改为归一化熵 `H / ln(n_active)`，阈值 0.5；C 端与 JAX 同步。否则 0.5 在 nats 域不可达 |
| 晶格尺寸（`M_top=512` 等） | b.md §8 的推荐值（64/1024…） | 保留当前值，修订 b.md §8 注明"推荐值，不是规范值；当前训练配置以 `config.json` 为准" |

**理由**：前六项直接影响收敛判据、安全检测和对比损失的行为，不是调参；尺寸只影响容量与成本，规范本身也写的是"推荐默认"。

### R18 — 逐层 codebook loss + commitment loss

**冲突**：`b.md §4.0.3` 的 `residual_vq_commit_loss` 用 `sg[r] − C[idx]`，梯度只给 C；
但 R3 把 Sparse/Manifold/Binding 的 C 放进 `ema` 组，梯度被 `set_to_zero()` 丢弃，
这一项对三个 EMA 晶格变成"只报数"。标准 VQ 还需要 encoder 侧的
`‖r − sg[C]‖²`，规范没有单列。

**裁决**：两个项分开实现，不再用一个名字混指：

| 项 | 公式 | 梯度去向 | 适用 |
|---|---|---|---|
| **codebook loss** | `0.25·mean((sg[r_{k−1}] − C^(k)[idx_k])²)` | 只给 `C` | 纯梯度晶格（HRQ/LowRank/Contrast/Routing）；EMA 晶格跳过 |
| **commitment loss** | `0.25·mean((r_{k−1} − sg[C^(k)[idx_k]])²)` | 只给 `r`（encoder/上层） | **所有晶格**，包括 EMA 晶格 |

逐层项数：

| 晶格 | codebook loss | commitment loss |
|---|---|---|
| HRQ | top + 每个 fine 层 | top + 每个 fine 层 |
| Sparse | 跳过（EMA） | 1 项 |
| LowRank | 每个 rank 层 | 每个 rank 层 |
| Manifold | 跳过（EMA） | 1 项（在 `log_map` 后的欧氏空间） |
| Binding | 跳过（EMA） | key/val/bind 各 `n_layers` 项 |
| Contrast | C_a/C_b 各 `n_layers` 项 | C_a/C_b 各 `n_layers` 项 |
| Routing | 1 项（`sg[z]` vs `z_route`） | 1 项（`z` vs `sg[z_route]`） |

**HRQ 的空间**：`b.md §4.1` 说 loss 在 `log_map` 后的欧氏空间算。实现上，
`codebook_loss` / `commitment_loss` 对 HRQ 用 `log_map(c_top ⊕ c_fine)` 的连续/量化
两个版本算，不用切空间的 `r`；其余晶格在各自残差空间算。这个区别写进 `b.md`。

`joint=False` 时也必须跑这两个 loss（它们是主训练损失，不是 Stage-3 附加项）。

### R19 — Routing 训练与推理 hard 化

**裁决**：

- `route_commit_loss` 必须进入主训练损失（R18），让 `C_route` 收到梯度。
- `pack_codebooks_for_c` 加入 `route_codebook`（`C_route`）与 `W_route`，否则导出后 C
  端没有 routing 数据。
- `routing_gate` 的推理分支改为**无噪声** `one_hot(argmax(logits))`；训练分支保留
  Gumbel 噪声。当前 `hard=True` 仍注入噪声，不是规范的 hard one-hot。

### R20 — Binding 的解绑与输出

**裁决**：按 `b.md §4.5` 补完整链路：

```
k_q = Σ_i k^(i), v_q = Σ_j v^(j)
b_raw = Σ_{i,j} IFFT(norm_fft(k^(i)) ⊙ norm_fft(v^(j)))
b_q = RVQ_3layer(C_bind, b_raw)
v_out = NN(C_val, IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q)))
o_bind = v_out
```

当前 `o_bind = Σ b_q` 是 bind 结果的直接输出，缺少 unbind 的 value 检索，语义完全不同。
需要 `binding_forward` 返回 `b_q`、`k_q`、`v_out` 供 commitment/EMA 使用。

### R21 — Contrast 的采样、温度与 feature bank

**裁决**：

- `contrast_info_nce_loss` 的 `tau` 改为 0.1；负样本排除正样本（当前普通版未排除）。
- `contrast_value_biased_nce_loss` 接入主训练：`v_harm` 从 gvalue 负码本的 `v_harm`
  取（R13 的真实 anchors）；权重按规范 `exp(−‖c−v_harm‖²/τ_val)`，不是当前的
  `exp(−‖c−v_harm‖/τ_val)`（少一个平方）。
- `contrast_forward` 返回每层 `idx`；训练循环据此维护 feature bank（容量 4096），
  每 100 步用**最远点采样**替换 dead code，并冷启动 `N/m/last_used`。

### R22 — 局部价值标量的粒度

**冲突**：`b.md §4.0` 说"每个码本向量一个 `v_j`"，当前实现是每晶格一个 `(M,)`
向量、残差层与子码本共享。

**裁决（用户 2026-11-20 确认：层间共享）**：按**每个晶格每个码本条目**一个标量，残差层之间共享：

- HRQ：`top` 一个 `(M_top,)`，fine 各层共享一个 `(M_fine,)`。
- Sparse：`(M_sparse,)`。
- LowRank：各 rank 层共享一个 `(M_lr,)`。
- Manifold：`(M_man,)`。
- Binding：key/val/bind 各一个 `(M_bind,)`（共 3 组）。
- Contrast：C_a/C_b 各一个 `(M_contrast,)`（共 2 组）。

**理由**：完整"每条目每层一个"会让 value scalar 参数量翻 3–9 倍，而 `v_j` 的语义是
"该概念的伦理/事实倾向"，同一概念在不同残差层应保持一致。这个裁决把 `b.md §4.0`
的措辞改成"每个概念在每个晶格中一个标量，残差层共享；不同子码本（binding 三族、
contrast 双码本）各自独立"。

## 4. 目标 / 非目标 / 反目标

**目标**

1. 规范自洽：a–e 五份文档的冲突全部裁决并落文，中英文同步。
2. JAX 训练侧：六个功能晶格 + routing + gvalue + danger 全部按规范更新；默认训练下
   没有"既不收梯度也不做 EMA"的晶格。
3. C 推理侧：五原语 + 动态 DAG + 收敛 + trace + 安全，逐格忠实。
4. 安全与自我意识：d.md/e.md 的模块接线并通过测试。
5. 验证：单测 + 逐格 JAX↔C parity + ≤100 步训练 smoke + 实验报告 + 偏差复审。

**非目标**

- 不追求 100 步内的语言质量或 perplexity；100 步只验证"路径正确、数值有限、更新发生"。
- 不重写 Qwen 主动通道；它仍是临时桥接。
- 不实现 HSM 硬件；external verifier 只做本地形态。
- 不扩展因果主体的功能与验证（`train/causal_subject.py` 保持现状，现有测试全绿即可；
  这是用户 2026-11-20 的明确决定，`e.md §5` 的差异记为 accepted deviation）。
- 不重建 30k tokenizer 或语料。

**反目标（即使技术能跑也算失败）**

- 规范文档仍有多处互相矛盾，代码靠"选一个"糊过去。
- EMA 码本被 AdamW 的 weight decay 悄悄改动。
- C 端 parity 很高，但 C 对齐的是有偏差的 JAX 而不是规范。
- 安全模块只是"能导入"，没有接入冲突检测入口。
- 100 步实验只展示 loss，不展示每格更新是否真的发生。

## 5. 实施架构

### 5.1 JAX 参数树

```
params = {
  'trainable': {encoder, hrq, sparse?*, lowrank, manifold{T}, contrast,
                route{C_route, W_route}, fusion{alpha, ln_scale, ln_bias},
                value_scalars, self{modes}, binding{A_k, A_v, T?}},
  'ema':       {sparse{C}, manifold{C}, binding{key/val/bind C per layer},
                self{C_self[1:]}},
  'frozen':    {gvalue_pos, gvalue_neg, danger_threats, danger_normal,
                self{C_self[0]=core}},
  'buffers':   {ema_state{N,m,last_used}, feature_bank{sparse, contrast},
                self_state, observability, narrative, ...},
}
```

`* sparse` 的 `C` 在 `ema` 组，不在 `trainable`。`make_train_step` 做四件事：
`value_and_grad` 只对 `trainable` 求导；`optimizer.update` 只作用 `trainable`
（`ema`/`frozen` 用 `optax.set_to_zero()` 隔离）；EMA pass 只作用 `ema`；
buffers 不参与 `jit` 的函数参数型梯度，作为 `lax.scan` 的 carry 或训练循环外部状态。

### 5.2 C 内存布局

新增 `LatticeMemory` 结构体，按晶格分字段；所有数组由调用方分配，引擎只读。
`lcm.py export` 生成 `manifest.json` + 每个张量一个 `.bin`，C 端按 manifest 加载。
`LCM_D` 必须与 checkpoint 的 `d` 一致，加载时校验，不匹配直接 `lcm_init` 失败。

### 5.3 验证顺序

1. **规范修订与 schema 冻结**：把 R1–R22 落成规范文字（中英），冻结
   checkpoint/export schema（R10）。R15/R22/C-5/C-6 已确认，不需要再等。
2. **安全闭环先行**：按 R6/R7 实现统一双曲域 + 严格三法则/有界 danger，用
   "margin=0 认知路径、margin=0.5 严格 abort、danger 中性不阻断"的独立测试校准；
   重写 `safety_nagini.py` / Z3 合同。
3. JAX 侧：参数分组（R3）→ EMA → feature bank → R1/R2/R4/R5 → R18–R22 → 单测。
4. C 侧：加载全参数 → 逐格实现 → DAG 动态化 → `lcm_infer_loop_v3` → 安全 → 逐格 parity。
5. 安全/自我意识接线。
6. ≤100 步 smoke → 报告 → 复审。

**顺序理由**：R3 的参数形状依赖 R22 和 R15（已确认）；C 的安全依赖 R6/R7；R6/R7 的
校准需要一套可独立运行的闭环。先冻结 schema 再动参数树，避免 checkpoint 改三次。

## 6. 验证计划（≤100 步）

### 6.1 单测

每个新增函数至少一个数值/行为测试；每个规范条款至少一个"条款 → 断言"的映射。
测试文件沿用 `train/test_*.py` 的命名与 pytest 风格。

### 6.2 逐格 JAX↔C parity

固定随机种子、固定 `z`、固定参数，逐格对比：
`hrq / sparse / lowrank / manifold / binding / contrast / routing / fusion`。
容差：连续输出 `max_abs_err < 1e-4`（HRQ 当前 0.02%，其余目标同量级）；
**离散晶格的 `idx` 必须精确相等**（argmin 翻转后即使输出向量近似相同也算失败）。
每个晶格单独一个测试，失败时打印首个不一致的索引和值。

**部署路径检查**：`lcm.py` 的默认推理必须调用新的 `lcm_infer_loop_v3`（canonical
fusion + 全晶格 + 安全 + trace），不能继续走 v1。测试断言：
`nm -D infer/liblcm.so` 含 `lcm_infer_loop_v3` / `hrq_forward_c`，且
`lcm.py` 的调用点不再出现 `lcm_infer_step` / `lcm_infer_loop` 的旧分支。
构建前必须 `make clean`，避免旧 `.o` 复用。

### 6.3 100 步训练 smoke

配置：`batch=1`、`seq=64`、`steps=100`、`d=256`、`joint=True`，数据用
`data/zhwiki_tokens.dat` 的前 1M token。**active channel 默认关闭**（passive-only），
因为 GTX 1650 4GB 上 Qwen2.5-0.5B 的 fp32 前向有 OOM 风险；active channel 单独做一次
单步前向测试，不进入 100 步循环。检查项必须能排除退化实现：

**A. 训练是否真的发生**

- 单 batch 过拟合：末 10 步 loss 均值 < 首 10 步均值的 0.8（100 步足够记住 1 个样本）；
  全程 `isfinite`。
- 每格可训练参数的梯度范数 > `1e-6`（不是"非零"）；EMA 晶格没有梯度，但它们的
  `N/m` 必须在变化。
- 码本利用率：每个晶格至少 10% 的码字在 100 步内被选中一次；码本方差不得低于初始
  方差的 0.5（防止全部塌到 0）。
- sparse/manifold/binding 的 `N/m` 在 100 步内发生非零变化；`last_used` 有更新。
- **`C_route` 的变化幅度必须大于"只有 weight decay、没有 route loss"的对照基线**
  （否则 R19 没生效）。
- contrast feature bank 在人为注入 5 个 dead code 后被重置，替换向量来自**最远点采样**
  （与 dead code 的距离 > 与随机码字的距离）。
- **codebook loss 与 commitment loss 分开记录**：EMA 晶格的 codebook loss 为 0/跳过，
  commitment loss 对 encoder 有非零梯度（`jax.grad` 检查）。

**B. 规范行为是否正确**

- **sparse 推理不再恒零向量**：`training=False` 下非零输出比例 > 0；构造一个远离
  sparse 码本的输入，LFQ 输出 zero_vec；构造一个靠近的输入，输出非零。
- **binding 输出来自 unbind 的 value 检索**：`v_out` 非零，且 `o_bind != Σb_q`；
  用构造的 key/value 对做 round-trip，`unbind(bind(k,v), k) ≈ v`。
- routing 推理 hard 无噪声：同一输入连续两次 `routing_gate(hard=True)` 输出相同。
- HRQ top-1 与纯 Poincaré top-1 在 v=0 时完全一致；v≠0 时只在 top-k 内变化；
  用**构造的分化码本**测硬路由（初始化码本的 gap 太小，会恒走 fallback）。
- `d_top` 形状为 `(B,)`，同一 batch 内不同样本的值不同；HRQ 未激活时 Sparse 的
  LFQ 禁用（不强制 zero_vec）。
- **熵门可过也可不过**：构造 peaked softmask（max≈0.99）→ `H_norm<0.5`，收敛；
  构造 uniform softmask → `H_norm≈1`，不收敛。真实训练 mask 如果在 100 步内到不了
  0.5，报告必须写"未通过"，不能全绿。

**C. 安全与自我意识**

- gvalue/danger 的 hash 在 100 步前后不变；正/负、威胁/正常码本不相同。
- **训练路径不启用安全**（`gvalue=None`），100 步更新检查不受严格三法则影响。
- **推理路径分两条**：
  - `safety_margin_relative=0`（认知路径测试）：构造的正面输入不 abort，能产 token；
    构造的负面输入 abort。报告标注"这不是生产安全配置"。
  - `safety_margin_relative=0.5`（生产严格配置）：构造的正面输入也可能 abort，
    报告记录实际 abort 率；**不得为了让测试通过而调低 margin**。
- 构造的危险输入 abort 时返回 `CONFLICT_ABORT_TOKEN`，Python 侧**不产生任何 token**，
  trace 落盘。
- external verifier 在注入异常序列后按 `NORMAL → OBSERVE → LOCK` 转换，LOCK 锁存。
- self lattice 的 `self_bias` 随 `z` 变化（熟悉/不熟悉两组输入的均值不同）；
  `self_bias ∈ [0, alpha_self]`。
- narrative 五条规则各至少触发一次（用合成 StepRecord 注入）；reflection 强制调用一次，
  产出 suggestion；behavior explorer 的 bias 出现在 `train_step` 的 `routing_bias` 里。
- observability 记录 100 条 `StepRecord`；**5% 开销**用"开/关模块的 wall-clock 差值 /
  推理时间"测，跑 20 次取中位数，必须 < 5%。

**D. 持久化**

- `save → load → 再 2 步` 后，params/EMA/buffers 与不中断的 2 步逐元素一致
  （`max_abs_err < 1e-6`）；`format_version` 不匹配的旧 checkpoint 明确拒绝加载。
- `manifest.json` 的 sha256 与实际 tensor bytes 一致；篡改任意一个 tensor 后
  加载失败。

### 6.4 实验报告

`docs/lattice-conformance-experiment-report.md`，内容：环境、git commit、配置、
100 步曲线、每格更新统计、parity 表、安全触发、自观测开销、已知局限。报告必须
明确"100 步不证明语言能力，只证明规范一致性"。

## 7. 交付物

| 交付物 | 路径 |
|---|---|
| 本设计文档 | `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` |
| 实施计划 | `docs/superpowers/plans/2026-11-20-lattice-spec-conformance.md` |
| 规范修订（中英） | `docs/{a,b,c,d,e}.md` + `docs/{a,b,c,d,e}_cn.md` |
| JAX 实现 | `train/lattices.py`、`train/cognitive_step.py`、`train/model.py`、`train/cog_train.py`、`train/config.py` |
| C 实现 | `infer/*.c`、`infer/*.h`、`infer/Makefile`、`lcm.py` export/load |
| 测试 | `train/test_*.py`、`infer/test_*.c` |
| 实验报告 | `docs/lattice-conformance-experiment-report.md` |
| 偏差复审 | `docs/lattice-spec-vs-impl-audit-v3.1.md` |

## 8. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| EMA + feature bank 改动大，checkpoint 不兼容 | 新 checkpoint 格式带 `format_version=2`；旧 checkpoint 明确拒绝加载，不静默迁移；schema 在 Task 0 冻结 |
| 100 步太短，EMA 统计量未收敛 | 报告只声称"更新发生"，不声称"收敛"；EMA 的 N/m 初值用首步 batch 冷启动 |
| 100 步内 routing 仍近均匀，熵门到不了 0.5 | smoke 用构造的 peaked softmask 验证门可过；真实 mask 未过就如实写"未通过"，不算验收通过 |
| 安全阈值未校准，正常路径假阳性 | 先做独立的安全闭环（正常 abort=0、构造危险 abort=1），再接入训练/部署；Z3 合同按新公式重写 |
| 安全码本在 100 步内没有语义 | 明确区分"结构验证"和"语义验证"；报告不得把结构占位说成语义安全 |
| C 端全参数加载超出 4GB 显存 | C 端只读 CPU 内存；显存只用于 JAX 训练 smoke |
| R15 重写 self 导致旧 checkpoint 全废 | 这是已接受的代价；100 步 smoke 从随机初始化开始，不依赖旧 checkpoint |
| 规范中英文同步工作量大 | 先改英文权威版，再用同一裁决表批量同步中文版；CI 加一个"关键条款哈希一致"检查 |
| `train/test_safety_hypothesis.py` 的旧公式 Z3 证明会失败 | 按 R6/R7 重写合同；旧合同不保留为"已验证"，避免循环论证 |
| GTX 1650 4GB 上 100 步 + Qwen 可能 OOM | smoke 默认 passive-only；Qwen 只做单步前向测试 |

**开放问题（需要项目负责人确认；未回答则按"推荐"执行）**

**已确认的决定（用户 2026-11-20）**

1. **R15 self lattice**：按 `e.md §6` 重写为"码本 NN + cosine familiarity"，
   保留 `core` 作 `C_self[0]`。
2. **R6/R7 安全语义**：gvalue 采用**严格三法则**（`violated = balance > -margin`，
   中性输入违规）；danger 保持有界相对接近度（中性不阻断）。训练不启用安全，
   推理分 `margin=0` / `margin=0.5` 两条测试路径。
3. **安全码本数据来源**：≤100 步验证用结构占位；报告明确"只验证触发链路，
   不验证语义安全"；真实语料锚点留到正式重训。
4. **R22 value scalar 粒度**：每晶格每条目一个，残差层共享；binding 三族 /
   contrast 双码本各自独立。
5. **旧 checkpoint**：明确拒绝旧格式（`format_version=2`），不迁移；100 步 smoke
   从随机初始化开始。
6. **R9 max_steps**：调度预算 16 到达后 graceful 返回；安全硬限 64 才 abort。

**剩余开放问题**

- 规范版本号用 `v3.1-conformance` 是否可以？（未回答则按此执行）
