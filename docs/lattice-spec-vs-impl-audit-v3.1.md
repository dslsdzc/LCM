# 晶格规范 vs 实现 偏差复审 v3.1（conformance）

日期：2026-11-21
基线：`docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` 的裁决 **R1–R22**
旧审计：`docs/lattice-spec-vs-impl-audit.md`（2026-10-01，对 `docs/b.md` v3.0）
范围：JAX 训练侧、C 推理侧、安全子系统、自我意识子系统、≤100 步 smoke

本复审只回答一个问题：**旧审计列出的 P0/C/F 每一条，现在是什么状态？**
状态只有三种：

| 状态 | 含义 |
|---|---|
| **fixed** | 代码已按 R 编号实现，并有测试或 smoke 证据 |
| **accepted** | 用户/设计明确接受的偏差，记录理由与代价 |
| **deferred** | 未实现或只实现了一部分；记录影响与触发条件 |

> 不能把 "结构存在" 当作 "已实现"。本复审把 "有字段/有函数/能导入" 与
> "运行时触发/有梯度/有 EMA/有 abort" 分开写。

---

## 1. 旧审计 P0 条目

| # | 旧审计问题 | 裁决 | 状态 | 证据 / 说明 |
|---|---|---|---|---|
| P0-1 | 三个晶格的 EMA 整体缺失；binding EMA 函数不存在；feature bank 无人消费 | R3 | **fixed** | `train/ema_state.py`（sparse/manifold/binding/self，decayed-sum + 球面等价说明）；`train/param_groups.py` 三组隔离；`cog_train._apply_ema_updates` 是唯一 writer；`FeatureBank` 在训练循环内接线。`test_ema_state.py`、`test_feature_bank.py`、`test_param_groups.py` 覆盖 |
| P0-2 | HRQ 顶层度量是欧氏价值 argmax，不是 Poincaré | R1 | **fixed** | `hrq_forward` 两阶段：Poincaré 几何 top-k → top-k 内价值重排；STE 前向硬 top-1、反向 top-k softmax；相对 gap fallback。`test_hrq_routing.py` + `test_parity_lattices.py` 的 v=0 等同 pure Poincaré、v≠0 只在 top-k 内变化 |
| P0-3 | Sparse 的 `d_top` 是 batch 平均的 `1-mean(sim)` | R2 | **fixed** | `hrq_forward` 返回逐样本 `(B,)` `top_sim`；`sparse_forward` 拒绝标量/1 元素 `d_top`；`d_sparse_min` 与 zero_vec 同域；`build_dag` 加 HRQ→Sparse 依赖边。`test_sparse_lfq.py`、`test_c_loop.py` 的 “Sparse d_top is exactly HRQ's Poincaré top_sim” |
| P0-4 | C 端晶格是降维版（LowRank rank0 / Contrast 第一块 / Binding 平铺块） | R10 | **fixed** | v3 内存布局 + 全层加载（Task 7–9）；C `test_runner` 251 个 `[PASS]` 全过；`test_parity_lattices.py` 逐格对比连续输出与离散 idx |
| P0-5 | 规范 §4.8 无 LayerNorm、§7.1 有 | R4 | **fixed** | `fuse_lattices_with_aux` 最后 `layer_norm`；`test_fusion_conformance.py`。规范文字已同步：`b.md §4.8` 补 LayerNorm 并声明 §4.8 是简化写法、§7.1 是完整写法 |
| P0-6 | 安全链运行时基本不生效（danger 双码本相同、hash 两套、gvalue 禁用、alert 为空、abort 后继续解码） | R6/R7/R11/R13 | **fixed（结构链路）；语义安全 deferred** | C `danger_init`/`danger_verify` 共用 SHA-256；`relative_gap` 有界；`detect_any_conflict` 返回 danger/gvalue/consistency/scheduler，并同时检查进入步的 `z_cur` 与离开步的 `z_next`（只查 `z_next` 会让被 DAG 一步变换掉的威胁输入逃逸）；`CONFLICT_ABORT_TOKEN=-2`；`lcm.py` 非零即停；alert callback + `infer/alerts.log` + trace。gvalue/danger 码本现在是**互不相同的结构占位**：danger 威胁半是固定随机基，normal 半来自安全 encoder 状态库；不是语义安全锚点。C 一致性阈值已重新标定为 0.4，`value_consistency = clip(cos,0,1)`（R17）；占位 value 标量 `max|v|<0.1` 时一致性返回 1.0（无欺骗证据），避免把未训练 value 当成语义判决 |
| P0-7 | 外部验证器与 Self 激活状态机缺失 | R14 | **fixed（库/本地形态）；生产接线 deferred** | `train/external_verifier.py` 纯标准库独立进程 + shared memory + 文件签名；四态 + Path A/B/C；Path C 只允许 `DANGER_FLAGGED → FULL_ACTIVATION`；fail-closed LOCK 永久锁存。`test_external_verifier.py`、`test_self_activation.py`。**`SelfActivationState` 尚未门控生产训练写入，C identity core 未 mprotect**（§4 R14 / D-12）；HSM/物理隔离是非目标 |

---

## 2. 设计审计的规范冲突 C-1 … C-15

| # | 冲突 | 裁决 | 状态 | 处理 / 证据 |
|---|---|---|---|---|
| C-1 | 融合是否带 `exp(β_val·value_signal)` | R5 | **fixed** | 融合权重只由 `soft_mask·α` 决定；`gvalue`/`beta_val`/`tau_val` 参数保留但被忽略；`test_fusion_conformance.py` 用 `gvalue=object()` 证明不消费 |
| C-2 | danger 阈值与相似度量纲不自洽 | R7 | **fixed** | `relative_gap=(sim_normal-sim_threat)/(sum)`，阈值 0.05；`danger_assess` 中性不阻断；C 测试 `test_danger_threshold_relative_gap` |
| C-3 | `safety_margin_relative` 用绝对减法 | R6 | **fixed** | `balance=(pos_sim-neg_sim)/(pos_sim+neg_sim+ε)`；严格三法则 `violated = balance > -margin`；JAX/C 同式；`test_safety_semantics.py`、C `test_gvalue_strict_neutral` |
| C-4 | `build_dag` 伪代码实现级缺陷 | R8 | **fixed** | C `build_dag_v3` 临时数组收集后追加；bind 依赖所有上游 retrieve；unbind 两输入；per-lattice `theta*mean_recent_d_min`；trace 记录 op/lattice/dist/activated。`test_c_loop.py` 的 DAG 组 |
| C-5 | Sparse LFQ 依赖 HRQ top-1 但没有依赖边 | R2 | **fixed** | `build_dag_v3` 加 HRQ→Sparse 边；HRQ 未激活时 Sparse 的 LFQ 禁用（`d_top=0` / 无 HRQ 边）。C 测试 “inactive HRQ ⇒ Sparse d_top = 0” |
| C-6 | Sparse 训练/推理分支不一致 | R2 | **fixed** | 训练 `C_search=C`，推理 `concat(zero_vec,C)`；EMA 只在训练分支累积，zero 行永不进入 EMA。`test_sparse_lfq.py`、C sparse 组 |
| C-7 | 规范版本基线混乱（v3.0/v4.0，中英不同步） | — | **fixed** | 五份规范（a–e）与中文版统一标记 `v3.1-conformance`，头部 changelog 引用 R1–R22；Task 0 + 本次 review 修正后中英文逐条同步。关键条款 grep 见 §7 |
| C-8 | 训练融合与引擎融合两套公式未说明 | R9 | **fixed（代码 + 规范）** | canonical fusion `layer_norm(Σ softmask·α·o)`；engine fusion `w=1/(d+ε)`；两套分别 parity。`test_parity_lattices.py` 的 fusion 组；`b.md §4.8/§7.1` 与 `c.md §3/§4.3` 已写清阶段差异 |
| C-9 | 两个 "Self Lattice" 语义不同 | R15/R14 | **fixed（代码 + 规范）** | `C_self`（认知自我格，EMA 模式 + frozen core）与 `SelfActivationState`（安全激活状态机）分离；`test_self_lattice.py`、`test_self_activation.py`；`d.md §3` 与 `e.md §6` 已明确命名区分 |
| C-10 | `c.md §3` 的 `softmax(-d)` 与 `§4.3` `1/(d+ε)` | R9 | **fixed（C + 规范）** | 引擎融合统一 `1/(d+ε)`；C 测试 `lcm_fuse_only` / 零权重语义；`c.md §3` 改为"概念描述"，`§4.3` 标明权威公式 |
| C-11 | 部署路径走 v1，faithful 路径只在 v2 | R10 | **fixed** | `lcm.py` 默认 `lcm_infer_loop_v3_flat`；`nm` 断言 `lcm_infer_loop_v3`/`hrq_forward_v31`；`test_c_loop.py` |
| C-12 | 逐层 commitment loss 缺失；HRQ 细层残差未用 | R18 | **fixed** | `train/losses.py::cognitive_vq_losses` 分开 codebook/commitment；EMA 晶格跳过 codebook、保留 commitment；HRQ 在 `log_map(c_top⊕·)` 空间。`test_vq_losses.py`、`test_vq_step_losses.py` |
| C-13 | `C_route` 梯度恒零 | R19 | **fixed** | `route_commit_loss` 进入主损失（`vq_route_codebook`）；`pack_codebooks_for_c` 含 `C_route`/`W_route`；推理 hard `one_hot(argmax)` 无噪声。smoke `route_beats_weight_decay_baseline` 实测通过 |
| C-14 | Binding 只 bind 不 unbind | R20 | **fixed** | `binding_forward` 返回 `b_q/k_q/v_out`；`v_out=NN(C_val, unbind(b_q,k_q))`；输出 STE `o=z+sg(v_out-z)` 保留 `dz`。`test_binding_unbind.py` + C binding 组 |
| C-15 | 价值标量每晶格一个向量、各层共享 | R22 | **fixed** | `init_value_scalars` 按 R22 粒度：HRQ top/fine、sparse、lowrank、manifold、binding 三族、contrast 双码本；残差层共享。`test_parity_lattices.py` 的 alpha=0.1 全晶格覆盖 |

---

## 3. 实现侧 F-1 … F-7

| # | 问题 | 裁决 | 状态 | 证据 |
|---|---|---|---|---|
| F-1 | 融合仍保留 `exp(β_val·value_signal)` 软调制 | R5 | **fixed** | 同 C-1；`fuse_lattices_with_aux` 明确忽略 `gvalue` |
| F-2 | `d_top=1-mean(hrq_top_sim)` 方向反了且是 batch 标量 | R2 | **fixed** | 同 P0-3；逐样本 `(B,)`，与 `d_sparse_min` 同域 |
| F-3 | EMA/frozen 参数留在 AdamW，weight decay 会改动它们 | R3 | **fixed** | `partition_params` + `set_to_zero` 语义（优化器只见 trainable）；`test_param_groups.py` 的 round-trip/隔离测试。`danger.C/C_normal` 在 frozen 组 |
| F-4 | gvalue `pos==neg` placeholder，C 端直接禁用检查 | R6/R13 | **fixed（结构）；语义 accepted-deferred** | `make_global_value_vectors` 给出互不相同的 R6 锚点；`save_cog_checkpoint` 导出两半；`lcm.py::validate_safety_codebooks` 拒绝相等占位；`GValueCodebook.verify_integrity`。真实训练出的语义锚点留到正式重训 |
| F-5 | LowRank/Contrast/Binding export 只取第一层/第一块 | R10 | **fixed** | manifest v2 全层导出；`test_export.py`、`test_c_load.py`、`test_parity_lattices.py` |
| F-6 | `entropy_threshold=2.0` 使熵门几乎恒过 | R9/R17 | **fixed** | `train/config.py` 改为归一化阈值 0.5；`make_train_step` 用 `H/ln(n_active)`；C v3 同样是归一化 0.5；smoke 的 `entropy_gate.pass` 包含真实 mask，crafted peaked/uniform 与真实值都记录 |
| F-7 | 多个默认值与规范不一致 | R17 | **fixed** | `max_inference_steps=16`、`max_retrievals_per_step=128`、`tau_contrast=0.1`、`tau_val=0.05`、`consistency_threshold=0.4`；C `LCM_CONSISTENCY_THRESHOLD=0.4`，`value_consistency=clip(cos,0,1)` |

---

## 4. R1–R22 状态总表

| 裁决 | 主题 | 状态 | 关键证据 |
|---|---|---|---|
| R1 | HRQ Poincaré 主路由 + top-k 价值重排 | fixed | §2 C-13、`test_hrq_routing.py`、parity |
| R2 | Sparse LFQ 逐样本 `d_top` + 同域阈值 | fixed | §2 C-5/C-6、`test_sparse_lfq.py`、smoke |
| R3 | EMA 接线与三组参数隔离 | fixed | §1 P0-1、`test_ema_state.py`、smoke EMA |
| R4 | 融合以 §7.1 为准 | fixed | §1 P0-5、`test_fusion_conformance.py` |
| R5 | 删除融合价值软惩罚 | fixed | §2 C-1、`test_fusion_conformance.py` |
| R6 | 统一双曲域 + gvalue 严格三法则 | fixed | §2 C-3、`test_safety_semantics.py`、C gvalue 组 |
| R7 | Danger 有界相对接近度 | fixed | §2 C-2、C danger 组 |
| R8 | `build_dag` 修正 | fixed | §2 C-4、C DAG 组 |
| R9 | 训练/引擎融合、熵门、max_steps | fixed | 融合 parity；C 循环 graceful/hard；JAX 归一化熵；F-6 |
| R10 | C 端逐格 + schema v2 | fixed | §1 P0-4、§3 F-5、parity/C load；`train/c_load.py` 逐 tensor sha256 校验（`test_c_load.py::test_manifest_tensor_sha256_rejects_tampered_payload`）；pickle 项见 D-10 |
| R11 | 安全子系统（d.md） | fixed（本地形态）；语义安全 accepted | §1 P0-6、`test_external_verifier.py`、smoke safety |
| R12 | 自我意识（e.md） | fixed（causal accepted deviation） | `test_self_awareness_wiring.py`、smoke overhead/self |
| R13 | 安全链真正生效 | fixed（结构链路）；真实语义码本 deferred | `init_danger_params` 双半、`save_cog_checkpoint`、`lcm.py` 拒占位；gvalue/danger 的 trailing SHA-256 与 manifest 逐 tensor hash 均验证，C init 后 `gvalue_verify`/`danger_verify` 再与保存 digest 比对；`gv_n=0`/`danger_M=0` 不再 fail-open；smoke 用**最终 checkpoint** 的 `LCMInferEngine` 跑真实 v3 loop：ret=-2、0 token、alert、trace；gvalue strict margin=0.5 probe。`--active` 与非 loop 路径也走同一 abort/LOCK 门 |
| R14 | 外部验证器 + Self 激活状态机 | **partially implemented / deferred** | `train/self_activation.py` 状态机库存在且有测试；**未接入生产训练循环**，`C_self` 的 EMA 写入不受状态机门控；C 端 identity core 也**没有 mprotect**。本地形态的外部验证器已接线（§1 P0-7），完整 R14 生产执行留待后续 |
| R15 | Self lattice 码本 NN + cosine familiarity | fixed | `test_self_lattice.py`、smoke self_bias |
| R16 | 自我意识模块接线 + 5% | fixed（模块）；5% 实测见 smoke | `test_self_awareness_wiring.py`、`smoke_report.json.overhead` |
| R17 | 参数默认值对齐 | fixed | §3 F-7；C 一致性阈值 0.4（clip 域） |
| R18 | 逐层 codebook/commitment loss | fixed | §2 C-12；HRQ top 与 manifold commitment 的连续侧均为 forward 的 tangent pointer `safe_unit(z)`（因为 `z_P=exp_map(safe_unit(z))` 且 `log_map(z_P)=safe_unit(z)`），与 quantised 侧同域；`test_safety_semantics.py` 用 ‖z‖>1 校验独立手算值（`d=2→0.125`、`d=4→0.0625`）与 `‖cont‖=1.0` |
| R19 | Routing 训练与 hard 推理 | fixed | §2 C-13、smoke route |
| R20 | Binding unbind | fixed | §2 C-14 |
| R21 | Contrast 采样/温度/feature bank | **partially implemented / deferred** | 普通 InfoNCE、负样本排除正样本、温度、feature bank 已修复并有 `test_contrast_conformance.py`、smoke 证据；**value-biased NCE 只在调用方提供 gvalue 时可用**，训练路径没有 gvalue，因此训练期间不会运行该变体（不伪造路径） |
| R22 | value scalar 粒度 | fixed | §2 C-15 |

---

## 5. Task 15 发现并修复的新问题

这些不在旧审计里，是 100 步 smoke 准备过程中暴露的阻断项：

| # | 问题 | 修复 | 证据 |
|---|---|---|---|
| N-1 | `joint=True` 直接崩溃：`manifold_orth_loss` 对 traced `indices` 调 `jnp.unique`，JAX 0.10 需要静态 `size=` | 去掉 `unique`，active + sampled 直接拼接；重复项只是同一 loss 项的重复权重，形状静态 | `train/lattices.py`；`test_smoke_conformance.py` 的 mini smoke 以 `joint=True` 跑通 |
| N-2 | 默认 `max_inference_steps=32` 时，32 步展开循环的反向梯度非有限（69 个 trainable leaf 中 39 个 NaN），参数被污染 | 默认改为 R17 的 16；16 步时 0/69 非有限 | `train/config.py`；梯度检查见任务报告 |
| N-3 | `cog_train` 导出的 gvalue `pos==neg`、danger `threat==normal`，`lcm.py` 按 R13 拒绝加载 | `init_danger_params` 生成互不相同的双半；`FROZEN_PATHS` 加 `C_normal`；首步用 batch z 播种 normal 半；gvalue 导出真实 R6 锚点；`_save_danger` 拒绝相等写入 | `train/lattices.py`、`train/param_groups.py`、`train/cog_train.py`、`train/checkpoint.py`；smoke `safety.export` |
| N-4 | smoke 需要训练后诊断，但 `train_cog` 不暴露 per-step 状态 | 新增可选 `on_step` 回调（默认 None，不改变训练行为）和 `tokenizer_path` 覆盖（测试用临时 tokenizer） | `train/cog_train.py` |

---

## 6. Deferred / 已解决清单（状态逐条标注）

| # | 项 | 影响 | 触发条件 / 后续 |
|---|---|---|---|
| D-1 | ~~规范文档 a–e 的 v3.1-conformance 文字同步（Task 0）未完成~~ **已解决（Task 0 + review 修正）** | 五份英文 + 五份中文已按 R1–R22 同步，头部 changelog 引用设计文档；关键条款 grep 见 §7 | — |
| D-4 | gvalue/danger 码本是结构占位，不是语义锚点 | gvalue 是固定手工 ± 轴；danger 威胁半是固定随机基、normal 半是安全 encoder 状态库。触发链路可验证，但不验证语义安全 | 正式重训时用真实正/负、威胁/正常语料锚点替换 |
| D-5 | 外部验证器是本地形态（shared memory + 文件签名），不是 HSM | 不是生产安全边界 | 部署到有 HSM 的环境时替换签名层 |
| D-6 | `train/causal_subject.py` 的功能与验证不扩展 | `e.md §5` 的反事实/边界行为与实现有差异 | **accepted deviation（用户 2026-11-20 决定）**；现有 causal 测试全绿即可 |
| D-7 | 100 步 smoke 不证明语言能力 | loss 曲线只用于“训练发生/数值有限/更新发生” | 正式语言质量评估需要 >100 步与独立评测集 |
| D-8 | `config.consistency_threshold` 与 C 常量可能漂移 | C 仍使用编译期 `LCM_CONSISTENCY_THRESHOLD`，不读取运行期 `config.json`；Python 字段只在训练/导出侧使用 | **已加漂移测试**：`test_safety_semantics.py::test_python_consistency_threshold_matches_c_header` 断言 `LCMConfig().consistency_threshold` 与 `infer/lcm.h` 的 `LCM_CONSISTENCY_THRESHOLD` 相等。改阈值必须同时改两处并重编译 C |
| D-9 | 100 步训练的原始 streaming probe 未过（HRQ 5.86%、Sparse 7.03%、LowRank 1.30%、Manifold 6.25%、Binding 0.20%、Contrast 1.33%、Self 1.56%；Route 83.33%）；旧版 post-hoc 校准也逐层失败（bind 层 1/2 0.195%、contrast 最低 6.84%、manifold 9.96%） | 不能声称“训练出的每个码本 10% 都被覆盖”；不能声称 conformance complete | **open research**：见 `docs/d9-codebook-utilization-research-plan.md`。审查后修正了校准前向模拟：`b_raw` 必须用投影后的 `q_key/q_val` 走真实 residual VQ，Manifold 必须在 R6 域校准，LowRank 必须用 `C=U@V^T` 的残差，Contrast 必须用欧氏域，且每层用 bounded distinct 行 + 重复填充。修正后的 `checkpoints_calibrated_fixed5` 在官方 streaming probe 和独立 `research_d9_probe` 上逐层全过（最低 Contrast 11.13%、Binding 12.89%，HRQ fine 已纳入，0 个 <10% 层）。但这是 **post-hoc 码本替换**，不是训练；是否接受为官方 memory construction 仍需行为/语义审查。原始 100 步训练路线仍 FAIL，不能混为一谈 |
| D-10 | manifest 的 `params.*` pickle 项没有逐 tensor sha256（C loader 只读 `.bin`） | 训练 params 的 C 侧完整性由 pickle schema/`load_cog_state` 负责，不是 C 张量 | C 端只接受 flat float32 `.bin`；pickle 项在 `_read_tensor` 直接拒绝。若未来导出 pickle 索引，需要单独的签名层 |
| D-11 | R21 value-biased NCE 在训练路径不可用 | 训练状态没有 gvalue，`compute_contrast_loss` 只走普通 InfoNCE | **deferred**：不伪造 gvalue。普通 InfoNCE 的采样/温度/feature bank 已 fixed；调用方显式提供 gvalue 时才运行 value-biased 变体 |
| D-12 | R14 SelfActivationState 未接入生产训练 | `train/self_activation.py` 状态机库有测试，但 `ema_update_self` 的写入不受它门控；C identity core 未 mprotect | **deferred / accepted deviation**：本 pass 不接线；代码注释与 §4 R14 已标注 |
| D-13 | DAG trigger 距离与检索域不同 | HRQ/contrast 的触发筛选用欧氏最小距离，真正检索在 Poincaré 域；跨 lattice 距离不可直接比较 | 已在 `infer/loops_v3.c::build_dag_v3` 注释中记录；阈值按 lattice 各自维护。**已补 v3 active trace**：`lcm_get_trace_v3_flat` 导出每步各 lattice `active`；`scripts/validate_c_dag_parity.py` 在 40 个输入、5 种 mask 上重建 C 融合，最大误差 9.5e-6、0 失败。`safe_unit` 的加性 eps 已在 C 的 `unit_copy`/`v3_unit_copy` 与 JAX 对齐，并有 `test_safe_unit_boundary.py` |
| D-14 | `train/causal_student_train.py` / `train/eval_lattice_storage.py` 已从 retired generic-codebook 路径移植到 `cog_loop_scan(z, params, cfg)` | 旧 `student.pkl` 若没有 `route`/`fusion` 参数会明确报错，不再静默或 `NotImplementedError`；该 standalone trainer 仍是单一 AdamW 树，不实现 R3 的 EMA/分组 | 需要旧 checkpoint 时重新训练/导出 canonical 六格参数；canonical 训练请用 `train_cog`。新 checkpoint 带 route/fusion/self/value_scalars |

---

## 7. 验收命令与结果

在本 worktree（HEAD 见任务报告）执行：

```bash
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
  train/test_smoke_conformance.py train/test_parity_lattices.py -q
make -C infer clean && make -C infer test
python train/smoke_conformance.py --steps 100 --batch 1 --seq 64 \
  --data /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat \
  --shape /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json \
  --spans /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy \
  --output checkpoints/conformance_smoke --passive-only
```

结果：

- `JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest train/test_smoke_conformance.py train/test_parity_lattices.py train/test_c_load.py train/test_vq_losses.py train/test_binding_unbind.py train/test_sparse_lfq.py -q`
  → **63 passed in 269.94s**（mini smoke + 24 parity + 11 C loader + R18/R20/R2 coverage）。
- `JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest train/test_safety_hypothesis.py -q`
  → **12 passed in 76.19s**（formal constants now 128/0.4）。
- `JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest train/test_c_loop.py -q`
  → **12 passed in 2.23s**（含真实 v3 loop 的 danger abort `-2` / 无 token 路径、
  active 路径 abort/LOCK、bridge digest 校验）。
- `JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest train/test_vq_losses.py train/test_safety_semantics.py train/test_ema_state.py -q`
  → **42 passed in 145.06s**（R18 独立手算：`d=2→0.125`、`d=4→0.0625`，
  连续侧范数 1.0；含 100-step grouped vs ungrouped）。
- `make -C infer clean && make -C infer test` → **ALL TESTS PASSED**；
  日志含 **249** 个 `[PASS]` 检查，退出码 0。
- `nm -D infer/liblcm.so | grep -E "lcm_infer_loop_v3|hrq_forward_v31"` →
  `lcm_infer_loop_v3`、`lcm_infer_loop_v3_flat`、`hrq_forward_v31` 均存在。
- 100 步 smoke（复跑未加 `--allow-fail`；已知 FAIL 使退出码为 1）→
  完成 100/100 步；loss 首 10 步均值 **18.3368** → 末 10 步均值 **11.9986**
  （比值 **0.6543**，< 0.8），全程有限；**20/22 checks PASS**。
  - **FAIL：10% 码本利用率**。training probe（固定单 batch）HRQ 5.08%/
    Sparse 4.69%/Binding 1.17%；Ruling 51 的 100-batch streaming probe 仍失败：
    HRQ 5.86%、Sparse 7.03%、LowRank 1.30%、Manifold 6.25%、Binding 0.20%、
    Contrast 1.33%、Self 1.56%；Route 83.33%。报告为 D-9 unresolved。
  - **FAIL：真实 mask 熵门**。`real_mask_H_norm=0.5652 > 0.5`；crafted
    peaked/uniform 分别通过/不通过，说明门本身可判别，但 100 步真实路由
    尚未收敛到 0.5 以下。没有调低阈值。
  - **PASS：safety split（Ruling 50）**。导出 `danger_codebook.bin` 的直接
    `danger_assess(value_consistency=1.0)` 对 threat 返回
    `THREAT_PATTERN_MATCH`（type 1），normal 不阻断；真实 v3 loop 的
    `safety_abort_any_conflict` 为 hard-stop（-2、0 token、alert、trace），
    威胁输入的实际 type 是 `THREAT_PATTERN_MATCH`，因为 `detect_any_conflict`
    同时检查 `z_cur` 与 `z_next`。normal 输入在 `use_safety=True` 时按严格
    三法则由 `THREAT_THREE_LAWS` 中断；这是用户确认的预期行为，不宣称
    normal input 的语义安全通过。value 标量为占位（`max|v|<0.1`）时一致性
    返回 1.0，不产生 deception 误报。
  - 其余 PASS：真实 `jax.grad` 每格范数 >1e-6、EMA N/m、码本方差 ≥0.5、
    C_route baseline（仅 `C_route`，不含 `W_route`）、sparse LFQ、HRQ 路由、
    VQ codebook/commitment split、routing hard、HRQ-inactive LFQ、
    observability 100 条、feature bank、gvalue strict margin=0.5
    （pos 1.0 safe / neutral 1.0 abort / neg 1.0 abort）、frozen hashes、
    self_bias、5% 开销（0.0398%）、save/load/旧 checkpoint 拒绝。
  - `--allow-fail` 只影响退出码；`checks` 表仍是 verdict。
- Artifact：`checkpoints/conformance_smoke_final2/smoke_report.json` +
  `smoke_report.md` + `step_000100/`（149 MB，不提交）。

**规范文字 grep（本次 Task 0）**：

```bash
grep -n "λ_lfq\|λ_shrink" docs/b.md docs/b_cn.md          # 两版命中
grep -n "H_norm\|n_active" docs/c.md docs/c_cn.md          # 两版命中
grep -n "balance > -safety_margin_relative" docs/b.md docs/d.md docs/b_cn.md docs/d_cn.md
grep -n "C_self\[0\]\|C_self\[1:\]" docs/e.md docs/e_cn.md
grep -n "v3.1-conformance" docs/{a,b,c,d,e}.md docs/{a,b,c,d,e}_cn.md
```

五份英文规范 + 五份中文版均带 v3.1-conformance changelog；`λ_sparse` 已拆为
`λ_shrink`/`λ_lfq`，`β_val` 标记 deprecated，`max_steps` 拆为调度预算与安全硬限，
`C_self` 与 `SelfActivationState` 命名已区分。C 参考测试日志现为 **249** 个
`[PASS]` 检查（不是 234）。

> 本次 100 步复跑运行在 `661a26b`（I7 修复尚未提交的工作树）；smoke JSON 的
> `meta.git_commit` 记录的是运行时 HEAD `661a26b`。I7 修复与 evidence 更新随后续
> 提交落库；`docs/lattice-conformance-experiment-report.md` 已同步本次数值。

---

## 8. 结论

- 旧审计的 7 条 P0：**7 条 fixed**；其中 P0-6 的语义安全码本归 D-4（deferred）。
- C-1…C-15：**15 条 fixed**；C-7 的规范文字同步、C-8/C-9/C-10 的文档歧义已在本次 Task 0 一并落文。
- F-1…F-7：**7 条 fixed/结构 fixed**；F-6 的 JAX 归一化熵门已改，
  F-7 的 C 一致性阈值已重新标定为 0.4（clip 域）。
- 本次新增并修复 4 个阻断项（N-1…N-4），其中 N-2/N-3 直接影响 100 步 smoke
  能否运行与 C 端能否加载。
- 100 步 smoke 的 10% 码本利用率未通过（D-9，streaming probe 也未过），
  真实训练 mask 的归一化熵 0.5652 > 0.5 也未通过。两项**不**被算作验收通过；
  artifact 见实验报告。
