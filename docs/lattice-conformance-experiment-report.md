# 晶格规范一致性实验报告（≤100 步 smoke）

日期：2026-11-21
smoke 运行 HEAD：`f6b2e4c`（value consistency 实际索引、安全门 z_cur/z_next、D-9 诊断与文档修订之后）
基线：`docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` R1–R22
偏差复审：`docs/lattice-spec-vs-impl-audit-v3.1.md`

> **结论边界**：100 步 smoke **不证明语言能力**。它只证明：训练路径发生、每格都有真实梯度或 EMA 更新、数值有限、稀疏/路由/熵/安全/自我/持久化等规范路径能被触发。gvalue/danger 是**结构占位**，不是语义安全锚点。

> 本次复跑 **20/22 checks PASS**；两项 FAIL 仍是 `codebook_utilization_10pct` 与 `entropy_gate`（真实 mask），详见 §5/§8。复跑未加 `--allow-fail`，退出码 1；`checks` 表是 verdict。

## 1. 环境

| 项 | 值 |
|---|---|
| 平台 | Linux-7.3.4-zen2-1-zen-x86_64-with-glibc2.44 |
| Python | 3.14.10 |
| JAX | 0.10.0 |
| 设备 | CPU（`JAX_PLATFORMS=cpu`；GTX 1650 4GB 不参与） |
| smoke 运行 HEAD | `f6b2e4c`（value consistency 实际索引、安全门 z_cur/z_next、D-9 诊断与文档修订） |
| 总耗时 | 152.3s |
| C 参考 | `make -C infer clean && make -C infer test`（251 PASS） |

## 2. 配置

| 参数 | 值 |
|---|---|
| `d_model` | `256` |
| `vocab_size` | `30000` |
| `n_heads` | `8` |
| `d_ff` | `384` |
| `n_encoder_layers` | `2` |
| `M_top` | `512` |
| `M_fine` | `256` |
| `n_hrq_layers` | `3` |
| `M_sparse` | `512` |
| `M_lr` | `256` |
| `ranks` | `[2, 4, 8]` |
| `M_man` | `512` |
| `t_dim` | `4` |
| `M_bind` | `512` |
| `n_bind_layers` | `3` |
| `M_contrast` | `512` |
| `n_contrast_layers` | `3` |
| `M_danger` | `256` |
| `n_self_codes` | `64` |
| `max_inference_steps` | `16` |
| `lambda_lfq` | `1.5` |
| `alpha_val` | `0.1` |
| `alpha_self` | `0.05` |
| `beta_vq` | `0.25` |
| `gamma_sparse` | `0.99` |
| `gamma_man` | `0.99` |
| `gamma_bind` | `0.99` |
| `entropy_threshold` | `0.5` |
| steps / batch / seq | 100 / 1 / 64 |
| lr | 0.0003 |
| joint | True |
| self_awareness | True |
| passive_only | True |
| overfit_one_batch | True |

`overfit_one_batch=True` 复用同一个窗口，满足设计 §6.3 A 的“单 batch 过拟合”；它也意味着路由每步看到同一个 z，是 §5 利用率偏低的直接原因。

## 3. 数据身份

| 项 | 值 |
|---|---|
| `.dat` | `/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat` |
| `_shape.json` | `/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json` |
| `_docs.npy` | `/home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy` |
| tokenizer | `bpe30k` (`/home/DslsDZC/data/lcm/tokenizers/bpe30k/tokenizer.json`) |
| tokenizer sha256 | `8749577fc889fa4d3a4a0b3f737d4a7ee61a7b3dadd34c159df92166fe80038d` |
| token data sha256 | `8b6ac8859560aa6fa124db591e48bca9a461d2569a573827684dfb0889bd9b1a` |
| document spans sha256 | `7263b3c82aa46cd165f8430add5a69dba9eeb56758016fead8c89d545eebc3f4` |
| tokens / docs | 43036 / 829 |
| separator | id 3 (`[EOS]`) |
| model vocab | 30000 |

旧 `data/zhwiki_tokens.dat` 只有 `n_tokens` 的 identity，缺 tokenizer/separator/spans 摘要，`resolve_dataset_identity` 会拒绝。本次用 `scripts/build_conformance_corpus.py` 从 bpe30k tokenizer 重建了合规语料。

## 4. 100 步 loss

- 首 10 步均值：**18.3368**
- 末 10 步均值：**11.9986**
- 比值：**0.6543**（< 0.8：PASS）
- 全部有限：PASS

| 步区间 | 均值 | 最小 | 最大 |
|---|---|---|---|
| 0–9 | 18.3368 | 17.4823 | 19.3431 |
| 10–19 | 17.6552 | 16.4572 | 20.3163 |
| 20–29 | 16.3323 | 15.0985 | 21.7605 |
| 30–39 | 15.6059 | 14.1383 | 21.5206 |
| 40–49 | 15.3877 | 13.7310 | 17.7664 |
| 50–59 | 13.3281 | 12.7299 | 14.2777 |
| 60–69 | 12.9188 | 12.5025 | 13.5287 |
| 70–79 | 15.5383 | 12.5509 | 20.1645 |
| 80–89 | 12.5461 | 12.0204 | 14.3196 |
| 90–99 | 11.9986 | 11.6236 | 13.2558 |

完整逐 step 曲线在 `checkpoints/conformance_smoke_final2/smoke_report.json` 的 `loss.per_step`。

## 5. 逐格更新、真实梯度、方差与利用率

参数 delta 单独不能区分真实梯度和 AdamW weight decay；本节以 `make_train_step(..., return_grads=True)` 的一次真实 `jax.grad` 为主判据，delta 只作 secondary。

| 晶格 | grad norm | grad>1e-6 | Δmax abs (secondary) | var ratio | utilization |
|---|---|---|---|---|---|
| encoder | 1090.5130 | PASS | 0.0152 | — | — |
| hrq | 677.0636 | PASS | 0.0162 | 4.5002 | 5.08% |
| sparse | — | — | 3.0744 | 49.3075 | 4.69% |
| lowrank | 1.8931 | PASS | 0.0084 | 0.7463 | 6.77% |
| manifold | 0.0023 | PASS | 3.0520 | 49.8755 | 4.69% |
| binding | 0.7223 | PASS | 1.4081 | 76.2772 | 1.17% |
| contrast | 1.9307 | PASS | 0.0200 | 298.9470 | 2.96% |
| route | 39.1918 | PASS | 0.0149 | 1.0535 | 100.00% |
| fusion | 22.7596 | PASS | 0.0166 | — | — |
| self | — | — | 3.1184 | 41.3112 | 37.50% |
| value_scalars | 55.0926 | PASS | 0.0159 | — | — |

sparse/self 无 trainable 子树，用 N/m 判据；manifold/binding 同时有 `T`/`A_k`/`A_v` 的梯度和码本 EMA。

### 5.1 10% 码本利用率（设计 §6.3 A / Ruling 51）

| 晶格 | training probe (fixed batch) | streaming probe (100 batches) | 阈值 | verdict |
|---|---|---|---|---|
| hrq | 5.08% | 5.86% | 10% | **FAIL** |
| sparse | 4.69% | 7.03% | 10% | **FAIL** |
| lowrank | 6.77% | 1.30% | 10% | **FAIL** |
| manifold | 4.69% | 6.25% | 10% | **FAIL** |
| binding | 1.17% | 0.20% | 10% | **FAIL** |
| contrast | 2.96% | 1.33% | 10% | **FAIL** |
| route | 100.00% | 83.33% | 10% | PASS |
| self | 37.50% | 1.56% | 10% | **FAIL** |

Streaming verdict：**FAIL**；batches=100，B=1。
Training probe（固定单 batch，secondary）：**FAIL**。
Ruling 51 的 streaming probe 仍低于 10%：这不是“码本没更新”，而是路由/码本在 100 步内高度集中。正式验收需要更多步数、B≥4 或路由探索；本次如实记为 unresolved（D-9），不调低阈值。只读诊断（`train/research_d9_probe.py`）进一步显示 Binding 与 Self 的选择输入无关、LowRank 每层只有 2 个吸引子；见 `docs/d9-codebook-utilization-research-plan.md`。

### 5.1.1 Post-training FPS 校准的严格逐层复核（2026-11-23）

`scripts/calibrate_coverage.py` 把每个码本替换成最终 encoder query 的 FPS 覆盖行。旧版原型导出的 `checkpoints_calibrated` 在 **family mean** 下全绿，但按 D-9 原文“每个码本 ≥10%”逐层判定**不通过**；该旧结果保留为失败证据。`_streaming_utilization_probe` 已改为用 `min_fraction` 判定，family mean 仅作诊断。下面是旧版结果和审查后的修正。

| 晶格 | mean | 最低层 | 阈值 | verdict |
|---|---:|---:|---:|---|
| hrq（仅 top） | 16.99% | 16.99% | 10% | PASS |
| sparse | 15.04% | 15.04% | 10% | PASS |
| lowrank | 16.54% | 12.89% | 10% | PASS |
| manifold | 9.96% | 9.96% | 10% | **FAIL（边缘）** |
| binding | 13.28% | **0.195%**（bind 层 1/2 各 1/512，恒定） | 10% | **FAIL** |
| contrast | 12.11% | **6.84%**（a1 7.03%，b1/b2 7.81%） | 10% | **FAIL** |
| route | 100% | 100% | 10% | PASS |
| self | 32.81% | 32.81% | 10% | PASS |

根因不在 FPS 的随机初始点，而在 `calibrate_coverage.py::_residual_rows` 的逐层全覆盖：

1. bind 的真实 query `b_raw` 在 1000 个窗口里只有 436 个唯一值；
2. 第一层 FPS 选了 512 行，覆盖了全部 436 个唯一值，`r - rr[idx]` 的残差**全为零**；
3. 第二、三层在零残差上继续 FPS，只能安装零行；`_selection_query(normalize_selection=True)` 下零行成为所有 query 的最近行，于是层 1/2 恒定选中 1 行。

**以上 0.195% / 6.84% 的根因复核（2026-11-23，审查后修正）**：

上一版 `calibrate_coverage.py` 的 bind 构造并不等于真实前向：它用原始 `z` 在每层独立选 key/val 行，没有用投影后的 `q_key/q_val`，也没有走 residual VQ 的逐层残差更新。修正为：

1. `q_key = z @ (V @ A_k)^T`、`q_val = z @ (V @ A_v)^T`；
2. 先用校准后的 key/val 码本跑 `_residual_vq_chain`，取得每层真实量化行 `C[idx]`；
3. 再用这些行做 9 对 HRR bind，得到真实 `b_raw`；
4. 最后校准 bind 码本。

同时修正三处与正式 forward 不一致的校准：

- Contrast：`contrast_forward` 用原始欧氏选择，校准从 `cosine=True` 改为 `cosine=False`；
- Manifold：检索在 R6 域 `exp_map(safe_unit(z))`，校准改为在该域 FPS，并安装 `log_map(z_P)` 到 tangent-space `C`；
- LowRank：实际码本是 `C = U @ V^T`，残差更新必须用 `C_rows` 而不是 FPS 原始行。

修正后，**bounded-distinct post-hoc 校准可以让逐层 D-9 全部通过**。`checkpoints_calibrated_fixed5`（samples=1500，binding distinct=256，contrast distinct=320）的官方 streaming probe 与独立 `research_d9_probe` 均为：

| 晶格 | family mean | 最低层 | 阈值 | verdict |
|---|---:|---:|---:|---|
| HRQ（top + 3 fine） | 24.71% | 17.58% | 10% | PASS |
| Sparse | 17.19% | 17.19% | 10% | PASS |
| LowRank | 20.83% | 17.19% | 10% | PASS |
| Manifold | 17.58% | 17.58% | 10% | PASS |
| Binding | 14.63% | 12.89% | 10% | PASS |
| Contrast | 13.05% | 11.13% | 10% | PASS |
| Route | 100% | 100% | 10% | PASS |
| Self | 37.50% | 37.50% | 10% | PASS |

独立 `research_d9_probe` 的逐层扫描为 0 个 <10% 层。HRQ fine 层已纳入统计（`hrq_vq['fine_idx']`），不再是只追踪 top 的缺口。

需要撤回上一版的“Contrast 需要 104 个互异簇、不可达”推论：那是基于错误的 `b_raw` 模拟和 600 样本实验；真实前向模拟 + 1500 样本 + bounded distinct 后，Contrast 最低层达到 11.13%。两层各 52 行并不要求原始输入有 104 个簇，因为两层的 residual query 分布不同。

**仍未解决/不能声称的**：

- 这是 post-hoc 码本替换，不是训练出来的记忆；是否接受为官方 memory construction 仍需行为/语义审查。D-9 的 PASS 只证明利用率，不证明语义质量。
- 同样算法放进初始化 + 100 步训练仍然更差（encoder 漂移）：HRQ 0.78%、Sparse 2.93%、LowRank min 7.03%、Manifold 4.88%、Binding min 1.56%、Contrast min 0.20%。所以不能把 `train_binding_fps.py` 的路径当作已验证的官方训练步骤。
- 官方判据只有 **per-layer `min_fraction`**；family mean 仅作诊断。不得再用 mean 代替逐层验收。

### 5.2 码本方差（final/initial ≥ 0.5）

| 晶格 | ratio |
|---|---|
| hrq | 4.5002 |
| sparse | 49.3075 |
| lowrank | 0.7463 |
| manifold | 49.8755 |
| binding | 76.2772 |
| contrast | 298.9470 |
| route | 1.0535 |
| self | 41.3112 |

Result: PASS

## 6. EMA、VQ split、routing hard、observability

### 6.1 EMA

| 分支 | ΔN abs | N sum | 使用行 | Δm norm | changed |
|---|---|---|---|---|---|
| sparse | 63.3968 | 63.3968 | 24/512 | 573.9488 | PASS |
| manifold | 63.3968 | 63.3968 | 24/512 | 573.9488 | PASS |
| self | 63.3968 | 63.3968 | 24/64 | 573.9488 | PASS |
| binding | 570.5711 | 570.5711 | 54/4608 | 377.4845 | PASS |

EMA 总结果：PASS

### 6.2 Codebook / commitment split (R18)

- EMA codebook loss zero：PASS
- commitment loss positive：PASS
- `hrq_codebook`：0.0033098
- `hrq_commitment`：0.0033098
- `sparse_codebook`：0.0000
- `sparse_commitment`：0.0001511
- `lowrank_codebook`：0.7620
- `lowrank_commitment`：0.7620
- `manifold_codebook`：0.0000
- `manifold_commitment`：0.0076786
- `binding_codebook`：0.0000
- `binding_commitment`：0.022789
- `contrast_codebook`：1.4140
- `contrast_commitment`：1.4140
- `route_codebook`：0.25021
- `route_commitment`：0.25021
- pass：PASS

### 6.3 Routing hard (R19)

- deterministic：PASS
- one-hot：PASS
- pass：PASS

### 6.4 Observability (R16)

- raw records：100/100
- long-term records：195
- errors：{}
- pass：PASS

## 7. C_route vs weight-decay baseline

- `C_route` Δmax：1.5537e-02
- weight-decay-only `C_route` Δmax：1.066e-05
- 实际 / baseline：1457.8×
- full route tree Δmax（含 `W_route`，secondary）：1.5537e-02
- 结论：PASS（C_route only; W_route is reported separately）

## 8. Sparse / LFQ、HRQ 路由、熵门、feature bank、self

### 8.1 Sparse LFQ

- 推理非零比例：100.00% (8/8)
- near 非零：PASS；far 强制 zero_vec：PASS
- far d_top=2.5766, d_min=2.8352, lambda_lfq=1.5
- 无 c_top 时 LFQ 禁用：PASS
- d_top shape=[2]，跨样本不同=PASS

### 8.2 HRQ 路由

- v=0 top-1 == pure Poincaré：PASS
- v≠0 仍在几何 top-k：PASS
- 价值偏置改变 top-1：PASS (alpha=10.0, k=4)

### 8.3 熵门（真实 mask 是 verdict 的一部分）

- crafted peaked：0.0396 → PASS
- crafted uniform：1.0000 → 不通过=PASS
- **真实训练 mask**：H_norm=0.5652 → **FAIL**（阈值 0.5）
- 总 pass：**FAIL**；JAX 训练门按归一化 0.5 解释。

### 8.4 Feature bank（synthetic policy probe）

- synthetic=True：constructed 8-d FeatureBank and 5 injected dead rows; this is the replacement policy test, not a 100-step training event
- contrast 5 dead → 5 replaced，farthest-point=PASS
- sparse 5 dead → 5 replaced，cold-start=PASS

### 8.5 Self lattice

- self_bias [0.0036, 0.0500]，alpha=0.05，in_range=PASS
- 熟悉 vs 不熟悉：0.0500 vs 0.0036，varies=PASS
- 训练步内范围：[0.004925, 0.0500]

## 9. 安全：pattern probe 与 v3 hard-stop 分开

| 检查 | 结果 |
|---|---|
| 训练 is_safe 全 True（无安全 oracle） | PASS |
| 导出 danger pattern：threat blocked / type | PASS / 1 (1=PATTERN_MATCH) |
| 导出 danger pattern：normal not blocked / type | PASS / 0 |
| 导出 danger pattern pass | PASS |
| 真实 v3 loop 威胁输入 ret | -2 (CONFLICT_DANGER/THREAT_PATTERN_MATCH) |
| `generate` 威胁输入 ret / tokens | -2 / 0 |
| alert callback | PASS (CONFLICT_DANGER/THREAT_PATTERN_MATCH) |
| trace file written | PASS |
| v3 abort any conflict | PASS |
| C gvalue margin=0.5：pos safe / neutral abort / neg abort | 1.0000 / 1.0000 / 1.0000 (n=12) |
| gvalue hash unchanged | PASS |
| danger hash unchanged after step 0 | PASS |
| 导出 gvalue pos≠neg / danger threat≠normal | PASS / PASS |

Ruling 50：v3 loop 的 threat 输入现在由 `CONFLICT_DANGER/THREAT_PATTERN_MATCH` 中断：`detect_any_conflict` 同时检查进入步的 `z_cur` 和离开步的 `z_next`。只查 `z_next` 会让被 DAG 一步变换掉的威胁输入逃逸；这个洞此前被 value consistency 的占位误报掩盖。`safety_abort_any_conflict` 记录该 hard-stop；导出的 danger pattern probe 仍单独验证 `THREAT_PATTERN_MATCH`。normal 输入在 `use_safety=True` 时按严格三法则由 `CONFLICT_GVALUE/THREE_LAWS` 中断，这是已确认的预期行为；`use_safety=False` 时不触发 gvalue，且不匹配 danger pattern。

gvalue/danger are structural placeholders built from fixed anchors and a bank of safe encoder states. They exercise the trigger chain; they do not establish semantic safety.

## 10. 逐格 JAX↔C parity

`train/test_parity_lattices.py`：**24 passed**。

| 组 | 覆盖 | 容差 / 判据 | 结果 |
|---|---|---|---|
| HRQ | alpha 0.0 / 0.1 | 连续 1e-4，idx 精确 | PASS |
| Sparse | alpha 0.0 / 0.1 + value bias + LFQ | 连续 1e-4，idx 精确 | PASS |
| LowRank | alpha 0.0 / 0.1 | 连续 1e-4，idx 精确 | PASS |
| Manifold | alpha 0.0 / 0.1 | 连续 1e-4，idx 精确 | PASS |
| Binding | alpha 0.0 / 0.1 + projection 方向 | 连续 1e-4，idx 精确 | PASS |
| Contrast | alpha 0.0 / 0.1 | 连续 1e-4，idx 精确 | PASS |
| Routing | alpha 0.0 / 0.1（hard 无噪声） | 连续 1e-4，idx 精确 | PASS |
| canonical fusion | 六格 + softmask + alpha | 1e-5 | PASS |
| engine fusion | 六格 + distance | 1e-5 | PASS |
| 判别性 | Poincaré≠Euclidean、相对 gap、binding 投影、NaN 不可假通过 | — | PASS |

C `test_runner` 251 PASS；`nm` 含 `lcm_infer_loop_v3`/`hrq_forward_v31`。Manifest per-tensor sha256 由 `train/c_load.py` 校验；`test_c_load.py::test_manifest_tensor_sha256_rejects_tampered_payload` 和 `test_manifest_missing_sha256_is_rejected` 覆盖篡改/缺 hash。

### 10.1 校准 checkpoint 的 C↔JAX 端到端复核

旧版 `checkpoints_calibrated`：单步组件 `max_abs = 8.0e-7`；用 JAX 六格输出 + C trace 的 active/fallback/self 调度可以复现每步 `z_next` 到 `5.7e-6`。

修正后的 `checkpoints_calibrated_fixed5`：

- 单步组件 parity：`max_abs = 9.5e-7`，融合权重一致；route/fusion/self/HRQ/Sparse 张量逐项 `maxdiff = 0`。
- 用 C trace 的调度复现 JAX 输出：多数步 `1e-6 – 2e-6`。但发现一个 **all-zero-route 边界**：C 的 trace 显示该步融合权重全零，JAX 侧按“无 routed lattice、无 self”的 identity 回退重建时与 C 的 `z_next` 仍差 `2.37e-2`（cos `0.999994`）。这说明 `lcm_get_trace` 暴露的 v1 weights 在该边界不足以还原 active-set/fallback 的完整语义，需要导出 v3 DAG 节点 `activated` 标志才能定论；这是整体执行语义的未关闭缺口，不能用一个随机输入的单步 parity 覆盖。
- C 引擎：`fixed5` 的 8 个中性 prompt × 8 token `use_safety=False` 生成 0 abort；3 个 12-token prompt 也 0 abort。`max_steps=4` 时有一个随机输入未收敛（返回非零），属于 graceful budget，不是硬限 abort，但需要单列。
- 语言：`fixed5` 与校准前一样只产出 100 步模型的低质量 token 串；符合 D-7，不能作为校准有效证据。
- `safe_unit` 的极小非零向量边界（JAX `norm+1e-8` vs C `unit_copy` 阈值除法）尚未补 parity 测试，是已知缺口。

## 11. 自观测 5% 开销

- 真实 `_self_observe_step` 20 次中位数：0.4465 ms
- 生产 `train_step` 20 次中位数（无 on_step，block_until_ready）：1073.67 ms
- callback interval 参考值：1110.79 ms
- overhead：**0.0416%**，<5%：PASS

20-run median wall-clock of the real _self_observe_step path divided by a 20-run median of the production train_step measured without the smoke on_step callback and with jax.block_until_ready per run. The callback interval is only a reference. The first JIT step is a warmup, not a measurement.

## 12. Save → load → 2 步一致性

| 检查 | 结果 |
|---|---|
| sampling_ok | PASS |
| params round-trip | 0.000e+00 |
| EMA round-trip | 0.000e+00 |
| bank round-trip | 0.000e+00 |
| 2 步 params | 0.000e+00 |
| 2 步 EMA | 0.000e+00 |
| old format_version=1 rejected | PASS |

Optimizer moments are deliberately not part of the v2 checkpoint (R10: params/EMA/buffers). Both sides use a fresh optimizer, so the comparison tests state round-trip, not AdamW continuity.

## 13. 覆盖表（设计 §6.3）

| item | smoke check | focused test |
|---|---|---|
| codebook_commitment_split | vq_codebook_commitment_split | train/test_vq_losses.py::test_ema_codebook_loss_skipped_but_commitment_trains_r |
| binding_unbind_roundtrip | — | train/test_binding_unbind.py::test_unbind_recovers_value_code |
| routing_hard_determinism | routing_hard_deterministic | train/test_vq_losses.py::test_routing_hard_is_deterministic |
| hrq_inactive_sparse_lfq_disabled | hrq_inactive_lfq_disabled | train/test_sparse_lfq.py::test_no_lfq_without_c_top |
| codebook_variance | codebook_variance_50pct | — |
| gvalue_danger_hash_unchanged | frozen_safety_hash | — |
| observability_step_records | observability_records | train/test_self_awareness_wiring.py:black-box recording tests |

这些项没有被算进 “14/14” 之类的总数；`checks` 表逐项列出 smoke 实际判定，binding unbind 由 focused test 覆盖，不在 smoke 里硬编码 PASS。

## 14. 已知局限

1. **100 步不证明语言能力**。
2. **gvalue/danger 是结构占位**，不是语义安全锚点；导出 danger pattern path 用 value_consistency=1.0 单独验证，真实 v3 loop 的威胁输入由 `CONFLICT_DANGER/THREAT_PATTERN_MATCH` 中断（同时检查 `z_cur`/`z_next`）。normal input 在 `use_safety=True` 时按严格三法则 abort，这是已确认的预期行为，不是语义安全通过。
3. **10% 码本利用率未通过**：streaming probe 全列为 HRQ 5.86%、Sparse 7.03%、LowRank 1.30%、Manifold 6.25%、Binding 0.20%、Contrast 1.33%、Self 1.56%（Route 83.33% PASS）。结构原因与诊断见 §5.1 和 `docs/d9-codebook-utilization-research-plan.md`。
4. **真实 mask 的归一化熵 0.5652 > 0.5**，熵门未通过；crafted peaked/uniform 只证明门本身可过可不过。
5. ~~规范文字未同步到 v3.1（D-1）~~ **已在 Task 0 + review 修正中解决**：`docs/a–e.md` 与 `docs/a–e_cn.md` 均按 R1–R22 同步并带 v3.1-conformance changelog。
6. **外部验证器是本地形态**（未做 HSM）；`causal_subject.py` 不扩展（accepted deviation）。
7. **优化器矩没有进 checkpoint**；2 步一致性用 fresh optimizer 测状态 round-trip。
8. **manifest 的 pickle params 没有 per-tensor sha**；C loader 只读 `.bin` 张量。
9. **feature-bank 替换是 synthetic policy probe**，不是 100 步训练事件。
10. **training is_safe 无安全 oracle**；Stage-2 只检查非有限 loss。

## 15. 复现命令

```bash
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
  train/test_smoke_conformance.py train/test_parity_lattices.py \
  train/test_c_load.py train/test_vq_losses.py \
  train/test_binding_unbind.py train/test_sparse_lfq.py -q
make -C infer clean && make -C infer test

JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python scripts/build_conformance_corpus.py
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python train/smoke_conformance.py \
  --steps 100 --batch 1 --seq 64 \
  --data /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat \
  --shape /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json \
  --spans /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy \
  --output checkpoints/conformance_smoke_final2 --passive-only
```

本次复跑未加 `--allow-fail`，因此两个已知 FAIL（利用率/真实熵门）使退出码为 1；`checks` 表是 verdict。只有在需要强制写出完整 artifact 时才使用 `--allow-fail`。

## 16. Checks 汇总

| check | 结果 |
|---|---|
| loss_all_finite | PASS |
| loss_decreased | PASS |
| grad_norms_gt_1e-6 | PASS |
| ema_statistics_changed | PASS |
| all_lattices_updated | PASS |
| codebook_utilization_10pct | **FAIL** |
| codebook_variance_50pct | PASS |
| vq_codebook_commitment_split | PASS |
| routing_hard_deterministic | PASS |
| hrq_inactive_lfq_disabled | PASS |
| route_beats_weight_decay | PASS |
| sparse_lfq | PASS |
| hrq_value_routing | PASS |
| entropy_gate | **FAIL** |
| feature_bank | PASS |
| safety_chain | PASS |
| safety_abort_any_conflict | PASS |
| frozen_safety_hash | PASS |
| observability_records | PASS |
| self_bias | PASS |
| overhead_under_5pct | PASS |
| persistence | PASS |

artifact：`checkpoints/conformance_smoke_final2/smoke_report.json`、`smoke_report.md`、`step_000100/`（149 MB，不提交）。
