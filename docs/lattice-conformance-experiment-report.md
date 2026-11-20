# 晶格规范一致性实验报告（≤100 步 smoke）

日期：2026-11-21
运行 HEAD：`88feec345d7d781a401355fa34a0a43abfb7a010`（Task 15 修复前的运行 HEAD；修复与报告在同一后续提交中）
基线：`docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` R1–R22
偏差复审：`docs/lattice-spec-vs-impl-audit-v3.1.md`

> **结论边界**：100 步 smoke **不证明语言能力**。它只证明：训练路径发生、每格都有真实梯度或 EMA 更新、数值有限、稀疏/路由/熵/安全/自我/持久化等规范路径能被触发。gvalue/danger 是**结构占位**，不是语义安全锚点。

> 本次运行 **19/21 checks PASS**；两项 FAIL 是 `codebook_utilization_10pct` 与 `entropy_gate`（真实 mask），详见 §5/§8。命令使用了 `--allow-fail` 以保留完整 artifact，退出码本身不表示全部通过。

## 1. 环境

| 项 | 值 |
|---|---|
| 平台 | Linux-7.3.4-zen2-1-zen-x86_64-with-glibc2.44 |
| Python | 3.14.10 |
| JAX | 0.10.0 |
| 设备 | CPU（`JAX_PLATFORMS=cpu`；GTX 1650 4GB 不参与） |
| 运行 HEAD | `88feec345d7d781a401355fa34a0a43abfb7a010` |
| 总耗时 | 138.8s |
| C 参考 | `make -C infer clean && make -C infer test`（235 PASS） |

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

- 首 10 步均值：**19.0291**
- 末 10 步均值：**12.0008**
- 比值：**0.6307**（< 0.8：PASS）
- 全部有限：PASS

| 步区间 | 均值 | 最小 | 最大 |
|---|---|---|---|
| 0–9 | 19.0291 | 18.2452 | 19.6813 |
| 10–19 | 18.1085 | 16.6639 | 21.5405 |
| 20–29 | 16.8085 | 15.8604 | 20.4704 |
| 30–39 | 16.0041 | 15.1494 | 17.0854 |
| 40–49 | 14.8288 | 13.8446 | 18.5375 |
| 50–59 | 13.9908 | 13.4843 | 14.7696 |
| 60–69 | 17.4041 | 13.4157 | 23.5527 |
| 70–79 | 13.4862 | 12.8819 | 14.2139 |
| 80–89 | 12.4041 | 11.5306 | 12.8711 |
| 90–99 | 12.0008 | 11.1231 | 14.4545 |

完整逐 step 曲线在 `checkpoints/conformance_smoke/smoke_report.json` 的 `loss.per_step`。

## 5. 逐格更新、真实梯度、方差与利用率

参数 delta 单独不能区分真实梯度和 AdamW weight decay；本节以 `make_train_step(..., return_grads=True)` 的一次真实 `jax.grad` 为主判据，delta 只作 secondary。

| 晶格 | grad norm | grad>1e-6 | Δmax abs (secondary) | var ratio | utilization |
|---|---|---|---|---|---|
| encoder | 1090.4057 | PASS | 0.0158 | — | — |
| hrq | 679.2744 | PASS | 0.0160 | 4.7166 | 6.25% |
| sparse | — | — | 3.0818 | 35.5111 | 3.91% |
| lowrank | 0.4115 | PASS | 0.0081 | 0.8552 | 5.47% |
| manifold | 0.0023 | PASS | 3.1703 | 36.0782 | 3.91% |
| binding | 0.4066 | PASS | 1.3391 | 90.8786 | 1.56% |
| contrast | 1.4179 | PASS | 0.0206 | 241.3321 | 3.45% |
| route | 39.1816 | PASS | 0.0155 | 1.0816 | 83.33% |
| fusion | 22.7596 | PASS | 0.0164 | — | — |
| self | — | — | 3.0693 | 29.8935 | 31.25% |
| value_scalars | 55.2943 | PASS | 0.0164 | — | — |

sparse/self 无 trainable 子树，用 N/m 判据；manifold/binding 同时有 `T`/`A_k`/`A_v` 的梯度和码本 EMA。

### 5.1 10% 码本利用率（设计 §6.3 A）

| 晶格 | unique/M | 阈值 | 结果 |
|---|---|---|---|
| hrq | 6.25% | 10% | **FAIL** |
| sparse | 3.91% | 10% | **FAIL** |
| lowrank | 5.47% | 10% | **FAIL** |
| manifold | 3.91% | 10% | **FAIL** |
| binding | 1.56% | 10% | **FAIL** |
| contrast | 3.45% | 10% | **FAIL** |
| route | 83.33% | 10% | PASS |
| self | 31.25% | 10% | PASS |

总结果：**FAIL**。HRQ 只统计 top codebook（top codebook only; fine-layer selections are not tracked by this smoke）。原因不是“码本没更新”，而是 `batch=1 × 100 步 × 固定单 batch` 最多 100 次选择、同一个 z 高度重复；512 行的 10% 需要 51 个不同行。正式利用率验收必须用 streaming batch 或更多步数，不能调低阈值。

### 5.2 码本方差（final/initial ≥ 0.5）

| 晶格 | ratio |
|---|---|
| hrq | 4.7166 |
| sparse | 35.5111 |
| lowrank | 0.8552 |
| manifold | 36.0782 |
| binding | 90.8786 |
| contrast | 241.3321 |
| route | 1.0816 |
| self | 29.8935 |

Result: PASS

## 6. EMA、VQ split、routing hard、observability

### 6.1 EMA

| 分支 | ΔN abs | N sum | 使用行 | Δm norm | changed |
|---|---|---|---|---|---|
| sparse | 63.3968 | 63.3968 | 20/512 | 658.1293 | PASS |
| manifold | 63.3968 | 63.3968 | 20/512 | 658.1293 | PASS |
| self | 63.3968 | 63.3968 | 20/64 | 658.1293 | PASS |
| binding | 570.5711 | 570.5711 | 72/4608 | 252.8772 | PASS |

EMA 总结果：PASS

### 6.2 Codebook / commitment split (R18)

- EMA codebook loss zero：PASS
- commitment loss positive：PASS
- `hrq_codebook`：0.2543
- `hrq_commitment`：0.2543
- `sparse_codebook`：0.0000
- `sparse_commitment`：0.0001
- `lowrank_codebook`：0.7617
- `lowrank_commitment`：0.7617
- `manifold_codebook`：0.0000
- `manifold_commitment`：0.1482
- `binding_codebook`：0.0000
- `binding_commitment`：0.0217
- `contrast_codebook`：1.4368
- `contrast_commitment`：1.4368
- `route_codebook`：0.2493
- `route_commitment`：0.2493
- pass：PASS

### 6.3 Routing hard (R19)

- deterministic：PASS
- one-hot：PASS
- pass：PASS

### 6.4 Observability (R16)

- raw records：100/100
- long-term records：207
- errors：{}
- pass：PASS

## 7. C_route vs weight-decay baseline

- `C_route` Δmax：9.974e-03
- weight-decay-only `C_route` Δmax：1.066e-05
- 实际 / baseline：935.5×
- full route tree Δmax（含 `W_route`，secondary）：1.546e-02
- 结论：PASS（C_route only; W_route is reported separately）

## 8. Sparse / LFQ、HRQ 路由、熵门、feature bank、self

### 8.1 Sparse LFQ

- 推理非零比例：100.00% (8/8)
- near 非零：PASS；far 强制 zero_vec：PASS
- far d_top=1.9341, d_min=2.8724, lambda_lfq=1.5
- 无 c_top 时 LFQ 禁用：PASS
- d_top shape=[2]，跨样本不同=PASS

### 8.2 HRQ 路由

- v=0 top-1 == pure Poincaré：PASS
- v≠0 仍在几何 top-k：PASS
- 价值偏置改变 top-1：PASS (alpha=10.0, k=4)

### 8.3 熵门（真实 mask 是 verdict 的一部分）

- crafted peaked：0.0396 → PASS
- crafted uniform：1.0000 → 不通过=PASS
- **真实训练 mask**：H_norm=0.5624 → **FAIL**（阈值 0.5）
- 总 pass：**FAIL**；JAX 训练门按归一化 0.5 解释。

### 8.4 Feature bank（synthetic policy probe）

- synthetic=True：constructed 8-d FeatureBank and 5 injected dead rows; this is the replacement policy test, not a 100-step training event
- contrast 5 dead → 5 replaced，farthest-point=PASS
- sparse 5 dead → 5 replaced，cold-start=PASS

### 8.5 Self lattice

- self_bias [0.0342, 0.0500]，alpha=0.05，in_range=PASS
- 熟悉 vs 不熟悉：0.0500 vs 0.0342，varies=PASS
- 训练步内范围：[0.0058, 0.0500]

## 9. 安全：真实 checkpoint → 真实 C loop

| 检查 | 结果 |
|---|---|
| 训练 is_safe 全 True（无安全 oracle） | PASS |
| 真实 v3 loop 威胁输入 ret | -2 (CONFLICT_DANGER) |
| `generate` 威胁输入 ret / tokens | -2 / 0 |
| alert callback | PASS (CONFLICT_DANGER) |
| trace file written | PASS |
| C gvalue margin=0.5：pos safe / neutral abort / neg abort | 1.0000 / 1.0000 / 1.0000 (n=12) |
| gvalue hash unchanged | PASS |
| danger hash unchanged after step 0 | PASS |
| 导出 gvalue pos≠neg / danger threat≠normal | PASS / PASS |

真实 v3 loop 的 normal input 也返回 `-2`（source=CONFLICT_DANGER）。这是结构占位 danger 码本的过阻断，**不是**语义安全结论；gvalue 的 strict 三法则 probe 单独隔离在上一行。

gvalue/danger are structural placeholders built from fixed anchors and the first batch's z. They exercise the trigger chain; they do not establish semantic safety.

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

C `test_runner` 235 PASS；`nm` 含 `lcm_infer_loop_v3`/`hrq_forward_v31`。Manifest per-tensor sha256 由 `train/c_load.py` 校验；`test_c_load.py::test_manifest_tensor_sha256_rejects_tampered_payload` 和 `test_manifest_missing_sha256_is_rejected` 覆盖篡改/缺 hash。

## 11. 自观测 5% 开销

- 真实 `_self_observe_step` 20 次中位数：0.430 ms
- 生产 `train_step` 20 次中位数（无 on_step，block_until_ready）：1100.9 ms
- callback interval 参考值：1083.6 ms
- overhead：**0.039%**，<5%：PASS

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
2. **gvalue/danger 是结构占位**，不是语义安全锚点；真实 v3 loop 的 normal input 也被 danger 过阻断，需正式重训/校准 normal 码本。
3. **10% 码本利用率未通过**（HRQ 6.25%、Sparse 3.91%、Manifold 3.91%、Binding 1.56%）。结构原因见 §5.1。
4. **真实 mask 的归一化熵 0.5624 > 0.5**，熵门未通过；crafted peaked/uniform 只证明门本身可过可不过。
5. **规范文字未同步到 v3.1**（D-1）。
6. **外部验证器是本地形态**（未做 HSM）；`causal_subject.py` 不扩展（accepted deviation）。
7. **优化器矩没有进 checkpoint**；2 步一致性用 fresh optimizer 测状态 round-trip。
8. **manifest 的 pickle params 没有 per-tensor sha**；C loader 只读 `.bin` 张量。
9. **feature-bank 替换是 synthetic policy probe**，不是 100 步训练事件。
10. **training is_safe 无安全 oracle**；Stage-2 只检查非有限 loss。

## 15. 复现命令

```bash
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
  train/test_smoke_conformance.py train/test_parity_lattices.py \
  train/test_c_load.py -q
make -C infer clean && make -C infer test

JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python scripts/build_conformance_corpus.py
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python train/smoke_conformance.py \
  --steps 100 --batch 1 --seq 64 \
  --data /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat \
  --shape /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json \
  --spans /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy \
  --output checkpoints/conformance_smoke --passive-only --allow-fail
```

`--allow-fail` 让命令在已知 FAIL（利用率/真实熵门）时仍写出完整 artifact；`checks` 表是 verdict，不是退出码。

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
| frozen_safety_hash | PASS |
| observability_records | PASS |
| self_bias | PASS |
| overhead_under_5pct | PASS |
| persistence | PASS |

artifact：`checkpoints/conformance_smoke/smoke_report.json`、`smoke_report.md`、`step_000100/`（149 MB，不提交）。
