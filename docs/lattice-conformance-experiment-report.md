# 晶格规范一致性实验报告（≤100 步 smoke）

日期：2026-11-21
smoke 运行 HEAD：`c32eea9`；最终代码提交：`ef7136b`（文档在后续提交）
基线：`docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` R1–R22
偏差复审：`docs/lattice-spec-vs-impl-audit-v3.1.md`

> **结论边界**：100 步 smoke **不证明语言能力**。它只证明：训练路径发生、每格都有真实梯度或 EMA 更新、数值有限、稀疏/路由/熵/安全/自我/持久化等规范路径能被触发。gvalue/danger 是**结构占位**，不是语义安全锚点。

> 本次运行 **20/22 checks PASS**；两项 FAIL 是 `codebook_utilization_10pct` 与 `entropy_gate`（真实 mask），详见 §5/§8。命令使用了 `--allow-fail` 以保留完整 artifact，退出码本身不表示全部通过。

## 1. 环境

| 项 | 值 |
|---|---|
| 平台 | Linux-7.3.4-zen2-1-zen-x86_64-with-glibc2.44 |
| Python | 3.14.10 |
| JAX | 0.10.0 |
| 设备 | CPU（`JAX_PLATFORMS=cpu`；GTX 1650 4GB 不参与） |
| smoke 运行 HEAD | `c32eea9`（最终代码提交 `ef7136b`） |
| 总耗时 | 142.7s |
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

- 首 10 步均值：**18.8378**
- 末 10 步均值：**12.0619**
- 比值：**0.6403**（< 0.8：PASS）
- 全部有限：PASS

| 步区间 | 均值 | 最小 | 最大 |
|---|---|---|---|
| 0–9 | 18.8378 | 17.5791 | 19.5480 |
| 10–19 | 17.9662 | 16.8932 | 20.6202 |
| 20–29 | 16.8646 | 15.1954 | 19.6887 |
| 30–39 | 15.8370 | 13.5775 | 23.3209 |
| 40–49 | 13.9577 | 12.8126 | 19.4712 |
| 50–59 | 14.5387 | 12.8902 | 24.7764 |
| 60–69 | 18.0618 | 12.7357 | 24.7903 |
| 70–79 | 16.5140 | 12.3887 | 24.8466 |
| 80–89 | 13.3967 | 11.7650 | 22.0993 |
| 90–99 | 12.0619 | 11.5746 | 13.0399 |

完整逐 step 曲线在 `checkpoints/conformance_smoke/smoke_report.json` 的 `loss.per_step`。

## 5. 逐格更新、真实梯度、方差与利用率

参数 delta 单独不能区分真实梯度和 AdamW weight decay；本节以 `make_train_step(..., return_grads=True)` 的一次真实 `jax.grad` 为主判据，delta 只作 secondary。

| 晶格 | grad norm | grad>1e-6 | Δmax abs (secondary) | var ratio | utilization |
|---|---|---|---|---|---|
| encoder | 1090.6358 | PASS | 0.0164 | — | — |
| hrq | 677.2792 | PASS | 0.0165 | 4.9977 | 5.08% |
| sparse | — | — | 3.4040 | 32.6588 | 3.12% |
| lowrank | 2.3269 | PASS | 0.0088 | 1.0741 | 5.08% |
| manifold | 0.0023 | PASS | 3.3699 | 33.2298 | 3.12% |
| binding | 0.8542 | PASS | 1.1924 | 113.5821 | 1.61% |
| contrast | 1.3658 | PASS | 0.0201 | 256.9729 | 3.55% |
| route | 39.1918 | PASS | 0.0150 | 1.0941 | 100.00% |
| fusion | 22.7596 | PASS | 0.0164 | — | — |
| self | — | — | 3.3616 | 27.5803 | 25.00% |
| value_scalars | 55.2141 | PASS | 0.0165 | — | — |

sparse/self 无 trainable 子树，用 N/m 判据；manifold/binding 同时有 `T`/`A_k`/`A_v` 的梯度和码本 EMA。

### 5.1 10% 码本利用率（设计 §6.3 A / Ruling 51）

| 晶格 | training probe (fixed batch) | streaming probe (100 batches) | 阈值 | verdict |
|---|---|---|---|---|
| hrq | 5.08% | 7.03% | 10% | **FAIL** |
| sparse | 3.12% | 6.05% | 10% | **FAIL** |
| lowrank | 5.08% | 1.17% | 10% | **FAIL** |
| manifold | 3.12% | 6.84% | 10% | **FAIL** |
| binding | 1.61% | 0.20% | 10% | **FAIL** |
| contrast | 3.55% | 0.68% | 10% | **FAIL** |
| route | 100.00% | 83.33% | 10% | PASS |
| self | 25.00% | 1.56% | 10% | **FAIL** |

Streaming verdict：**FAIL**；batches=100，B=1。
Training probe（固定单 batch，secondary）：**FAIL**。
Ruling 51 的 streaming probe 仍低于 10%：这不是“码本没更新”，而是路由/码本在 100 步内高度集中。正式验收需要更多步数、B≥4 或路由探索；本次如实记为unresolved（D-9），不调低阈值。

### 5.2 码本方差（final/initial ≥ 0.5）

| 晶格 | ratio |
|---|---|
| hrq | 4.9977 |
| sparse | 32.6588 |
| lowrank | 1.0741 |
| manifold | 33.2298 |
| binding | 113.5821 |
| contrast | 256.9729 |
| route | 1.0941 |
| self | 27.5803 |

Result: PASS

## 6. EMA、VQ split、routing hard、observability

### 6.1 EMA

| 分支 | ΔN abs | N sum | 使用行 | Δm norm | changed |
|---|---|---|---|---|---|
| sparse | 63.3968 | 63.3968 | 16/512 | 675.2273 | PASS |
| manifold | 63.3968 | 63.3968 | 16/512 | 675.2273 | PASS |
| self | 63.3968 | 63.3968 | 16/64 | 675.2273 | PASS |
| binding | 570.5711 | 570.5711 | 74/4608 | 240.6789 | PASS |

EMA 总结果：PASS

### 6.2 Codebook / commitment split (R18)

- EMA codebook loss zero：PASS
- commitment loss positive：PASS
- `hrq_codebook`：0.2553
- `hrq_commitment`：0.2553
- `sparse_codebook`：0.0000
- `sparse_commitment`：0.0000
- `lowrank_codebook`：0.7621
- `lowrank_commitment`：0.7621
- `manifold_codebook`：0.0000
- `manifold_commitment`：0.1483
- `binding_codebook`：0.0000
- `binding_commitment`：0.0130
- `contrast_codebook`：1.3749
- `contrast_commitment`：1.3749
- `route_codebook`：0.2491
- `route_commitment`：0.2491
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

- `C_route` Δmax：1.252e-02
- weight-decay-only `C_route` Δmax：1.066e-05
- 实际 / baseline：1174.5×
- full route tree Δmax（含 `W_route`，secondary）：1.502e-02
- 结论：PASS（C_route only; W_route is reported separately）

## 8. Sparse / LFQ、HRQ 路由、熵门、feature bank、self

### 8.1 Sparse LFQ

- 推理非零比例：100.00% (8/8)
- near 非零：PASS；far 强制 zero_vec：PASS
- far d_top=2.3013, d_min=2.8511, lambda_lfq=1.5
- 无 c_top 时 LFQ 禁用：PASS
- d_top shape=[2]，跨样本不同=PASS

### 8.2 HRQ 路由

- v=0 top-1 == pure Poincaré：PASS
- v≠0 仍在几何 top-k：PASS
- 价值偏置改变 top-1：PASS (alpha=10.0, k=4)

### 8.3 熵门（真实 mask 是 verdict 的一部分）

- crafted peaked：0.0396 → PASS
- crafted uniform：1.0000 → 不通过=PASS
- **真实训练 mask**：H_norm=0.5676 → **FAIL**（阈值 0.5）
- 总 pass：**FAIL**；JAX 训练门按归一化 0.5 解释。

### 8.4 Feature bank（synthetic policy probe）

- synthetic=True：constructed 8-d FeatureBank and 5 injected dead rows; this is the replacement policy test, not a 100-step training event
- contrast 5 dead → 5 replaced，farthest-point=PASS
- sparse 5 dead → 5 replaced，cold-start=PASS

### 8.5 Self lattice

- self_bias [0.0036, 0.0500]，alpha=0.05，in_range=PASS
- 熟悉 vs 不熟悉：0.0500 vs 0.0036，varies=PASS
- 训练步内范围：[0.0083, 0.0500]

## 9. 安全：pattern probe 与 v3 hard-stop 分开

| 检查 | 结果 |
|---|---|
| 训练 is_safe 全 True（无安全 oracle） | PASS |
| 导出 danger pattern：threat blocked / type | PASS / 1 (1=PATTERN_MATCH) |
| 导出 danger pattern：normal not blocked / type | PASS / 0 |
| 导出 danger pattern pass | PASS |
| 真实 v3 loop 威胁输入 ret | -2 (CONFLICT_DANGER/THREAT_DECEPTION) |
| `generate` 威胁输入 ret / tokens | -2 / 0 |
| alert callback | PASS (CONFLICT_DANGER/THREAT_DECEPTION) |
| trace file written | PASS |
| v3 abort any conflict | PASS |
| C gvalue margin=0.5：pos safe / neutral abort / neg abort | 1.0000 / 1.0000 / 1.0000 (n=12) |
| gvalue hash unchanged | PASS |
| danger hash unchanged after step 0 | PASS |
| 导出 gvalue pos≠neg / danger threat≠normal | PASS / PASS |

Ruling 50：v3 loop 的 abort 实际是 `THREAT_DECEPTION`（source 仍是 CONFLICT_DANGER；consistency 分支先触发，value_consistency < 0.7），所以 `safety_chain` 只由导出 danger pattern probe 的 `THREAT_PATTERN_MATCH` 决定；`safety_abort_any_conflict` 单独记录 hard-stop。normal input 的 abort 也是 consistency/deception，不是 danger pattern 过阻断。

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

- 真实 `_self_observe_step` 20 次中位数：0.440 ms
- 生产 `train_step` 20 次中位数（无 on_step，block_until_ready）：1101.5 ms
- callback interval 参考值：1109.6 ms
- overhead：**0.040%**，<5%：PASS

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
2. **gvalue/danger 是结构占位**，不是语义安全锚点；导出 danger pattern path 用 value_consistency=1.0 单独验证，真实 v3 loop 的 abort 是 consistency/deception。normal input 的 abort 不是 danger pattern 结果。
3. **10% 码本利用率未通过**：streaming probe 全列为 HRQ 7.03%、Sparse 6.05%、LowRank 1.17%、Manifold 6.84%、Binding 0.20%、Contrast 0.68%、Self 1.56%（Route 83.33% PASS）。结构原因见 §5.1。
4. **真实 mask 的归一化熵 0.5676 > 0.5**，熵门未通过；crafted peaked/uniform 只证明门本身可过可不过。
5. ~~规范文字未同步到 v3.1（D-1）~~ **已在本次 Task 0 解决**：`docs/a–e.md` 与 `docs/a–e_cn.md` 均按 R1–R22 同步并带 v3.1-conformance changelog。
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
| safety_abort_any_conflict | PASS |
| frozen_safety_hash | PASS |
| observability_records | PASS |
| self_bias | PASS |
| overhead_under_5pct | PASS |
| persistence | PASS |

artifact：`checkpoints/conformance_smoke/smoke_report.json`、`smoke_report.md`、`step_000100/`（149 MB，不提交）。
