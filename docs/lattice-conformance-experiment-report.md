# 晶格规范一致性实验报告（≤100 步 smoke）

日期：2026-11-21  
运行 HEAD：`2558c81c2bbe8f80d6ba9fbfda60e7fe93b736b5`（Task 15 的修复在同一提交/工作树中；完整提交见 `git log -1`）  
基线：`docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md` R1–R22  
偏差复审：`docs/lattice-spec-vs-impl-audit-v3.1.md`

> **结论边界**：100 步 smoke **不证明语言能力**。它只证明：训练路径发生、每格都有更新、数值有限、稀疏/路由/熵/安全/自我/持久化等规范路径能被触发。gvalue/danger 是**结构占位**，不是语义安全锚点。

## 1. 环境

| 项 | 值 |
|---|---|
| 平台 | Linux-7.3.4-zen2-1-zen-x86_64-with-glibc2.44 |
| Python | 3.14.10 |
| JAX | 0.10.0 |
| 设备 | CPU（`JAX_PLATFORMS=cpu`；GTX 1650 4GB 不参与本次 smoke） |
| 运行 HEAD | `2558c81c2bbe8f80d6ba9fbfda60e7fe93b736b5` |
| 总耗时 | 146.2s |
| 参考 C 库 | `infer/liblcm.so`（`make -C infer clean && make -C infer test`） |

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
| `entropy_threshold` | `2.0` |
| steps / batch / seq | 100 / 1 / 64 |
| lr | 0.0003 |
| joint | True |
| self_awareness | True |
| passive_only | True |
| overfit_one_batch | True |

`overfit_one_batch=True` 让 100 步复用同一个 `(inputs, targets)` 窗口，对应设计 §6.3 A 的“单 batch 过拟合”检查。它同时意味着路由看到的是**同一个 z**，所以码本利用率会偏低——见 §5 与 §13。

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

旧 `data/zhwiki_tokens.dat` 只有 `n_tokens` 的 identity，缺 tokenizer/separator/spans 摘要，`resolve_dataset_identity` 会拒绝。本次用 bpe30k tokenizer 重建了一份 43,036 token / 829 doc 的合规语料（`/home/DslsDZC/data/tmp/opencode/conformance_corpus/`）。

## 4. 100 步 loss

- 首 10 步均值：**19.1135**
- 末 10 步均值：**14.5163**
- 比值：**0.7595**（< 0.8：PASS）
- 全部有限：PASS

| 步区间 | 均值 | 最小 | 最大 |
|---|---|---|---|
| 0–9 | 19.1135 | 18.4077 | 19.4798 |
| 10–19 | 17.2747 | 16.5181 | 18.8734 |
| 20–29 | 16.2872 | 15.4438 | 18.5129 |
| 30–39 | 15.7686 | 14.0297 | 21.6168 |
| 40–49 | 14.6274 | 13.6658 | 16.0568 |
| 50–59 | 14.0590 | 13.2592 | 15.1336 |
| 60–69 | 14.2660 | 12.7249 | 20.6058 |
| 70–79 | 15.2951 | 12.2420 | 19.0277 |
| 80–89 | 15.4116 | 14.8502 | 15.9641 |
| 90–99 | 14.5163 | 14.2005 | 14.9179 |

完整逐 step 曲线在 `checkpoints/conformance_smoke/smoke_report.json` 的 `loss.per_step`。

## 5. 逐格更新、码本方差与利用率

| 晶格 | Δmax abs | ΔL2 | 更新>1e-6 | 码本方差比 | 利用率 |
|---|---|---|---|---|---|
| encoder | 0.0165 | 18.5424 | PASS | — | — |
| hrq | 0.0155 | 2.4453 | PASS | 4.9998 | 6.45% |
| sparse | 3.2568 | 34.4696 | PASS | 23.6566 | 2.93% |
| lowrank | 0.0091 | 0.1238 | PASS | 0.7804 | 4.56% |
| manifold | 3.2131 | 34.4226 | PASS | 24.2275 | 2.93% |
| binding | 1.2523 | 51.7457 | PASS | 65.4458 | 1.26% |
| contrast | 0.0203 | 12.1863 | PASS | 213.2288 | 1.92% |
| route | 0.0154 | 0.2808 | PASS | 1.0871 | 100.00% |
| fusion | 0.0155 | 0.1380 | PASS | — | — |
| self | 3.2451 | 37.6414 | PASS | 24.2663 | 25.00% |
| value_scalars | 0.0160 | 0.4516 | PASS | — | — |

码本方差比 ≥ 0.5 全部通过；没有晶格塌到 0。

### 5.1 10% 码本利用率检查（设计 §6.3 A）

| 晶格 | unique/M | 阈值 | 结果 |
|---|---|---|---|
| hrq | 6.45% | 10% | **FAIL** |
| sparse | 2.93% | 10% | **FAIL** |
| lowrank | 4.56% | 10% | **FAIL** |
| manifold | 2.93% | 10% | **FAIL** |
| binding | 1.26% | 10% | **FAIL** |
| contrast | 1.92% | 10% | **FAIL** |
| route | 100.00% | 10% | PASS |
| self | 25.00% | 10% | PASS |

总结果：**FAIL**。
原因不是“码本没更新”，而是 `batch=1 × 100 步 × 固定单 batch` 最多只有 100 次选择，且同一个 z 在 100 步内高度重复。512 行的 10% 需要 51 个**不同**行；本次 HRQ 约覆盖 33 行、Sparse/Manifold 约覆盖 15/15 行。把利用率当独立验收项时必须用streaming batch 或更多步数；本次如实记为未通过，而不是调低阈值。

## 6. EMA 统计

| 分支 | ΔN abs | N sum | 使用行 | Δm norm | changed |
|---|---|---|---|---|---|
| sparse | 63.3968 | 63.3968 | 15/512 | 645.6061 | PASS |
| manifold | 63.3968 | 63.3968 | 15/512 | 645.6061 | PASS |
| self | 63.3968 | 63.3968 | 16/64 | 577.1770 | PASS |
| binding | 570.5711 | 570.5711 | 58/4608 | 156.0567 | PASS |

EMA 总结果：PASS。sparse/manifold/self 的 N/m 都从 0 变为非零，binding 的三族逐层也更新；`last_used` 最大到 99。

## 7. C_route vs weight-decay baseline

- 实际 `C_route` Δmax：1.537e-02
- 只有 weight decay 的 baseline Δmax：3.323e-05
- 实际 / baseline：462.6×
- 结论：PASS（R19 的 route loss 确实进入了梯度）

## 8. Sparse / LFQ、HRQ 路由、熵门、feature bank、self

### 8.1 Sparse LFQ

- 推理非零比例：100.00% (8/8)
- near 输入非零：PASS；far 输入强制 zero_vec：PASS
- far: d_top=2.5905, d_min=2.8553, lambda_lfq=1.5
- 无 c_top 时 LFQ 禁用：PASS
- d_top shape=[2]，跨样本不同=PASS

### 8.2 HRQ 路由

- v=0 top-1 == pure Poincaré：PASS
- v≠0 top-1 仍在几何 top-k 内：PASS
- 价值偏置实际改变了 top-1：PASS (probe alpha=10.0, k=4)

### 8.3 熵门

- crafted peaked：H_norm=0.0397 → PASS
- crafted uniform：H_norm=1.0000 → 不通过=PASS
- **真实训练 mask**：H_norm=0.4854 → PASS（本次真实 mask 通过；只报事实，不做保证）
- JAX 训练收敛仍用 raw-nats 阈值 2.0；R9 的归一化 0.5 在 C v3 与本 probe 中实现，JAX 侧列为 deferred（见复审 F-6）。

### 8.4 Feature bank

- contrast 注入 5 个 dead code → 替换 5 个，最远点采样=PASS
- sparse 注入 5 个 dead code → 替换 5 个，cold start（N/m=0）=PASS

### 8.5 Self lattice

- `self_bias` ∈ [0.0036, 0.0500]，`alpha_self`=0.05，范围内=PASS
- 熟悉 vs 不熟悉输入均值不同：PASS (0.0500 vs 0.0036)
- 训练步内 self_bias 范围：[0.0056, 0.0500]

## 9. 安全触发

| 检查 | 结果 |
|---|---|
| 训练 100 步 `is_safe` 全 True | PASS |
| C `detect_any_conflict` 构造威胁输入阻断 | PASS |
| 冲突源（1=DANGER） | 1 |
| C 构造正常输入不阻断 | PASS |
| danger hash 在 init 前后稳定 | PASS |
| danger 阈值 | 0.05000000074505806（R7=0.05） |
| Python `generate` 在 ret=-2 时不产 token | PASS |
| 导出 gvalue pos≠neg | PASS |
| 导出 danger threat≠normal | PASS |
| gvalue integrity | PASS |
| gvalue margin=0.5 中性输入违规（严格三法则） | PASS |
| gvalue 负向输入 margin=0.5 违规 | PASS |

安全码本是**结构占位**：threat = 固定随机基，normal = 首个 batch 的 z；它们只验证“威胁更近 → abort”的链路。不得把本表读成语义安全。`CONFLICT_ABORT_TOKEN=-2` 的 v3 loop 真实路径由 `train/test_c_loop.py` `test_dangerous_vector_returns_empty` 覆盖并在最终 pytest 中重跑。

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
| Routing | alpha 0.0 / 0.1（hard，无噪声） | 连续 1e-4，idx 精确 | PASS |
| canonical fusion | 六格 + softmask + alpha | 1e-5 | PASS |
| engine fusion | 六格 + distance | 1e-5 | PASS |
| 判别性 | Poincaré≠Euclidean、相对 gap、binding 投影、NaN 不能假通过 | — | PASS |

C 单测：`make -C infer clean && make -C infer test` → **ALL TESTS PASSED**（234 个 `[PASS]` 检查）。`nm -D infer/liblcm.so` 含 `lcm_infer_loop_v3` 与 `hrq_forward_v31`。

## 11. 自观测 5% 开销

- 方法：20-run median wall-clock of the real _self_observe_step path divided by the median callback-to-callback training-step interval. The latter is device-synced wall-clock; train_cog's own step_time_ms is only dispatch time. The first JIT step is excluded because its callback precedes any steady-state interval.
- 真实 `_self_observe_step` 20 次中位数：0.462 ms
- 训练 step callback-to-callback 中位数：1088.2 ms
- `train_cog` 自己的 dispatch 中位数（仅参考）：4.594 ms
- overhead：**0.042%**，<5%：PASS

这里没有用 `train_cog` 的 `step_time_ms` 做分母：JAX 异步 dispatch 使那个数只有几毫秒，不是真实计算时间。分母用相邻 callback 的 `perf_counter` 差值（callback 进入时前一步已完成 device sync），因此是包含 host 观测、bank、保存的端到端 step wall-clock。

## 12. Save → load → 2 步一致性

| 检查 | 结果 |
|---|---|
| params round-trip max_abs_err | 0.000e+00 |
| EMA round-trip max_abs_err | 0.000e+00 |
| feature/contrast bank round-trip | 0.000e+00 |
| 2 步 params max_abs_err | 0.000e+00 |
| 2 步 EMA max_abs_err | 0.000e+00 |
| format_version=1 旧 checkpoint 被拒 | PASS |

Optimizer moments are deliberately not part of the v2 checkpoint (R10: params/EMA/buffers). Both sides use a fresh optimizer, so the comparison tests state round-trip, not AdamW continuity.

`old checkpoint` 的错误信息：

```
/home/DslsDZC/.local/share/opencode/worktree/c16568/spec-conformance/checkpoints/conformance_smoke/old_ckpt_3aocazbd/cog_params.pkl: old checkpoint format; re-export required (format_version=1, expected 2)
```

## 13. 已知局限

1. **100 步不证明语言能力**。loss 曲线只证明训练发生、数值有限、每格更新；不证明 perplexity、生成质量或语义。
2. **gvalue/danger 是结构占位**，不是语义安全锚点。它们验证触发链路，不验证“真实有害输入会被拦截”。
3. **10% 码本利用率未通过**（HRQ 6.4%、Sparse 2.9%、LowRank 4.6%、Manifold 2.9%、Binding 1.3%、Contrast 1.9%；Route/Self 通过）。`batch=1`、100 步、固定单 batch 的选择次数上限与重复 z 使 512 行码本无法覆盖 51 个不同行。这是 smoke 设计的结构限制，不应通过调低阈值来掩盖。
4. **真实训练 mask 的归一化熵 0.4854**（阈值 0.5，通过）；crafted peaked/uniform 证明门本身可过可不过。JAX 训练收敛仍用 raw-nats 2.0（复审 F-6 deferred）。
5. **规范文字未同步到 v3.1**（复审 D-1）。代码以设计文档 R1–R22 为准。
6. **C 一致性阈值仍 0.3**（R17 要求 0.7，复审 D-3）；`value_consistency` 的数值范围未重新校准。
7. **外部验证器是本地形态**（shared memory + 文件签名），不是 HSM；`causal_subject.py` 的功能与验证不扩展（accepted deviation）。
8. **优化器矩没有进 checkpoint**（R10 只要求 params/EMA/buffers）；2 步一致性用 fresh optimizer 在两侧各初始化一次，测的是状态 round-trip，不是 AdamW 连续性。

## 14. 复现命令

```bash
# 1) tokenizer + 合规语料（一次性）
mkdir -p /home/DslsDZC/data/lcm/tokenizers/bpe30k
cp /home/DslsDZC/LCM/data/tokenizer.json \
   /home/DslsDZC/data/lcm/tokenizers/bpe30k/tokenizer.json
# 用 train.data.build_corpus + resolve_tokenizer_spec('[EOS]', 30000) 从
# zhwiki_final.txt 前 2000 行构建 43,036 token / 829 doc 语料到
# /home/DslsDZC/data/tmp/opencode/conformance_corpus/

# 2) 100 步 smoke
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python train/smoke_conformance.py \
  --steps 100 --batch 1 --seq 64 \
  --data /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance.dat \
  --shape /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_shape.json \
  --spans /home/DslsDZC/data/tmp/opencode/conformance_corpus/conformance_docs.npy \
  --output checkpoints/conformance_smoke --passive-only

# 3) 最终验证
JAX_PLATFORMS=cpu /home/DslsDZC/LCM/be/bin/python -m pytest \
  train/test_smoke_conformance.py train/test_parity_lattices.py -q
make -C infer clean && make -C infer test
nm -D infer/liblcm.so | grep -E 'lcm_infer_loop_v3|hrq_forward_v31'
```

执行结果：

- `pytest train/test_smoke_conformance.py train/test_parity_lattices.py -q`
  → **25 passed in 121.93s**。
- `pytest train/test_c_loop.py -q` → **5 passed in 2.20s**（真实 v3 loop 的
  danger abort `-2` / 无 token）。
- `make -C infer clean && make -C infer test` → **ALL TESTS PASSED**，
  234 个 `[PASS]`，退出码 0。
- `nm` → `lcm_infer_loop_v3`、`lcm_infer_loop_v3_flat`、`hrq_forward_v31` 均存在。

## 15. Checks 汇总

| check | 结果 |
|---|---|
| loss_all_finite | PASS |
| loss_decreased | PASS |
| all_lattices_updated | PASS |
| ema_changed | PASS |
| codebook_utilization_10pct | **FAIL** |
| route_beats_weight_decay | PASS |
| sparse_lfq | PASS |
| hrq_value_routing | PASS |
| entropy_gate | PASS |
| feature_bank | PASS |
| safety_chain | PASS |
| self_bias | PASS |
| overhead_under_5pct | PASS |
| persistence | PASS |

artifact：`checkpoints/conformance_smoke/smoke_report.json`、`checkpoints/conformance_smoke/smoke_report.md`、`checkpoints/conformance_smoke/step_000100/`（149 MB，不提交）。
