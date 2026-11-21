# D-9 码本利用率与真实熵门：研究计划

日期：2026-11-21
状态：**open / research**。本计划不改变验收阈值，也不把失败写成通过。
相关证据：`docs/lattice-conformance-experiment-report.md`、`infer/loops_v3.c`、`train/smoke_conformance.py`。

## 1. 问题定义

100 步 conformance smoke 有两项已披露的 FAIL：

| 检查 | 当前值 | 阈值 | 结论 |
|---|---|---|---|
| `codebook_utilization_10pct`（streaming，100 个不同 batch） | HRQ 5.86%、Sparse 7.03%、LowRank 1.30%、Manifold 6.25%、Binding 0.20%、Contrast 1.33%、Self 1.56%、Route 83.33% | 每个码本 ≥10% | FAIL |
| `entropy_gate`（真实 routing mask） | `H_norm=0.5652` | ≤0.5 | FAIL |

两项都按原阈值保留。研究目标是解释并修复根因，而不是调低 10% / 0.5。

## 2. 已有诊断（只读探针）

新增 `train/research_d9_probe.py`，对固定 checkpoint、100 个不同语料窗口做推理，输出每个码本层的：

- 唯一选中行数 / M
- 选择分布的归一化熵
- top-1 行占比
- `constant_across_batches`：是否每个 batch 都选中完全相同的行集合

示例（100 个 batch，`conformance_smoke_final2/step_000100`）。探针从磁盘重新加载 checkpoint，而 smoke 使用训练结束时的内存参数，因此计数可能相差几行；结论不变：

| 码本 | 唯一行 / M | top-1 占比 | 是否输入无关 |
|---|---|---|---|
| Binding key/val/bind（3 层） | 1 / 512 每层 | 100% | **是** |
| Self | 1 / 64 | 100% | **是** |
| LowRank layer 0/1/2 | 2–3 / 256 每层 | 71–97% | 否 |
| Contrast a/b | 2–18 / 512 | 42–79% | 否 |
| HRQ top | 29 / 512 | 24% | 否 |
| Sparse | 29 / 512（去掉 zero_vec 后） | 21% | 否 |
| Manifold | 34 / 512 | 18% | 否 |
| Route | 5 / 6 | 63% | 否 |

几何信号：

- HRQ top 行范数 `min 0.0221 / mean 0.0413 / max 0.3255`，码本尺度跨一个数量级。
- Route 行两两 cosine `min -0.123 / mean 0.017 / max 0.113`，接近正交但只有 6 行；83.33% 只是 5/6，不构成多样性证据。
- LowRank layer-0 的 `U` 只有 rank 2，effective rank 1.96，行间几何高度集中。

**关键结论：Binding 和 Self 不是“路由探索不足”，而是前向选择对输入不敏感。** 每层、每个 batch 都选同一行，说明增加训练步数或 batch size 不会自动解决它。LowRank 也接近常数。HRQ/Sparse/Manifold 是分布覆盖不足，和 Binding/Self 不是同一类问题。

## 3. 假设与优先级

### H1（P0）Binding / Self 选择退化

`binding_forward` 的 key/value 选择可能被 phase-only `normalize_fft`、固定投影或 codebook 行几何支配，导致 query 变化不改变 argmax。Self 的 cosine familiarity 也可能总落在同一 frozen core / EMA 行。先做只读诊断（E1），再决定是否修 forward。验收：Binding 三个 family 与 Self 的唯一行数都 ≥10% M。

### H2（P0）LowRank 码本几何/初始化塌缩

LowRank `U` 的 rank 很低且行间距离小，256 行实际只有 2 个吸引子。可能是初始化尺度、commitment 梯度或 EMA 把行拉在一起。需要检查每层 `U` 的行范数、pairwise distance、assignment entropy，并做小规模初始化对照（E2）。

### H3（P1）HRQ/Sparse/Manifold 的输入覆盖不足

100 步、B=1、单一语料窗口的 overfit 让 encoder 的 z 分布很窄。Ruling 51 的 streaming probe 已经用 100 个不同窗口，仍低，说明不只是“单 batch”。需要比较 B=1 与 B=4、100 步与 300 步，判断是步数不足还是码本几何问题（E3/E5）。

### H4（P1）真实 mask 熵门

路由 soft mask 在真实 batch 上 `H_norm=0.5637`，接近但高于 0.5。可能是 route 只有 6 行、softmax 温度/距离尺度不合适，或 6 个 lattice 的激活数/权重分布偏斜。需要记录每次真实 mask 的 `n_active`、最大权重、最小权重，区分“6 个都活跃但略不均”与“少数 lattice 支配”。不通过调低 0.5 解决。

### H5（P2）Contrast 的写入与选择

Contrast 的 feature bank / usage clock 可能让选择集中；Self 已归入 H1。先确认选择索引来自真实前向，而不是统计口径问题（P1 修复后 value consistency 已改为实际 forward idx，可作为先例）。

## 4. 实验矩阵

| # | 实验 | 改动 | 成本 | 通过标准 |
|---|---|---|---|---|
| E0 | 只读诊断 | `train/research_d9_probe.py`，不改参数 | 低（20–100 batch 推理） | 输出每层 histogram/entropy/constant 标记 |
| E1 | Binding/Self 前向追踪 | 记录 key/value/bind 与 self 的 query、FFT 相位、argmax 距离 | 低 | 找到选择输入无关的具体算子 |
| E2 | LowRank 初始化对照 | 小规模 20–50 步，比较默认/正交/单位半径初始化 | 中 | 至少一个初始化让有效吸引子 >10%，且不破坏 parity |
| E3 | Batch size 对照 | B=1 vs B=4，100 步，同数据/种子 | 中 | 利用率提升但 loss/EMA 不崩 |
| E4 | 路由熵诊断 | 记录真实 mask 的 n_active/权重分布；只调温度/距离尺度，不调阈值 | 低–中 | `H_norm ≤ 0.5` 且路由不塌成单格 |
| E5 | 长步数诊断 | 300/1000 步（研究，不作为 conformance） | 高 | 判断 10% 是否只是 100 步不可达 |
| E6 | 多样性辅助目标 | 仅当 E1–E5 证明是优化问题时，加入小权重 usage-entropy/commitment 项 | 高 | 10% + 熵门同时过，且不降语言 loss 边界 |

优先顺序：E0（已完成）→ E1 → E2 → E3 → E4 → E5/E6。E1/E2 是 P0，因为 Binding/Self/LowRank 是结构性塌缩，不是数据量问题。

## 5. 验收与报告规则

1. **不降低阈值**：10% 和 `H_norm=0.5` 不变。
2. **分母写清楚**：每个 lattice / 每个 layer 单独报告；Route 只有 6 行，100% 不构成“多样性证据”。
3. **区分训练选择与推理选择**：Ruling 51 的 streaming probe 是 verdict；固定单 batch 只作次要诊断。
4. **不把初始化覆盖当成训练利用率**：初始化时每行都“存在”不等于被选中。
5. **不把结构触发当语义安全**：D-9 与安全锚点无关。
6. 最终 100 步 conformance 必须同时满足：每码本 ≥10%、真实 mask `H_norm ≤ 0.5`、loss/EMA/数值检查不退化。

## 6. 当前决定

- 用户已确认：**保留严格三法则**；normal 输入 abort 是预期，语义锚点训练是后续任务。
- 用户已确认：**开 D-9 研究任务**，不降低阈值。
- 本计划只建立诊断工具和实验顺序；没有修改训练行为，也没有声称 D-9 通过。

## 7. 复现命令

```bash
JAX_PLATFORMS=cpu python -m train.research_d9_probe \
  --checkpoint checkpoints/conformance_smoke_final2/step_000100 \
  --data /path/conformance.dat \
  --shape /path/conformance_shape.json \
  --spans /path/conformance_docs.npy \
  --batches 100 \
  --output /tmp/d9_probe.json
```
