# 晶格规范 vs 实现 审计（对 `docs/b.md` v3.0）

日期：2026-10-01
基准：`docs/b.md`（Lattice Cognitive Model Technical Specification v3.0）
方法：逐格三列对照 —— 规范怎么说 / JAX 实际做什么 / C 部署实际做什么

## 为什么做这个

此前的工作方向是"让 C 对齐 JAX"，做 JAX↔C 单步数值 parity（已从 99.59% 收到
0.02%）。但那个方向有一个隐含前提：**JAX 是正确参考**。对表之后这个前提不成立
——规范里有几处是 JAX 自己也没实现的。

所以正确的参考是 `b.md`，不是当前的 JAX 实现。C↔JAX parity 只能是**结果**，
不能是目标：否则就是在忠实地把偏移复制到部署端。

## 一、规范说了、实现没做（P0）

### 1. 三个晶格的 EMA 更新整体缺失

`b.md` §6.4 的更新规则表明确写：

| 晶格 | 规范规定的更新方式 |
|---|---|
| Sparse | **EMA (γ_sparse) + 软阈值收缩 + feature bank 重置** |
| Manifold | **主码本 EMA (γ_man) + 球面重投影**，T 走纯梯度 |
| Binding | **三个子码本逐层 EMA (γ_bind)**，A_k/A_v 走纯梯度 |

同节末尾还有一份独立的 EMA 更新清单，重复列了这三项。

**实现侧实测：**

- `train/cog_train.py` **零 EMA 调用**
- `sparse_ema_update` / `manifold_ema_update`（`train/lattices.py`）**全仓库零调用点**
- **binding 的 EMA 连函数都不存在**
- manifold 的 EMA 后球面重投影不存在
- feature bank 的 `maybe_reset_dead`：`LCMConfig` 里有 `bank_capacity=4096` /
  `bank_check_interval=100` / `bank_dead_threshold=1000` 三个配置项，但没有任何
  消费它们的代码路径被 `cog_train` 调用

三个晶格在 `b.md` 里都是 **EMA 管理**（即设计上**不**收梯度、由 EMA 独立更新）。
实现里它们既不收梯度（检索路径刻意 detach，见 `ste_relax` 的文档）也不做 EMA
——**两边都没有**。

`joint=False`（两个 CLI 入口的默认值）下更是连 Stage-3 的 `vq_total` /
`contrast_info_nce_loss` / `manifold_orth_loss` 都不跑。

**这是规范与实现差距最大的一处，也是"默认认知训练下 5/6 晶格不学习"的根因。**

### 2. HRQ 顶层选择的度量与规范不符

`b.md` §4.1 步骤 1：

> 计算 `z_P = exp_map(z)` 与 `C_top` 的相似度（`poincare_similarity`），
> 选择相似度最高的原型 `c_top*` 作为路由目标（top-1 硬路由）

实现（`train/lattices.py::hrq_forward`）：

```python
if value_scalars is not None and alpha_val > 0:
    c_top, top_idx, top_scores = value_biased_retrieve(z, C_top, value_scalars, alpha_val, C_out=C_top_P)
else:
    top_idx = sims.argmin(axis=-1)      # 才是规范说的 Poincaré
```

`value_biased_retrieve` 用 `value_biased_scores`，即
`-‖z−c‖² + α_val·tanh(v)·avg_dist²` —— **欧氏**度量，作用在切空间的 `z` 上，
而不是球面上的 `z_P`。

`value_scalars` 在 `init_all_params` 中总是存在、`alpha_val = 0.1 > 0`，所以
**默认走的是欧氏分支，与 §4.1 相反**。

注意 `b.md` 自身在这里是不自洽的：§4.0 的局部价值标量定义给的是欧氏分数，
§4.1 的 HRQ 路由给的是 Poincaré 相似度，规范没说两者同时成立时谁优先。
实现选了欧氏。**这需要一次裁决，不是简单改代码。**

### 3. Sparse 的 LFQ 动态阈值 `d_top` 换了量

`b.md` §4.2：

```
d_top = poincare_similarity(exp_map(z), c_top*)   # 到 HRQ 顶层最近原型的双曲相似度
```

实现（`train/model.py`，`six_lattice_step` 沿用了它）：

```python
d_top = None if training else (1.0 - jnp.mean(hrq_top_sim))
```

`hrq_top_sim` 是该 batch 每个样本 top-1 的相似度，取 `1 - mean` —— 一个**批次
平均**的量，不是"当前样本到其 top-1 原型"的相似度。

### 4. C 部署端的晶格是规范描述的降维版

| 晶格 | `b.md` 描述 | C 实际 |
|---|---|---|
| HRQ | Poincaré 层次残差量化（§4.1） | **纯欧氏最近邻**（本会话已修为 `hrq_forward_c`，见下） |
| LowRank | `n_layers` 层递增 rank 残差链，`C_lr^(k) = U_k @ V[:, :r_k]^T`（§4.3） | `lcm.py::_load_codebooks()` **只重建第一个 rank** |
| Contrast | dual codebook `(o_a+o_b)/2`，各自 n 层残差（§4.6） | **只取第一块 `C_a`** |
| Binding | key/value/bind 三层残差 + 9 对跨层 HRR 绑定（§4.5） | **一个平铺 codebook 块** |
| Manifold | 测地滑移 + 切空间投影（§4.4） | `slide_manifold`（未逐行核对） |
| Sparse | 训练/推理分支 + LFQ（§4.2） | `retrieve_single`，与规范训练分支一致 |

## 二、规范自身内部不一致

### §4.8 与 §7.1 对融合的描述不同

- §4.8：`w_i = softmask_i · α_i`，`z_q = Σ_i w_i · o_i` —— **没有 LayerNorm**
- §7.1：`z_q = sum(...)`，然后 `z_q = layer_norm(z_q)` —— **有 LayerNorm**

实现（`train/fusion.py::fuse_lattices_with_aux`）**有** LayerNorm，即跟随 §7.1。
C 的 `lcm_infer_step_v2` 也对齐了 §7.1。

审计建议：以 §7.1 为准并把 §4.8 补全（LayerNorm 是量级正确的必要条件——本会话
实测过：去掉它输出落到 1e-3 量级而 JAX 是 1 量级）。

## 三、已核实**一致**的部分

逐行核对过的：

| 项 | 结论 |
|---|---|
| `simvq_codebook`（§4.0.1） | 与实现逐行一致 |
| Poincaré 工具（§4.0.2）`poincare_similarity` / `exp_map` / `log_map` / `mobius_add` | 公式一致；C 侧在 c=1 时等价（本会话逐条核对） |
| 局部价值标量初值（§4.0） | `jnp.zeros(M)` ✓；实现用 `tanh(v)` 强制 ±1 饱和，规范写 `v_j ∈ [-1,+1]`，一致 |
| Routing 门（§4.0） | `argmin ‖z−C_route‖²` + STE + GumbelSoftmax(tau=0.5) ✓ |
| Sparse 训练/推理分支（§4.2） | 训练不含 zero_vec、推理含 ✓ |
| Contrast 输出 `(o_a+o_b)/2`（§4.6） | ✓ |
| Frozen LLM 前 4 层（§5.1） | ✓（`n_layers=4`） |
| GValue 冻结 + hash 校验（§4.7） | 机制在，但当前是 placeholder（pos == neg），README 已诚实说明 |

## 四、未核实

`train/lattices.py` 的 `lowrank_forward` / `manifold_forward` / `binding_forward`
三个函数体**尚未逐行读过**，§4.3 / §4.4 / §4.5 的三列对照因此不完整。

`b.md` §8 的参数计数与 §4.x 的推荐值（`M_top=64`、`M_lr=1024` 等）与
`LCMConfig` 的当前值（`M_top=512`、`M_lr=256` 等）也不一致，但那属于调参范围，
不是语义分歧。

## 五、因此的正确顺序

1. **裁决 §4.1 的度量冲突**（HRQ 顶层：欧氏价值分数 vs Poincaré 相似度）——
   规范自相矛盾，必须先定
2. **补上三个晶格的 EMA**，或明确改为纯梯度并**同步修订 `b.md` §6.4**——
   二选一，但不能再维持"规范说 EMA、实现两边都没有"
3. 修正 §4.2 的 `d_top`
4. 补 §4.8 的 LayerNorm（或确认以 §7.1 为准）
5. 上述定案之后，再谈 C 侧逐格忠实实现——那时 C 是对齐**规范**，不是对齐一个
   自己也偏了的 JAX

C↔JAX 的单步 parity 已经做到 0.02%，但那证明的是"两个实现算得一样"，
**不是"算得对"**。本文件之前的那条结论需要按这个区分重新表述。
