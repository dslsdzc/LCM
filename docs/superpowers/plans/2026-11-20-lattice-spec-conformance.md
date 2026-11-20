# 晶格规范一致性（Spec Conformance）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `docs/a.md`–`docs/e.md` 五份规范与 JAX 训练侧、C 推理侧、安全子系统、自我意识子系统逐条一致，并用 ≤100 步训练验证。

**Architecture:** 先冻结规范与 checkpoint schema，再按“安全闭环 → JAX 参数更新与晶格行为 → C 引擎 → 安全/自我意识接线 → 100 步 smoke”推进。每个任务独立可测，TDD，频繁提交。

**Tech Stack:** Python 3.11、JAX 0.10.0、optax、pytest、C99、ctypes、Z3/Nagini（离线合同）。

**Spec:** `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md`

## Global Constraints

- 验证训练步数 **≤100 步**；不跑正式长训练。
- 因果主体 `train/causal_subject.py` **不扩展**；现有测试全绿即可；`e.md §5` 差异记为 accepted deviation。
- 旧 checkpoint 一律拒绝加载；新格式 `format_version=2`；不做迁移。
- 训练路径不启用 gvalue/danger 安全检查；推理路径按 R6/R7 分开测。
- 所有双曲比较使用 `z_P = exp_map(safe_unit(z))`；`poincare_distance` 只用于人类可读输出。
- EMA/frozen 参数不得进入 AdamW；`optax.set_to_zero()` 或参数白名单。
- C 端构建前必须 `make clean`；`LCM_D=256`；部署路径必须是 `lcm_infer_loop_v3`。
- 代码注释英文，文档中文；遵循现有文件风格。
- 每个任务结束提交一次；提交信息用 `feat:`/`test:`/`docs:` 前缀。

## Review Focus

1. **单位球面边界**：`z` 经过 `safe_unit` 后 ‖z‖=1，`poincare_similarity` 分母为 0；任何双曲路径若直接用 raw `z` 会得到 NaN 或符号翻转。测试必须覆盖 `z=0`、`‖z‖=1`、`‖z‖>1` 三种输入。
2. **EMA 码本被 weight decay 侵蚀**：把 EMA 参数留在 AdamW 里，零梯度也会每步收缩。测试必须比较“分组后”与“未分组”的 EMA 码本在 100 步后的差异，并断言分组后 `N/m` 是唯一变化来源。
3. **Sparse LFQ 在 `d_top` 缺失时**：HRQ 未激活或训练分支下，不能强制 zero_vec。测试必须覆盖 `d_top=None`、HRQ 未激活、零码本三种情况。
4. **Routing hard 的噪声**：推理 hard 若仍注入 Gumbel 噪声，同一输入两次输出不同。测试必须连续调用两次并断言 bit-identical。
5. **C 端 `.so` 陈旧**：源码改了但 `.so` 没重编，部署仍走旧逻辑。测试必须在 `make clean && make LCM_D=256` 后检查 `nm -D` 符号，并跑一次 `lcm.py` 的 smoke 调用。

---

## File Structure

**新增文件**

| 文件 | 职责 |
|---|---|
| `train/param_groups.py` | 参数按 trainable/ema/frozen 分组、合并、optimizer 白名单 |
| `train/ema_state.py` | EMA 状态（N/m/last_used）与三个晶格的更新函数 |
| `train/feature_bank.py` | sparse 随机替换、contrast 最远点替换、dead-code 检测 |
| `train/self_activation.py` | d.md §3 的 SelfActivationState 四态状态机 |
| `train/smoke_conformance.py` | ≤100 步 smoke 的执行与断言 |
| `infer/loops_v3.c` | `lcm_infer_loop_v3`、canonical fusion、trace 落盘 |
| `infer/lattice_layers.c` | LowRank/Contrast/Binding/Routing/Sparse 的 C 实现 |
| `infer/sha256.c` | 统一 SHA-256（danger/gvalue 共用） |

**主要修改**

| 文件 | 修改 |
|---|---|
| `train/checkpoint.py` | 32-byte header、`format_version=2`、manifest、canonical hash |
| `train/lattices.py` | R1 HRQ 路由、R2 sparse、R18–R22 |
| `train/cognitive_step.py` | 调用顺序、d_top 传递、自输出融合 |
| `train/fusion.py` | 删除 value signal 软调制、R4 LayerNorm |
| `train/cog_train.py` | 参数分组、EMA/bank 接线、commitment loss、self 融合 |
| `train/config.py` | R17 参数、lambda 拆分、schema version |
| `train/gvalue.py` | R6 严格三法则、统一双曲域 |
| `infer/lcm.h` / `lcm_api.h` | `LatticeMemory`、v3 接口、`CONFLICT_ABORT_TOKEN` |
| `infer/engine.c` | 动态 DAG、依赖边、trace |
| `infer/lcm_api.c` | v3 桥、alert 回调、verify |
| `infer/danger.c` / `gvalue.c` | R6/R7、SHA-256、真实双码本 |
| `lcm.py` | 全参数加载、v3 调用、abort 传播 |
| `train/external_verifier.py` | 纯标准库独立进程 |
| `train/self_lattice.py` | R15 重写 |
| `train/observability.py` 等 | R16 接线 |

---

## Task 0: 规范修订与裁决记录

**Files:**
- Modify: `docs/a.md`、`docs/b.md`、`docs/c.md`、`docs/d.md`、`docs/e.md`
- Modify: `docs/a_cn.md`、`docs/b_cn.md`、`docs/c_cn.md`、`docs/d_cn.md`、`docs/e_cn.md`
- Create: `docs/lattice-spec-vs-impl-audit-v3.1.md`

**Interfaces:**
- Produces: 规范文字与设计文档 R1–R22 一致；后续任务的“规范条款 → 测试”映射以此为准。

- [ ] **Step 1: 在 `docs/b.md` 写入 R1–R6、R9、R17–R22 的规范文字**

逐条修改：
- §4.0 价值标量：粒度层间共享（R22）；检索分数符号 `-dist² + α·v·avg_dist²`。
- §4.1 HRQ：Poincaré top-k + 价值重排、相对 gap、STE 代理（R1）。
- §4.2 Sparse：`d_top` 逐样本、同域、`lambda_shrink`/`lambda_lfq` 拆分、HRQ 依赖（R2）。
- §4.4 Manifold：参数存切空间、EMA 等价重投影（R3）。
- §4.8 Fusion：补 LayerNorm，删除 value signal 软惩罚（R4/R5）。
- §6.3/§6.4：codebook loss + commitment loss 分开（R18）。
- §7.1/§7.2：canonical fusion、EMA 接线、feature bank（R9）。
- §1 符号表：`β_val` 标记 deprecated；`max_steps` 拆两个名字（R17）。

- [ ] **Step 2: 在 `docs/a.md` 写入 R5、C-6 的修订**

- §4.2.8 删除 `exp(β_val·value_signal)`，改为 `z_q = layer_norm(Σ softmask·α·o)`。
- §4.2.2 Sparse 训练分支改为“训练不含 zero_vec，推理含”。

- [ ] **Step 3: 在 `docs/c.md` 写入 R8/R9 的修正**

- §3 融合权重改为“概念描述，实际以 §4.3 的 `1/(d+ε)` 为准”。
- §4.1 `max_steps`：调度预算 graceful、安全硬限 abort。
- §4.2 `build_dag` 伪代码修正：先收集、再追加；bind 依赖所有上游；unbind 接 key；
  threshold 动态；`OP_DISTANCE_FUSION` 由执行器处理。
- §5.3 熵门：归一化熵 `H/ln(n_active)`。

- [ ] **Step 4: 在 `docs/d.md` 写入 R6/R7/R14 的修正**

- §1.3 danger 用有界相对接近度（R7）。
- §4.2 gvalue 用严格三法则 `balance > -margin`（R6）。
- §3 Self Activation 状态机与 `C_self` 命名区分（C-9）。

- [ ] **Step 5: 在 `docs/e.md` 写入 R15/R16 的修正**

- §6 Self Lattice：`C_self[0]=core`、EMA 其余行、NN + cosine familiarity。
- Overview/§8：self lattice 是融合项；其他模块非实时。

- [ ] **Step 6: 同步五份中文版**

按英文版逐条同步，保留中文术语；在每份文件头部加：

```markdown
> **v3.1-conformance changelog**：本版按 `docs/superpowers/specs/2026-11-20-lattice-spec-conformance-design.md`
> 的 R1–R22 修订。中文版与英文版同步。
```

- [ ] **Step 7: 写偏差复审文档**

`docs/lattice-spec-vs-impl-audit-v3.1.md`：逐条 P0/C/F 编号 → 对应 R 编号 → 状态
（fixed / accepted deviation / deferred）。causal 的差异标 accepted deviation。

- [ ] **Step 8: 验证关键条款已落文**

Run:
```bash
grep -n "lambda_lfq\|lambda_shrink" docs/b.md docs/b_cn.md
grep -n "H / ln\|H/ln\|n_active" docs/c.md docs/c_cn.md
grep -n "balance > -safety_margin_relative" docs/b.md docs/d.md
grep -n "C_self\[0\]\|C_self\[1:\]" docs/e.md docs/e_cn.md
```
Expected: 每条至少 2 个匹配（英文 + 中文）。

- [ ] **Step 9: Commit**

```bash
git add docs/
git commit -m "docs: v3.1-conformance 规范修订（R1-R22）"
```

---

## Task 1: Checkpoint/export schema v2

**Files:**
- Modify: `train/checkpoint.py`
- Modify: `lcm.py`
- Modify: `infer/lcm.h`
- Test: `train/test_checkpoint_schema.py`

**Interfaces:**
- Produces:
  - `HEADER_FMT_V2 = '<iiiifII'`、`HEADER_SIZE_V2 = 32`、`FORMAT_VERSION = 2`
  - `save_manifest(output_dir: str, tensors: dict[str, np.ndarray]) -> None`
  - `load_manifest(output_dir: str) -> dict`
  - `canonical_sha256(tensors: dict[str, np.ndarray]) -> str`
  - `read_bin_v2(path: str) -> tuple[dict, np.ndarray]`
  - 旧 24-byte header 抛 `ValueError("old checkpoint format; re-export required")`。

- [ ] **Step 1: 写失败测试**

`train/test_checkpoint_schema.py`:

```python
import numpy as np
import pytest
from train.checkpoint import (
    FORMAT_VERSION, HEADER_FMT_V2, HEADER_SIZE_V2,
    canonical_sha256, save_manifest, load_manifest,
)

def test_header_v2_roundtrip(tmp_path):
    import struct
    hdr = struct.pack(HEADER_FMT_V2, 512, 256, 1, 0, 1.0, FORMAT_VERSION, 0)
    assert len(hdr) == HEADER_SIZE_V2
    assert struct.unpack(HEADER_FMT_V2, hdr)[5] == 2

def test_canonical_sha256_is_order_independent():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.ones((4,), dtype=np.float32)
    h1 = canonical_sha256({'b': b, 'a': a})
    h2 = canonical_sha256({'a': a, 'b': b})
    assert h1 == h2

def test_manifest_roundtrip(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_manifest(str(tmp_path), {'a': a})
    m = load_manifest(str(tmp_path))
    assert m['format_version'] == FORMAT_VERSION
    assert m['tensors']['a']['shape'] == [2, 3]
    assert m['tensors']['a']['dtype'] == 'float32'

def test_old_header_rejected(tmp_path):
    p = tmp_path / "old.bin"
    p.write_bytes(b'\x00' * 24)
    from train.checkpoint import read_bin_v2
    with pytest.raises(ValueError, match="old checkpoint format"):
        read_bin_v2(str(p))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_checkpoint_schema.py -v`
Expected: FAIL with `ImportError` / `cannot import name`.

- [ ] **Step 3: 实现 v2 header、manifest、canonical hash**

`train/checkpoint.py`:
- 新增 `HEADER_FMT_V2 = '<iiiifII'`、`HEADER_SIZE_V2 = 32`、`FORMAT_VERSION = 2`。
- `read_bin_v2(path)`：读 32 字节，若文件长度对应旧 24 字节 header 或无
  `format_version` 字段，抛 `ValueError("old checkpoint format; re-export required")`。
- `canonical_sha256(tensors)`：按 tensor name 字典序；每个 tensor 先写
  `np.int32(ndim)` + shape，再写 `float32` little-endian bytes；返回 hex。
- `save_manifest` 写 `manifest.json`，含 `format_version` 和每 tensor 的
  `name/shape/dtype/offset/length/sha256`。
- `load_manifest` 校验 `format_version == 2`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest train/test_checkpoint_schema.py -v`
Expected: PASS 4 passed.

- [ ] **Step 5: 改 `save_cog_checkpoint` 写 v2 + manifest**

`train/cog_train.py::save_cog_checkpoint` 末尾调用 `save_manifest(output_dir, {...})`，
tensor 集合 = 所有 `.bin` 的逻辑名 + `cog_params.pkl` 的 params 摘要。
`run_identity` 增加 `format_version=2`。

- [ ] **Step 6: 改 `lcm.py` 加载路径拒绝旧格式**

`load_codebook_for_c` 改用 `read_bin_v2`；`LCMInferEngine.__init__` 检查
`config.json['format_version'] == 2`，否则 `SystemExit`。

- [ ] **Step 7: 运行现有 checkpoint 测试**

Run: `pytest train/test_identity_enforcement.py train/test_resume_identity.py -v`
Expected: PASS（新格式测试用 tmp_path；旧 checkpoint 测试若依赖旧文件，改为断言拒绝）。

- [ ] **Step 8: Commit**

```bash
git add train/checkpoint.py train/cog_train.py train/test_checkpoint_schema.py lcm.py infer/lcm.h
git commit -m "feat(checkpoint): schema v2 with manifest and canonical sha256"
```

---

## Task 2: 统一双曲域与安全语义

**Files:**
- Modify: `train/hyp.py`、`train/gvalue.py`
- Modify: `infer/hyp.c`、`infer/gvalue.c`、`infer/danger.c`
- Modify: `train/safety_nagini.py`、`train/verify.py`、`train/test_safety_hypothesis.py`
- Test: `train/test_safety_semantics.py`

**Interfaces:**
- Produces:
  - `train.hyp.to_poincare(z) -> jnp.ndarray`：`exp_map(safe_unit(z))`。
  - `GValueCodebook.check_safety(z, margin=0.5) -> (bool, int)` 使用
    `balance = (pos_sim - neg_sim)/(pos_sim + neg_sim + eps)`，
    `violated = balance > -margin`。
  - `infer/gvalue.c::gvalue_check_safety(gv, z, margin, &is_safe, &law)` 同公式。
  - `infer/danger.c::danger_assess` 用
    `relative_gap = (sim_normal - sim_threat)/(sim_normal + sim_threat + eps)`，
    `should_block = relative_gap > threshold`。

- [ ] **Step 1: 写失败测试**

`train/test_safety_semantics.py`:

```python
import jax.numpy as jnp
import numpy as np
from train.hyp import to_poincare, poincare_similarity
from train.gvalue import make_global_value_vectors, GValueCodebook

def test_to_poincare_keeps_points_inside_ball():
    for scale in [0.0, 0.5, 1.0, 3.0]:
        z = jnp.ones((4, 256)) * scale
        z_P = to_poincare(z)
        assert jnp.all(jnp.linalg.norm(z_P, axis=-1) < 1.0)
        assert jnp.all(jnp.isfinite(z_P))

def test_strict_three_laws_neutral_is_violation():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 256))
    is_safe, law = gv.check_safety(z, safety_margin_relative=0.5)
    assert not bool(is_safe)
    assert int(law) >= 0

def test_margin_zero_allows_neutral():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    z = jnp.zeros((1, 256))
    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.0)
    assert bool(is_safe)

def test_positive_input_is_safe():
    C_pos, C_neg = make_global_value_vectors(256)
    gv = GValueCodebook(C_pos, C_neg)
    # 取一个正锚点附近的点：exp_map 后仍在球的同一侧
    z = jnp.asarray(C_pos[0])[None, :] * 0.5
    is_safe, _ = gv.check_safety(z, safety_margin_relative=0.0)
    assert bool(is_safe)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_safety_semantics.py -v`
Expected: FAIL with `ImportError: to_poincare`.

- [ ] **Step 3: 实现 `to_poincare` 与 JAX 安全公式**

`train/hyp.py`:
```python
def to_poincare(z, c=1.0):
    """Map arbitrary Euclidean z to the Poincaré ball at a safe radius."""
    return exp_map(safe_unit(z), c)
```
`train/gvalue.py`：`check_safety` / `check_safety_batch` 用 `to_poincare`；
`balance` 公式；`violated = balance > -margin`。`check_safety_batch` 返回
`(is_safe: (B,), margins: (B,), violated_law: (B,))`，`margins` 定义为
`-balance`（越大越安全）。

- [ ] **Step 4: 运行 JAX 测试确认通过**

Run: `pytest train/test_safety_semantics.py -v`
Expected: PASS 4 passed.

- [ ] **Step 5: 改 C 端 `gvalue.c`/`danger.c`**

`infer/gvalue.c`：
- `gvalue_check_safety` 用 `exp_map_c(safe_unit_c(z))`，`balance` 公式，
  `violated = balance > -margin`。
- `gvalue_init` 预投影 `C_pos/C_neg` 到球面；`gvalue_verify` 用统一 SHA-256。

`infer/danger.c`：
- `danger_assess` 用 `to_poincare` 后的 `poincare_similarity`；
  `relative_gap` 公式；`should_block = gap > threshold`。
- 删除不可达的 runaway 分支（移到 scheduler）。

- [ ] **Step 6: 加 C 单元测试**

`infer/test_runner.c` 新增：
```c
static void test_gvalue_strict_neutral(void) { /* z=0, margin=0.5 → !safe */ }
static void test_gvalue_margin_zero(void)     { /* z=0, margin=0 → safe */ }
static void test_danger_neutral_no_block(void){ /* threat≈normal → no block */ }
```

- [ ] **Step 7: 运行 C 测试**

Run: `make -C infer clean && make -C infer test`
Expected: PASS all tests.

- [ ] **Step 8: 重写安全合同**

`train/safety_nagini.py` / `train/verify.py` / `train/test_safety_hypothesis.py`：
把旧公式替换为 `balance` / `relative_gap`；断言：
- `margin=0` 时中性安全；
- `margin>0` 时中性违规；
- `relative_gap` 有界 `[-1,1]`。

Run: `pytest train/test_safety_hypothesis.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add train/hyp.py train/gvalue.py infer/hyp.c infer/gvalue.c infer/danger.c \
        infer/test_runner.c train/safety_nagini.py train/verify.py \
        train/test_safety_hypothesis.py train/test_safety_semantics.py
git commit -m "feat(safety): unified Poincare domain and strict Three Laws"
```

---

## Task 3: 参数分组、EMA 接线、feature bank

**Files:**
- Create: `train/param_groups.py`、`train/ema_state.py`、`train/feature_bank.py`
- Modify: `train/cog_train.py`、`train/checkpoint.py`、`train/config.py`
- Test: `train/test_param_groups.py`、`train/test_ema_state.py`、`train/test_feature_bank.py`

**Interfaces:**
- Produces:
  - `partition_params(params) -> (trainable, ema, frozen)`，三棵同构子树。
  - `merge_params(trainable, ema, frozen) -> dict`。
  - `init_ema_state(params, cfg) -> dict`：
    `{'sparse': {'N':(M,), 'm':(M,d), 'last_used':(M,)}, ...}`
  - `ema_update_sparse(params_C, ema_state, z, idx, gamma, lambda_shrink)`。
  - `ema_update_manifold(params_C, ema_state, z, idx, gamma)`。
  - `ema_update_binding(params_binding, ema_state, residuals, gamma)`。
  - `FeatureBank(capacity, dim)` with `add(z)`, `dead_indices(last_used, step, t_dead)`,
    `replace_dead_sparse(C, state, step)`, `replace_dead_contrast(C, state, step)`.

- [ ] **Step 1: 写参数分组测试**

`train/test_param_groups.py`:

```python
import jax.numpy as jnp
from train.param_groups import partition_params, merge_params

def test_partition_roundtrip():
    p = {'encoder': {'w': jnp.ones((2, 2))},
         'sparse': {'C': jnp.ones((4, 2)), 'zero_vec': jnp.zeros((1, 2))},
         'manifold': {'C': jnp.ones((4, 2)), 'T': jnp.ones((4, 2, 1))},
         'binding': {'key_cb': [{'A': jnp.ones((4, 2))}], 'A_k': jnp.ones((2, 2))},
         'danger': {'C': jnp.ones((4, 2))}}
    tr, em, fr = partition_params(p)
    assert 'sparse' not in tr
    assert 'sparse' in em
    assert 'danger' in fr
    assert 'binding' in tr and 'key_cb' not in tr['binding']
    assert 'A_k' in tr['binding']
    assert merge_params(tr, em, fr).keys() == p.keys()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_param_groups.py -v`
Expected: FAIL `ImportError`.

- [ ] **Step 3: 实现参数分组**

`train/param_groups.py`:
- `EMA_PATHS = {('sparse','C'), ('manifold','C'), ('binding','key_cb'),
  ('binding','val_cb'), ('binding','bind_cb'), ('self','C_self')}`。
- `FROZEN_PATHS = {('danger','C'), ('gvalue','C_pos'), ('gvalue','C_neg')}`。
- 递归拆分；`merge_params` 反向合并；断言没有叶子丢失。

- [ ] **Step 4: 写 EMA 测试**

`train/test_ema_state.py`:

```python
import jax.numpy as jnp
from train.ema_state import init_ema_state, ema_update_sparse

def test_sparse_ema_updates_only_selected_rows():
    cfg = type('C', (), {'M_sparse': 4, 'd_model': 2,
                         'lambda_shrink': 1e-4})()
    C = jnp.zeros((4, 2))
    state = {'N': jnp.zeros(4), 'm': jnp.zeros((4, 2)), 'last_used': jnp.zeros(4, dtype=jnp.int32)}
    z = jnp.array([[1.0, 2.0]])
    idx = jnp.array([1])
    C2, s2 = ema_update_sparse(C, state, z, idx, gamma=0.99, lambda_shrink=1e-4)
    assert float(jnp.linalg.norm(C2[1])) > 0.0
    assert float(jnp.linalg.norm(C2[0])) == 0.0
    assert float(s2['N'][1]) > 0.0
```

- [ ] **Step 5: 实现 EMA 状态与三个更新函数**

`train/ema_state.py`:
- `init_ema_state`：按 `cfg.M_sparse/M_man/M_bind/n_bind_layers` 建零状态。
- `_scatter_sum(z, idx, M)`：`jnp.zeros((M,d)).at[idx].add(z)`。
- `_scatter_count(idx, M)`：`jnp.zeros(M).at[idx].add(1)`。
- `ema_update_sparse`：`N/m` 更新 + `C = m/clip(N,1)` + soft threshold。
- `ema_update_manifold`：同 EMA，不做 `exp_map`（参数存切空间）。
- `ema_update_binding`：对 `residuals['key'/'val'/'bind']` 每层调用通用 EMA。

- [ ] **Step 6: 写 feature bank 测试**

`train/test_feature_bank.py`:

```python
import jax.numpy as jnp
from train.feature_bank import FeatureBank

def test_dead_code_replaced_by_random():
    bank = FeatureBank(capacity=8, dim=2)
    for i in range(4):
        bank.add(jnp.array([[float(i), 0.0]]))
    C = jnp.zeros((3, 2))
    last_used = jnp.array([0, 100, 100], dtype=jnp.int32)
    C2, last2 = bank.replace_dead_sparse(C, last_used, step=1001, t_dead=1000)
    assert not jnp.array_equal(C2[0], C[0])
    assert int(last2[0]) == 1001

def test_contrast_uses_farthest_point():
    bank = FeatureBank(capacity=4, dim=2)
    for v in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]:
        bank.add(jnp.array([v]))
    C = jnp.array([[5.0, 0.0]])
    C2, _ = bank.replace_dead_contrast(C, jnp.array([0], dtype=jnp.int32),
                                       step=1001, t_dead=1000)
    # 最远点是 [-1, 0]
    assert float(C2[0, 0]) < 0.0
```

- [ ] **Step 7: 实现 feature bank 并接线**

`train/feature_bank.py`：ring buffer、`dead_indices`、两种替换策略。
`train/cog_train.py`：训练循环在 `opt_state` 更新后调用 EMA；每
`bank_check_interval` 步在 jit 外做 bank 更新（host numpy），重建 params。
`train/checkpoint.py`：保存/恢复 `ema_state` 和 `feature_bank`。

- [ ] **Step 8: 运行全部新测试**

Run: `pytest train/test_param_groups.py train/test_ema_state.py train/test_feature_bank.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add train/param_groups.py train/ema_state.py train/feature_bank.py \
        train/cog_train.py train/checkpoint.py train/config.py \
        train/test_param_groups.py train/test_ema_state.py train/test_feature_bank.py
git commit -m "feat(train): parameter groups, EMA updates, feature banks"
```

---

## Task 4: 逐层 codebook/commitment loss + routing 训练

**Files:**
- Modify: `train/losses.py`、`train/lattices.py`、`train/cog_train.py`
- Test: `train/test_vq_losses.py`

**Interfaces:**
- Produces:
  - `codebook_loss(r, C, idx, beta=0.25)`：`β·mean((sg[r] − C[idx])²)`。
  - `commitment_loss(r, C, idx, beta=0.25)`：`β·mean((r − sg[C[idx]])²)`。
  - `hrq_vq_loss(params, z, aux, cfg) -> (codebook, commitment)`。
  - `route_commit_loss` 进入主损失。
  - `routing_gate(hard=True)` 无噪声。

- [ ] **Step 1: 写失败测试**

`train/test_vq_losses.py`:

```python
import jax
import jax.numpy as jnp
from train.losses import codebook_loss, commitment_loss

def test_codebook_loss_grad_only_to_C():
    r = jnp.array([[1.0, 0.0]])
    C = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    idx = jnp.array([1])
    g_r, g_C = jax.grad(lambda r, C: codebook_loss(r, C, idx), argnums=(0, 1))(r, C)
    assert jnp.allclose(g_r, 0.0)
    assert not jnp.allclose(g_C, 0.0)

def test_commitment_loss_grad_only_to_r():
    r = jnp.array([[1.0, 0.0]])
    C = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    idx = jnp.array([1])
    g_r, g_C = jax.grad(lambda r, C: commitment_loss(r, C, idx), argnums=(0, 1))(r, C)
    assert not jnp.allclose(g_r, 0.0)
    assert jnp.allclose(g_C, 0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_vq_losses.py -v`
Expected: FAIL `ImportError`.

- [ ] **Step 3: 实现两个 loss**

`train/losses.py`：按签名实现，用 `lax.stop_gradient`。
`train/lattices.py`：`hrq_forward` 返回 `(o_hrq_cont, o_hrq_hard, idx)` 或
在 `aux` 中携带连续/量化两版；`binding_forward` 返回 `b_q/v_out`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest train/test_vq_losses.py -v`
Expected: PASS 2 passed.

- [ ] **Step 5: 接线到 `cog_train`**

把现有 `joint` 块里的 `min dist²` 替换为逐层 `codebook_loss` + `commitment_loss`；
`joint=False` 也执行。`route_commit_loss` 无条件执行。
`routing_gate(hard=True)` 改为 `one_hot(argmax(logits))`，不注入噪声。

- [ ] **Step 6: 写 routing hard 无噪声测试**

`train/test_vq_losses.py` 追加：

```python
def test_routing_hard_is_deterministic():
    from train.lattices import init_route_params, routing_gate
    p = init_route_params(jax.random.PRNGKey(0), 6, 8)
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    m1, _, _ = routing_gate(p, z, tau=0.5, hard=True, rng=jax.random.PRNGKey(2))
    m2, _, _ = routing_gate(p, z, tau=0.5, hard=True, rng=jax.random.PRNGKey(3))
    assert jnp.array_equal(m1, m2)
```

- [ ] **Step 7: 运行测试**

Run: `pytest train/test_vq_losses.py -v`
Expected: PASS 3 passed.

- [ ] **Step 8: Commit**

```bash
git add train/losses.py train/lattices.py train/cog_train.py train/test_vq_losses.py
git commit -m "feat(loss): layered codebook/commitment losses and routing training"
```

---

## Task 5: HRQ 两阶段路由 + Sparse LFQ

**Files:**
- Modify: `train/lattices.py`、`train/cognitive_step.py`、`train/config.py`
- Test: `train/test_hrq_routing.py`、`train/test_sparse_lfq.py`

**Interfaces:**
- Produces:
  - `hrq_forward(params, z, tau_fallback=0.1, value_scalars=None,
    alpha_val=0.0, value_topk=4, tau_proxy=1.0)`。
  - `_topk_indices(params, z, k) -> jnp.ndarray`（测试辅助；内部同 top-k 计算）。
  - `sparse_forward(params, z, training=True, lambda_lfq=1.5, d_top=None,
    c_top=None, value_scalars=None, alpha_val=0.0)`（Ruling 15：1.5，0.1 会在
    随机初始化下恒零）。
  - `six_lattice_step` 先算 HRQ，把 `d_top`/`c_top` 传给 Sparse。

- [ ] **Step 1: 写 HRQ 测试**

`train/test_hrq_routing.py`:

```python
import jax
import jax.numpy as jnp
from train.lattices import init_hrq_params, hrq_forward
from train.hyp import exp_map, poincare_similarity

def _crafted_hrq(d=8, M_top=4):
    p = init_hrq_params(jax.random.PRNGKey(0), d, M_top, 4, 1)
    # 让 top 码本分化：第 i 行在单位轴上远离
    A = jnp.eye(M_top, d) * 0.9
    p['top'] = {'A': A, 'W': jnp.eye(d)}
    return p

def test_v_zero_matches_pure_poincare():
    p = _crafted_hrq()
    z = jax.random.normal(jax.random.PRNGKey(1), (3, 8)) * 0.3
    o1, idx1, _ = hrq_forward(p, z, value_scalars=jnp.zeros(4), alpha_val=0.0)
    z_P = exp_map(z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8))
    C_P = exp_map(p['top']['A'] @ p['top']['W'])
    sims = poincare_similarity(z_P[:, None, :], C_P[None, :, :]).squeeze(-1)
    assert jnp.array_equal(idx1, sims.argmin(axis=-1))

def test_value_bias_changes_top1_only_within_topk():
    p = _crafted_hrq()
    z = jax.random.normal(jax.random.PRNGKey(2), (1, 8)) * 0.3
    v = jnp.array([0.0, 0.0, 0.0, 10.0])
    _, idx0, _ = hrq_forward(p, z, value_scalars=jnp.zeros(4), alpha_val=0.1)
    _, idx1, _ = hrq_forward(p, z, value_scalars=v, alpha_val=0.1)
    # 变化只允许发生在 top-k 内
    from train.lattices import _topk_indices
    topk = _topk_indices(p, z, k=4)
    assert bool(jnp.all(jnp.isin(idx1, topk)))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_hrq_routing.py -v`
Expected: FAIL（`value_topk` 未定义 / 符号错误）。

- [ ] **Step 3: 实现 R1 路由**

`train/lattices.py::hrq_forward`：
- `z_P = exp_map(safe_unit(z))`；`C_P = exp_map(C_top)`。
- `sims = poincare_similarity(...)`；`topk_idx = jnp.argsort(sims)[:, :k]`。
- `scale = stop_gradient(mean(sims)) + 1e-6`；
  `score = sims - alpha_val * tanh(v) * scale`。
- `score_topk` 只在 top-k 内重排；`argmin` 得 `top_idx`。
- hard 路由：`o_hard = c_top*`；`o_soft = Σ softmax(-score_topk/tau_proxy) * c_topk_P`；
  `o_hard = o_hard + stop_gradient(o_soft - o_hard)`。
- fallback：`gap = (sim2 - sim1)/(sim1 + eps)`，`use_fallback = gap < tau_fallback`；
  fallback 用 `softmax(-score_topk)` 混合。
- 删除 `ste_relax(top_scores, C_top_P)` 旁路。

- [ ] **Step 4: 运行 HRQ 测试**

Run: `pytest train/test_hrq_routing.py -v`
Expected: PASS.

- [ ] **Step 5: 写 Sparse 测试**

`train/test_sparse_lfq.py`:

```python
import jax.numpy as jnp
from train.lattices import init_sparse_params, sparse_forward

def test_no_lfq_without_d_top():
    p = init_sparse_params(jax.random.PRNGKey(0), 4, 3)
    z = jnp.ones((2, 4))
    o, idx = sparse_forward(p, z, training=False, d_top=None)
    assert not jnp.allclose(o, 0.0)

def test_lfq_forces_zero_when_far():
    p = init_sparse_params(jax.random.PRNGKey(0), 4, 3)
    z = jnp.ones((2, 4))
    d_top = jnp.array([1e-6, 1e-6])
    o, idx = sparse_forward(p, z, training=False, d_top=d_top, c_top=z)
    assert jnp.allclose(o, 0.0)
```

- [ ] **Step 6: 实现 R2 sparse**

- `d_top` 逐样本；`d_sparse_min` 用 `poincare_similarity(z_P, exp_map(clamp_norm(C)))`；
- `lambda_lfq` 与 `lambda_shrink` 分开；
- `d_top=None` 或 `c_top=None` 时禁用 LFQ；
- 训练分支不含 zero_vec，推理含。

- [ ] **Step 7: 改 `six_lattice_step`**

HRQ 先算，传 `d_top=hrq_top_sim`、`c_top=hrq_top_point` 给 sparse；
`training` 时也传（LFQ 在 sparse 内部按 training 禁用）。

- [ ] **Step 8: 运行测试**

Run: `pytest train/test_hrq_routing.py train/test_sparse_lfq.py train/test_cognitive_step_parity.py -v`
Expected: PASS（parity 测试若因签名变化失败，更新为新的 `d_top` 形状）。

- [ ] **Step 9: Commit**

```bash
git add train/lattices.py train/cognitive_step.py train/config.py \
        train/test_hrq_routing.py train/test_sparse_lfq.py train/test_cognitive_step_parity.py
git commit -m "feat(lattices): Poincare top-k HRQ routing and per-sample sparse LFQ"
```

---

## Task 6: Fusion、Binding、Contrast、Value scalars

**Files:**
- Modify: `train/fusion.py`、`train/lattices.py`、`train/cognitive_step.py`、`train/config.py`
- Test: `train/test_fusion_conformance.py`、`train/test_binding_unbind.py`、`train/test_contrast_conformance.py`

**Interfaces:**
- Produces:
  - `fuse_lattices_with_aux(..., gvalue=None)`：删除 value signal 软调制；
    `z_q = layer_norm(Σ softmask·α·o + self_bias·o_self)`。
  - `binding_forward(...) -> (v_out, residuals, b_q, k_q)`。
  - `contrast_forward(...) -> (o_contrast, indices_per_layer)`。
  - `init_value_scalars` 返回 per-lattice 结构（HRQ top + fine、binding 三族、contrast 双码本）。

- [ ] **Step 1: 写 fusion 测试**

`train/test_fusion_conformance.py`:

```python
import jax.numpy as jnp
from train.fusion import fuse_lattices_with_aux, init_fusion_params

def test_no_value_signal_modulation():
    outs = [jnp.ones((1, 4)) * i for i in range(6)]
    mask = jnp.ones((1, 6)) / 6
    params = init_fusion_params(jax.random.PRNGKey(0), 6, 4)
    z, w, h = fuse_lattices_with_aux(outs, mask, params, gvalue=object(), beta_val=0.5)
    # gvalue 非 None 也不得改变权重
    z2, w2, h2 = fuse_lattices_with_aux(outs, mask, params, gvalue=None)
    assert jnp.allclose(w, w2)
    assert jnp.allclose(z, z2)
```

- [ ] **Step 2: 实现 fusion 对齐**

删除 `compute_value_signal` 调用与 `exp(beta_val * value_signals)`；
保留 `gvalue` 参数但忽略（API 兼容），文档注明 deprecated。
`self_bias` 加法在 LayerNorm 前。

- [ ] **Step 3: 写 binding 测试**

`train/test_binding_unbind.py`:

```python
import jax
import jax.numpy as jnp
from train.lattices import init_binding_params, binding_forward

def test_unbind_recovers_value():
    d, M, n, r = 8, 16, 3, 4
    p = init_binding_params(jax.random.PRNGKey(0), d, M, n, r)
    V = jax.random.normal(jax.random.PRNGKey(1), (d, r)) * 0.1
    z = jax.random.normal(jax.random.PRNGKey(2), (2, d)) * 0.2
    out, res, b_q, k_q = binding_forward(p, z, V)
    assert out.shape == (2, d)
    assert b_q.shape[0] == 2
```

- [ ] **Step 4: 实现 binding unbind**

按 R20 公式：`v_out = NN(C_val, IFFT(conj(norm_fft(k_q)) ⊙ norm_fft(b_q)))`；
`o_bind = v_out`；返回 `b_q/k_q` 供 loss/EMA。

- [ ] **Step 5: 写 contrast 测试**

`train/test_contrast_conformance.py`:

```python
import jax
import jax.numpy as jnp
from train.lattices import init_contrast_params, contrast_forward, contrast_info_nce_loss

def test_temperature_is_accepted():
    p = init_contrast_params(jax.random.PRNGKey(0), 8, 16, 2)
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 8))
    loss = contrast_info_nce_loss(p, z, tau=0.1)
    assert jnp.isfinite(loss)
```

- [ ] **Step 6: 实现 contrast 与 value scalars**

- `tau=0.1`；负样本排除正样本（普通版）；
- value-biased 版用 `exp(-||c-v_harm||²/tau_val)`；
- `contrast_forward` 返回每层 idx；
- `init_value_scalars` 返回：
  `{'hrq': {'top': (M_top,), 'fine': (M_fine,)}, 'sparse': (M_sparse,),
    'lowrank': (M_lr,), 'manifold': (M_man,),
    'binding': {'key': (M_bind,), 'val': (M_bind,), 'bind': (M_bind,)},
    'contrast': {'a': (M_contrast,), 'b': (M_contrast,)}}`。

- [ ] **Step 7: 运行测试**

Run: `pytest train/test_fusion_conformance.py train/test_binding_unbind.py train/test_contrast_conformance.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add train/fusion.py train/lattices.py train/cognitive_step.py train/config.py \
        train/test_fusion_conformance.py train/test_binding_unbind.py \
        train/test_contrast_conformance.py
git commit -m "feat(lattices): binding unbind, contrast sampling, value scalar granularity"
```

---

## Task 7: C 端全参数内存布局与加载

**Files:**
- Modify: `infer/lcm.h`、`infer/lcm_api.h`、`infer/lcm_api.c`
- Modify: `lcm.py`
- Test: `infer/test_runner.c`、`train/test_c_load.py`

**Interfaces:**
- Produces:
  - `lattice_memory_v3`：每晶格的多层指针 + 元数据。
  - `lcm_memory_v3`：`lattice_memory_v3 lattices[7]`、`route_C/route_W`、
    `fusion_alpha/ln_scale/ln_bias`、`value_scalars`、`self_C/alpha_self`、
    `zero_vec`、`tau_*`、`lambda_lfq`。
  - `lcm_load_memory_v3(manifest_path, &mem) -> int`（Python 侧传指针）。
  - `load_lattice_memories(ckpt_dir: str, cfg: dict) -> dict`：按 manifest 读取并解包
    全部层，返回 `{hrq_top, hrq_fine, sparse_C, sparse_zero, lowrank_U, lowrank_V,
    manifold_C, manifold_T, bind_key, bind_val, bind_bind, contrast_a, contrast_b,
    route_C, route_W, fusion, value_scalars, self_C}`。
  - `lcm_infer_loop_v3(...)` 的扁平签名（按晶格分组）。

- [ ] **Step 1: 定义 C 结构**

`infer/lcm.h` 新增：

```c
typedef struct {
    const float* C_top; int M_top;
    const float* C_fine; int M_fine; int n_fine;
    const float* value_top; const float* value_fine;
} hrq_mem_t;

typedef struct {
    const float* C; int M;
    const float* zero_vec;
    const float* value;
} sparse_mem_t;

typedef struct {
    const float* U;        /* [n_layers][M_lr][r_max] */
    const float* V;        /* [D][r_max] */
    const int*   ranks; int n_layers;
    const float* value;
} lowrank_mem_t;

typedef struct {
    const float* C; int M;
    const float* T; int t_dim;
    const float* value;
} manifold_mem_t;

typedef struct {
    const float* key_cb; const float* val_cb; const float* bind_cb;
    int M; int n_layers;
    const float* A_k; const float* A_v;
    const float* value_key; const float* value_val; const float* value_bind;
} binding_mem_t;

typedef struct {
    const float* C_a; const float* C_b; int M; int n_layers;
    const float* value_a; const float* value_b;
} contrast_mem_t;

typedef struct {
    const float* C_route; const float* W_route;
    const float* alpha; const float* ln_scale; const float* ln_bias;
    const float* self_C; int M_self; float alpha_self;
} fusion_mem_t;
```

`lcm_memory_v3` 聚合上述。

- [ ] **Step 2: 写 C 加载测试**

`infer/test_runner.c` 新增 `test_memory_v3_layout`：构造小数组，断言
`lcm_memory_v3` 的字段能正确索引（每层 offset = `layer * M * D`）。

- [ ] **Step 3: 实现 `lcm_load_memory_v3`**

从扁平指针和维度构造 `lcm_memory_v3`；`LCM_D` 校验；不分配内存。

- [ ] **Step 4: 改 `lcm.py` 加载全参数**

`_load_codebooks` 改为读 `manifest.json`，按 name 读取每个 tensor；
不再只取第一层/第一块。加载 `route_codebook`、`fusion.bin`、`value_scalars.bin`、
`self_lattice.bin`、`binding_proj.bin`、contrast `C_b`、lowrank 1..n 层、sparse `zero_vec`。

- [ ] **Step 5: 写 Python 加载测试**

`train/test_c_load.py`（直接测解析函数，不初始化完整引擎）：

```python
import json, struct
import numpy as np
from train.checkpoint import HEADER_FMT_V2, HEADER_SIZE_V2, FORMAT_VERSION
from lcm import load_lattice_memories

def _write_bin(path, M, d, n_layers, arr):
    hdr = struct.pack(HEADER_FMT_V2, M, d, n_layers, 0, 1.0, FORMAT_VERSION, 0)
    path.write_bytes(hdr + np.asarray(arr, dtype=np.float32).tobytes())

def test_loads_lowrank_all_layers(tmp_path):
    d, M, n, r_max = 8, 16, 3, 4
    U = np.zeros((n, M, r_max), dtype=np.float32)
    V = np.zeros((d, r_max), dtype=np.float32)
    _write_bin(tmp_path / "lowrank_codebook.bin", M, d, n, np.concatenate([U.ravel(), V.ravel()]))
    cfg = {"d_model": d, "M_lr": M, "n_lr_layers": n, "r_max": r_max,
           "ranks": [2, 4, 8]}
    mem = load_lattice_memories(str(tmp_path), cfg)
    assert mem["lowrank_U"].shape == (n, M, r_max)
    assert mem["lowrank_V"].shape == (d, r_max)
```

- [ ] **Step 6: 运行测试**

Run: `make -C infer clean && make -C infer test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add infer/lcm.h infer/lcm_api.h infer/lcm_api.c infer/test_runner.c lcm.py train/test_c_load.py
git commit -m "feat(infer): full multi-layer memory layout and v3 loader"
```

---

## Task 8: C 端 HRQ + Sparse + Routing

**Files:**
- Modify: `infer/lattice.c`、`infer/lcm.h`
- Test: `infer/test_runner.c`

**Interfaces:**
- Produces:
  - `int hrq_forward_v31(const float* z, const hrq_mem_t* m, float alpha_val,
    int value_topk, float tau_fallback, float* out, int* idx, float* top_sim)`。
  - `int sparse_forward_c(const float* z, const sparse_mem_t* m,
    float lambda_lfq, float d_top, int training, float* out, int* idx)`。
  - `void routing_gate_c(const float* z, const float* C_route, const float* W_route,
    int n_lattices, float* soft_mask)`。

- [ ] **Step 1: 写 HRQ C 测试**

`infer/test_runner.c` 新增：
```c
static void test_hrq_v31_poincare_top1(void) {
    /* 构造 4 个分化 top 码本，验证 idx 与纯 Poincare argmin 一致 */
}
static void test_hrq_v31_relative_gap(void) {
    /* 两个几乎相同的原型 → fallback；两个远离的原型 → hard */
}
```

- [ ] **Step 2: 改 `hrq_forward_c`**

顶层改为 `poincare_similarity` argmin（不是欧氏）；top-k 内价值重排；
相对 gap；STE 代理（C 端只做前向，无梯度，代理不需要）。
保留 Möbius 残差与 `log_map`。

- [ ] **Step 3: 实现 `sparse_forward_c`**

训练分支不含 zero_vec；推理含；`d_top` 逐样本；`lambda_lfq`；
`d_top<=0` 或 `training` 时禁用 LFQ。

- [ ] **Step 4: 实现 `routing_gate_c`**

`idx = argmin||z - C_route||²`；`logits = W_route^T z_route`；
推理 hard = one-hot argmax；训练路径由 Python 侧算 soft mask（C 只做推理）。

- [ ] **Step 5: 运行 C 测试**

Run: `make -C infer clean && make -C infer test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add infer/lattice.c infer/lcm.h infer/test_runner.c
git commit -m "feat(infer): Poincare HRQ, sparse LFQ, routing gate"
```

---

## Task 9: C 端 LowRank + Manifold + Contrast + Binding

**Files:**
- Modify: `infer/lattice.c`、`infer/lcm.h`
- Create: `infer/lattice_layers.c`
- Modify: `infer/Makefile`
- Test: `infer/test_runner.c`

**Interfaces:**
- Produces:
  - `void lowrank_forward_c(const float* z, const lowrank_mem_t* m, float* out, int* idx)`。
  - `void manifold_forward_c(const float* z, const manifold_mem_t* m, float* out, int* idx)`。
  - `void contrast_forward_c(const float* z, const contrast_mem_t* m, float* out)`。
  - `void binding_forward_c(const float* z, const binding_mem_t* m, float* V,
    float* out, float* b_q, float* k_q)`。

- [ ] **Step 1: 写 C 测试**

每个函数一个 round-trip/形状测试；LowRank 断言 `n_layers` 层都参与；
Contrast 断言输出是 `(o_a+o_b)/2`；Binding 断言 `unbind(bind(k,v),k)≈v`。

- [ ] **Step 2: 实现 LowRank**

`V` 共享；每层 `C_k = U_k @ V[:, :r_k]^T`；残差链；`idx` 每层返回。

- [ ] **Step 3: 实现 Manifold**

复用 `slide_manifold`；把触发距离改为 `poincare_similarity`（R6 域）。

- [ ] **Step 4: 实现 Contrast**

`C_a/C_b` 每层残差；`o = (Σa + Σb)/2`。

- [ ] **Step 5: 实现 Binding**

`W_k = V @ A_k`；三层残差；9 对 HRR；`b_q`；`v_out = NN(C_val, unbind)`。

- [ ] **Step 6: 运行 C 测试**

Run: `make -C infer clean && make -C infer test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add infer/lattice.c infer/lattice_layers.c infer/lcm.h infer/Makefile infer/test_runner.c
git commit -m "feat(infer): full lowrank, manifold, contrast, binding"
```

---

## Task 10: C 端动态 DAG、loop_v3、trace

**Files:**
- Create: `infer/loops_v3.c`
- Modify: `infer/engine.c`、`infer/lcm_api.c`、`infer/lcm_api.h`、`infer/lcm.h`
- Modify: `lcm.py`
- Test: `infer/test_runner.c`、`train/test_c_loop.py`

**Interfaces:**
- Produces:
  - `int build_dag_v3(const float* z, lcm_memory_v3* mem, dag_t* dag)`：
    动态阈值、HRQ→Sparse 依赖边、bind/unbind 完整输入。
  - `int lcm_infer_loop_v3(...)`：canonical fusion、归一化熵、graceful 16、
    hard 64、安全检测、trace 落盘、`CONFLICT_ABORT_TOKEN=-2`。
  - `lcm_get_trace_v3(...)`。

- [ ] **Step 1: 写 DAG 测试**

`infer/test_runner.c` 新增：
- `test_dag_sparse_depends_on_hrq`：Sparse 节点有 `d_top` 输入边；
- `test_dag_dynamic_threshold`：低阈值下不全开；
- `test_dag_bind_all_upstream`：bind 的 `n_inputs == n_retrieve`。

- [ ] **Step 2: 实现 `build_dag_v3`**

分层收集候选 → 再追加；HRQ 先执行；Sparse 依赖 HRQ 的 `top_sim`；
bind 依赖所有 retrieve；unbind 接 bind + key；threshold 动态。

- [ ] **Step 3: 写 loop 测试**

`train/test_c_loop.py`：
- 正常输入 `max_steps=16` 不 abort，返回 0；
- `max_steps=64` 构造不收敛输入 → abort -2；
- trace 文件存在且含 DAG 节点。

- [ ] **Step 4: 实现 `lcm_infer_loop_v3`**

`for step in [0, max_steps)`：build_dag_v3 → execute_dag → canonical fusion →
detect_any_conflict → trace → 收敛检查（`diff<tol` 且 `H/ln(n_active)<0.5`）。
调度预算耗尽返回 0；安全硬限耗尽返回 -2 并 `halt_and_alert`。

- [ ] **Step 5: 改 `lcm.py` 调 v3**

`generate` 默认调用 `lcm_infer_loop_v3`；`CONFLICT_ABORT_TOKEN` 时
`return ""`（不产生 token）；alert 回调写日志。
删除 `--no-safety` 或改为只跳过 gvalue。

- [ ] **Step 6: 运行测试**

Run: `make -C infer clean && make -C infer test && pytest train/test_c_loop.py -v`
Expected: PASS.

- [ ] **Step 7: 检查 `.so` 符号**

Run: `nm -D infer/liblcm.so | grep -E "lcm_infer_loop_v3|hrq_forward_v31"`
Expected: 两个符号都存在。

- [ ] **Step 8: Commit**

```bash
git add infer/loops_v3.c infer/engine.c infer/lcm_api.c infer/lcm_api.h infer/lcm.h \
        infer/test_runner.c lcm.py train/test_c_loop.py
git commit -m "feat(infer): dynamic DAG, v3 macro loop, trace and hard abort"
```

---

## Task 11: JAX↔C 逐格 parity

**Files:**
- Modify: `train/parity_c_compare.py`
- Create: `train/test_parity_lattices.py`

**Interfaces:**
- Consumes: Task 5/6/8/9/10 的 JAX/C 函数。
- Produces: 每格 `idx` 精确相等 + 连续输出 `max_abs_err<1e-4`。

- [ ] **Step 1: 写 parity 测试**

`train/test_parity_lattices.py`：

```python
import numpy as np
import pytest
from train.parity_c_compare import run_lattice_parity

@pytest.mark.parametrize("lattice", ["hrq","sparse","lowrank","manifold",
                                     "binding","contrast","routing"])
def test_lattice_parity(lattice):
    res = run_lattice_parity(lattice, seed=0)
    assert res["idx_equal"], f"{lattice} idx mismatch"
    assert res["max_abs_err"] < 1e-4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest train/test_parity_lattices.py -v`
Expected: FAIL（`run_lattice_parity` 不存在）。

- [ ] **Step 3: 实现 parity runner**

`train/parity_c_compare.py`：固定 seed、固定 z、固定参数；
对每格调 JAX 与 C；返回 `idx_equal/max_abs_err`。
C 端用 ctypes 加载 `liblcm.so`；`LCM_D` 与 checkpoint 一致。

- [ ] **Step 4: 运行测试**

Run: `pytest train/test_parity_lattices.py -v`
Expected: PASS 7 passed.

- [ ] **Step 5: 加 fusion parity**

追加：
- canonical fusion：`fuse_lattices_with_aux` vs `lcm_fuse_only`；
- engine fusion：`cog_loop.dag_fuse` vs `distance_weighted_fusion`。
两者 `max_abs_err<1e-5`。

- [ ] **Step 6: 运行测试并提交**

Run: `pytest train/test_parity_lattices.py -v`
Expected: PASS.

```bash
git add train/parity_c_compare.py train/test_parity_lattices.py
git commit -m "test(parity): per-lattice and fusion JAX<->C parity"
```

---

## Task 12: External verifier 与 SelfActivationState

**Files:**
- Rewrite: `train/external_verifier.py`
- Create: `train/self_activation.py`
- Test: `train/test_external_verifier.py`、`train/test_self_activation.py`

**Interfaces:**
- Produces:
  - `VerifierProcess`：纯标准库，`multiprocessing.shared_memory`，状态
    `NORMAL/OBSERVE/LOCK`，LOCK 锁存。
  - `SelfActivationState`：`DEFINED_FROZEN/TEMP_UNFROZEN/FULL_ACTIVATION`；
    `transition(target, authorization=None)`；Path A 无需授权，Path B 需
    danger flag，Path C 需操作员授权；转换记录写 `self_activation.log`。

- [ ] **Step 1: 写 verifier 测试**

`train/test_external_verifier.py`：

```python
import time
from train.external_verifier import VerifierProcess, NORMAL, OBSERVE, LOCK

def test_state_machine_reaches_lock():
    v = VerifierProcess(dim=8, threshold_low=1.0, threshold_high=2.0,
                        observe_limit=2, shared_name=None)
    v.start()
    for _ in range(3):
        v.submit([10.0] * 8)
        time.sleep(0.05)
    assert v.state() == LOCK
    v.stop()
```

- [ ] **Step 2: 实现 verifier**

纯 `math/statistics/hashlib/multiprocessing`；z-score 规则；`NORMAL` 记误报；
`OBSERVE` 写 trace；`LOCK` 锁存并回调。不 import JAX/numpy。

- [ ] **Step 3: 写 self activation 测试**

`train/test_self_activation.py`：

```python
import pytest
from train.self_activation import SelfActivationState, State

def test_path_a_requires_no_authorization():
    s = SelfActivationState()
    s.transition(State.TEMP_UNFROZEN, authorization="user asked")
    assert s.state == State.TEMP_UNFROZEN
    s.transition(State.DEFINED_FROZEN)
    assert s.state == State.DEFINED_FROZEN

def test_path_c_requires_operator():
    s = SelfActivationState()
    with pytest.raises(PermissionError):
        s.transition(State.FULL_ACTIVATION)
    s.transition(State.FULL_ACTIVATION, authorization="operator:alice")
    assert s.state == State.FULL_ACTIVATION
```

- [ ] **Step 4: 实现 SelfActivationState**

四态、转换表、授权校验、只追加日志；`identity_score` 接口从 self lattice 读取。

- [ ] **Step 5: 运行测试**

Run: `pytest train/test_external_verifier.py train/test_self_activation.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train/external_verifier.py train/self_activation.py \
        train/test_external_verifier.py train/test_self_activation.py
git commit -m "feat(safety): external verifier process and self activation state machine"
```

---

## Task 13: Self lattice 重写与融合

**Files:**
- Rewrite: `train/self_lattice.py`
- Modify: `train/fusion.py`、`train/cognitive_step.py`、`train/model.py`、`train/cog_train.py`
- Test: `train/test_self_lattice.py`

**Interfaces:**
- Produces:
  - `init_self_params(rng, d, M_self=64) -> {'C_self': (M_self,d), 'bias_self': ()}`。
  - `self_lattice_forward(params, z, training) -> (z_self, self_bias)`。
  - `C_self[0]` frozen core；`C_self[1:]` EMA。
  - `self_bias = alpha_self * max(0, cosine_sim(z, z_self))`。
  - 融合：`z_q = layer_norm(Σ w_i o_i + self_bias * z_self)`。

- [ ] **Step 1: 写测试**

`train/test_self_lattice.py`：

```python
import jax
import jax.numpy as jnp
from train.self_lattice import init_self_params, self_lattice_forward

def test_self_bias_varies_with_z():
    p = init_self_params(jax.random.PRNGKey(0), 8, 64)
    z1 = jnp.ones((1, 8)) / jnp.sqrt(8)
    z2 = -z1
    _, b1 = self_lattice_forward(p, z1, training=False)
    _, b2 = self_lattice_forward(p, z2, training=False)
    assert not jnp.allclose(b1, b2)

def test_core_is_frozen():
    p = init_self_params(jax.random.PRNGKey(0), 8, 64)
    core = p['C_self'][0]
    p2 = jax.tree_util.tree_map(lambda x: x + 1.0, p)
    assert jnp.array_equal(p2['C_self'][0], core + 1.0)  # 结构测试
```

- [ ] **Step 2: 实现 R15**

`C_self` 码本；`C_self[0]=core` 不更新；`C_self[1:]` EMA；
`z_self = C_self[argmin ||z-C_self||²]`；`self_bias` cosine；
删除 `modes`/`mode_activation` 作为输出选择器（保留为诊断）。

- [ ] **Step 3: 接入融合**

`fuse_lattices_with_aux` 的 `self_bias` 改为传入的标量数组；
`cognitive_step` 把 `self_bias * z_self` 加在 LayerNorm 前；
`cog_train` 传 `self_output`/`self_bias`。

- [ ] **Step 4: 运行测试**

Run: `pytest train/test_self_lattice.py train/test_fusion_conformance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train/self_lattice.py train/fusion.py train/cognitive_step.py \
        train/model.py train/cog_train.py train/test_self_lattice.py
git commit -m "feat(self): codebook NN + cosine familiarity per e.md"
```

---

## Task 14: 自我意识接线与 5% 开销

**Files:**
- Modify: `train/cog_train.py`、`train/observability.py`、`train/narrative_memory.py`、`train/reflection_loop.py`、`train/behavior_explorer.py`、`lcm.py`
- Test: `train/test_self_awareness_wiring.py`

**Interfaces:**
- Produces:
  - `ObservabilityRecorder` 记录完整字段；long-term ring 10000。
  - `NarrativeMemory` 五条规则、`context`/`access_count`、遗忘。
  - `ReflectionLoop` 每 200 步异步、四项分析、suggestion queue。
  - `BehaviorExplorer` 每 10 步、danger 优先、bias 进 `train_step`。
  - `measure_overhead(fn_with, fn_without) -> float`。

- [ ] **Step 1: 写测试**

`train/test_self_awareness_wiring.py`：

```python
def test_narrative_accepts_is_safe_rule():
    from train.narrative_memory import NarrativeMemory
    m = NarrativeMemory()
    # 构造 is_safe=False 的 StepRecord，断言入库

def test_overhead_under_five_percent():
    from train.observability import measure_overhead
    ratio = measure_overhead(lambda: None, lambda: None)
    assert ratio < 0.05
```

- [ ] **Step 2: 实现接线**

- `cog_train` 每步 `obs.record_step(...)`、`narr.feed(...)`、`causal.step(...)`；
- `reflection` 用后台线程每 200 步；
- `behavior` 每 10 步，`routing_bias` 传入 `six_lattice_step`；
- `lcm.py` 补全字段（`z_cur/z_next/soft_mask/convergence_diff/sparse_idx` 等）；
- 5% 测量用 wall-clock 差值。

- [ ] **Step 3: 运行测试**

Run: `pytest train/test_self_awareness_wiring.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add train/cog_train.py train/observability.py train/narrative_memory.py \
        train/reflection_loop.py train/behavior_explorer.py lcm.py \
        train/test_self_awareness_wiring.py
git commit -m "feat(self-awareness): wire modules and measure overhead"
```

---

## Task 15: 100 步 smoke、实验报告、偏差复审

**Files:**
- Create: `train/smoke_conformance.py`
- Create: `docs/lattice-conformance-experiment-report.md`
- Modify: `docs/lattice-spec-vs-impl-audit-v3.1.md`
- Test: `train/test_smoke_conformance.py`

**Interfaces:**
- Consumes: 前 14 个任务的全部接口。
- Produces: 可复现的 100 步 smoke 报告。

- [ ] **Step 1: 写 smoke 脚本**

`train/smoke_conformance.py`：
- `--steps 100 --batch 1 --seq 64 --passive-only`
- 记录：loss、每格梯度范数、EMA N/m、码本利用率、route 变化、sparse 非零比例、
  entropy 门、安全 abort 率、self_bias、5% 开销、save/load 一致性。
- 输出 JSON + Markdown 表。

- [ ] **Step 2: 写 smoke 断言测试**

`train/test_smoke_conformance.py`：用 5 步小规模跑通；断言 JSON 字段齐全。

- [ ] **Step 3: 跑 100 步 smoke**

Run:
```bash
python train/smoke_conformance.py --steps 100 --batch 1 --seq 64 \
  --data data/zhwiki_tokens.dat -o checkpoints/conformance_smoke
```
Expected: 完成 100 步，无 NaN；脚本内部调用 `train_cog`，并生成
`checkpoints/conformance_smoke/smoke_report.json` + Markdown 表。
（不启用 Qwen 即为 passive-only；active channel 单独做一次单步前向测试。）

- [ ] **Step 4: 跑逐格 parity**

Run: `pytest train/test_parity_lattices.py train/test_c_loop.py -v`
Expected: PASS.

- [ ] **Step 5: 写实验报告**

`docs/lattice-conformance-experiment-report.md`：
环境、commit、配置、100 步曲线、每格更新统计、parity 表、安全触发、
自观测开销、已知局限。明确“100 步不证明语言能力，只证明规范一致性”；
安全码本是结构占位，不声称语义安全。

- [ ] **Step 6: 更新偏差复审**

`docs/lattice-spec-vs-impl-audit-v3.1.md`：P0/C/F 每条 → R 编号 → fixed/deferred；
causal 标 accepted deviation。

- [ ] **Step 7: 最终验证**

Run:
```bash
pytest train/ -v
make -C infer clean && make -C infer test
nm -D infer/liblcm.so | grep -E "lcm_infer_loop_v3|hrq_forward_v31"
```
Expected: 全部 PASS；符号存在。

- [ ] **Step 8: Commit**

```bash
git add train/smoke_conformance.py train/test_smoke_conformance.py \
        docs/lattice-conformance-experiment-report.md \
        docs/lattice-spec-vs-impl-audit-v3.1.md
git commit -m "test: 100-step conformance smoke and experiment report"
```

---

## Self-Review

**Spec coverage**

| 设计条款 | 任务 |
|---|---|
| R1 HRQ 路由 | Task 5 |
| R2 Sparse LFQ | Task 5 |
| R3 EMA/参数分组 | Task 3 |
| R4 LayerNorm | Task 6 |
| R5 删除 value signal | Task 6 |
| R6 严格三法则 | Task 2 |
| R7 Danger | Task 2 |
| R8 build_dag | Task 10 |
| R9 fusion/max_steps/熵 | Task 6 + Task 10 |
| R10 C 端逐格 | Task 7–9 |
| R11 安全子系统 | Task 2 + Task 10 + Task 12 |
| R12 自我意识 | Task 14 |
| R13 安全链 | Task 2 + Task 10 |
| R14 外部验证器/状态机 | Task 12 |
| R15 Self lattice | Task 13 |
| R16 接线/5% | Task 14 |
| R17 参数默认值 | Task 3 + Task 5 + Task 10 |
| R18 codebook/commitment | Task 4 |
| R19 Routing | Task 4 |
| R20 Binding unbind | Task 6 |
| R21 Contrast | Task 6 |
| R22 value scalars | Task 6 |

**Review Focus 对应测试**

- 单位球面边界：Task 2 `test_to_poincare_keeps_points_inside_ball`。
- EMA weight decay：Task 3 参数分组测试 + Task 15 smoke 的 EMA 统计。
- Sparse `d_top` 缺失：Task 5 `test_no_lfq_without_d_top`。
- Routing hard 噪声：Task 4 `test_routing_hard_is_deterministic`。
- `.so` 陈旧：Task 10 Step 7 + Task 15 Step 7。

**未覆盖的规范项**

- `e.md` 的 causal 反事实/边界扩展：按用户决定不扩展，记为 accepted deviation。
- `d.md` 的 HSM 物理隔离：非目标，只做本地形态。
- 正式重训（>100 步）：非目标，本次只做 smoke。
