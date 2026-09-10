# LCM 第 3 命题：Qwen 仅训练期塑形、学生可剥离独立运行

## 命题

> Qwen 仅用于训练期产生语言软目标并通过蒸馏损失塑形学生；导出的学生 checkpoint 不携带 Qwen，推理时不导入或读取 Qwen，仍能独立形成认知状态并自由生成输出。

这一命题验证的是**训练/推理解耦与运行独立性**，不是声称当前小规模学生已经达到 Qwen 的开放域能力。

## 被验证对象

- 三个实际训练保存的 `lattice_only` 学生：seed 7、17、27。
- 每个 checkpoint 累计 1,800 步；前 800 步来自 Qwen2.5-0.5B-Instruct 软目标训练，后续为晶格瓶颈专项训练。
- 每个学生参数张量总大小：1,690,364 bytes，约 1.61 MiB。
- 每个 checkpoint 顶层参数组：`encoder`、`hrq`、`sparse`、`lowrank`、`manifold`、`binding`、`contrast`、`gen_head`。

## 训练路径审计

对 `train/causal_student_train.py` 的 AST 与损失路径进行审计：

1. `teacher_targets()` 调用 `load_qwen_params()`。
2. 教师前向调用 `qwen_forward()`。
3. Qwen 在学生局部词表上的分布进入蒸馏项 `kd`。
4. 学生总损失包含 `ce + 0.02 * kd + margin`，因此 `kd` 对学生参数参与反向传播。
5. 保存逻辑仅序列化学生参数 `p`，没有把教师参数写入 `student.pkl`。

五项源代码审计均通过。Qwen 在这里是训练期教师，而不是推理期主动通道。

## 序列化边界审计

递归扫描三个 `student.pkl` 中全部张量路径：

- Qwen 张量：0。
- teacher 张量：0。
- Qwen 配置或权重路径依赖：0。
- 三个学生均可仅凭 `student.pkl`、`global_token_ids.npy` 与 tokenizer 恢复。

## 纯学生运行时修复

旧评估器虽然没有加载 Qwen 权重，但通过 `causal_student_train.z_state` 间接 import 训练模块。进一步追踪发现 `cog_train.pack_codebooks_for_c` 又会 import `qwen_lm`。这不构成权重依赖，却使“运行时代码完全剥离”不够严格。

本次新增 `train/student_only_runtime.py`：

- 只导入 encoder、认知循环、fusion 和生成头。
- 在文件内部实现推理期 codebook 展平。
- 不 import `causal_student_train`、`cog_train`、`qwen_lm` 或 `qwen_lang_lcm`。
- `eval_causal_student.py` 已切换到该纯学生运行时。

## 冷启动隔离实验

在新进程中安装两层硬保护：

1. import hook：任何 `qwen_lm`/Qwen 模块导入立即抛出异常。
2. 文件访问 hook：任何读取路径中带 `qwen` 的文件立即拒绝。

在保护生效后加载三个学生并对 20 条未见问法执行自由自回归生成。

| Seed | 非空输出率 | 不同输出数 | EOS 完成率 | Qwen 模块加载 |
|---:|---:|---:|---:|---:|
| 7 | 100% | 14/20 | 100% | 否 |
| 17 | 100% | 17/20 | 95% | 否 |
| 27 | 100% | 11/20 | 95% | 否 |

三个进程均未触发保护器，说明没有尝试访问 Qwen。

## 输出一致性

对相同权重和输入分别执行普通纯学生推理与受限冷启动推理，逐条输出序列做 SHA-256：

| Seed | 输出摘要 | 是否一致 |
|---:|---|---:|
| 7 | `30d2d4fe...1df5cfb` | 是 |
| 17 | `c6c8994a...44a4816` | 是 |
| 27 | `fa6e9ec9...e44a4816` | 是 |

三组输出逐字节一致，保护器没有改变计算结果。

## 行为能力边界

这些学生确实能独立形成并输出完整中文因果句，但当前泛化质量仍有限。此前 s1800 未见问法结果为每个种子约 3–4/20 完全正确，且状态干预 token accuracy 为约 56.8%–62.7%。因此本实验支持“剥离后可工作”，不支持“剥离后已经达到开放域高质量对话”。

这种区分很重要：结构独立性已经成立；能力规模仍取决于后续训练、数据覆盖和晶格路由质量。

## 判据与结论

| 判据 | 结果 |
|---|---:|
| Qwen 真实参与软目标构建 | 通过 |
| 蒸馏项进入学生训练损失 | 通过 |
| checkpoint 不含教师张量 | 通过 |
| 运行时不导入 Qwen | 通过 |
| 运行时不读取 Qwen 文件 | 通过 |
| 三学生均独立自由生成 | 通过 |
| 受限/普通输出完全一致 | 通过 |

命题在当前实验边界内**完整通过**：Qwen 的作用止于训练期软目标塑形，导出的学生具有真实的权重、代码路径和行为层独立性。

## 复现文件

- 纯学生运行时：`train/student_only_runtime.py`
- 剥离验证器：`train/verify_qwen_detachment.py`
- 修改后的学生评估器：`train/eval_causal_student.py`
- 受限运行原始结果：`checkpoints/qwen_detachment_validation.json`
- 普通运行原始结果：`checkpoints/qwen_detachment_unguarded.json`
- 被验证权重：`artifacts/lattice_bottleneck_v2/lattice_bottleneck_seed{7,17,27}_s1800/`

```bash
python -m train.verify_qwen_detachment \
  --guard \
  --root artifacts/lattice_bottleneck_v2 \
  --tokenizer artifacts/lattice_bottleneck_v2/tokenizer.json
```
