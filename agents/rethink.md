# Phase 2 失败复盘（Masked Diffusion Head）

这份文档给后续实现 agent 使用：总结目前实验为何看起来“训出来垃圾”，以及最可能的实现偏差点。

## 1) 现象总结（基于当前结果）

- 参考 `agents/results.md`，Phase 2 Run A/B（`ViT_base_mdh`）在 ep39 时有学习信号，但非常弱：
  - accuracy 仅从 ~0.13% 提升到 ~0.52%
  - loss 从 ~0.69 降到 ~0.66
- 这说明模型不是完全不学，但学习效率和最终质量明显不符合预期。
- 当前评估指标是“10 bit 全对才算对”（sequence-level exact match），随机基线约 0.097%，所以早期指标很低是正常的；但如果实现正确，通常不会长期卡在这种量级。

## 2) 当前实现（关键路径）

### 2.1 Model（`models.py`）

- `ViT_base_mdh` 使用 DeiT backbone + `MaskedDiffusionHead`（`models.py:309-330`）。
- `MaskedDiffusionHead` 实现：
  - bit token vocab `{0,1,MASK(=2)}` embedding（`models.py:153`）
  - 拼接 CLS 条件 token + bit tokens，经若干 `DiffusionLayer`（默认 2 层）
  - 输出 `(B, n_bits, 2)`（`models.py:160-171`）
- 类别编码：`class_to_bits`/`bits_to_class` 为 **LSB-first**（`models.py:105-114`）。

### 2.2 Train（`train.py`）

- diffusion train step: `train_step_diffusion`（`train.py:252-299`）
- 当前 mask 采样：
  - 每样本 `t~Uniform(0,1)`
  - 每 bit 独立 `Bernoulli(t)` 决定是否 mask（`train.py:264-268`）
- loss：仅在 masked positions 上做 bit-wise CE（`train.py:30-40`, `train.py:277`）。

### 2.3 Eval / Decode（`train.py`）

- 当前 eval 用 `eval_step_diffusion`（`train.py:301-327`）：
  - 输入全 mask（不传 `masked_bits`）
  - **single-step greedy** decode（`diffusion_decode_greedy`, `train.py:43-51`）
- 存在 iterative decode 函数 `diffusion_decode_iterative`（`train.py:53-84`），但未接入 eval 主流程。

## 3) 与 BAR 论文做法的关键偏差（高优先级）

> 结论：当前代码抓住了“bit-level masked prediction”的外形，但 BAR 的两个关键点（masking schedule + iterative decoding）没有真正落地到 eval/training protocol。

### 偏差 A：训练标签与 Mixup/CutMix 冲突（最严重）

- 在 `train_step_diffusion` 中，`batch['label']` 是 soft label（mixup/cutmix 后），但代码直接 `argmax` 成单一 hard class（`train.py:259-261`）。
- 这会形成“混合图像 + 单标签 bit 监督”的监督噪声，严重拖累 diffusion head 学习。
- 这是目前最像“训出来垃圾”的主因。

### 偏差 B：推理策略太弱（single-step all-mask）

- 当前 eval 使用 all-mask + 单步解码。
- BAR 的核心是 progressive unmasking（多步迭代），论文中 sampling steps 与 unmask schedule 是核心 ablation（2->3 步有明显提升）。
- 只用 single-step 会显著低估 head 能力，尤其在 early/mid training 阶段。

### 偏差 C：mask ratio/schedule 过于单一

- 当前只实现了 uniform `t` + independent bit mask。
- BAR 对 mask ratio sampling（uniform / arccos / logit-normal）做了系统对比，并指出 schedule 会影响效果。
- 当前实现未给出可切换 schedule，也没有固定每步解开 bit 数的显式 unmask schedule。

### 偏差 D：无效类别码处理不干净

- 10 bit 可表示 0..1023，但真实类别只有 0..999。
- 当前 decode 后 `clip(0,999)`（`train.py:50`, `train.py:83`），1000..1023 被压到 999，造成类别 999 偏置。
- 这不是最致命问题，但会污染上限与误差分布。

## 4) BAR 论文里真正重要的点（供实现 agent 对齐）

- MBM head 不是大词表 softmax，而是 bit-level 条件生成。
- 训练：对 bit 序列做随机 masking，优化 bit-wise CE。
- 推理：通过 **多步 progressive unmasking** 生成，而不是单步。
- sampling step 数和每步解 mask 的 schedule 是 performance-sensitive 设计，不是可忽略细节。

## 5) 建议的修复优先级（按 ROI 排序）

### P0（必须先做）

1. **先关掉 diffusion 分支的 mixup/cutmix** 做 sanity run（保证图像-标签一致）。
2. eval 同时报告：
   - all-mask single-step
   - all-mask iterative（例如 3/4 步）
3. decode 不再简单 clip 到 999；统计 invalid code rate（`pred_class > 999`）。

### P1（直接针对 BAR 核心）

4. 加可配置的 mask ratio sampling（至少 `uniform` / `logit-normal` 两种）。
5. 实现显式 unmask schedule（例如 `[4,4,2]`、`[3,3,2,2]`），避免阈值 ties 导致每步解 mask 数失控。

### P2（结构增强）

6. 增强 head capacity（层数/宽度）并与 step/schedule 联动搜索。
7. 可选：加入 bit validity 约束（例如训练或解码时处理 1000..1023 无效区域）。

## 6) 最小实验矩阵（建议一晚内可跑）

- 固定 backbone 与优化器，先只改 protocol：
  - Exp-A: no mixup/cutmix + single-step eval
  - Exp-B: no mixup/cutmix + iterative eval(3 steps)
  - Exp-C: no mixup/cutmix + iterative eval(4 steps) + schedule variant
  - Exp-D: 在 C 基础上切换 mask ratio sampling（uniform vs logit-normal）
- 每个实验至少看：`eval_loss`, `eval_accuracy`, `invalid_code_rate`，并记录到 `agents/results.md`。

## 7) 一句话结论

目前不是“完全训不动”，而是 **训练监督噪声 + 推理策略过弱 + schedule 未对齐 BAR** 叠加，导致表现看起来像失败。优先修 protocol（P0/P1）比盲目加大模型更关键。
