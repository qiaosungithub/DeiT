# Auto Research Notes — DeiT + Masked Diffusion Head

## Research Goal

The goal is to replace the standard logit head + CE loss in image classification (DeiT-Base on ImageNet-1K) with a **masked diffusion head**, inspired by the BAR paper.

- Classical head: outputs `(B, N)` logits for N=1000 classes, inference cost O(N).
- Masked diffusion head: represents the label as `log2(N)` bits (~10 bits for 1000 classes). Training uses masked diffusion loss; inference is iterative decoding. More robust to large N.

**Phase 1**: Reproduce DeiT-Base baseline → target **81.8% top-1 on ImageNet-1K**.
**Phase 2**: New branch — implement masked diffusion head, run ablations.

Ablation axes for Phase 2:
1. Different masked diffusion head implementations
2. Sampling strategy / number of sampling steps
3. Training-time mask ratio sampling schedule
4. Training duration (may need longer)

---

## Codebase Overview

- **Framework**: JAX + Flax + Optax. NOT PyTorch.
- **Model**: `models.py` — `ViT` class, `ViT_base` partial. The cls token's output goes through `final_ln → fc (logit head)`.
- **Training loop**: `train.py` — `train_and_evaluate()`. Uses `jax.pmap` over devices.
- **Config system**: `configs/default.py` (ml_collections). Loaded via `configs/load_config.py`.
- **Entry point**: `main.py` — calls `train.train_and_evaluate`.
- **Data**: ImageNet at `/kmh-nfs-ssd-us-mount/data/imagenet`. Input pipeline in `input_pipeline.py`.
- **Logging**: WandB (configured in config). Also local logs.

Key config fields (see `configs/default.py`):
- `model = 'ViT_base'`
- `learning_rate = 0.0005`, scaled as `lr * batch_size / 512`
- `num_epochs = 330`, `warmup_epochs = 5`
- `batch_size = 1024`
- `optimizer = 'adamw'`, `weight_decay`, `stochastic_depth_rate`
- `dataset.use_rand_augment`, `dataset.use_mixup_cutmix`, `dataset.label_smoothing`

Differences from reference DeiT-Base (tracked in `report_deit_base.md`):
- `qkv_bias = True`, `out_proj bias = True` (in Attention, currently biases are disabled — check!)
- LayerNorm: scale only, no bias
- Learned scaling in residual connections (non-standard — this might be a deviation worth checking)
- Stochastic depth: default 0.1
- cls token init: `truncated_normal(0.02)`

---

## Infrastructure: How to Run Experiments

### Find available TPUs
```bash
tou
```
Constraint: only use **v5p ≤ 64** or **v6e ≤ 32** chips.

### Launch a job
```bash
ftmd <ka> <alias>
tpu run <ka> sqa dir=7
```
- `ftmd` = `tpu zhan $ka sqa && tpu fang $ka $alias && tmd $alias` (claims TPU, sets alias, mounts disk)
- `dir=7` is the index for this DeiT directory (verify with `tpu ls sqa`; use `tpu set-cur <idx> sqa` to set if needed — **don't overwrite**)
- Job runs from staged code in `/kmh-nfs-ssd-us-mount/staging/sqa/...`

### Monitor jobs
```bash
tcs            # = tpu check sqa — shows all job statuses
tms            # = tpu monitor sqa col=2
```
- If a job errored due to **code bug**: fix and relaunch.
- If a job errored due to **preemption**: do nothing. Auto-resume script handles it (`/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/MONITOR.py`). **Do not touch resume logic.**

### View logs
- Each job has a tmux window `sqa:<window_id>`.
- Logdir is at `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/...` (TASKNAME from config.sh is currently "paligemma-baseline" — probably should be updated for DeiT).
- Stagedir is under `/kmh-nfs-ssd-us-mount/staging/sqa/`.
- Use `cl` alias (opens logfile in vscode) or read the log files directly.

### Config override at launch
Configs are passed as `--config=configs/load_config.py:remote_run` plus extra `--config.key=value` overrides.

### Run multiple jobs in parallel
Recommended when multiple TPUs are available. Typical hyperparam search: double LR each step (don't over-tune).

---

## Workflow for Phase 2 (New Feature)

1. Create a new git branch (or copy the folder).
2. Modify `models.py`: replace `self.fc = special_linear(num_classes)` and the final `self.fc(x)` call with the masked diffusion head.
3. Modify `train.py`: replace `categorical_cross_entropy_loss` with masked diffusion loss; add iterative decoding for eval.
4. Add new config fields as needed.
5. Run baseline check first (does it train at all?), then ablations.

---

## Record-Keeping

After each experiment completes:
- Record results + WandB link in `agents/results.md`.
- Plan next jobs based on findings.

---

## Research Cycle (每次醒来必须执行)

1. **重新读本文件** (防止忘记初心)
2. **检查 job 状态**: `python /kmh-nfs-ssd-us-mount/code/hanhong/miniforge3/bin/python /home/jzc/zhichengjiang/working/xibo_tpu_manager/tpu.py check sqa` → grep for "DeiT" / relevant jobs
3. **读最新日志**: `python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/see_log.py <window_id>` → 得到 logdir → `grep "eval_accuracy\|eval epoch\|Error" <logdir>/output.log`
4. **整理进度**: 把最新 eval 结果记到 `agents/results.md`
5. **如果有 job error**: 看日志判断是代码 bug 还是 preempt；preempt 不用管
6. **规划下一步**: 根据当前结果决定是否启动新实验
7. **找空闲卡**: `python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/wrap_master.py` → grep IDLE，只用 v5p≤64 或 v6e≤32
8. **启动新实验**:
   - `ftmd <full_tpu_name> <alias>` （ftmd 在 ~/.bashrc 里，需要 source ~/.bash_aliases 才能用）
   - 等几秒后 `tpu run <full_tpu_name> sqa dir=7 --config.xxx=yyy`
   - alias 格式: asia-northeast1-b 的 v6e-8 用 `v6e-8-tmp201~208`（已用: 201=vgda24, 202=axuxm0当前job）
9. **更新 results.md** 记录 WandB link、实验 tag、结果

## Phase 1 Status (Baseline Reproduction)

目标: DeiT-Base 81.8% top-1 on ImageNet-1K, 330 epochs.

| Run | Window | Model | biases | LearnedScale | Status |
|-----|--------|-------|--------|--------------|--------|
| baseline | 6323 | ViT_base | False | True (1e-4) | Running, ep~70, eval@59=60.6% |
| fixed-arch | TBD | ViT_base_v2 | True | False | Planned |
| bias+ls | TBD | ViT_base_v3 | True | True | Planned |

架构差异 (vs reference DeiT-B):
- **Missing**: Q/K/V/out_proj biases (`use_bias=False`) → Reference has these
- **Missing**: LayerNorm bias (`use_bias=False`) → Reference has this
- **Extra**: LearnedScale (LayerScale) at 1e-4 → NOT in DeiT, from CaiT paper
- MLP: has bias ✓, attn_dim: 768 ✓, stochastic_depth: 0.1 ✓

---

## Data / Region

- **跨区读数据不是问题**：codebase 已自动处理 —— 启动时会从本 TPU 同区的 GCS bucket 拉取 ImageNet，放进 `/dev/shm`（本地内存），完全不跨区传输大量数据。任意区域的卡都可以用。
- 所以 asia-northeast1-b、us-east5、us-central1 的卡均可正常用于训练。

---

## Constraints and Gotchas

- **DeiT 同时使用 TPU 不能超过 8 个**（跨所有 DeiT 实验，包括 v2/v3/baseline 所有 job）。用户已将限制从5提升到8。
- **Checkpoint 保存时可能出现假 Error**：tcs 把 checkpoint log 里某些打印判定为 error，其实是正常的。每次醒来检查 Error 时要看实际 log，判断是真 code error 还是假 error。
- **清理无用 window**：如果一个 window 有明显 error 且已修复重跑，用 `tmux kill-window -t sqa:<window_id>` 关掉旧 window。
- **循环间隔：每 30 分钟醒一次**。

- **Do not modify resume logic** in `tpu_manager/MONITOR.py` or related scripts.
- **Do not run `tpu set-cur` carelessly** — dir=7 is the DeiT directory index, don't overwrite.
- **Only v5p ≤ 64 or v6e ≤ 32** TPUs are allowed.
- The `label_smoothing` in `train_step_sqa` is hardcoded to `0.0` (line ~333 in train.py) — label smoothing is applied via Mixup in the dataset pipeline (`dataset.label_smoothing=0.1`).
- `grad_norm_clip` in AdamW currently asserts it must be `None` — don't set it.
- `tpu run` always requires `dir=7` to use the DeiT directory (default is dir=1).
- `tpu fang <new_machine> <alias>` requires the alias to already exist in data.json's `tpu_aliases`.
- mount-disk lock file: `/tmp/xibo_mount_<tpu_name>.lock` — if stale, delete it and retry.
- wandb `set_wandb` step may fail with local Python env issues (missing `annotated_types`) — this is non-critical, training still works.
- **AdamW b2 = 0.95 (not 0.999)**: hardcoded in original code; now configurable via `--config.adamw_b2=0.999`. Reference DeiT uses b2=0.999. This may cause slight divergence from reference trajectory. When launching new runs, consider testing b2=0.999.
- **Registered v6e-8 aliases**:
  - asia-northeast1-b: v6e-8-tmp201→j3rqvs, v6e-8-tmp202→axuxm0, v6e-8-tmp203→p1u4mx
  - us-east5-b: v6e-8-tmp51→c8umw4 (Phase 2 Run A), v6e-8-tmp52→cz2ivo (Phase 2 Run B), v6e-8-tmp53→8507kk (Run D), v6e-8-tmp205/206→yq00yh (Run C)
  - asia-northeast1-b (new): v6e-8-tmp207→i91hh1 (Run E), v6e-8-tmp208→3djlis (Run F), v6e-8-tmp209→06q7u9 (Run G)
  - **Alias convention**: asia-northeast1-b → tmp201+; us-east5-b → tmp51+. ALWAYS check zone before picking alias range.
  - `tpu register` (interactive, no args) only writes data.json — does NOT write to spreadsheet. Let user handle registration.
  - **All Run E/F/G aliases registered** in data.json. User needs to: `ftmd <full_tpu_name> <alias> && tpu run <full_tpu_name> sqa dir=7 --config=configs/load_config.py:<mode>`
  - Run E: `ftmd kmh-tpuvm-v6e-8-spot-gzy-i91hh1 v6e-8-tmp207 && tpu run kmh-tpuvm-v6e-8-spot-gzy-i91hh1 sqa dir=7 --config=configs/load_config.py:remote_run_E`
  - Run F: `ftmd kmh-tpuvm-v6e-8-spot-gzy-3djlis v6e-8-tmp208 && tpu run kmh-tpuvm-v6e-8-spot-gzy-3djlis sqa dir=7 --config=configs/load_config.py:remote_run`
  - Run G: `ftmd kmh-tpuvm-v6e-8-spot-gzy-06q7u9 v6e-8-tmp209 && tpu run kmh-tpuvm-v6e-8-spot-gzy-06q7u9 sqa dir=7 --config=configs/load_config.py:remote_run_G`

## Critical Bugs Found and Fixed (2026-05-09)

- **`class ViT(nn.Module):` was missing** (commit e54ec1c accidentally dropped it when inserting MLPDiffusionHead). The fields appeared as part of MLPDiffusionHead class body, causing NameError on import. Fix: add `class ViT(nn.Module):` before `channels: int` field. **Always check after adding a new class above ViT.**
- **`remote_run_config.yml` had `use_mixup_cutmix: true`** — leftover bug that would have poisoned Run F (MLP baseline). Fixed to `false`. **All diffusion training configs must have `use_mixup_cutmix: false`.**
- **`ViT_base_mdh_mlp` was removed** when the previous agent added MLPDiffusionHead but replaced it with ViT_debug incorrectly. Restored.

## Phase 2 Progress Summary (as of 2026-05-09 ~04:10)

| Run | Architecture | ep | single-step | iter | notes |
|-----|-------------|-----|-------------|------|-------|
| A | attention head, mixup BUG | 120 | ~20%? | N/A | old code; ep=119=17.95%, ep=139 upcoming |
| C | attention head, uniform | 66 | ~11%? | ~17%? | LEADING; ep=59=9.18%/14.04%; ep=79 upcoming |
| D | attention head, logit-normal | 65 | ~10%? | ~15%? | ep=59=8.46%/12.46%; behind C |
| E | attention head, zero-init | TBD | TBD | TBD | i91hh1 MOUNTED+IDLE; user needs ftmd+tpu_run |
| F | MLP head baseline | TBD | TBD | TBD | config ready; need slot (7/8 used) |
| G | large head (512-dim, 4L) | TBD | TBD | TBD | config ready; need slot |

**Key trajectory (Run A, old-code baseline)**: ep79=5.78% → ep99=12.21% → ep119=17.95% → still accelerating
**Key trajectory (Run C, fixed-code)**: ep39=2.12% → ep59=9.18% — much faster, next check ep=79
