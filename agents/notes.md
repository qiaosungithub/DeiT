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
  - asia-northeast1-b (new): v6e-8-tmp207→i91hh1 (Run E), v6e-8-tmp208→3djlis (Run F), v6e-8-tmp209→06q7u9 (Run G), v6e-8-tmp210→qxxa8y (Run H)
  - **Alias convention**: asia-northeast1-b → tmp201+; us-east5-b → tmp51+. ALWAYS check zone before picking alias range.
  - `tpu register` (interactive, no args) only writes data.json — does NOT write to spreadsheet. Let user handle registration.
  - **All Run E/F/G/H aliases registered** in data.json. User needs to: `ftmd <full_tpu_name> <alias> && tpu run <full_tpu_name> sqa dir=7 --config=configs/load_config.py:<mode>`
  - Run E: **i91hh1 preempted** — use axuxm0 (v6e-8-tmp202, already IDLE+MOUNTED): `tpu run kmh-tpuvm-v6e-8-spot-gzy-axuxm0 sqa dir=7 --config=configs/load_config.py:remote_run_E`
  - Run F: `ftmd kmh-tpuvm-v6e-8-spot-gzy-3djlis v6e-8-tmp208 && tpu run kmh-tpuvm-v6e-8-spot-gzy-3djlis sqa dir=7 --config=configs/load_config.py:remote_run`
  - Run G: `ftmd kmh-tpuvm-v6e-8-spot-gzy-06q7u9 v6e-8-tmp209 && tpu run kmh-tpuvm-v6e-8-spot-gzy-06q7u9 sqa dir=7 --config=configs/load_config.py:remote_run_G`
  - Run H: `ftmd kmh-tpuvm-v6e-8-spot-gzy-qxxa8y v6e-8-tmp210 && tpu run kmh-tpuvm-v6e-8-spot-gzy-qxxa8y sqa dir=7 --config=configs/load_config.py:remote_run_H`

## Critical Bugs Found and Fixed (2026-05-09)

- **`class ViT(nn.Module):` was missing** (commit e54ec1c accidentally dropped it when inserting MLPDiffusionHead). The fields appeared as part of MLPDiffusionHead class body, causing NameError on import. Fix: add `class ViT(nn.Module):` before `channels: int` field. **Always check after adding a new class above ViT.**
- **`remote_run_config.yml` had `use_mixup_cutmix: true`** — leftover bug that would have poisoned Run F (MLP baseline). Fixed to `false`. **All diffusion training configs must have `use_mixup_cutmix: false`.**
- **`ViT_base_mdh_mlp` was removed** when the previous agent added MLPDiffusionHead but replaced it with ViT_debug incorrectly. Restored.

## Phase 2 Progress Summary (as of 2026-05-09 20:21)

| Run | Architecture | TPU | Window | Status | notes |
|-----|-------------|-----|--------|--------|-------|
| A | attention head, mixup BUG | c8umw4 (us-east5-b) | 6352 | ⚠️ ERROR | Preempted; TPU deleted |
| C | attention head, uniform | yq00yh (us-east5-b) | 6363 | ⚠️ ERROR | Preempted; TPU deleted |
| D | attention head, logit-normal | 8507kk (us-east5-b) | 6364 | ⚠️ ERROR | Preempted; TPU deleted |
| E | attention head, zero-init | axuxm0 | 6377 | ✅ RUNNING | Restarted fresh ep=0; logdir `20260509_200405_88jyh8_...axuxm0...` |
| F | MLP head baseline | 3djlis | 6375 | ✅ RUNNING | ep~1.5; logdir `20260509_200228_jwwv8y_...3djlis...` |
| G | large head (512-dim, 4L) | 06q7u9 | 6376 | ✅ RUNNING | ep~1.4; logdir `20260509_200306_h9blvq_...06q7u9...` |
| H | attention + aux CE (λ=0.1) | qxxa8y | 6372 | ✅ RUNNING | ep=0 launching; logdir `20260509_202006_qqjdbo_...qxxa8y...` |
| I | pretrained backbone + diff head | j3rqvs | 6378 | ✅ RUNNING | Resumed from ep~8; logdir `20260509_195827_ote0gj_...j3rqvs...` |
| sanity | ViT_base_v3 (full align) | p1u4mx | 6378 | ✅ RUNNING | ep=0 launching; logdir `20260509_201805_rtww0x_...p1u4mx...` |

**Phase 1 COMPLETE**: Run 3 (biases+LS) = **73.14%** BEST, Run 1 = 71.96%, Run 2 = 65.96%.
**Run A/C/D**: TPUs deleted (us-east5-b spot wave). No auto-resume possible; abandoned.
**Sanity run**: New run testing full alignment with reference (WD mask, b2=0.999, LR schedule, CLS init, stochastic depth linear, final LN bias) — target 81.8%.
**All 6 Phase 2 slots filled** as of 20:21.

## MONITOR.py Auto-Resume Pipeline (2026-05-09)

MONITOR.py lives at `/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/MONITOR.py`.

### Job classification
`_is_gpt_b_job(job)` checks if `'deit'` is in the job's wandb_notes string. Jobs classified as DeiT get routed to v6e-8 TPUs. **This is why wandb_notes must contain 'deit' for auto-resume to work.**

### Auto-resume flow
1. `mainloop()` runs periodically, calls `python tpu.py check sqa` to find Error-status windows.
2. `check_job_status(job)` checks the error type: `preempted`, `deleted`, `tpu_still_exists`, or `resume_next_round`.
3. For `preempted` jobs: detects via gcloud status → finds idle v6e-8 TPU → runs `ftmd + tpu run`.
4. For `resume_next_round` jobs (seen error last round, TPU still alive): runs `tpu resume` or `tpu rerun` directly.

### Queue system
`queue.json` holds pending jobs. `_process_queue()` is called each loop iteration.
- `_get_queue_job_wandb_notes()` reads from **`configs/remote_run_config.yml`** specifically (not the actual launched config). This is the file that must be kept up-to-date.

### Manual resume
```bash
python /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/tpu.py resume sqa window=<W> tpu=<full_tpu_name>
```
This calls `jobs.resume_rerun_job(job, load_ckpt=True)` which:
1. Looks up the job's `stage_dir` and `extra_configs` from data.json
2. Creates a new tmux window
3. SSHes to the TPU and runs from the staged dir with `--config.load_from=<log_dir>`

### Launch convention (IMPORTANT)
**Always copy the experiment config to `remote_run_config.yml` before launching**, then launch with `--config=configs/load_config.py:remote_run`. This ensures `tpu.py check` shows the correct tag.

```bash
cp configs/remote_run_F_config.yml configs/remote_run_config.yml
tpu run kmh-tpuvm-v6e-8-spot-gzy-3djlis sqa dir=7 --config=configs/load_config.py:remote_run
```

### Gotchas
- `staging.sh` blocks until the remote training completes (gcloud ssh blocks). Queued shell commands execute only after the preceding job finishes (could be 330 epochs later).
- MONITOR.py only tracks jobs registered via `tpu run` in data.json. Direct `source staging.sh` launches are invisible to MONITOR.py.
- `tpu.py resume` correctly handles checkpoint loading by appending `--config.load_from=<logdir>` to `extra_configs`.
- Don't pass `--config.load_from` manually to `staging.sh` — it causes `flag defined twice` error.

## New Feature: load_backbone_from (2026-05-09)
- Config field `load_backbone_from` in default.py
- In train.py: after state creation, if `load_backbone_from != ''`, calls `ckpt_util.load_backbone_params()`
- `ckpt_util.load_backbone_params()`: loads raw checkpoint, copies all params except `fc`/`diffusion_head` into Phase 2 state, fresh optimizer state (step=0)
- Run I config: `configs/remote_run_I_config.yml` — uses Phase 1 Run 3 logdir as `load_backbone_from`
