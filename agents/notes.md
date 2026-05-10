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
- **AdamW b2 policy**: keep `adamw_b2=0.95` as the project default for stability (avoid 0.999 spike issues).
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

## Phase 2 Progress Summary (as of 2026-05-09 22:15)

| Run | Architecture | TPU | Window | Status | ep=0 eval | notes |
|-----|-------------|-----|--------|--------|-----------|-------|
| A | attention head, mixup BUG | c8umw4 (us-east5-b) | 6352 | ❌ DEAD | 0.132% | Preempted; TPU deleted |
| C | attention head, uniform | yq00yh (us-east5-b) | 6363 | ❌ DEAD | 0.090% | Preempted; TPU deleted |
| D | attention head, logit-normal | 8507kk (us-east5-b) | 6364 | ❌ DEAD | 0.130% | Preempted; TPU deleted |
| E | attention head, zero-init | axuxm0 | 6382 | ✅ RUNNING | 0.100% | ep~20; ep=19=**0.186%**/0.212%iter — near random, diffusion warmup slow |
| F | MLP head baseline | 3djlis | 6375 | ✅ RUNNING | 0.206% | ep~48; ep=19=**5.58%**/9.67%; ep=39=**14.90%**/21.23% — MLP scales well early |
| G | large head (512-dim, 4L) | 06q7u9 | 6383 | ✅ RUNNING | 0.108% | ep~20; ep=19=**0.162%**/0.238%iter — large head slower than baseline at ep=19 |
| H | attention + aux CE (λ=0.1) | qxxa8y | 6381 | ✅ RUNNING | 0.100% | ep~35; ep=19=**23.83%**/28.00%iter 🚀 aux CE gives massive boost |
| I | pretrained backbone + diff head | j3rqvs | 6388 | ✅ RUNNING | 0.184%/invalid=49.8% | ep~28; ep=19=**47.08%**/50.80%iter 🚀🚀 BEST — warm backbone dominates everything |
| sanity | ViT_base_v3 (full align, CE) | favaxa | 6380 | ✅ RUNNING | 0.096% CE | ep~35; train_acc=37%, loss=4.41 — CE training progressing |

**Phase 1 COMPLETE**: Run 3 (biases+LS) = **73.14%** BEST, Run 1 = 71.96%, Run 2 = 65.96%.
**Run A/C/D**: TPUs deleted (us-east5-b spot wave). No auto-resume possible; abandoned.
**Sanity run**: ViT_base_v3 (CE) with full DeiT-B alignment — target 81.8%.
**All 6 Phase 2 + sanity slots active** as of 22:15.

## Critical Bugs Found and Fixed (2026-05-09, this session)

- **Run E/G/I were all launched with wrong MLP config** (previous agent used Run F config for all). Fixed by killing wrong runs and relaunching with correct configs (commits 537e610, 37daeb8, 2f57c6c, 48514a4, 05264ce).
- **`fc` not initialized during model.init()**: When `head_aux_ce=True`, `self.fc` was only called when `return_aux_ce=True`. During `model.init()` (default `return_aux_ce=False`), fc was never called, so params never included fc/kernel. Fix: always call `self.fc(cls)` when `head_aux_ce=True`, gate the return (not the call) on `return_aux_ce` (commit 537e610).
- **`load_backbone_params` repeated failures**: Phase 1 Run 3 checkpoint has `final_ln: ['scale']` (no bias, despite `use_ln_bias=True` — trained before the fix), and `embedding: {'_model': ...}` nesting. Fixed with recursive copy that preserves Phase 2 param tree structure (commit 2f57c6c). Additionally: `state.params` is plain dict (not FrozenDict); using `freeze()` caused node type mismatches throughout (commit 05264ce).

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

## 2026-05-10 Planning Update (Takeover)

- Remaining time: **4 days**; single full run takes ~1 day+, so prioritize high-ROI changes only.
- New P0 direction (user-proposed): **multi-mask per image with CLS reuse**.
  - For each image, run backbone once to get CLS.
  - Sample K different masks (`n_masks_per_image=K`), repeat CLS K times, train diffusion head on all K masks.
  - Goal: increase diffusion supervision density ~Kx with limited overhead (mainly head compute, not backbone).
- Priority plan:
  1. Implement `n_masks_per_image` in diffusion train step (start K=2/4).
  2. Mainline experiment: warm-start backbone + aux CE + multi-mask.
  3. Keep uniform mask schedule default; deprioritize large-head/logit-normal expansions.


## 2026-05-10 Re-Plan v2 (per user feedback)

- User decision:
  - Keep **pure diffusion** direction; do **not** prioritize aux CE multitask path ("投敌").
  - `n_masks_per_image` target starts at **10** (maximize if memory allows).
  - Also try reducing augmentation/regularization because diffusion convergence is slow.

### Clarifications
- `warm-start backbone` = initialize Phase-2 backbone weights from a finished Phase-1 CE checkpoint (`load_backbone_from`), while keeping diffusion head randomly initialized.
- `multitask` here means optimizing two losses together on shared backbone: diffusion bit loss + CE loss. This equals the `head_aux_ce` branch and is deprioritized now.

### Memory/Compute rough estimate for multi-mask
- For ViT-B/16 + 2-layer diffusion head (256-dim):
  - rough activation-element ratio: `backbone : head ≈ 437 : 1`
  - rough FLOPs ratio: `backbone : head ≈ 1000 : 1`
- Implication:
  - `K=10` adds ~`10/437 ≈ 2.3%` activation-equivalent overhead (rough)
  - head compute still far below backbone compute
  - practical bottleneck may come from framework buffers / sharding, not head math itself
- Plan: set `n_masks_per_image=10` first; if OOM then fallback to `K=6` then `K=4`.

### New priority (4 days)
1. Implement and run `K=10` multi-mask with CLS reuse (encode once, train head on 10 masks/image).
2. Mainline = warm-start backbone + pure diffusion + multi-mask (no aux CE).
3. Regularization/augmentation down-tuning run:
   - `dataset.use_rand_augment: false`
   - `dataset.reprob: 0.0`
   - `dataset.repeated_aug: 1`
   - `weight_decay: 0.02` (or 0.0 as aggressive variant)
   - `stochastic_depth_rate: 0.05` (or 0.0 as aggressive variant)
   - note: with pure diffusion loss, `dataset.label_smoothing` has near-zero effect because labels are converted by argmax.


## 2026-05-10 Execution Update (after user clarification)

- **Sanity run (window 6380) check**:
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_212142_mxfpdz_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval`
  - Active training confirmed at `2026-05-10 01:57 UTC`, around `ep=43.56` (not stalled, no manual resume needed now).
  - Latest evals in log: `ep19=38.77%`, `ep39=54.47%`.

- User direction update:
  - Do **not** prioritize warm-start/aux-CE mainline (considered "投敌").
  - Highest-priority ablation remains `n_masks_per_image=10` with CLS reuse.
  - Second ablation: further reduce augmentation/regularization on top of no-mixup baseline.
  - Add technical ablation: diffusion loss normalization variant (`token_mean` vs `sample_mean`).
  - Final-stage eval-only trick: multi-sample majority vote decoding.

- Loss normalization note:
  - Current implementation uses masked-token mean:
    `sum(mask * CE) / sum(mask)`.
  - This implicitly gives higher weight to higher mask-ratio samples.
  - Keep as default for stability; add `sample_mean` as explicit ablation option.

- Config updates applied locally:
  - Unified diffusion config `adamw_b2` to `0.95` in `default.py` and all `remote_run*.yml` variants.
  - Added low-aug configs for ablation-2:
    - `configs/remote_run_J_low_aug_config.yml` (mild)
    - `configs/remote_run_K_low_aug_aggressive_config.yml` (aggressive)
