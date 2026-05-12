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

## 2026-05-10 New Architecture Directions (User-Decided)

### Direction A: Tiny-Bit MLP Head (pure CLS decoding)

- Motivation:
  - For classification, decisive information should mainly be in CLS (768-d).
  - Bit token embedding does not need to be large (`head_inner_dim=256` may be overkill for bit side).
- Design intent:
  - Keep a **small bit embedding** (target: <=10 dims, e.g. 8 or 10).
  - Keep **large MLP capacity** for decoding CLS to 1000-way semantics (e.g. 768 -> 3072 scale).
- High-level structure:
  1. Encode image once with backbone, get CLS (768-d).
  2. Bit side uses compact embeddings for {0,1,MASK} + positional bits (`n_bits=10`, bit emb dim ~8-16).
  3. Concatenate CLS-conditioned vector with compact bit context.
  4. Use large MLP to output `(n_bits, 2)` logits.
- Key hypothesis:
  - Performance bottleneck is decoder capacity over CLS semantics, not bit token embedding size.

### Direction B: Cross-Attention Diffusion Head over Patch Tokens

- Motivation:
  - Current attention head only takes CLS + bit tokens; may discard fine-grained patch evidence.
  - For hard classes, decoding bits may benefit from direct patch-token access.
- Design intent:
  - Head consumes **all encoder tokens** (CLS + patches) as memory.
  - Bit queries perform **cross-attention** to token memory.
- High-level structure:
  1. Backbone outputs token sequence (not CLS-only path for this head).
  2. Build bit query tokens (MASK/0/1 + bit position), typically in 768-d to match token dim.
  3. Stack ~4 layers of cross-attention (bit queries attend to patch-token memory).
  4. Project query states to per-bit binary logits `(n_bits, 2)`.
- Key hypothesis:
  - Direct access to patch tokens improves ambiguous class separation vs CLS-only decoding.

## 2026-05-10 Loop #1 Execution Log

- Full notes re-read: done.
- Cleaned obsolete DeiT windows via `tmux kill-window`:
  - removed: 6323, 6352, 6363, 6364, 6375, 6379
  - remaining active DeiT windows: 6380, 6381, 6382, 6383, 6388
- Current active count: 5 jobs (<= 8 limit).
- Current progress snapshot:
  - 6380 sanity-run: running, ep~46.9
  - 6381 Run H (aux CE): running, ep~46.3
  - 6382 Run E (zero-init): running, ep~44
  - 6383 Run G (large head): running, ep~44
  - 6388 Run I (warm-start): running, ep~39

## Priority Order (starting now)

1. Keep sanity run (6380) uninterrupted until baseline target check.
2. Implement `n_masks_per_image=10` (with fallback K=6/4 if OOM) as top ablation.
3. Implement Direction A (tiny-bit MLP + large CLS decoder capacity).
4. Implement Direction B (cross-attention over full patch tokens, ~4 layers).
5. Launch low-aug/low-reg ablations after 2/3 are runnable.
6. Final stage: eval-only majority vote decoding trick.
- Loop cadence: every 30 minutes. After this cycle at 2026-05-10 02:19 UTC, next check target is ~2026-05-10 02:49 UTC.

## 2026-05-10 Loop #2 (Implementation Start)

- Implemented `n_masks_per_image` in code:
  - `models.ViT.decode_diffusion_from_cls()` added.
  - `train_step_diffusion` now supports encode-once + K-mask decode (`n_masks_per_image`).
  - Config field added: `n_masks_per_image` (default=1).
  - p_train_step wiring updated to read `config.n_masks_per_image`.
- Prepared run config:
  - `configs/remote_run_L_mask10_config.yml` (pure diffusion, K=10, wandb notes includes `deit`).
- Launch attempts:
  - tried `tpu.py run` on `p1u4mx` and `cz2ivo` with `config=configs/load_config.py:remote_run --auto` after copying L config to `remote_run_config.yml`.
  - both attempts stall after `kill_jobs_tpu: Sending C-c ... No main.py processes found.` and no new DeiT window appears yet.
- Next loop action:
  - retry launch on another eligible card (prefer v6e-16/v5p-32, fallback smaller) and verify window creation in `tpu.py check sqa`.


## 2026-05-10 Loop #2 Launch Summary (after user 'stop launch first' instruction)

- Implemented user-proposed architecture forms BEFORE launching:
  1. MLP form:
     - tiny bit embedding (`head_bit_dim`, default=8)
     - large CLS decoder MLP (`head_mlp_hidden_dim`, default=3072)
  2. Attention form:
     - cross-attention head over full token memory (`encode_tokens`)
     - bit queries cross-attend to CLS+patch tokens
- `n_masks_per_image` path kept and integrated with both head types.

### Launch outcomes
- Run L (cross-attn + K10) launched on p1u4mx → window 6389.
- Run M (tiny-bit MLP + K10) launched on cz2ivo → window 6390.
- Run N (tiny-bit MLP + K10 + low-aug mild) launched on 3djlis → window 6392.
- Cleaned duplicate killed window 6391.
- Current active DeiT count reached 8/8.

### TPU policy note
- User clarified: prefer v6e-16 / v5p-32; if unavailable, smaller cards are allowed.
- This round used available registered v6e-8 cards to avoid idle time and keep throughput high.


## 2026-05-10 Persistent 30-min Loop Process

- Started persistent loop script: `agents/deit_auto_loop.sh`
- Launch mode: detached session (`setsid -f bash agents/deit_auto_loop.sh`)
- Current process (at setup time): `bash agents/deit_auto_loop.sh` (check by `pgrep -af deit_auto_loop.sh`)
- Cadence: every 1800s (30 minutes)
- Each loop does:
  1. read `agents/notes.md`
  2. run `python .../tpu.py check sqa`
  3. append DeiT window snapshot to `agents/auto_loop_status.md`
- Logs:
  - heartbeat: `agents/auto_loop.log`
  - status snapshots: `agents/auto_loop_status.md`


## Loop Update (2026-05-10 11:37 UTC)

- 按要求本轮开始先完整重读 `agents/` 全部文件（含 `notes/results/idea/experiments/rethink/report/auto_loop*`）。
- 修复并清理异常窗口：
  - `6389`（Run L 旧链路）= SSH timeout failed。
  - `6398`（Run L resume 链路）= 不是模型 bug，而是启动命令里 `--config` 被传了两次，触发 `absl` 报错：`The flag 'config.load_from' is defined twice`。
  - 已 kill `6389` 与 `6398`，避免错误链条继续污染。
- 已在空闲且已挂载卡 `kmh-tpuvm-v6e-8-spot-gzy-z169mq` 直接重新 launch Run L，创建新窗口 `6400`（干净重启）。
- 当前 DeiT 窗口（`6380/6381/6382/6383/6388/6390/6392/6400`）共 8 个，其中 7 个明确 Running，`6400` 当前处于 early startup（日志到 env/data 准备阶段，`tpu check` 显示 Unknown）。
- 已修改 `agents/deit_auto_loop.sh`：每轮不只读 `notes.md`，而是读 `agents/` 下所有文件，满足“每轮全量记忆刷新”要求；并固定 `tpu.py check sqa user=sqa`。
- 重启了 auto loop：新进程 `bash agents/deit_auto_loop.sh` 已启动。

## Loop Update (2026-05-10 12:00 UTC) — z169mq reboot/mount fix

- 用户反馈正确：重启后 `z169mq` 未挂载 SSD。
- 实际错误链：
  - `6402` 报错 `cp: cannot stat .../staging/...`，确认是未 mount 导致。
  - 执行 `tpu.py mount-disk kmh-tpuvm-v6e-8-spot-gzy-z169mq`，完成环境恢复。
  - 重拉后 `6403` 新错误：`ModuleNotFoundError: No module named 'timm'`。
  - 在 `z169mq` 手动安装：`python3 -m pip install --user timm`。
  - 再次重拉创建 `6404`。
- `6404` 当前状态：`Unknown`（日志已进入 JAX/W&B 启动并到 worker startup 阶段，但尚未打印 epoch）。
- 其余 7 个 DeiT 任务持续 Running（6380/6381/6382/6383/6388/6390/6392）。

## Loop Update (2026-05-10 12:45 UTC) — Takeover

- 本轮开始已完整重读 `agents/` 下所有文件（notes/results/idea/experiments/rethink/report/status/log/script）。
- 当前 DeiT jobs 正好 8/8 占满：`6380, 6381, 6382, 6383, 6388, 6390, 6392, 6405`。最新 `tpu check` 显示 7 个 Running，`6405` 为 Unknown，但 output.log 仍在更新，所以本轮无需 resume。
- 本轮不需要 resume；也不应 launch 新 job，因为已达 DeiT 上限。
- 已把上个 agent 的全部主要工作整理成 HTML report：`agents/report_takeover_2026-05-10.html`。

### Latest Results Snapshot

- `6380` sanity CE: ep139 `63.356%`，仍低于 81.8% target 轨迹。
- `6381` Run H aux CE: ep139 `52.506%` / iter `54.816%`。
- `6382` Run E zero-init: ep139 `1.326%` / iter `3.172%`，低 ROI。
- `6383` Run G large old attention head: ep139 `2.952%` / iter `6.636%`，低 ROI。
- `6388` Run I warm-start: ep139 `58.986%` / iter `62.470%`，强 diagnostic，但非当前纯 diffusion 主线。
- `6390` Run M tiny-bit MLP K10: ep79 `51.186%` / iter `56.400%`，等待 ep99/119。
- `6392` Run N tiny-bit MLP K10 low-aug: ep119 `59.060%` / iter `62.800%`，当前最佳纯 diffusion。
- `6405` Run L cross-attn full tokens K10: clean relaunch，ep0 `0.100%` / iter `0.116%`，等待 ep19/39。

### Current Decision

- 主线继续围绕 pure diffusion 的 `tiny-bit MLP + K10 + lower augmentation/regularization`。
- 不新增 job；等待 M/N/L 下一个 eval。
- 如果需要腾 slot，优先考虑 E/G（已明显低 ROI），但不要自动 kill，除非用户明确允许或后续 policy 允许。
- 下一批候选：aggressive low-aug/low-reg、`sample_mean` loss normalization、K>10/K schedule、eval-only majority vote。

## Loop Policy Update (2026-05-10) — Manual Agent Loop Only

- User clarified: the required 30-minute loop must be performed by the agent itself, not by an autonomous shell script.
- `agents/deit_auto_loop.sh` is stopped; current process check shows no running `deit_auto_loop.sh` process.
- Do not rely on the script as trusted research automation.
- Future loops must be manual agent actions: re-read all `agents/` files, check TPU state, inspect logs, update `results.md`/`notes.md`, then decide whether to resume/launch/stop jobs.

## Manual Sleep Loop Contract (2026-05-10)

- User clarified that “loop” means the agent itself must manually run `sleep 1800` in the background/session and wake up after 30 minutes.
- Do not replace this with an autonomous checking script.
- When user says to loop: start `sleep 1800; echo "DEIT_AGENT_WAKE $(date -u '+%Y-%m-%d %H:%M:%S UTC')"`, then after it returns, personally execute the full research cycle:
  1. read all files under `agents/`
  2. run TPU status check
  3. inspect relevant logs
  4. update `results.md` and `notes.md`
  5. analyze results and decide whether to resume/launch/stop jobs
  6. if continuing, start the next `sleep 1800`
- Current manual timer session at time of writing: `87996`.

## Manual Loop Update (2026-05-10 14:35 UTC)

### Wake mechanism correction

- Previous manual timer session `87996` completed at `2026-05-10 13:50:47 UTC`, but because the agent had already sent `final`, the tool completion was not surfaced automatically.
- Correct method: when user asks for a loop, start `sleep 1800` and keep the assistant turn open until the sleep command returns. Do not send `final` before the sleep completes.
- This is the only reliable in-agent wake mechanism available in this environment. A detached shell script can wake itself, but it cannot make the agent reason or send a message.

### This loop

- Full `agents/` re-read completed: `/tmp/deit_agents_full_read_20260510_143149.txt`.
- `tpu.py check sqa user=sqa` run at `2026-05-10 14:31 UTC`.
- `6381` is Error, but log indicates SSH connection closed/refused/timed out after active training at ep~164.5; no Python traceback. Treat as preemption/network failure and do not touch resume logic.
- Active/running DeiT windows: `6380, 6382, 6383, 6388, 6390, 6392, 6405`.
- Error/preemption-like: `6381`.

Latest evals:
- `6380` sanity CE: ep159 `65.530%`.
- `6382` Run E zero-init: ep159 `2.384%` / iter `4.706%`.
- `6383` Run G large attention: ep159 `4.614%` / iter `9.764%`.
- `6388` Run I warm-start: ep159 `61.020%` / iter `63.754%`.
- `6390` Run M tiny-bit MLP K10: ep99 `52.132%` / iter `56.992%`.
- `6392` Run N tiny-bit MLP K10 low-aug: ep139 `59.908%` / iter `63.410%`.
- `6405` Run L cross-attn K10: ep19 `19.220%` / iter `29.756%`.

Decision:
- No launch now.
- Do not manually resume `6381`; preemption-like, let MONITOR handle.
- Run N remains best pure-diffusion mainline.
- Wait for Run L ep39 before deciding whether cross-attn deserves continuation.

## Manual Loop Update (2026-05-10 15:08 UTC)

- Woke correctly from in-turn `sleep 1800` at `2026-05-10 15:03:01 UTC`; this confirms the reliable wake mechanism is to keep the assistant turn open and poll the same sleep session until completion.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_150315.txt`.
- Current DeiT status: 7 running + `6381` Error/preemption-like.
- `6381` latest eval before failure: ep159 `53.206%` / iter `55.122%`; failure is SSH closed/refused/timed out, no code traceback.
- Run M ep119: `52.990%` / iter `57.468%`, plateauing and behind Run N.
- Run N ep139: `59.908%` / iter `63.410%`, remains best pure diffusion.
- Run L ep19: `19.220%` / iter `29.756%`; cross-attn learns but needs ep39 before decision.

### Run O queued locally

- Created `configs/remote_run_O_mlp_k10_lowaug_aggressive_config.yml` and copied it to `configs/remote_run_config.yml`.
- Design: tiny-bit MLP + K10 + aggressive low-aug/low-reg (`wd=0`, `sd=0`, no RA/reprob, repeated_aug=1).
- Launch attempt on `kmh-tpuvm-v6e-8-spot-gzy-vtcoc1` failed because tpu manager could not find that TPU in its known zone/sheet data.
- Do not force registration. Launch Run O later when a manager-visible registered slot is available.

## Manual Loop Update (2026-05-10 15:38 UTC)

- Woke correctly from second in-turn `sleep 1800` at `2026-05-10 15:36:06 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_153613.txt`.
- TPU status: 7 DeiT windows Running (`6380, 6382, 6383, 6388, 6390, 6392, 6405`), `6381` still Error/preemption-like.
- No new eval checkpoint yet for Run N or Run L since previous loop.
- Run N is around ep157; wait for ep159 eval.
- Run L is around ep32; wait for ep39 eval.
- No launch: Run O config is ready locally, but last launch failed because candidate TPU was not manager-visible. Do not force registration or random retry.

## Manual Loop Update (2026-05-10 16:10 UTC)

- Woke correctly from in-turn sleep at `2026-05-10 16:07:00 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_160714.txt`.
- TPU status: 7 DeiT running (`6380, 6382, 6383, 6388, 6390, 6392, 6405`), `6381` still Error/preemption-like.
- Run N ep159 arrived: `59.974%` / iter `63.478%`, loss `0.4401`; almost flat vs ep139 `59.908%` / iter `63.410%`.
- Run N appears to be plateauing around 60% single-step / 63.5% iterative.
- Run I warm-start ep159: `61.020%` / iter `63.754%`; still slightly ahead but not pure-from-scratch mainline.
- Run L has not reached ep39 eval yet; tpu check shows ep~37. Need next loop to evaluate cross-attn.
- No launch. Run O remains queued locally; last launch attempt failed because candidate TPU was not manager-visible.

## User-Directed Update (2026-05-10 16:45 UTC) — Report + K40 Launches

- User reviewed progress and correctly identified Run N/K10 as the strongest pure-diffusion direction so far.
- Wrote ImageNet-22K collaborator handoff report:
  - `agents/report_imagenet22k_mdh_handoff_2026-05-10.html`
  - Includes architecture, code-level implementation details, ImageNet-22K bit/class changes, K memory estimates, recommended 22K configs, and invalid-code handling.
- Answer to user question on E/G:
  - E/G were old attention-head experiments, not the successful tiny-bit MLP K10 path.
  - E: `ViT_base_mdh_zero_init`, old CLS+bit-token attention head, `inner_dim=256`, `head_n_layers=2`, `head_n_heads=4`, zero-init final projection, K=1, full aug/reg.
  - G: `ViT_base_mdh_large`, old CLS+bit-token attention head, `inner_dim=512`, `head_n_layers=4`, `head_n_heads=8`, K=1, full aug/reg.
  - Main reasons for low acc: K=1 sparse supervision, full augmentation/regularization, weaker old attention-head inductive bias, and E's zero-init slowing early learning.
- E/G have been killed/freeing their slots:
  - old E window `6382` on `axuxm0` no longer active.
  - old G window `6383` on `06q7u9` no longer active.
- Run L ep39 arrived:
  - `39.676%` single-step / `49.884%` iterative, loss `0.43349`, invalid `1.978%`.
  - It learns, but at ep39 it trails Run N at ep39 (`49.738%` / `55.010%`), so cross-attn is lower priority than MLP/K scaling.
- K scaling decision:
  - Rough memory estimate supports K40 safely; K40 mostly scales the MLP head because backbone encode is reused.
  - Launched K40 before K80 to test stability and signal.
- New local configs:
  - `configs/remote_run_P_mlp_k40_lowaug_mild_config.yml`
  - `configs/remote_run_Q_mlp_k40_lowaug_aggressive_config.yml`
- Run P launched:
  - window `6409`, TPU `kmh-tpuvm-v6e-8-spot-gzy-axuxm0`.
  - design: tiny-bit MLP, K40, low-aug mild (`wd=0.02`, `sd=0.05`).
  - logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_163819_kdc043_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`
  - status at check: parameter init complete, first batch ready, initial compilation; no immediate OOM.
- Run Q launched:
  - window `6410`, TPU `kmh-tpuvm-v6e-8-spot-gzy-06q7u9`.
  - design: tiny-bit MLP, K40, low-aug aggressive (`wd=0.0`, `sd=0.0`).
  - logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_164147_hf92wd_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`
  - status at check: process created and reading config/W&B setup; not yet past first batch at 16:43 UTC.
- `configs/remote_run_config.yml` currently contains Run Q because it was the last launched config.

Next loop priorities:
1. Check whether P/Q compile and reach train step 100; if either OOMs, reduce K to 20 or try gradient/checkpointing only if necessary.
2. Record Run M ep139 and Run N ep179 if available.
3. Keep Run L only if later eval narrows the gap; otherwise MLP/K40 variants have priority.
4. Continue manual in-turn sleep loop; do not use `agents/deit_auto_loop.sh` as trusted automation.

### Immediate correction (2026-05-10 16:46 UTC)
- Run Q also reached `First batch ready` at `16:43:26 UTC`; like Run P, it has no immediate OOM before initial compilation.

## User Question Analysis (2026-05-10 16:58 UTC) — Why E/G Were So Bad + REPA MLP Read

### E/G architecture clarification
- E/G are NOT the user-proposed MLP architecture.
- E/G config files explicitly use `head_type: attention`:
  - `configs/remote_run_E_config.yml`
  - `configs/remote_run_G_config.yml`
- The user-proposed MLP path is M/N/P/Q with `head_type: mlp`.
- E/G were launched before the later full-token cross-attention rewrite; they correspond to the old attention-head family, not the current MLP family.

### E architecture
- Model: `ViT_base_mdh_zero_init`
- Head family: old attention diffusion head
- `head_inner_dim=256`, `head_n_layers=2`, `head_n_heads=4`
- `head_zero_init_proj=true`
- `n_masks_per_image=1`
- Full augmentation/regularization: RandAugment on, reprob 0.25, repeated_aug 3, wd 0.05, stochastic depth 0.1
- Latest before kill: ep159 `2.384%` / iter `4.706%`

### G architecture
- Model: `ViT_base_mdh_large`
- Head family: old attention diffusion head
- `head_inner_dim=512`, `head_n_layers=4`, `head_n_heads=8`
- `n_masks_per_image=1`
- Full augmentation/regularization: RandAugment on, reprob 0.25, repeated_aug 3, wd 0.05, stochastic depth 0.1
- Latest before kill: ep159 `4.614%` / iter `9.764%`

### Why they were worse than the earlier fixed MDH K=1 baseline
- Earlier fixed Run C K=1 baseline reached ep119 `33.276%` / iter `39.138%`, so E/G being worse is not just because K=1.
- E-specific likely cause: zero-init final bit projection blocks gradient flow into the head/backbone at the very beginning; initially only the final projection receives useful gradients. For a sparse exact-bit objective this can trap/sluggishly bootstrap the run.
- G-specific likely cause: larger/deeper old attention head did not solve the supervision bottleneck; with K=1 each image supplies one random mask view, and a deeper bit-token transformer adds optimization burden/attention dilution instead of cleaner CLS-to-bits supervision.
- Shared cause: full DeiT augmentation/regularization appears harmful for diffusion supervision. Run N vs M already shows low aug beats normal aug by a large margin.
- Shared architecture issue: old attention head spends capacity modeling interactions among only 10 bit tokens; the successful MLP path uses the semantic CLS vector directly and treats bit conditioning as a small side input.
- Metric amplification: exact 10-bit sequence accuracy collapses if per-bit error is moderate; poor bit calibration in E/G looks much worse in exact-match acc.

### REPA reading
- Paper downloaded to `../readings/repa_2410.06940.pdf` from arXiv `2410.06940`.
- GitHub cloned temporarily to `/tmp/REPA` from `https://github.com/sihyun-yu/REPA`.
- REPA paper: aligns projections of noisy diffusion hidden states with clean representations from pretrained visual encoders; reports major training speedups.
- REPA code projector in `/tmp/REPA/models/sit.py` and `/tmp/REPA/models/mmdit.py` is exactly a plain 3-linear MLP:
  - `Linear(hidden_size, projector_dim)`
  - `SiLU()`
  - `Linear(projector_dim, projector_dim)`
  - `SiLU()`
  - `Linear(projector_dim, z_dim)`
- No LayerNorm inside the projector MLP.
- REPA projection loss normalizes both projected features and target features with L2 normalization before negative cosine similarity.
- Implication for our MLP head:
  - Adding LayerNorm inside the MLP is not directly supported by REPA's projector design.
  - More faithful REPA-inspired ablation is: switch MLP activations from GELU to SiLU and keep a 3-linear no-LN projector/head.
  - If we add normalization, prefer output/logit-side calibration or feature L2/cosine auxiliary loss as a separate ablation, not arbitrary LayerNorm between MLP layers.

### Current TPU status snapshot
- P/K40 mild (`6409`) is Running at ep~2; ep0 `0.078%` / iter `0.098%`, invalid `2.176%`.
- Q/K40 aggressive (`6410`) is Running at ep~1.5; ep0 `0.082%` / iter `0.096%`, invalid `2.200%`.
- Both K40 runs compiled and reached eval0/training, so K40 does not immediately OOM.
- M is around ep139 but ep139 eval not printed yet in inspected log.

Next design candidate after P/Q first meaningful eval:
- `MLP-REPA-style`: same as Run N/P but use SiLU activations, 3 Linear layers, no internal LayerNorm; optionally compare against current GELU head.

## Report Constraint Update (2026-05-10 17:03 UTC)

- User asked to emphasize in the ImageNet-22K report: do not "投敌".
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` with explicit hard constraints:
  - do not warm-start from CE classifier checkpoints,
  - do not use `load_backbone_from`,
  - do not enable `head_aux_ce`,
  - do not add auxiliary `num_classes` CE loss,
  - do not rely on a standard class-logit head.
- Report now labels Run I (warm-start) and Run H (aux CE) as diagnostic-only and says not to copy them for ImageNet-22K reproduction.
- The collaborator should reproduce pure masked-diffusion from scratch using the Run N/P/Q family, not CE-assisted shortcuts.

## Manual Loop Update (2026-05-10 17:39 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 17:32:59 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_173317.txt`.
- Updated ImageNet-22K report earlier this loop to emphasize no CE shortcut / no "投敌": no warm start, no `load_backbone_from`, no `head_aux_ce`, no auxiliary CE loss, no standard class-logit head training.

### New evals/status
- Run M ep139: `52.826%` / iter `57.356%`, loss `0.46864`; normal augmentation path is flat/declining and clearly behind Run N.
- Run N ep179: `60.576%` / iter `64.094%`, loss `0.45628`; still improving slowly after apparent ep159 plateau.
- Run L latest remains ep39: `39.676%` / iter `49.884%`; trails MLP path.
- Run P K40 mild (`6409`) is Running around ep10; eval0 `0.078%` / iter `0.098%`, invalid `2.176%`.
- Run Q K40 aggressive (`6410`) is Running around ep9.7; eval0 `0.082%` / iter `0.096%`, invalid `2.200%`.
- K40 is confirmed to compile/run through startup on both P and Q.

### REPA-style MLP ablation prepared
- Added configurable MLP activation:
  - `models.MLPDiffusionHead.activation`, supports `gelu` (default) and `silu`.
  - `ViT.head_mlp_activation` field.
  - `train.py` passes `head_mlp_activation` from config.
  - `configs/default.py` default `head_mlp_activation='gelu'`.
- Local init/apply smoke test passed for `ViT_base_mdh_mlp(head_mlp_activation='silu')`, output shape `(1, 10, 2)`.
- Created `configs/remote_run_R_mlp_k40_lowaug_silu_config.yml`:
  - pure masked diffusion, no CE shortcut,
  - K40 low-aug mild like Run P,
  - only architecture change: REPA-style SiLU MLP activation, no LayerNorm.
- Launch attempt for Run R on `kmh-tpuvm-v6e-8-spot-gzy-qxxa8y` failed because TPU manager reports `unknown state deleted`; no window created.
- `configs/remote_run_config.yml` currently contains Run R because it was copied before the failed launch.

### Decision
- Do not launch R on random/unregistered TPUs.
- Wait for P/Q ep19 before judging K40 and wd/sd effect.
- If a manager-visible slot opens, launch Run R as the next pure no-CE ablation.

## Manual Loop Update (2026-05-10 18:08 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 18:05:58 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_180615.txt`.

### TPU status
- DeiT running: `6380`, `6388`, `6390`, `6392`, `6405`, `6409`, `6410`.
- Error/stale: `6381` (Run H aux CE) remains Error; underlying `qxxa8y` launch attempt already showed TPU state `deleted`, so do not use it for Run R.
- No manager-visible empty slot identified for Run R this loop.

### Key results/status
- Run P K40 mild (`6409`): Running ep~17; no ep19 eval yet. Startup eval0 remains `0.078%` / iter `0.098%`.
- Run Q K40 aggressive (`6410`): Running ep~16.5; no ep19 eval yet. Startup eval0 remains `0.082%` / iter `0.096%`.
- Run N latest remains ep179 `60.576%` / iter `64.094%`; next eval ep199 later.
- Run L latest remains ep39 `39.676%` / iter `49.884%`; next eval ep59 later.
- Run M latest remains ep139 `52.826%` / iter `57.356%`; normal aug remains weak.

### Decision
- No new launch this loop.
- Wait one more 30-min cycle for P/Q ep19 results. These will decide whether K40 is better than Run N K10 at matched early epoch and whether aggressive wd0/sd0 helps.
- Keep Run R queued locally; launch only when a manager-visible TPU is available.

## Manual Loop Update (2026-05-10 18:40 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 18:36:42 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_183704.txt`.

### Key new evals
- Run P K40 mild ep19: `41.366%` / iter `47.526%`, loss `0.43401`, invalid `1.486%`.
- Run Q K40 aggressive ep19: `41.540%` / iter `47.992%`, loss `0.43054`, invalid `1.542%`.
- Run N K10 low-aug ep19 baseline for comparison: `38.328%` / iter `44.502%`.
- Interpretation: K40 gives a clear early gain of about +3.0 to +3.5 points over K10 at matched ep19. Q is slightly ahead of P, so wd0/sd0 is not hurting early and may help slightly.
- Run L cross-attn ep59: `47.802%` / iter `55.968%`, loss `0.40572`, invalid `0.766%`.
  - At ep59 it still trails Run N ep59 (`54.634%` / iter `59.000%`), so full-token cross-attn remains lower priority than MLP/K scaling.
- Run I warm-start ep199: `64.028%` / iter `66.462%`; diagnostic only and not allowed as main recipe because it is CE warm-start.

### Report update
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` with P/Q ep19 results and the conclusion that K40 is promising.

### Decision
- Continue P/Q until at least ep39; this is the next decisive checkpoint.
- Do not launch new job this loop. No manager-visible empty slot; `qxxa8y` remains unusable/deleted.
- If needing a slot after next eval, candidate to stop first is Run M after its ep159 eval, because normal augmentation is dominated by Run N/P/Q.
- Run R (SiLU/no-LN/K40) remains queued locally and should launch on the next real available slot.

## Manual Loop Action (2026-05-10 19:12 UTC) — Retire M, Launch Run R

- Run M normal-aug MLP K10 reached ep159: `54.266%` / iter `58.480%`, loss `0.46328`.
- This confirms the normal augmentation/reg path is dominated by low-aug MLP runs:
  - Run N ep159: `59.974%` / iter `63.478%`.
  - Run P/Q K40 already beat Run N at ep19.
- Stopped Run M / window `6390` on `cz2ivo` to free slot.
- Launched Run R on the freed manager-visible TPU `kmh-tpuvm-v6e-8-spot-gzy-cz2ivo`.
- Run R details:
  - window `6411`
  - config `configs/remote_run_R_mlp_k40_lowaug_silu_config.yml`
  - pure masked diffusion; no CE warm-start; no aux CE; no standard class-logit objective
  - K40, low-aug mild like Run P
  - REPA-style MLP activation: `head_mlp_activation: silu`, no LayerNorm
  - logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_190919_lm2dnh_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval`
- Run R reached `First batch ready` at `2026-05-10 19:10:26 UTC`; no immediate startup OOM.
- `configs/remote_run_config.yml` currently contains Run R.

Next loop priorities:
1. P/Q ep39 results.
2. Run R eval0 and startup compile status.
3. N ep199 if available.

## Manual Loop Update (2026-05-10 19:43 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 19:40:57 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_194115.txt`.

### TPU status
- Running pure/mainline diffusion: N (`6392`), L (`6405`), P (`6409`), Q (`6410`), R (`6411`).
- M was retired and is no longer active.
- H (`6381`) remains stale/error; ignore (aux CE, not mainline).
- I (`6388`) still running but diagnostic-only CE warm-start; not part of main recipe.

### New results/status
- Run R SiLU/no-LN/K40 eval0: `0.080%` / iter `0.094%`, loss `0.69092`, invalid `1.676%`; status running around ep6. Startup is healthy.
- Run N ep199: `60.792%` / iter `64.684%`, loss `0.46998`, invalid `0.544%`. N continues slow improvement but loss is rising.
- P/Q have not reached ep39 eval yet:
  - P status around ep37.
  - Q status around ep36.
- L no new eval beyond ep59 `47.802%` / iter `55.968%`.

### Decision
- No new launch. Current useful queue is full enough: P/Q/R are the key K40/activation ablations.
- Next loop should capture P/Q ep39, likely the most important checkpoint of the current stage.
- If P/Q ep39 maintain lead over N K10 ep39 (`49.738%` / `55.010%`), K40 should be treated as current best recommendation for ImageNet-22K.

## User-Directed Mixup/CutMix Run S (2026-05-10 20:05 UTC)

- User observed current low-aug runs converge faster early but may not grow as strongly late, and asked to test adding full baseline augmentation back with a diffusion-correct mixup/cutmix target.
- Implemented `diffusion_label_mode: soft_top2`:
  - default remains `argmax`, preserving previous no-mixup diffusion behavior.
  - for mixed labels, use `jax.lax.top_k(labels_soft, 2)`.
  - renormalize the top-2 weights and split the `K = n_masks_per_image` repeated diffusion mask views deterministically by weight.
  - example: K10 + 50/50 mix -> 5 views label A and 5 views label B; 0.8/0.2 -> 8/2.
  - this keeps image-level mixup/cutmix but avoids the old bug of supervising a mixed image with only the argmax label.
- Changed training loop so diffusion only calls `apply_mixup_cutmix_batch` when `diffusion_label_mode == 'soft_top2'`; old diffusion configs still skip mixup/cutmix.
- Local checks:
  - `python -m py_compile train.py models.py input_pipeline.py` passed.
  - smoke test of `make_diffusion_bits(..., K=10, soft_top2)` produced expected 5/5 and 8/2 allocations.

### Run S config
- File: `configs/remote_run_S_mlp_k10_fullaug_mixlabel_config.yml`
- Current launch config copied to `configs/remote_run_config.yml`.
- Architecture: `ViT_base_mdh_mlp`, tiny-bit MLP, `head_bit_dim=8`, `head_mlp_hidden_dim=3072`, `head_mlp_activation=gelu`, `head_n_layers=2`.
- Diffusion: `n_masks_per_image=10`, `mask_schedule=uniform`, `eval_iter_steps=4`, `diffusion_label_mode=soft_top2`.
- Aug/reg: full baseline-style augmentation/reg: RandAugment on, reprob 0.25, repeated_aug 3, mixup alpha 0.8, cutmix alpha 1.0, label_smoothing 0.1, wd 0.05, sd 0.1.
- No CE shortcut: `load_backbone_from=''`, `head_aux_ce=false`, `aux_ce_loss_weight=0.0`.

### Slot decision
- Stopped Run I/window `6388` on `j3rqvs` to free a slot. Run I is warm-start from CE backbone, diagnostic-only, and explicitly not part of the no-投敌 recipe.
- Launched Run S on `kmh-tpuvm-v6e-8-spot-gzy-j3rqvs`.
- Window: `6412`.
- Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_200200_2v7ro7_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`.
- Status at launch check: Staging/initial compilation; config loaded correctly with `diffusion_label_mode: soft_top2` and `use_mixup_cutmix: true`.

### New P/Q checkpoint before launch
- Run P K40 mild ep39: `51.626%` / iter `56.988%`, loss `0.38740`, invalid `0.998%`.
- Run Q K40 wd0/sd0 ep39: `50.968%` / iter `56.540%`, loss `0.40941`, invalid `0.888%`.
- P slightly overtook Q by ep39; mild regularization looks safer than fully aggressive by this checkpoint.
- K40 still beats Run N K10 at ep39 (`49.738%` / iter `55.010%`) but the margin shrank from ep19.

Next loop priorities:
1. Confirm Run S reaches first batch/eval0 without code error.
2. Continue P/Q/R to next checkpoints; P is currently the best K40 branch at ep39.
3. Watch whether Run S full augmentation with correct mixed labels closes the late-growth gap vs low-aug runs.

### Run S startup confirmation (2026-05-10 20:08 UTC)
- Run S reached `First batch ready` at `2026-05-10 20:03:41 UTC`.
- This confirms data loading, full augmentation config, and `diffusion_label_mode=soft_top2` path passed startup. Await initial compilation/eval0.
- Updated ImageNet-22K handoff report to note: default remains low-aug/no mixup, but if mixup/cutmix is enabled it must use K-view label allocation rather than argmax.

## Manual Loop Update (2026-05-10 20:37 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 20:34:21 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_203440.txt`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, P `6409`, Q `6410`, R `6411`, S `6412`.
- H `6381` remains Error/stale and is aux-CE diagnostic, not mainline.
- No mainline job requires resume.

### New results/status
- Run S full-aug + soft_top2 mixup/cutmix:
  - Initial compilation completed at `20:04:29 UTC`.
  - eval0: `0.074%` / iter `0.116%`, loss `0.69222`, invalid `0.056%`.
  - Status running around ep4.3; no code error so far.
- Run L cross-attn K10 ep79: `50.770%` / iter `58.048%`, loss `0.42392`, invalid `0.802%`.
  - Still behind MLP K10 low-aug Run N at ep79 (`56.896%` / iter `60.954%`). Cross-attn remains lower priority.
- Run P K40 mild latest remains ep39: `51.626%` / iter `56.988%`, loss `0.38740`.
- Run Q K40 wd0/sd0 latest remains ep39: `50.968%` / iter `56.540%`, loss `0.40941`.
- Run R SiLU/no-LN/K40 is running around ep17; no ep19 eval yet.
- Run N is running around ep217; latest eval remains ep199 `60.792%` / iter `64.684%`.
- Sanity CE ep219: `68.538%`.

### Interpretation
- Run S successfully validates the corrected augmentation code path at startup. The important checkpoint is ep19; eval0 is random as expected.
- K40 remains promising, but P/Q margin over K10 is smaller at ep39 than at ep19. Need ep59 before deciding whether K40 actually improves the trajectory or only front-loads learning.
- Mild regularization (P: wd0.02/sd0.05) is now better than fully aggressive low-reg (Q: wd0/sd0) by ep39.
- L is learning but still not competitive with MLP; do not allocate new slots to cross-attn variants.

### Decision
- No new kill/launch this loop.
- Keep S until ep19 to test the user's full-augmentation/mix-label hypothesis.
- Keep P/Q to ep59 and R to ep19.
- If a slot opens later, next candidate is K80 low-aug mild or K40 full-aug soft_top2 depending on S ep19.

## Manual Loop Update (2026-05-10 21:08 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 21:05:30 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_210548.txt`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, P `6409`, Q `6410`, R `6411`, S `6412`.
- H `6381` remains Error/stale and is aux-CE diagnostic; ignore for mainline.
- No mainline job needs resume.

### New results/status
- Run R SiLU/no-LN/K40 ep19: `42.032%` / iter `48.254%`, loss `0.42070`, invalid `1.206%`.
  - This is slightly ahead of P ep19 (`41.366%` / `47.526%`) and Q ep19 (`41.540%` / `47.992%`). SiLU/no-LN is promising early, but needs ep39 before conclusion.
- Run N K10 low-aug ep219: `61.480%` / iter `65.148%`, loss `0.45957`, invalid `0.492%`.
  - N still improves slowly late; low-aug K10 remains strong and stable.
- Run S full-aug soft_top2 is running around ep9.3; no ep19 yet.
- Run P K40 mild around ep54; next eval ep59 soon.
- Run Q K40 wd0/sd0 around ep53; next eval ep59 soon.
- Run L latest remains ep79 `50.770%` / iter `58.048%`, running around ep85.
- Sanity CE latest remains ep219 `68.538%`.

### Interpretation
- REPA-style SiLU has the best K40 ep19 so far, but the difference is small. Wait for R ep39.
- N's late improvement argues against killing it yet; it is still useful as the mature K10 baseline.
- S is the main test of the user's hypothesis about full baseline augmentation when mixed labels are handled correctly; wait ep19.

### Decision
- No kill/launch this loop.
- Next loop should likely capture P/Q ep59 and S ep19; R ep39 may be later.
- If P/Q ep59 plateau below N's matched/mature trajectory, favor K10 low-aug or SiLU over blindly scaling K. If P or R keeps improving, consider K80 low-aug mild when a slot opens.

## Manual Loop Action (2026-05-10 21:42 UTC) — Retire Q, Launch Run T K80

- Full `agents/` read completed earlier this loop: `/tmp/deit_agents_full_read_20260510_213653.txt`.

### New evals this loop
- Run P K40 mild ep59: `55.920%` / iter `60.550%`, loss `0.38654`, invalid `0.916%`.
- Run Q K40 wd0/sd0 ep59: `54.176%` / iter `58.888%`, loss `0.44758`, invalid `0.688%`.
- Run N K10 low-aug matched ep59: `54.634%` / iter `59.000%`.

### Interpretation
- P remains ahead of N at matched ep59 by about +1.29 single-step / +1.55 iter, but the K40 advantage has shrunk compared with ep19.
- Q is now dominated by P and slightly behind N at ep59. Aggressive low-reg (`wd=0`, `sd=0`) is not attractive.
- Since Q has answered its ablation and lost, it is the best slot to recycle.

### Action
- Stopped Run Q/window `6410` on `kmh-tpuvm-v6e-8-spot-gzy-06q7u9`.
- Created `configs/remote_run_T_mlp_k80_lowaug_mild_config.yml`.
- Copied Run T config to `configs/remote_run_config.yml` for launch.
- Launched Run T on `kmh-tpuvm-v6e-8-spot-gzy-06q7u9`.
- Run T details:
  - window `6415`
  - K80, low-aug mild like Run P (`wd=0.02`, `sd=0.05`, no RA, no reprob, repeated_aug=1)
  - GELU tiny-bit MLP, `head_bit_dim=8`, hidden 3072
  - pure masked diffusion, no CE warm-start, no aux CE, no mixup/cutmix
  - logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_213853_66dm3n_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`
- Startup status: remote script reached `config.batch_size: 1024`; awaiting first batch/compile/eval0.
- K80 memory rationale: compared with K40, only the repeated MLP head path doubles. Previous estimate for K80 head activations is around ~2.21 GiB global / ~276 MiB per 8-way device before backward/XLA overhead; K40 was stable, so K80 is a reasonable risk.

Next loop priorities:
1. Confirm Run T reaches first batch/eval0 or catch OOM early.
2. Capture Run S ep19 full-aug soft_top2 result.
3. Capture Run R ep39 and continue P to ep79 if still running.

### Run T startup confirmation (2026-05-10 21:43 UTC)
- Run T reached `First batch ready` at `2026-05-10 21:40:34 UTC`.
- K80 passed data/first-batch construction; no immediate OOM seen yet. Await initial compilation/eval0.

## Manual Loop Action (2026-05-10 22:16 UTC) — Retire S, Launch Run U K80+SiLU

- Woke from in-turn `sleep 1800` at `2026-05-10 22:11:06 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_221123.txt`.

### New results/status
- Run T K80 GELU low-aug:
  - Initial compilation completed at `21:41:21 UTC`.
  - eval0: `0.090%` / iter `0.124%`, loss `0.69085`, invalid `2.406%`.
  - K80 is confirmed to compile and run through eval0; no immediate OOM.
- Run S full-aug soft_top2 K10 ep19: `2.364%` / iter `5.054%`, loss `0.64609`, invalid `0.778%`.
- Comparison for S:
  - N low-aug K10 ep19: `38.328%` / iter `44.502%`.
  - M full-aug/no-mixup K10 ep19: `21.244%` / iter `28.400%`.
  - Therefore full baseline augmentation plus correctly allocated mixup/cutmix labels is far worse than both low-aug and normal-aug/no-mixup.
- Run N manager status briefly showed `Unknown` due TPU driver log permission noise, but the output log is actively updating at ep236; no resume needed.

### Interpretation
- The user's mixup/cutmix idea was implemented correctly, but the result is strongly negative at ep19. The problem is not just the old argmax label bug; mixed images/targets appear fundamentally too noisy or too hard for this bit-diffusion objective at this stage.
- Default recommendation remains no mixup/cutmix and low augmentation.
- K80 is now validated at startup via Run T; next question is whether K80 improves ep19/ep39 over K40.

### Action
- Stopped Run S/window `6412` after ep19 failure.
- Created `configs/remote_run_U_mlp_k80_lowaug_silu_config.yml`.
- Copied Run U config to `configs/remote_run_config.yml`.
- Launched Run U on `kmh-tpuvm-v6e-8-spot-gzy-j3rqvs`.
- Run U details:
  - window `6417`
  - K80, low-aug mild like Run T/P
  - REPA-style `head_mlp_activation=silu`, no LayerNorm
  - pure masked diffusion; no CE shortcut; no mixup/cutmix
  - logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_221310_8gp2lr_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`
- Startup status: remote script reached `config.batch_size: 1024`; awaiting first batch/compile/eval0.

### Decision
- No further launch this loop.
- Next loop priorities: confirm Run U first batch/eval0, capture Run R ep39, continue T toward ep19, and watch P ep79.

### Run U startup confirmation (2026-05-10 22:18 UTC)
- Run U reached `First batch ready` at `2026-05-10 22:14:50 UTC`.
- K80+SiLU passed data/first-batch construction; no immediate OOM seen yet. Await initial compilation/eval0.

## Manual Loop Update (2026-05-10 22:49 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 22:45:27 UTC`.
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260510_224546.txt`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, P `6409`, R `6411`, T `6415`, U `6417`.
- H `6381` remains stale/error aux-CE diagnostic.
- P briefly showed `Unknown` in manager due log parsing noise, but its output log has recent evals/training and no code error; no manual resume.

### New results/status
- Run U K80+SiLU low-aug:
  - Initial compilation completed at `22:15:38 UTC`.
  - eval0: `0.088%` / iter `0.108%`, loss `0.69092`, invalid `1.930%`.
  - K80+SiLU runs through eval0; no immediate OOM.
- Run R K40+SiLU ep39: `53.514%` / iter `58.272%`, loss `0.38476`, invalid `0.840%`.
  - Beats P K40+GELU ep39 (`51.626%` / `56.988%`) by +1.888 single-step / +1.284 iter.
  - This upgrades SiLU from tentative to strong activation candidate.
- Run N K10 low-aug ep239: `62.212%` / iter `65.986%`, loss `0.45503`, invalid `0.552%`.
  - N continues slow late improvement.
- Run L cross-attn K10 ep99: `51.808%` / iter `58.396%`, loss `0.44191`, invalid `1.194%`.
  - Still below MLP K10 N ep99 (`57.566%` / `61.624%`), so cross-attn remains low ROI.
- Sanity CE ep239: `69.900%`.
- Run T K80+GELU is running around ep13; no ep19 yet.
- Run P K40+GELU is running around ep68; next eval ep79.

### Interpretation
- SiLU/no-LN is now the best-supported MLP architecture tweak: R beats P at ep19 and ep39 with the same K40 low-aug mild recipe.
- K80 is technically viable for both GELU and SiLU through eval0. Need T/U ep19 to decide whether K80 helps or only adds cost.
- Full-token cross-attn L should not get more variants; keep only if no slot pressure, otherwise it is a stop candidate.
- Full augmentation with mixup/cutmix is already negative from S and should not be retried before a substantially different objective.

### Decision
- No new kill/launch this loop.
- Keep R to ep59, P to ep79, T/U to ep19, and N as mature K10 reference.
- If a slot is needed, L is the first candidate to retire.

## Manual Loop Action (2026-05-10 23:24 UTC) — Retire P, Launch Run V K10+SiLU

- Woke from compensation loop at `2026-05-10 23:19:25 UTC` after the previous sleep ended early.
- Full `agents/` read completed and saved: `/tmp/deit_agents_full_read_20260510_232320.txt`.

### TPU status
- Running DeiT jobs after this loop: sanity CE `6380`, N `6392`, L `6405`, R `6411`, T `6415`, U `6417`, V `6419`.
- H `6381` remains stale/error aux-CE diagnostic; ignore for mainline.
- P `6409` was intentionally stopped after ep79 to recycle the `axuxm0` slot.
- Cleaned up the local `tpu.py run ... user=sqa` wrapper left by the V launch. The `--auto` manager process was left untouched.

### New results/status
- Run T K80+GELU low-aug ep19: `42.050%` / iter `48.410%`, loss `0.42973`, invalid `1.670%`.
  - This only modestly beats P K40+GELU ep19 (`41.366%` / `47.526%`) and is essentially tied with R K40+SiLU ep19 (`42.032%` / `48.254%`). K80 is memory-feasible, but early benefit over K40 is not large yet.
- Run P K40+GELU low-aug ep79: `57.226%` / iter `61.538%`, loss `0.39962`, invalid `0.888%`.
  - Compared with N K10+GELU ep79 (`56.896%` / `60.954%`), K40's matched-epoch advantage has shrunk to only +0.330 single-step / +0.584 iter. P looked useful for front-loaded learning but was no longer worth a slot versus SiLU/K80 tests.
- Run V K10+SiLU low-aug launched on `kmh-tpuvm-v6e-8-spot-gzy-axuxm0`.
  - Window: `6419`.
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_232107_h3cbub_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`.
  - First batch ready at `23:22:51 UTC`; initial compilation completed at `23:23:38 UTC`.
  - Purpose: isolate activation effect at K10 by comparing to N (`GELU -> SiLU`, otherwise low-aug mild K10).
- Run U K80+SiLU is running around ep13; ep19 pending.
- Run R K40+SiLU is running around ep51; ep59 pending.
- Run N K10+GELU is running around ep251; latest eval remains ep239 `62.212%` / iter `65.986%`.
- Run L cross-attn is running around ep107; latest eval remains ep99 `51.808%` / iter `58.396%` and remains low ROI.

### Interpretation
- K scaling from 10 -> 40 clearly speeds early convergence, but the advantage mostly compresses by ep79 for GELU. Scaling K alone may be front-loading rather than improving the final trajectory.
- SiLU/no-LN remains the highest-ROI architecture change because R beats P at matched ep19 and ep39, while K80 GELU at ep19 does not beat R K40 SiLU.
- The clean next comparison is V vs N at K10 and U vs T/R at K80/K40.
- Full augmentation with corrected mixup/cutmix labels remains a negative result from S and should not be recommended.

### Report update
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` with N ep239, P ep79, T ep19, V launch/compile, the negative S conclusion, and revised ImageNet-22K recommendations.
- Current handoff recommendation: reproduce Run N first as the mature no-shortcut baseline, then run SiLU/no-LN K40 or K80 low-aug variants. Do not warm start, do not use aux CE, do not use mixup/cutmix as the first 22K recipe.

### Next loop priorities
1. Capture V eval0 and early status.
2. Capture U ep19 and R ep59 if available.
3. Continue T to ep39 before deciding whether K80 GELU is worth keeping.
4. If slot pressure appears, L remains the first retirement candidate.

## Manual Loop Update (2026-05-11 00:05 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-10 23:55:38 UTC`.
- Full `agents/` read completed and saved: `/tmp/deit_agents_full_read_20260510_235547.txt`.
- Also did short waits to catch imminent R/N eval boundaries; R ep59 landed at `2026-05-11 00:04:45 UTC`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, R `6411`, T `6415`, U `6417`, V `6419`.
- H `6381` remains stale/error aux-CE diagnostic.
- No job is stuck, errored, or needs resume.

### New results/status
- Run U K80+SiLU low-aug ep19: `42.366%` / iter `48.870%`, loss `0.41920`, invalid `1.490%`.
  - This is the best ep19 among K40/K80 variants, but only slightly above T K80+GELU ep19 (`42.050%` / `48.410%`) and R K40+SiLU ep19 (`42.032%` / `48.254%`).
- Run R K40+SiLU low-aug ep59: `56.122%` / iter `60.246%`, loss `0.40242`, invalid `0.634%`.
  - Compared with P K40+GELU ep59 (`55.920%` / `60.550%`), R is +0.202 single-step but -0.304 iterative. SiLU's strong ep39 advantage did not clearly expand by ep59.
  - Compared with N K10+GELU ep59 (`54.634%` / `59.000%`), R is still ahead at matched epoch.
- Run V K10+SiLU low-aug eval0: `0.094%` / iter `0.110%`, loss `0.69097`, invalid `1.992%`; running around ep8.
- Run T K80+GELU is running around ep29; next important checkpoint is ep39.
- Run N K10+GELU is running around ep259; latest logged eval remains ep239 `62.212%` / iter `65.986%`.
- Run L cross-attn is running around ep114; no new eval since ep99.

### Interpretation
- K80+SiLU (U) is directionally best early, but the improvement is small. Continue U to ep39 before concluding.
- K40+SiLU (R) remains useful but no longer obviously dominates K40+GELU by ep59, especially on iterative accuracy. Continue R to ep79 to see whether it tracks P's later compression or recovers.
- No evidence supports reintroducing full augmentation/mixup/cutmix; S remains strongly negative.
- No slot pressure right now. Do not stop T/U/R/V before the next key checkpoints.

### Decision
- No new kill/launch this loop.
- Continue R to ep79, T to ep39, U to ep39, V to ep19, and N as mature K10 baseline.
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` with U ep19 and R ep59.

## Manual Loop Update (2026-05-11 00:37 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-11 00:35:57 UTC`.
- Full `agents/` read completed and saved: `/tmp/deit_agents_full_read_20260511_003608.txt`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, R `6411`, T `6415`, U `6417`, V `6419`.
- H `6381` remains stale/error aux-CE diagnostic.
- T `6415` briefly reports `Unknown`, but the manager output includes fresh train logging around ep35.8 and the logdir has no traceback/OOM. Treat as manager/log-parser noise, not a resume condition.

### New results/status
- Run N K10+GELU low-aug ep259: `62.594%` / iter `66.404%`, loss `0.45679`, invalid `0.422%`.
  - N continues slow late improvement and remains the best mature no-shortcut baseline.
- Run T K80+GELU low-aug is around ep35.8; next eval ep39 should arrive before/near the next loop.
- Run U K80+SiLU low-aug is around ep28.4; next eval ep39 likely next loop.
- Run V K10+SiLU low-aug is around ep14.3; ep19 should arrive before/near the next loop.
- Run R K40+SiLU low-aug is around ep66; next eval ep79 likely next loop.
- Run L cross-attn is around ep119 but no new eval since ep99; still low ROI.
- Sanity CE is around ep259 but no new eval since ep239 in the checked log.

### Interpretation
- Mature K10 GELU is still not saturated; even after ep239 it gained +0.382 single-step / +0.418 iter by ep259.
- This reinforces keeping N as the 22K handoff baseline until SiLU/K80 variants prove a clear matched-late advantage, not just faster early convergence.
- No run needs intervention. Do not relaunch T based on manager `Unknown` unless logs actually stop or error.

### Decision
- No kill/launch/resume this loop.
- Updated the ImageNet-22K handoff report with N ep259 as the current mature best.

## Manual Loop Update (2026-05-11 01:08 UTC)

- Woke from in-turn `sleep 1800` at `2026-05-11 01:07:06 UTC`.
- Full `agents/` read completed and saved: `/tmp/deit_agents_full_read_20260511_010716.txt`.

### TPU status
- Running DeiT jobs: sanity CE `6380`, N `6392`, L `6405`, R `6411`, T `6415`, U `6417`, V `6419`.
- H `6381` remains stale/error aux-CE diagnostic.
- T is back to manager-visible Running; previous `Unknown` was parser noise.
- No job needs resume.

### New results/status
- Run T K80+GELU low-aug ep39: `52.628%` / iter `57.866%`, loss `0.39504`, invalid `0.878%`.
  - Beats P K40+GELU ep39 (`51.626%` / `56.988%`) by +1.002 single-step / +0.878 iter.
  - Still trails R K40+SiLU ep39 (`53.514%` / `58.272%`) by -0.886 single-step / -0.406 iter.
  - K80 helps GELU at ep39, but does not dominate the SiLU K40 recipe.
- Run V K10+SiLU low-aug ep19: `37.934%` / iter `44.542%`, loss `0.44154`, invalid `1.748%`.
  - Compared with N K10+GELU ep19 (`38.328%` / `44.502%`), V is slightly worse single-step and basically tied iterative. SiLU is not a free win at K10.
- Run L cross-attn K10 ep119: `53.488%` / iter `59.382%`, loss `0.44323`, invalid `0.674%`.
  - Still below N K10 MLP ep119 (`59.060%` / `62.800%`), so cross-attn remains low ROI.
- Sanity CE ep259: `71.024%`, loss `1.48117`.
- Run U K80+SiLU is running around ep35; ep39 pending.
- Run R K40+SiLU is running around ep72; ep79 pending.
- Run N K10+GELU is running around ep272; latest eval remains ep259 `62.594%` / iter `66.404%`.

### Interpretation
- SiLU's effect is conditional: it helped K40 early/mid, but K10 SiLU does not beat K10 GELU at ep19.
- K80 GELU improves over K40 GELU at ep39, but not over K40 SiLU. The critical pending result is U ep39 (K80+SiLU) to see whether K scaling and SiLU combine or interfere.
- V should probably be kept to ep39 for a fair activation-isolation read; do not kill yet unless slot pressure appears.
- L remains the first stop candidate if a slot is needed.

### Decision
- No kill/launch/resume this loop.
- Updated the ImageNet-22K handoff report with T ep39, V ep19, and a less aggressive SiLU recommendation.

## Manual Loop Action (2026-05-11 01:48 UTC) — Retire L, Launch Run W K160

- Woke from in-turn `sleep 1800` at `2026-05-11 01:38:34 UTC`.
- Full `agents/` read completed and saved: `/tmp/deit_agents_full_read_20260511_013846.txt`.

### TPU status before action
- Running DeiT jobs before action: sanity CE `6380`, N `6392`, L `6405`, R `6411`, T `6415`, U `6417`, V `6419`.
- H `6381` remains stale/error aux-CE diagnostic.
- No active job needed resume.

### New results/status
- Run U K80+SiLU low-aug ep39: `52.828%` / iter `57.594%`, loss `0.39629`, invalid `0.668%`.
  - Compared with T K80+GELU ep39 (`52.628%` / `57.866%`), U is +0.200 single-step but -0.272 iterative.
  - Compared with R K40+SiLU ep39 (`53.514%` / `58.272%`), U is worse on both metrics. K80+SiLU does not show compounding benefit at ep39.
- Run R K40+SiLU low-aug ep79: `57.624%` / iter `61.282%`, loss `0.42108`, invalid `0.544%`.
  - Compared with P K40+GELU ep79 (`57.226%` / `61.538%`), R is +0.398 single-step but -0.256 iterative. SiLU remains mixed rather than clearly dominant.
- Run T K80+GELU latest remains ep39 `52.628%` / iter `57.866%`, running around ep50.
- Run V K10+SiLU latest remains ep19 `37.934%` / iter `44.542%`, running around ep29.
- Run N is running around ep280; latest logged eval remains ep259 `62.594%` / iter `66.404%`.

### Action
- Stopped Run L/window `6405` on `z169mq`; cross-attn was consistently behind MLP and remained first stop candidate.
- Created `configs/remote_run_W_mlp_k160_lowaug_mild_config.yml`.
- Copied it to `configs/remote_run_config.yml` for launch.
- Launched Run W on `kmh-tpuvm-v6e-8-spot-gzy-z169mq`.
- Run W details:
  - window `6424`
  - K160, low-aug mild, GELU tiny-bit MLP
  - pure masked diffusion, no CE warm-start, no aux CE, no mixup/cutmix
  - logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_014551_u2g9be_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval`
  - first batch ready at `2026-05-11 01:47:40 UTC`
- Cleaned up the local `tpu.py run ... user=sqa` wrapper after W registered. Remote tmux job left running.

### Interpretation
- SiLU is not a universal improvement: K10 SiLU failed to beat N at ep19, K80 SiLU did not beat K80 GELU on iterative ep39, and K40 SiLU's later advantage is small/mixed.
- K80 GELU still improves over K40 GELU at ep39. The most direct next scaling test is K160 GELU rather than another SiLU variant.
- W reached first batch, so data construction with K160 did not fail immediately. Await compilation/eval0 for real memory confirmation.

### Decision
- No further launch this loop.
- Next loop priorities: confirm W compile/eval0 or catch OOM, capture N ep279, T/U ep59 trajectories, V ep39, and R ep99 later.
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` with U ep39, R ep79, and W launch.

### Run W startup confirmation (2026-05-11 01:55 UTC)
- Run W K160+GELU reached initial compilation at `2026-05-11 01:48:43 UTC`.
- eval0: `0.096%` / iter `0.132%`, loss `0.69086`, invalid `2.574%`.
- K160 has now passed first batch, compile, and eval0 without immediate OOM. Continue to ep19 before judging scaling quality.
- Updated ImageNet-22K handoff report to note K160 eval0 success.

## User Correction and Run S Resume (2026-05-11 02:58 UTC)

- User pointed out that killing Run S after slow ep19 convergence was premature: full augmentation/mixup/cutmix was intended to test overfitting/late growth, so early slowness alone is not a sufficient stop criterion.
- Action: resumed Run S rather than restarting from scratch.
  - Original S window: `6412` on `j3rqvs`.
  - Original S logdir/checkpoint source: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_200200_2v7ro7_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`.
  - Verified GCS checkpoints existed, including `checkpoint_25020` (after ep19 eval).
  - Existing same-zone deleted registered slots `v6e-8-tmp203/p1u4mx` and `v6e-8-tmp51/c8umw4` could not be re-applied due zone capacity failures.
  - Registered existing READY/free same-zone TPU `kmh-tpuvm-v6e-8-spot-gzy-1dqe89` as `v6e-8-tmp213`; environment check passed.
  - Ran manager resume for `window=6412` onto `v6e-8-tmp213`, which created new window `6427`.
- Resume complications and fixes:
  - First S resume attempt in `6427` failed immediately with `ModuleNotFoundError: No module named 'timm'` on the newly registered TPU.
  - Installed `timm==0.9.16` on `1dqe89` via remote pip.
  - The manager-generated resume command included a duplicate `--config=...`; reran manually in the same `6427` tmux window with only `--config.load_from=<S old logdir>` so `run_remote.sh` supplies the single base `--config=configs/load_config.py:remote_run`.
  - Cleared the stale manager Error using `tpu.py ignore-error 6427 sqa` after confirming the second run was actually alive.
- Current S-resume logdir:
  - `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_024615_6lmien_kmh-tpuvm-v6e-8-spot-gzy-1dqe89_asia-northeast1-b__b_lr_ep_eval`
- S-resume verification:
  - Config is correct: full baseline augmentation, `use_mixup_cutmix: true`, `diffusion_label_mode: soft_top2`, `n_masks_per_image: 10`, no CE shortcut.
  - Restore confirmed: `Restoring orbax checkpoint from .../checkpoint_25020` at `2026-05-11 02:52:40 UTC`.
  - Initial compilation completed at `2026-05-11 02:54:12 UTC`.
  - Training resumed at epoch 20: first logs include `[25100] ep=20.061 train_accuracy=0.022567 train_loss=0.65389`; latest checked `[25800] ep=20.62 train_accuracy=0.024247 train_loss=0.65324`.
- Updated interpretation:
  - The ep19 S result (`2.364%` / iter `5.054%`) is still a strong early-negative signal, but it is not enough to answer the user's overfitting/late-growth hypothesis.
  - Keep S running to at least ep39/59 before drawing a final conclusion about full augmentation + corrected mixup/cutmix.
  - Do not kill slow augmentation runs solely because early convergence is poor unless there is code error, OOM, or clear late-stage domination evidence.

## Loop Results Update (2026-05-11 02:58 UTC)

- Full agents memory read for this cycle was completed earlier: `/tmp/deit_agents_full_read_20260511_022827.txt`.
- New main results:
  - Run T K80 GELU low-aug ep59: `56.584%` / iter `61.278%`, loss `0.39226`, invalid `0.596%`.
    - Compared with R K40 SiLU ep59 (`56.122%` / `60.246%`), T is +0.462 single-step and +1.032 iterative at matched epoch.
    - K80 GELU now looks useful by ep59, even though K80 SiLU was not clearly better at ep39.
  - Run V K10 SiLU low-aug ep39: `50.654%` / iter `56.088%`, loss `0.38271`, invalid `0.722%`.
    - This is behind N K10 GELU ep39 (`52.116%` / `57.364%` from prior table) and behind U/T K80 ep39. SiLU is not a universal activation win at K10.
  - Run W K160 GELU low-aug is healthy and training around ep14; no ep19 eval yet.
  - Run S full-aug soft_top2 is resumed and training around ep20.6; next meaningful read is ep39.
- Current decision:
  - Keep S, T, U, V, W, R, N all running for now.
  - Next checkpoints to capture: S ep39, W ep19, U ep59, R ep99, T ep79, V ep59, N ep299.

## Manual Loop Update (2026-05-11 03:31 UTC)

- Woke from manual `sleep 1800` at `2026-05-11 03:30:43 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_033052.txt`.
- Manager state:
  - Running/healthy: sanity CE `6380`, R `6411`, U `6417`, V `6419`, W `6424`, S resume `6427`.
  - N `6392` and T `6415` show manager `Unknown` due log permission/parser noise, but output logs are fresh and training is active. Do not resume/restart.
  - H `6381` remains stale Error diagnostic; ignore.
- New evals:
  - Run N K10 GELU low-aug ep299: `63.408%` / iter `67.480%`, loss `0.46508`, invalid `0.452%`.
    - N continues to improve from ep279 (`63.044/66.932`), so low-aug K10 is not fully plateaued.
  - Run R K40 SiLU ep99: `59.146%` / iter `62.334%`, loss `0.42650`, invalid `0.516%`.
    - R improves from ep79 (`57.624/61.282`), but still far behind mature N at later epochs.
  - Run U K80 SiLU ep59: `56.480%` / iter `60.426%`, loss `0.40562`, invalid `0.486%`.
    - Matched with T K80 GELU ep59 (`56.584/61.278`), U is slightly worse single-step and clearly worse iterative. SiLU does not help at K80.
  - Run W K160 GELU ep19: `42.846%` / iter `49.468%`, loss `0.42330`, invalid `1.370%`.
    - W beats T K80 GELU ep19 (`42.050/48.410`), U K80 SiLU ep19 (`42.366/48.870`), R K40 SiLU ep19 (`42.032/48.254`), and P K40 GELU ep19 (`41.366/47.526`). K160 looks promising early and should be kept.
  - Run S resumed full-aug soft_top2: no new eval yet; training active around ep25.9 with train accuracy ~3.5% and loss ~0.643.
- Interpretation:
  - K scaling may still help early if pushed to K160; W ep19 is the best early K-scaling result so far.
  - Activation: GELU is safer than SiLU at K10 and K80. SiLU's only useful-looking case remains K40 single-step, but not enough to prioritize.
  - Augmentation: S is still very slow but is now correctly kept alive for the late-growth/overfit hypothesis. Do not kill before ep39 unless it errors/OOMs.
- Decision:
  - No kill/launch/resume this loop.
  - Keep N/R/T/U/V/W/S all running; next read targets are S ep39, W ep39, T ep79, U ep79, V ep59, R ep119, N ep319.

## Manual Loop Update (2026-05-11 04:03 UTC)

- Woke from manual `sleep 1800` at `2026-05-11 04:02:44 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_040249.txt`.
- Manager state: all main DeiT jobs are Running: N `6392` ep307.8, R `6411` ep108, T `6415` ep78, U `6417` ep71, V `6419` ep57, W `6424` ep27, S resume `6427` ep31. H `6381` remains stale Error diagnostic.
- No new full eval beyond the 03:31 loop:
  - Latest N remains ep299 `63.408%` / iter `67.480%`.
  - Latest R remains ep99 `59.146%` / iter `62.334%`.
  - Latest T remains ep59 `56.584%` / iter `61.278%`; next ep79 is imminent.
  - Latest U remains ep59 `56.480%` / iter `60.426%`.
  - Latest V remains ep39 `50.654%` / iter `56.088%`; next ep59 is soon.
  - Latest W remains ep19 `42.846%` / iter `49.468%`.
  - S resume has no new eval yet; training is active around ep31 with train accuracy around 4-5% and loss around 0.63.
- Decision: no kill/launch/resume. Wait for T ep79, V ep59, S ep39, W ep39.

## Manual Loop Update (2026-05-11 04:34 UTC)

- Woke from manual `sleep 1800` at `2026-05-11 04:33:31 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_043338.txt`.
- Manager state: all main DeiT jobs are Running: N ep314, R ep114, T ep84, U ep77, V ep63, W ep33, S ep35. H remains stale Error diagnostic.
- New evals:
  - Run T K80 GELU ep79: `58.302%` / iter `62.476%`, loss `0.39220`, invalid `0.760%`.
    - This is now above P K40 GELU ep79 (`57.226/61.538`) and R K40 SiLU ep79 (`57.624/61.282`). K80 GELU is the best mid-epoch K-scaling branch so far.
  - Run V K10 SiLU ep59: `55.462%` / iter `59.876%`, loss `0.37677`, invalid `0.696%`.
    - This beats N K10 GELU at ep59 (`54.634/59.000`) after trailing at ep19/39. SiLU at K10 may catch up by ep59, but mature N GELU is still far ahead by ep299; keep V to later checkpoints before interpreting.
- No new S/W eval yet:
  - S full-aug soft_top2 is running around ep35.9; ep39 expected next loop. It is still slow but train loss/accuracy are improving.
  - W K160 GELU is running around ep33.7; ep39 expected next loop.
  - U K80 SiLU is running around ep77; ep79 expected soon and will decide whether U remains useful.
- Interpretation:
  - K scaling: K80 GELU has now become clearly useful by ep79. W/K160 remains promising based on ep19 and should be kept to ep39.
  - Activation: SiLU is mixed. It hurt K80 at ep59, but K10 SiLU recovered by ep59 vs N's matched early checkpoint. Need later V checkpoints.
  - Augmentation: keep S to ep39/59; do not judge from train accuracy alone.
- Decision: no kill/launch/resume. Wait for S ep39, W ep39, U ep79, V ep79, T ep99, N ep319.

## 2026-05-11 05:10 UTC - S retention rule after user correction
- User corrected an earlier mistake: do NOT stop Run S just because full augmentation / corrected soft_top2 mixup-cutmix has slow early convergence.
- Run S must be allowed to continue at least through ep59, preferably longer, because the purpose of augmentation is late-stage generalization / overfit prevention; slow early convergence is expected.
- Current S status checked this loop: window 6427 on `v6e-8-spot-gzy-1dqe89` is running around ep41. Latest eval: ep39 single 11.834%, iterative 17.946%, invalid 0.974%.
- Future loop action: never retire S on early-epoch speed alone. Compare it only after meaningful later checkpoints against low-augmentation runs.

## 2026-05-11 05:13 UTC - Run X launch and S correction
- S correction from user is active: S must not be killed due slow early convergence under full augmentation. S is currently running and should be evaluated at ep59+ before judgment.
- U was killed intentionally only because it was dominated by T at K80 and its TPU was needed for a higher-K scaling test.
- Run X uses K320 GELU low-augmentation mild config copied to `configs/remote_run_config.yml`; fixed notes to say Run W K160 -> K320.
- Launch needed `-f` because tpu_manager had stale/active U job metadata. `-f` allowed manager to mark U killed, run `kill_jobs_tpu`, and start window 6429.
- Duplicate `--config=configs/load_config.py:remote_run` appeared in the staging command but did not break this fresh run; X started training successfully.

## 2026-05-11 05:44 UTC loop decision
- Do not launch more augmentation branches until S reaches at least ep59; otherwise we risk overreacting to early slow convergence again.
- N is near final ep330 and will likely finish before/around the next loop. Consider reusing its TPU only after confirming completion and deciding whether the next priority is medium augmentation or another K-scaling run.
- X K320 is running without immediate OOM; wait for ep19 before deciding whether K320 is useful or whether K640 is worth trying.

## 2026-05-11 06:20 UTC decisions
- N finished and freed `3djlis`. Because Run M already tested full augmentation without mixup/cutmix and was worse than low-aug, do not duplicate that branch.
- Used freed slot for K-scaling: Run Y K640 GELU low-aug mild. This follows user request to push K upward after K10 success and after X K320 showed no immediate OOM.
- Continue to preserve S. S is running ep51+; do not kill before ep59+.
- Next important evals: W ep59, S ep59, X ep19, Y eval0/ep19, R ep139, T ep119, V ep99.

## 2026-05-11 07:06 UTC decisions after S ep59
- Keep S to ep79. User correction was directionally valid: full aug + mixup/cutmix has delayed convergence, and ep59 improved a lot over ep39.
- But S ep59 is still far below low-aug N and old full-aug/no-mix M, so do not promote S to best config; keep it as a late-growth/regularization test only.
- K scaling update: W/K160 ep59 and X/K320 ep19 do not show monotonic gains over K80/K160. Keep X/Y for later checkpoints before deciding whether high K is useful.
- No new launches this loop. Next useful decision points: S ep79, W ep79, X ep39, Y ep19, T ep119, V ep99.

## 2026-05-11 07:42 UTC decisions
- V/K10 SiLU is no longer worth its slot after ep99: it underperforms the K10 GELU trajectory and does not justify continuing.
- Launched Z/K80 full-aug soft_top2 to test whether more K can mitigate S/K10 full-aug mixup/cutmix slow convergence. This is a follow-up diagnostic, not a replacement for S.
- Continue S to ep79. Do not kill S.
- Continue W/X/Y until next key evals before deciding whether high-K scaling saturates or reverses.

## 2026-05-11 08:27 UTC - Run S resume repaired
- User correction remains active: do not stop Run S just because full augmentation converges slowly early. S is the full-baseline-augmentation + corrected `soft_top2` mixup/cutmix test and should be allowed to continue.
- Original S window 6427 on `1dqe89` errored due SSH/TPU connectivity after reaching around ep70; latest useful eval remains ep59 `30.338%` single / `37.420%` iterative.
- First S resume2 attempt window 6432 failed because `configs/remote_run_S_resume2_fullaug_mixlabel_config.yml` accidentally omitted `model: ViT_base_mdh_mlp`. The default model became `ViT_base`, while `use_diffusion_head: true` made training call diffusion decode, causing `AttributeError: "ViT" object has no attribute "diffusion_head"`.
- Fixed the resume config by adding `model: ViT_base_mdh_mlp`, copied it to `configs/remote_run_config.yml`, and relaunched on `favaxa` as window 6433.
- Run S resume2 window 6433 logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_082259_aoimz8_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval`.
- Startup verified: config printed `model: ViT_base_mdh_mlp`, `use_diffusion_head: true`, `diffusion_label_mode: soft_top2`, restored from GCS `checkpoint_75060`, completed initial compilation, and resumed training at step 75100 / ep60.024. Latest checked line: step 75300 / ep60.184, train_accuracy 0.27961, train_loss 0.55298.
- Next S decision point: ep79 eval. Keep S running unless it errors/OOMs.

## Manual Loop Update (2026-05-11 09:04 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T08:57:52Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_085758.txt`.
- S correction remains active: S is resumed and must not be killed for slow full-augmentation convergence. Current S resume2 window 6433 is running around ep65; latest eval remains original S ep59 `30.338%` / iter `37.420%`.
- New key evals since last loop:
  - W K160 GELU ep79: `58.320%` / iter `62.348%`, loss `0.42875`, invalid `0.542%`. This is effectively tied with T/K80 ep79 (`58.302%` / iter `62.476%`), so K160 is not clearly better than K80.
  - X K320 GELU ep39: `52.928%` / iter `58.350%`, loss `0.39195`, invalid `0.842%`. This is slightly below W/K160 ep39 (`53.186%` / iter `58.478%`), so K320 is not improving the matched checkpoint.
  - Y K640 GELU ep19: `42.734%` / iter `49.136%`, loss `0.42334`, invalid `1.304%`. Similar to X/K320 ep19 and below/near W/K160 ep19; high-K scaling is not monotonic.
  - R K40 SiLU ep159: `60.540%` / iter `63.454%`, loss `0.46266`, invalid `0.420%`. It improves slowly but its iterative accuracy is already below T/K80 ep119 (`63.574%`), so R is no longer a good use of a slot.
- Action: killed/replaced R on `cz2ivo` and launched Run AA window 6434.
- Run AA purpose: K80 GELU with baseline RA/reprob/repeated/WD/SD but no mixup/cutmix. This isolates whether mixup/cutmix soft-label splitting is the main cause of S/Z slow full-augmentation convergence.
- Run AA config: `configs/remote_run_AA_mlp_k80_strongaug_nomix_config.yml`; copied to `configs/remote_run_config.yml` for launch.
- Run AA logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_090038_1nzt77_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval`.
- Run AA startup verified: `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `use_rand_augment: true`, `reprob: 0.25`, `repeated_aug: 3`, `use_mixup_cutmix: false`, `diffusion_label_mode: argmax`, `load_from: ''`, `load_backbone_from: ''`, no aux CE. Compilation completed and train reached step 200 / ep0.159, loss `0.69302`.
- Decision: no further launch this loop. Wait for T ep139, AA ep0/19, Y ep39, Z ep19, S ep79, X ep59, W ep99.

## Manual Loop Update (2026-05-11 09:41 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T09:34:46Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_093453.txt`.
- New evals:
  - T K80 GELU low-aug ep139: `60.280%` / iter `64.014%`, loss `0.45873`, invalid `0.600%`. This confirms T/K80 is a stronger branch than retired R/K40 SiLU (R ep159 iter `63.454%`).
  - Y K640 GELU low-aug ep39: `52.898%` / iter `58.352%`, loss `0.38766`, invalid `0.914%`. This is essentially identical to X/K320 ep39 (`52.928/58.350`) and below W/K160 ep39 (`53.186/58.478`), so K640 is not useful enough to continue.
  - AA K80 strong-aug no mix/cut eval0: `0.106%` / iter `0.078%`, invalid `0.272%`; running around ep5 with train loss improving to ~0.647.
- Current S status: S resume2 window 6433 is running around ep71; latest eval still original S ep59 `30.338%` / iter `37.420%`. Keep S to ep79; do not kill due slow convergence.
- Current Z status: Z K80 full-aug + soft_top2 is running around ep18; no ep19 eval yet. Keep to ep19+.
- Action: killed/replaced Y K640 after ep39, because high-K scaling beyond K160/K320 is flat and the slot is better spent on an augmentation mechanism diagnostic.
- Launched Run AB window 6435 on `3djlis`.
- Run AB purpose: complete the K80 augmentation 2x2 design:
  - T = low aug, no mix/cut.
  - AB = low aug, corrected mix/cut soft_top2.
  - AA = strong aug, no mix/cut.
  - Z = strong aug, corrected mix/cut soft_top2.
- Run AB config: `configs/remote_run_AB_mlp_k80_lowaug_mixlabel_config.yml`; copied to `configs/remote_run_config.yml` for launch.
- Run AB logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_093651_zbntix_kmh-tpuvm-v6e-8-spot-gzy-3djlis_asia-northeast1-b__b_lr_ep_eval`.
- Run AB startup verified: `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `use_rand_augment: false`, `reprob: 0.0`, `repeated_aug: 1`, `use_mixup_cutmix: true`, `diffusion_label_mode: soft_top2`, `load_from: ''`, `load_backbone_from: ''`, no aux CE. Compilation completed and train reached step 200 / ep0.159, loss `0.69291`.
- Decision: no further launch this loop. Next checks: Z ep19, S ep79, X ep59, W ep99, AB ep0/19, AA ep19, T ep159.

## Manual Loop Update (2026-05-11 10:18 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T10:11:14Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_101120.txt`.
- New evals:
  - W K160 GELU low-aug ep99: `58.822%` / iter `62.762%`, loss `0.42818`, invalid `0.584%`. This is below T/K80 ep99 (`59.146%` / iter `63.188%`), so K160 is not improving over K80.
  - X K320 GELU low-aug ep59: `56.668%` / iter `61.136%`, loss `0.38906`, invalid `0.706%`. This is effectively tied/slightly worse than W/K160 ep59 and below T/K80 iterative at ep59; K320 is not useful.
  - Z K80 full-aug + corrected soft_top2 ep19: `3.694%` / iter `7.152%`, loss `0.63656`, invalid `1.066%`. Full augmentation + mix/cut remains extremely slow even with K80, but keep to later checkpoints before final judgment.
  - AB K80 low-aug + corrected soft_top2 eval0: `0.074%` / iter `0.100%`, invalid `1.404%`; running around ep6.2.
  - AA K80 strong-aug no mix/cut is running around ep11.2; eval0 remains `0.106%` / iter `0.078%`.
- S resume2 is running around ep77; latest eval still original S ep59. Keep S to ep79+ per user correction.
- Code change: added optional LayerNorm support for `MLPDiffusionHead` via `head_mlp_layer_norm` (default false). Dense -> optional LayerNorm -> activation for each MLP hidden projection. Existing runs are unaffected because their staged code/configs do not enable it.
- Verification: `python -m py_compile models.py train.py configs/default.py` passed.
- Action: killed/replaced X K320 after ep59 because high-K scaling is flat/dominated.
- Launched Run AC window 6436 on `j3rqvs`: K80 low-aug, SiLU MLP with LayerNorm, no warm-start, no aux CE, no CE shortcut. This tests whether the previous SiLU branch failed because REPA-style projector LayerNorm was missing.
- Run AC config: `configs/remote_run_AC_mlp_k80_lowaug_silu_ln_config.yml`; copied to `configs/remote_run_config.yml` for launch.
- Run AC logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_101355_psg7ix_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`.
- Run AC startup verified: `head_mlp_activation: silu`, `head_mlp_layer_norm: true`, `n_masks_per_image: 80`, low-aug config, `load_from: ''`, `load_backbone_from: ''`. Compilation completed and train reached step 300 / ep0.239, loss `0.69238`.
- Decision: no further launch this loop. Next checks: S ep79, Z ep39 later, AA/AB/AC ep19, W ep119, T ep159.

## Manual Loop Update (2026-05-11 10:54 UTC)
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_105443.txt`.
- User correction remains active: Run S must not be killed because full augmentation has slow early convergence. It is a late-growth/generalization test and should be kept at least through ep99 unless it errors/OOMs.
- New evals/status since the 10:18 record:
  - Run T K80 GELU low-aug ep159: `60.578%` / iter `64.212%`, loss `0.45579`, invalid `0.584%`. This remains the strongest active low-aug scaling branch, although still below completed Run N K10 GELU ep319 (`63.604%` / iter `67.726%`).
  - Run W K160 GELU low-aug ep99: `58.822%` / iter `62.762%`, loss `0.42818`, invalid `0.584%`. Since this is below T/K80 ep99 (`59.146%` / iter `63.188%`), W was killed and its slot reused.
  - Run Z K80 full baseline aug + corrected `soft_top2` mixup/cutmix ep19: `3.694%` / iter `7.152%`, loss `0.63656`, invalid `1.066%`. Keep to ep39 before interpreting too aggressively.
  - Run S resume2 full baseline aug + corrected `soft_top2` mixup/cutmix ep79: `43.392%` / iter `49.778%`, loss `0.409572`, invalid `0.580%`. This confirms the user's point: S is still improving sharply late (`11.834/17.946` at ep39 -> `30.338/37.420` at ep59 -> `43.392/49.778` at ep79). It remains far behind low-aug branches at matched epochs, but should continue to ep99.
  - Run AA K80 strong-aug no mix/cut is running around ep17; latest eval0 remains `0.106%` / iter `0.078%`.
  - Run AB K80 low-aug + corrected `soft_top2` mixup/cutmix is running around ep13; latest eval0 remains `0.074%` / iter `0.100%`.
  - Run AC K80 low-aug SiLU+LayerNorm is running around ep6; eval0 `0.112%` / iter `0.136%`, invalid `3.118%`.
- Action: killed/replaced W after confirming K160 is dominated by K80 at ep99.
- Launched Run AD window 6437 on `z169mq`: K80 low-aug, GELU MLP with LayerNorm, no warm-start, no aux CE, no CE shortcut. This pairs with AC to isolate LayerNorm vs activation effects.
- Run AD config: `configs/remote_run_AD_mlp_k80_lowaug_gelu_ln_config.yml`; copied to `configs/remote_run_config.yml` for launch.
- Run AD logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_104942_dt8wrk_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval`.
- Run AD startup verified: config printed `head_mlp_activation: gelu`, `head_mlp_layer_norm: true`, `n_masks_per_image: 80`, low augmentation, `load_from: ''`, `load_backbone_from: ''`. Compilation completed and training reached step 200 / ep0.159 with loss `0.69319`.
- Interpretation:
  - High-K scaling beyond K80 has now failed at K160/K320/K640 matched checkpoints. Do not spend more slots on larger K unless a new hypothesis appears.
  - S should not be retired for slow start; its late growth is real. But current matched-epoch accuracy remains far lower than low-aug K80/T, so augmentation is still diagnostic rather than best config.
  - The most useful active comparisons are now the augmentation 2x2 (T/AA/AB/Z) and LayerNorm MLP pair (AC/AD vs T).
- Next checks: S ep99, Z ep39, AA/AB/AC/AD ep19, T ep179.

## Manual Loop Update (2026-05-11 11:25 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T11:25:23Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_112531.txt`.
- Manager state: active DeiT jobs are T ep168, Z ep35, S resume2 ep88, AA ep21, AB ep19, AC ep12, AD ep5. No resume needed.
- New evals:
  - Run AA K80 strong augmentation no mix/cut ep19: `24.904%` / iter `33.344%`, loss `0.506714`, invalid `1.464%`.
  - Run AB K80 low augmentation + corrected `soft_top2` mixup/cutmix ep19: `32.838%` / iter `39.854%`, loss `0.471520`, invalid `2.032%`.
  - Run AD K80 low augmentation GELU+LayerNorm eval0: `0.124%` / iter `0.136%`, invalid `3.388%`; running healthy at ep6.55.
- Current running status:
  - T K80 GELU low-aug: running ep168; latest eval remains ep159 `60.578%` / iter `64.212%`.
  - S K10 full aug + `soft_top2`: running ep88.6; latest eval remains ep79 `43.392%` / iter `49.778%`. Keep to ep99.
  - Z K80 full aug + `soft_top2`: running ep35.8; latest eval remains ep19 `3.694%` / iter `7.152%`. Keep to ep39 before any decision.
  - AC K80 SiLU+LayerNorm: running ep14.1; latest eval0 `0.112%` / iter `0.136%`.
- Augmentation 2x2 interpretation at ep19:
  - T low-aug no mix/cut: `42.050%` / iter `48.410%`.
  - AB low-aug + mix/cut: `32.838%` / iter `39.854%`.
  - AA strong-aug no mix/cut: `24.904%` / iter `33.344%`.
  - Z strong-aug + mix/cut: `3.694%` / iter `7.152%`.
  - Both mix/cut and strong augmentation slow early convergence, but the catastrophic early drop is from their combination. Low-aug + mix/cut is worse than T but not dead; strong-aug no mix/cut is also worse but not dead. Z must be kept to ep39 because S showed delayed late growth, but Z is currently the weakest of the four.
- Decision: no kill/launch/resume this loop. Wait for S ep99, Z ep39, AC/AD ep19, AA/AB ep39, T ep179.

## Manual Loop Update (2026-05-11 11:56-12:10 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T11:56:23Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_115631.txt`.
- Manager state after wake: T running ep174, S resume2 running ep93, AA running ep26, AB running ep26, AC running ep19, AD running ep11. Z window 6431 changed to Error.
- Z failure diagnosis:
  - Z did not fail from training code. The log shows training reached ep37.165, then SSH/TPU connectivity failed (`Connection refused/timed out`, host key changed), followed by staging path errors.
  - Last available Z eval remains ep19 `3.694%` / iter `7.152%`; no ep39 eval was reached.
  - GCS contains Z checkpoints `checkpoint_1251` and `checkpoint_25020`; because `checkpoint_per_epoch: 20`, the next checkpoint would have been at ep39, so the best resumable point is ep19.
- Z resume attempt:
  - Removed the stale host key reported in the error: `ssh-keygen -f /home/sqa/.ssh/google_compute_known_hosts -R tpu.6934146544078258974-0-igfi4p`.
  - Tried `tpu.py resume window=6431 ka=kmh-tpuvm-v6e-8-spot-gzy-axuxm0 user=sqa`. It correctly located Z logdir and would have resumed from Z's own checkpoint, but failed because `axuxm0` status is `deleted`.
  - Tried `tpu.py applyy kmh-tpuvm-v6e-8-spot-gzy-axuxm0 user=sqa`. It failed/timed out due no spot v6e-8 capacity in `asia-northeast1-b`.
  - Decision: keep Z marked as needs-resume when a v6e-8 slot/capacity is available. This is not warm-start/投敌 because the planned load path is Z's own diffusion checkpoint.
- New evals/status:
  - Run AC K80 low-aug SiLU+LayerNorm ep19: `40.864%` / iter `46.078%`, loss `0.437251`, invalid `0.808%`. This is close to T/GELU-no-LN ep19 (`42.050%` / iter `48.410%`) and much healthier than earlier SiLU-no-LN concerns, but not an early win.
  - Run AD K80 low-aug GELU+LayerNorm is running ep15.9; no ep19 yet. Train accuracy/loss look healthy.
  - Run S is running ep95.8; latest eval remains ep79 `43.392%` / iter `49.778%`. Keep to ep99.
  - Run T is running ep177; latest eval remains ep159 `60.578%` / iter `64.212%`; ep179 expected soon.
  - Run AA is running ep30.6; latest eval remains ep19 `24.904%` / iter `33.344%`.
  - Run AB is running ep29.8; latest eval remains ep19 `32.838%` / iter `39.854%`.
- Interpretation:
  - Z remains important for the augmentation 2x2, but current infrastructure capacity blocks immediate resume.
  - AC shows LayerNorm+SiLU is viable but slightly behind plain GELU at ep19. Wait for AD ep19 and AC ep39 before deciding whether LayerNorm helps.
- Decision: no kill/launch. Resume Z only when a free/creatable v6e-8 is available, or when another diagnostic slot is retired.

## Manual Loop Update (2026-05-11 12:41 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T12:41:36Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_124147.txt`.
- Manager state: S resume2 running ep99, AA ep34, AB ep35, AC ep28, AD ep20. T is marked Unknown by manager due a `/tmp/tpu_logs/... Permission denied` issue, but the actual output log is still updating through ep183, so treat T as running.
- New evals:
  - Run S K10 full aug + corrected `soft_top2` ep99: `50.646%` / iter `56.452%`, loss `0.371894`, invalid `0.778%`.
  - Run T K80 GELU low-aug ep179: `60.734%` / iter `64.308%`, loss `0.463831`, invalid `0.552%`.
  - Run AD K80 low-aug GELU+LayerNorm ep19: `42.010%` / iter `47.516%`, loss `0.424386`, invalid `0.868%`.
- Current running status:
  - AA strong-aug no mix/cut is ep35.7; latest eval ep19 `24.904%` / iter `33.344%`; ep39 next.
  - AB low-aug + mix/cut is ep36; latest eval ep19 `32.838%` / iter `39.854%`; ep39 next.
  - AC SiLU+LayerNorm is ep29.8; latest eval ep19 `40.864%` / iter `46.078%`; ep39 later.
  - Z remains Error and needs resume from its own `checkpoint_25020`, but `axuxm0` re-apply previously failed due capacity.
- Interpretation:
  - S late growth is substantial: ep39 `11.834/17.946` -> ep59 `30.338/37.420` -> ep79 `43.392/49.778` -> ep99 `50.646/56.452`. User correction was valid: early convergence was misleading. However S is still below low-aug T at the same ep99 checkpoint (`59.146/63.188`), so full augmentation is not yet best.
  - T is growing very slowly now: ep159 `60.578/64.212` -> ep179 `60.734/64.308`. It may be plateauing and below completed N/K10 final, but keep at least to ep199 before retiring the main K80 low-aug reference.
  - AD GELU+LayerNorm does not improve early metrics over plain GELU T at ep19 (`42.010/47.516` vs `42.050/48.410`). AC SiLU+LayerNorm is slightly worse. Keep LayerNorm pair to ep39 before final judgment.
- Decision: no kill/launch/resume this loop. Next checks: AA/AB ep39, S ep119, T ep199, AC/AD ep39, and whether a v6e-8 slot/capacity appears for Z resume.

## Manual Loop Update (2026-05-11 13:12 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T13:12:49Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_131257.txt`.
- Manager state after wake: S running ep105, T log still running ep189 though manager says Unknown due temp log issue, AA ep39, AB ep41, AC ep34, AD ep27, Z original remains Error.
- New evals:
  - Run AA K80 strong augmentation no mix/cut ep39: `43.568%` / iter `50.788%`, loss `0.415536`, invalid `1.354%`.
  - Run AB K80 low augmentation + corrected `soft_top2` mixup/cutmix ep39: `57.148%` / iter `61.642%`, loss `0.338175`, invalid `0.860%`.
- Key interpretation:
  - AB is now the best augmentation result at matched ep39: it beats T low-aug/no-mix ep39 (`52.628%` / iter `57.866%`) by +4.520 single / +3.776 iterative.
  - AA is far below T/AB at ep39. Strong augmentation without mix/cut is not useful enough to keep occupying a slot, especially because historical full-aug/no-mix Run M was also weak later.
  - The augmentation 2x2 now reads:
    - T low-aug no mix/cut ep39: `52.628%` / iter `57.866%`.
    - AB low-aug + mix/cut ep39: `57.148%` / iter `61.642%`.
    - AA strong-aug no mix/cut ep39: `43.568%` / iter `50.788%`.
    - Z strong-aug + mix/cut needs resume to get ep39; original failed after ep37.165.
- Action: retired AA/window 6434 after ep39 and reused `cz2ivo` for Z resume, because AB made low-aug+mix/cut the more promising augmentation direction and Z is the user-requested full-augmentation corrected mix/cut control.
- Z resume details:
  - Created `configs/remote_run_Z_resume1_mlp_k80_fullaug_mixlabel_config.yml` from Z config with `load_from` set to Z's own `checkpoint_25020`; copied to `configs/remote_run_config.yml`.
  - Launched Z resume1 as window 6438 on `cz2ivo`, but it failed with GCS 403: us-east5 service account could not read the asia-northeast1-b bucket checkpoint.
  - Copied Z `checkpoint_25020` from `gs://kmh-gcp-asia-northeast1-b/...` to `gs://kmh-gcp-us-east5/..._copied_to_us-east5/checkpoint_25020` using `gsutil -m cp -r`.
  - Created `configs/remote_run_Z_resume2_mlp_k80_fullaug_mixlabel_config.yml` with the us-east5 checkpoint path and copied it to `configs/remote_run_config.yml`.
  - Launched Z resume2 as window 6439 on `cz2ivo`.
  - Startup verified: config printed `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `diffusion_label_mode: soft_top2`, `head_aux_ce: false`, `load_backbone_from: ''`, `load_from` the us-east5 copy of Z's own `checkpoint_25020`; log printed `Restoring orbax checkpoint ... checkpoint_25020` and reached `[25100] train_accuracy=0.025485, train_loss=0.64638, ep=20.061`.
  - This is a same-run diffusion checkpoint resume, not warm-start, not CE shortcut, and not backbone loading.
- Current plan:
  - Keep AB to ep59/79; it is now the strongest active augmentation branch.
  - Keep S despite full-aug slow start; next eval ep119.
  - Keep Z resume2 to ep39 to complete the full-augmentation corrected mix/cut control.
  - Keep AC/AD to ep39 for LayerNorm verdict.
  - Revisit T at ep199; it is plateauing but remains the K80 low-aug no-mix reference.

## Manual Loop Update (2026-05-11 14:24-14:27 UTC)
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_142405.txt`.
- Recovered the prior launch session and confirmed the pending launch sequence completed:
  - Run AE launched as window 6440 on `v6e-8-spot-gzy-j3rqvs`.
  - Run AF launched as window 6441 on `v6e-8-spot-gzy-z169mq`.
- Active manager state after launch:
  - T K80 GELU low-aug window 6415 running ep204.
  - S K10 full baseline aug + corrected `soft_top2` mixup/cutmix resume2 window 6433 running ep116-117.
  - AB K80 low-aug + corrected mixup/cutmix `soft_top2` window 6435 running ep55-56.
  - AE mixup-only and AF cutmix-only are now healthy startup runs.
- New evals since the previous 13:12/14:20 memory:
  - T ep199: `61.544%` single / `65.190%` iterative, loss `0.469239`, invalid `0.472%`. T is still improving after apparent plateau; keep at least to ep219 before deciding retirement.
  - S latest eval remains ep99: `50.646%` / iter `56.452%`; it is running ep117 and should continue to ep119. Do not kill S for slow full-augmentation convergence.
  - AB latest eval remains ep39: `57.148%` / iter `61.642%`; it is running ep56 and should continue to ep59/79.
- AE config/startup verified:
  - Purpose: isolate mixup-only under low augmentation after AB showed low-aug mix/cut is promising.
  - `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `use_mixup_cutmix: true`, `mixup_alpha: 0.8`, `cutmix_alpha: 0.0`, `switch_prob: 0.0`, `diffusion_label_mode: soft_top2`.
  - `load_from: ''`, `load_backbone_from: ''`, `head_aux_ce: false`. No warm-start, no aux CE, no CE shortcut.
  - Initial compilation completed; train reached step 200 / ep0.159 with loss `0.69289`.
- AF config/startup verified:
  - Purpose: isolate cutmix-only under low augmentation after AB showed low-aug mix/cut is promising.
  - `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `use_mixup_cutmix: true`, `mixup_alpha: 0.0`, `cutmix_alpha: 1.0`, `switch_prob: 1.0`, `diffusion_label_mode: soft_top2`.
  - `load_from: ''`, `load_backbone_from: ''`, `head_aux_ce: false`. No warm-start, no aux CE, no CE shortcut.
  - Initial compilation completed; train reached step 200 / ep0.159 with loss `0.69298`.
- Z status remains infrastructure-blocked: original Z/resume2 failed because of spot/SSH/TPU availability, not model code. Do not treat Z as a model failure; resume only from its own diffusion checkpoint when a slot/capacity exists.
- Decision: no further kill/launch this loop. Wait for S ep119, AB ep59, AE/AF ep0/19, and T ep219. Continue manual `sleep 1800` loop.

## Manual Loop Update (2026-05-11 14:58-15:00 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T14:58:12Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_145823.txt`.
- Active healthy jobs:
  - T K80 GELU low-aug window 6415 running ep211; latest eval remains ep199 `61.544%` / iter `65.190%`.
  - S K10 full baseline aug + corrected `soft_top2` mixup/cutmix resume2 window 6433 running ep122.
  - AB K80 low-aug + corrected mixup/cutmix `soft_top2` window 6435 running ep62.
  - AE K80 low-aug mixup-only `soft_top2` window 6440 running ep6.
  - AF K80 low-aug cutmix-only `soft_top2` window 6441 running ep6.
- New evals:
  - S ep119: `55.440%` single / `60.658%` iterative, loss `0.346927`, invalid `0.850%`. S continues the delayed full-augmentation catch-up trend, but remains below AB and below the low-aug/mix branch. Keep; do not kill due slow start.
  - AB ep59: `64.890%` single / `68.616%` iterative, loss `0.293254`, invalid `0.576%`. This is now the best observed pure-diffusion result, already above completed N/K10 low-aug final ep319 (`63.604%` / `67.726%`). AB should be preserved and run longer.
  - AE eval0: `0.080%` / iter `0.088%`, running ep6 with train acc around `0.122` at ep6.39.
  - AF eval0: `0.100%` / iter `0.138%`, running ep6 with train acc around `0.073` at ep6.15. Too early to judge mixup-only vs cutmix-only, but both are healthy.
- Manager artifact / pitfall:
  - New failed stale/rerun windows appeared: S stage3 window 6442 on `oulo2v`, Z stage2 window 6443 on `wzrxvx`, and Z resume1 rerun window 6444 on `v5p-16-spot-gzy-cwrrbs`.
  - These were not the active intended jobs. They are failed/stale manager rerun records and should not be interpreted as model failures or current active runs.
  - Effective S is still window 6433 and healthy. Effective Z remains blocked/failed due infrastructure; resume only from Z's own checkpoint when there is a real free v6e-8 slot/capacity.
- Interpretation:
  - Low augmentation + corrected mix/cut (`AB`) is no longer just promising; it is the current best config. The user hypothesis that mix/cut with label-token allocation can help is supported strongly at ep59.
  - Full augmentation (`S`) helps later relative to its slow start, but is still not competitive with AB at matched/nearby epochs.
  - Need wait for AE/AF ep19 to determine whether AB's gain comes primarily from mixup, cutmix, or their mixture.
- Decision: no kill/launch/resume this loop. Keep T until ep219, S to at least ep139, AB to ep79+, and AE/AF to ep19. Continue manual `sleep 1800` loop.

## Manual Loop Update (2026-05-11 15:29-15:31 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T15:29:34Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_152954.txt`.
- Active healthy jobs:
  - T K80 GELU low-aug window 6415 running ep217; latest eval still ep199 `61.544%` / iter `65.190%`. Ep219 is imminent but not printed yet.
  - S K10 full baseline aug + corrected `soft_top2` resume2 window 6433 running ep127; latest eval ep119 `55.440%` / iter `60.658%`.
  - AB K80 low-aug + corrected mix/cut `soft_top2` window 6435 running ep68; latest eval ep59 `64.890%` / iter `68.616%`.
  - AE K80 low-aug mixup-only `soft_top2` window 6440 running ep12; latest eval0 only.
  - AF K80 low-aug cutmix-only `soft_top2` window 6441 running ep12; latest eval0 only.
  - Z resume2 rerun window 6446 on `v5p-8-spot-katelyn-2-4` is running/resumed from Z's own checkpoint, currently around ep22.
- New useful evals: none since the 14:58 loop.
- Z 6446 verification:
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_150019_93fnd5_kmh-tpuvm-v5p-8-spot-katelyn-2-4_us-east5-a__b_lr_ep_eval`.
  - Config printed `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `use_mixup_cutmix: true`, `mixup_alpha: 0.8`, `cutmix_alpha: 1.0`, `switch_prob: 0.5`, `diffusion_label_mode: soft_top2`, `head_aux_ce: false`, `load_backbone_from: ''`.
  - `load_from` is the us-east5 copy of Z's own `checkpoint_25020`; log printed `Restoring orbax checkpoint ... checkpoint_25020`.
  - Initial compilation completed and training resumed at step 25100 / ep20.061; latest checked around step 27700 / ep22.139.
  - This is not warm-start/投敌. It is same-run checkpoint resume. Hardware is v5p-8 instead of the original v6e-8, so compare wall-clock throughput cautiously, but model eval is still useful.
- Decision: no kill/launch/resume this loop. Keep Z 6446 alive to recover the missing full-augmentation K80 control. Wait for T ep219, AE/AF ep19, AB ep79, and S ep139.

## Manual Loop Update (2026-05-11 16:01-16:07 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T16:01:06Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_160112.txt`.
- New eval:
  - T K80 GELU low-aug ep219: `62.162%` single / `65.682%` iterative, loss `0.479385`, invalid `0.554%`.
- Interpretation:
  - T is still slowly improving, but it is dominated by AB at a much earlier checkpoint: AB ep59 `64.890%` / iter `68.616%`.
  - T has served its role as low-aug no-mix K80 reference through ep219. Continuing T is lower value than testing whether AB's corrected mix/cut success scales to larger K.
- Action:
  - Retired/killed T window 6415 on `06q7u9` after ep219.
  - Created `configs/remote_run_AG_mlp_k160_lowaug_mixlabel_config.yml`.
  - Launched AG as window 6447 on `v6e-8-spot-gzy-06q7u9`.
- AG config/startup verified:
  - Purpose: AB follow-up with K160 under the same low-augmentation corrected mixup/cutmix `soft_top2` setting.
  - `model: ViT_base_mdh_mlp`, `n_masks_per_image: 160`, `use_mixup_cutmix: true`, `mixup_alpha: 0.8`, `cutmix_alpha: 1.0`, `switch_prob: 0.5`, `diffusion_label_mode: soft_top2`.
  - `load_from: ''`, `load_backbone_from: ''`, `head_aux_ce: false`. No warm-start, no aux CE, no CE shortcut.
  - First batch ready, initial compilation completed, and train reached step 300 / ep0.239 with loss `0.69280`. No OOM at startup.
- Other active status:
  - AB running ep75; latest eval ep59 `64.890%` / iter `68.616%`; wait ep79.
  - S running ep132; latest eval ep119 `55.440%` / iter `60.658%`; keep per user correction.
  - AE running ep18.8; latest eval0 only; ep19 imminent.
  - AF running ep18.8; latest eval0 only; ep19 imminent.
  - Z 6446 on v5p-8 running ep25.3 from its own checkpoint; no new eval yet.
- Decision: no further action this loop. Next loop should likely see AE/AF ep19 and possibly AB ep79. Continue manual `sleep 1800`.

## Manual Loop Update (2026-05-11 16:37-16:39 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T16:37:10Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_163751.txt`.
- Active healthy jobs: S window 6433 ep138, AB window 6435 ep82, AE window 6440 ep24, AF window 6441 ep24, Z window 6446 ep27-29 on v5p-8, AG window 6447 ep6.
- New evals:
  - AB ep79: `68.464%` single / `71.404%` iterative, loss `0.268052`, invalid `0.572%`. This is a large new best result, improving over AB ep59 `64.890/68.616` and far above completed N final `63.604/67.726`.
  - AE mixup-only ep19: `35.092%` / iter `41.990%`, loss `0.462858`, invalid `1.708%`.
  - AF cutmix-only ep19: `36.586%` / iter `43.614%`, loss `0.452960`, invalid `1.560%`.
- Interpretation:
  - AB remains the best overall branch by a large margin.
  - At ep19, both single augment controls are stronger than AB's own ep19 (`32.838/39.854`), and AF/cutmix-only is the fastest early branch. This means the AB mixture may not be the best early-speed setting, even though it is currently the best late branch.
  - Need keep AE and AF to ep39 before deciding whether mixup-only, cutmix-only, or mixed switch=0.5 is best.
  - S is around ep138 with latest eval still ep119 `55.440/60.658`; ep139 should be available next loop. Keep S per user correction.
  - Z v5p resume is around ep29 and still too slow/low-train-accuracy; no new eval yet. Keep to ep39 if it survives.
  - AG K160 mix/cut is healthy around ep6; wait ep19 to see whether K160 helps under AB-style corrected mix/cut.
- Decision: no kill/launch/resume this loop. Keep all active runs. Next likely checkpoints: S ep139, AE/AF ep39, AB ep99 later, AG ep19 later, Z ep39 later.

## Manual Loop Update (2026-05-11 17:11 UTC)
- Woke/continued loop at `2026-05-11T17:11:28Z` and re-read the full `agents/` directory: `/tmp/deit_agents_full_read_20260511_171128.txt`.
- Active healthy jobs from manager/logs:
  - S K10 full baseline aug + corrected `soft_top2` resume2 window 6433: running ep143 on `favaxa`.
  - AB K80 low-aug + corrected mix/cut `soft_top2` window 6435: running ep89 on `3djlis`.
  - AE K80 low-aug mixup-only `soft_top2` window 6440: running ep32 on `j3rqvs`.
  - AF K80 low-aug cutmix-only `soft_top2` window 6441: running ep32 on `z169mq`.
  - AG K160 low-aug + corrected mix/cut `soft_top2` window 6447: running ep13 on `06q7u9`.
  - Z K80 full-aug + corrected mix/cut `soft_top2` resume2 rerun window 6446: running ep32 on v5p-8 `katelyn-2-4` from Z's own checkpoint.
- New eval:
  - S ep139: `58.748%` single / `63.468%` iterative, loss `0.329652`, invalid `0.846%`.
- Still awaiting next decisive evals:
  - AB ep99; current best remains AB ep79 `68.464%` / iter `71.404%`.
  - AE/AF ep39; latest AE ep19 `35.092%` / iter `41.990%`, AF ep19 `36.586%` / iter `43.614%`.
  - AG ep19; currently healthy with eval0 `0.078%` / iter `0.096%` and training around ep13.
  - Z ep39; currently training around ep32, no new eval yet in the resume log.
- Interpretation:
  - S continues delayed full-augmentation catch-up: ep99 `50.646/56.452` -> ep119 `55.440/60.658` -> ep139 `58.748/63.468`. This supports not killing it early, but it remains clearly below AB's low-aug mix/cut trajectory.
  - AB remains the best branch and must be preserved.
  - AE/AF are not yet judgeable; both need ep39 before deciding whether mixup-only, cutmix-only, or mixed AB is best.
  - AG K160 should continue at least to ep19; no OOM or startup problem.
  - Z should continue to ep39 to complete the full-augmentation K80 control. It is a same-run checkpoint resume, not warm-start/投敌.
- Decision: no kill, no resume, no new launch this loop. Continue manual `sleep 1800` loop.

## Manual Loop Update (2026-05-11 17:42-18:09 UTC)
- Woke from manual `sleep 1800` at `2026-05-11T17:42:30Z` and re-read the full `agents/` directory: `/tmp/deit_agents_full_read_20260511_174230.txt`.
- New evals:
  - AE K80 low-aug mixup-only ep39: `55.496%` single / `59.818%` iterative, loss `0.347846`, invalid `0.970%`.
  - AF K80 low-aug cutmix-only ep39: `58.938%` single / `63.560%` iterative, loss `0.325430`, invalid `0.824%`.
  - AG K160 low-aug mixed mix/cut ep19: `32.340%` single / `39.472%` iterative, loss `0.478325`, invalid `1.754%`.
  - AB K80 low-aug mixed mix/cut ep99: `70.380%` single / `73.076%` iterative, loss `0.262941`, invalid `0.424%`. This is the new best pure diffusion result.
  - AH K80 low-aug cutmix-heavy p=0.75 eval0: `0.074%` / iter `0.100%`, invalid `1.440%`; healthy startup.
- Active status after interventions:
  - S window 6433 is running around ep152; latest eval ep139 `58.748/63.468`.
  - AB window 6435 is running after ep100; latest eval ep99 `70.380/73.076`.
  - AF window 6441 is running around ep43; latest eval ep39 `58.938/63.560`.
  - AG window 6447 is running around ep23; latest eval ep19 `32.340/39.472`.
  - Z window 6446 is running around ep38 on v5p-8; no new eval in resume log yet, but ep39 should be checked next loop.
  - AH window 6448 is running on `j3rqvs`, launched this loop.
- Interpretation:
  - AF dominates AE at both ep19 and ep39, and AF ep39 also beats AB ep39 (`58.938/63.560` vs `57.148/61.642`). CutMix is currently the useful half of the corrected mix/cut idea.
  - AE is therefore not worth keeping: mixup-only is below both cutmix-only and mixed AB at ep39.
  - AB remains the best mature trajectory by a large margin: ep99 `70.380/73.076` improves over ep79 `68.464/71.404`; keep AB running.
  - AG K160 mixed is not an early win versus AB K80 mixed at ep19 (`32.340/39.472` vs AB ep19 `32.838/39.854`) and is far below AF K80 cutmix-only at ep19. Still keep AG to ep39 for a matched K160 verdict unless slot pressure increases.
  - Z full-augmentation K80 is still slow; keep to ep39 because user asked not to judge full augmentation too early.
- Action:
  - Retired/killed AE window 6440 after ep39 because it is dominated by AF and AB.
  - Created `configs/remote_run_AH_mlp_k80_lowaug_cutmix75_config.yml` and copied it to `configs/remote_run_config.yml` for launch.
  - Launched AH as window 6448 on `v6e-8-spot-gzy-j3rqvs`.
  - AH config verified in log: `model: ViT_base_mdh_mlp`, `n_masks_per_image: 80`, `diffusion_label_mode: soft_top2`, `switch_prob: 0.75`, `mixup_alpha: 0.8`, `cutmix_alpha: 1.0`, `load_from: ''`, `load_backbone_from: ''`, `head_aux_ce: false`, `aux_ce_loss_weight: 0.0`, `head_mlp_layer_norm: false`.
  - AH startup verified: first batch ready, initial compilation completed, train reached ep1+ and eval0 printed. No warm-start, no aux CE, no CE shortcut.
- Decision: keep AB, AF, AG, S, Z, AH. Next decisive checks are Z ep39, AB ep119, AF ep59, S ep159, AG ep39, and AH ep19.

## Manual Intervention (2026-05-11 18:45-19:02 UTC): DeiT reference sanity alignment
- User asked to compare against `../papers/DeiT` and prioritize reproducing the CE sanity run.
- Code/config alignment applied locally:
  - `configs/default.py`: default `adamw_b2` changed from `0.95` to `0.999` to match `../papers/DeiT/train.py`.
  - `configs/remote_run_sanity_config.yml`: sanity model changed from `ViT_base_v3` to `ViT_base_v2` so LayerScale/LearnedScale is disabled; `adamw_b2: 0.999`; top-level CE `label_smoothing: 0.0`; notes/tags updated to include exact GELU and no LayerScale.
  - `models.py`: backbone/diffusion GELU now uses `partial(nn.gelu, approximate=False)`, matching `../papers/DeiT/models.py`.
  - `train.py`: CE extra label smoothing default changed to 0.0 and `train_step_sqa` now reads `config.label_smoothing` instead of hardcoded 0.0. Dataset Mixup still supplies `dataset.label_smoothing=0.1`, so no double smoothing in CE.
  - `input_pipeline.py`: removed the extra pre-Mixup `randperm`; reference has this disabled/commented.
  - `configs/remote_run_config.yml` was copied from the updated sanity config before launch.
- Verification:
  - `python -m py_compile train.py models.py input_pipeline.py configs/default.py configs/load_config.py` passed.
  - `configs.load_config:get_config('remote_run_sanity')` prints `model=ViT_base_v2`, `adamw_b2=0.999`, `label_smoothing=0.0`, `dataset.label_smoothing=0.1`, `use_mixup_cutmix=True`.
- Launch attempts:
  - Tried sanity on reserved `v5p-8-tmp201` as window `6449`; failed before training because the TPU was occupied by an unrelated `LAION400M_upload.py` process and `/tmp/libtpu_lockfile` owned by another user. Do not kill that process; treat `6449` as infrastructure/occupancy failure, not code failure.
  - No free v6e-8 slots existed. Killed AG window `6447` on `06q7u9` at about ep33.6 (last useful eval was ep19 `32.340/39.472`; AG was lower priority than sanity).
  - Launched new CE sanity run as window `6450` on `v6e-8-spot-gzy-06q7u9` using `remote_run_config.yml` and no extra `config=` argument, avoiding duplicate `--config` in `run_remote.sh`.
- Sanity `6450` status:
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_185613_4pinj8_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`.
  - Log confirms `model: ViT_base_v2`, `adamw_b2: 0.999`, dataset `label_smoothing: 0.1`, top-level `label_smoothing: 0.0`, `stochastic_depth_rate: 0.1`, `weight_decay: 0.05`, `steps_per_epoch: 1251`.
  - Training started: step 100 `train_accuracy=0.0010449`, `train_loss=6.9962`, `ep=0.079`; step 300 `train_accuracy=0.0018555`, `train_loss=6.9441`, `ep=0.23898`.
  - Manager `check` may show `Unknown` because of remote `/tmp/tpu_logs/... Permission denied`, but the training log is progressing.
- Active priority after intervention: preserve `6450` until at least ep19/39/59 and compare against failed `6380` trajectory. Sanity reproduction is now the top priority.

## Manual Loop Update (2026-05-11 20:35-20:40 UTC)
- Woke/continued at `2026-05-11T20:35:25Z`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260511_203525.txt`.
- Active/important jobs:
  - `6450` CE sanity reference-aligned retry on `06q7u9`: running ep15.3. Config already confirmed as `ViT_base_v2`, no LayerScale, exact GELU, `adamw_b2=0.999`, CE extra smoothing 0.0. Latest train step `[19100]` ep15.265: train_acc `0.094541`, train_loss `5.8746`, lr `0.00099561`. Eval0 after epoch 0: `0.570%`, loss `6.79978`. Preserve until at least ep19/39; sanity reproduction is top priority.
  - `6435` AB K80 low-aug mixed mix/cut `soft_top2`: running ep129. New eval ep119: `71.886%` single / `74.052%` iterative, loss `0.248336`, invalid `0.354%`. This is the new best pure-diffusion result and still improving. Keep.
  - `6441` AF K80 low-aug cutmix-only: running ep73.8. New eval ep59: `66.298%` single / `69.818%` iterative, loss `0.278813`, invalid `0.656%`. AF remains stronger than AB at matched ep59 (`66.298/69.818` vs AB `64.890/68.616`), so keep to ep79/99 to see whether cutmix-only catches AB later.
  - `6433` S K10 full baseline aug mixed mix/cut: running ep176.3. New eval ep159: `61.124%` single / `65.528%` iterative, loss `0.316067`, invalid `0.608%`. Still improving slowly and should not be killed just for slow full-aug convergence, but it remains far below AB.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75: running ep31.9. New eval ep19: `32.018%` single / `39.488%` iterative, loss `0.482421`, invalid `1.684%`. This is much worse than AF ep19 `36.586/43.614` and roughly AB-like early. Keep only to ep39 for a fair verdict, then likely retire if still dominated.
  - `6446`/`6452` Z K80 full baseline aug mixed mix/cut resume: Z ep39 eval from `6446` was only `19.730%` single / `27.324%` iterative, loss `0.530267`, invalid `0.888%`. Auto resume `6452` failed. Treat Z as retired/no further resume unless user explicitly asks; this branch is dominated and infrastructure-noisy.
- Interpretation:
  - AB is still the mature best branch; ep119 `71.886/74.052` is strong and should continue.
  - AF cutmix-only is the best early/mid branch at matched epoch, so it is the most important Phase 2 comparison after sanity and AB.
  - AH's cutmix-heavy mixture does not preserve AF's early advantage; wait ep39 only.
  - Full augmentation S is viable but slow and currently not competitive; continue because user explicitly wanted full aug not killed early.
  - New CE sanity is progressing normally; next critical checkpoint is ep19 to compare against old sanity `6380` ep19 `38.770%`.
- Action: no new launch/resume. No kill this loop beyond deciding not to resume Z. Continue AB/AF/S/AH and preserve CE sanity `6450`.

## Manual Z Resume Fix (2026-05-11 20:40-20:55 UTC)
- User pointed out latest Z log contained a bug. Inspected latest Z auto-resume window `6452`:
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_192250_lbq41t_kmh-tpuvm-v5p-8-spot-gzy-96acuy_us-east5-a__b_lr_ep_eval`.
  - Failure was `FATAL Flags parsing error: The flag 'config.load_from' is defined twice.`
  - Root cause: `tpu.py resume` reused old `extra_configs` already containing `--config.load_from=...` and appended a new `--config.load_from=...`.
- Fixed tpu_manager root cause in `/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/utils/jobs.py`:
  - Added `_drop_config_overrides(config_args, *keys)`.
  - `resume_rerun_job` now strips stale `config.load_from` and `config.stage` before appending fresh resume overrides.
  - Verified with `python -m py_compile /kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/utils/jobs.py` and a direct helper test.
- Manual resume action:
  - To preserve Z code fidelity, used the old Z stage directory `/kmh-nfs-ssd-us-mount/staging/sqa/260511150017-30amhc--code` instead of current working tree (current tree has sanity GELU/b2 changes).
  - Registered that stage as manager dir `88`.
  - Edited the stage-local `configs/remote_run_config.yml` to set `load_from: '/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_150019_93fnd5_kmh-tpuvm-v5p-8-spot-katelyn-2-4_us-east5-a__b_lr_ep_eval'` and notes `deit phase2 Run Z resume3 manual: MLP K80 full baseline aug soft_top2 from 6446 checkpoint_50040; fixed duplicate config.load_from bug; no CE shortcut`.
  - Confirmed GCS checkpoint exists: `gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files/paligemma-baseline/20260511_150019_93fnd5_kmh-tpuvm-v5p-8-spot-katelyn-2-4_us-east5-a__b_lr_ep_eval/checkpoint_50040/`.
  - Launched manual Z resume3 as window `6453` on `v5p-8-tmp202` / `kmh-tpuvm-v5p-8-spot-gzy-96acuy`, with no command-line `--config.load_from`.
- Z `6453` verification:
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_204334_8km7qk_kmh-tpuvm-v5p-8-spot-gzy-96acuy_us-east5-a__b_lr_ep_eval`.
  - Log shows single `load_from` in config, no duplicate flag error.
  - Restored from `gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files/paligemma-baseline/20260511_150019_93fnd5_kmh-tpuvm-v5p-8-spot-katelyn-2-4_us-east5-a__b_lr_ep_eval/checkpoint_50040`.
  - Continued at `epoch 40...`; train printed `[50100]` ep40.043 loss `0.57983`, `[50400]` ep40.282 loss `0.58955`, `[50600]` ep40.442 loss `0.58107`. Resume is live.
- Note: manager may lag and show `Compiling`; log confirms training steps after restore.

## Manual Loop Update (2026-05-12 03:01-03:05 UTC)
- Woke/continued loop at `2026-05-12T03:01:13Z` and re-read the full `agents/` directory. Full read snapshot: `/tmp/deit_agents_full_read_20260512_030326.txt` (`4065` lines, sha256 `c5fc7c2758280a23cf583cd75add43eb526e354cece8e2895a7047ee01a3bd91`).
- Manager/log status: active DeiT jobs are `6433` S, `6435` AB, `6441` AF, `6448` AH, `6450` CE sanity, and `6453` Z manual resume. All six active logs have current mtimes and no fresh fatal/error lines.
- New useful evals since the last recorded loop:
  - `6435` AB K80 low-aug mixed mix/cut `soft_top2`: ep139 `72.550%` / iter `74.542%`; ep159 `73.496%` / iter `75.538%`; ep179 `73.716%` / iter `75.582%`; ep199 `74.064%` / iter `75.886%`, loss `0.257361`, invalid `0.290%`. AB remains the best mature pure-diffusion branch.
  - `6441` AF K80 low-aug cutmix-only: ep79 `67.614%` / iter `71.588%`; ep99 `70.444%` / iter `73.084%`; ep119 `71.636%` / iter `74.050%`; ep139 `72.502%` / iter `74.584%`, loss `0.250571`, invalid `0.366%`. AF is close to AB at matched ep139 but no longer clearly ahead.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75: ep39 `57.566%` / iter `62.152%`; ep59 `65.572%` / iter `69.300%`; ep79 `68.572%` / iter `72.092%`; ep99 `70.798%` / iter `73.598%`, loss `0.252995`, invalid `0.422%`. AH is slightly better than AB/AF at matched ep99, so do not retire it.
  - `6433` S K10 full baseline aug mixed mix/cut: ep179 `63.072%` / iter `66.962%`; ep199 `64.112%` / iter `67.970%`; ep219 `65.662%` / iter `69.198%`, loss `0.296665`, invalid `0.574%`. S is still improving slowly but remains below AB/AF/AH K80 branches.
  - `6450` CE sanity reference-aligned retry: ep19 `29.264%`, ep39 `48.716%`, ep59 `50.392%`, loss `2.402908`; currently training around ep77. This does not reproduce old sanity `6380` trajectory yet: old v3 sanity had ep19 `38.770%`, ep39 `54.470%`, ep59 `59.132%`, ep99 `62.754%`, ep319 `72.814%`.
  - `6453` Z K80 full baseline aug mixed mix/cut manual resume: ep59 `40.768%` / iter `47.954%`, loss `0.428019`, invalid `0.832%`; currently training around ep77.5. This confirms the manual Z resume is live beyond the earlier duplicate-flag bug.
- Interpretation:
  - Best current branch remains AB, but AH has become important: p=0.75 cutmix-heavy is the best matched ep99 among AB/AF/AH and must continue at least to ep119/139.
  - AF's early lead over AB shrinks by ep139; pure cutmix-only is competitive but not obviously superior to mixed AB at later epochs.
  - S validates the user warning that full augmentation has delayed catch-up, but even at ep219 it is still behind K80 low-aug/mix branches. Keep for the full-aug curve unless slot pressure becomes severe.
  - Z full-aug K80 is much healthier after resume than its ep39 result suggested, but still substantially behind low-aug K80; keep to ep79/99 because it is the only K80 full-aug control.
  - CE sanity is still not reproduced. The only intentional delta from old `6380` that clearly changed performance is switching `ViT_base_v3` to `ViT_base_v2` (removing LayerScale) plus exact GELU/b2 alignment. Since old v3 was already better but still below DeiT reference, next useful sanity ablation when a slot opens is a v3 sanity with the current exact-GELU/b2/input-pipeline fixes, before deeper changes.
- Action/decision this loop: no kill, no resume, no new launch. No reliable free slot exists, and each active job still has a reason to continue. Next checks: AB ep219, AF ep159, AH ep119, S ep239, CE sanity ep79/99, Z ep79/99.

## Manual Loop Update (2026-05-12 03:34-03:37 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 03:34:07 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_033414.txt` (`4097` lines, sha256 `96b0adacafe73715a73c92d858195912f71d473bdae25f7001ca43b9fe29feb0`). Re-read `agents/notes.md` tail and active records before checking logs.
- Manager/log status: effective active DeiT windows remain `6433` S, `6435` AB, `6441` AF, `6448` AH, `6450` CE sanity, and `6453` Z. All six logs are live; no fresh fatal/error/OOM lines.
- New evals this half-hour:
  - `6433` S K10 full baseline aug mixed mix/cut: ep239 `66.574%` single / `70.134%` iterative, loss `0.297908`, invalid `0.474%`. Still improving slowly after ep219 `65.662/69.198`.
  - `6450` CE sanity reference-aligned retry: ep79 `54.122%`, loss `2.187716`. This is still below old `6380` v3 sanity at ep79 `59.342%`; sanity reproduction remains unresolved.
  - `6453` Z K80 full baseline aug mixed mix/cut manual resume: ep79 `52.408%` single / `58.076%` iterative, loss `0.359067`, invalid `0.584%`. Z is catching up substantially versus ep59 `40.768/47.954`, so do not kill before ep99.
- No new eval yet for:
  - AB: running ep210; latest ep199 `74.064/75.886`.
  - AF: running ep157-158; latest ep139 `72.502/74.584`; ep159 imminent.
  - AH: running ep114; latest ep99 `70.798/73.598`; ep119 next.
- Interpretation:
  - AB remains best overall, but the next useful comparison is whether AH maintains its ep99 matched advantage at ep119/139.
  - Z's full-aug K80 curve is no longer a total failure after the resume fix; it is still below low-aug K80, but now worth continuing to ep99.
  - S full-aug K10 continues delayed catch-up and now crossed 70% iterative at ep239; keep for complete full-aug curve.
  - CE sanity still underperforms the old v3 sanity by about 5.2 points at ep79. Keep to ep99 before deciding whether to free/replace with a v3+current-fixes sanity ablation.
- Action/decision: no kill, no resume, no new launch. Continue current six jobs. Next checks: AF ep159, AH ep119, AB ep219, CE ep99, Z ep99, S ep259.

## Manual Loop Update (2026-05-12 04:05-04:08 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 04:05:02 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_040510.txt` (`4129` lines, sha256 `f87657f737fa116a4414df91679c2f9576385c45881653061bd91f597c4ba05e`). Re-read `agents/notes.md` before status/log checks.
- Manager/log status: effective active DeiT windows remain healthy: `6433` S ep247, `6435` AB ep216, `6441` AF ep162, `6448` AH ep119-120, `6450` CE sanity ep87, `6453` Z ep83. No fresh fatal/error/OOM lines in active logs.
- New evals this half-hour:
  - `6441` AF K80 low-aug cutmix-only ep159: `72.946%` single / `74.926%` iterative, loss `0.252867`, invalid `0.406%`.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep119: `72.216%` single / `74.424%` iterative, loss `0.249503`, invalid `0.408%`.
- Matched-epoch interpretation:
  - AF ep159 is still below AB ep159 (`73.496%` / iter `75.538%`), so pure cutmix-only is competitive but not beating mixed AB at mature epochs.
  - AH ep119 is above AB ep119 (`71.886%` / iter `74.052%`) and AF ep119 (`71.636%` / iter `74.050%`), so cutmix-heavy p=0.75 remains the most interesting AB/AF/AH variant at matched ep119. Keep AH to ep139/159.
- No new eval yet for:
  - AB: running ep216; latest ep199 `74.064/75.886`; ep219 soon.
  - S: running ep247; latest ep239 `66.574/70.134`.
  - CE sanity: running ep87; latest ep79 `54.122`; still below old v3 sanity.
  - Z: running ep83; latest ep79 `52.408/58.076`.
- Action/decision: no kill, no resume, no new launch. Keep all six active jobs. Next checks: AB ep219, AH ep139, AF ep179, CE ep99, Z ep99, S ep259.

## Manual Loop Update (2026-05-12 04:35-04:38 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 04:35:52 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_043559.txt` (`4159` lines, sha256 `fe3bdd0a0253a88503e0f061f13fadf253e6509afcbafac0b5a458fc77b77413`). Re-read `agents/notes.md` before status/log checks.
- Manager/log status: effective active DeiT windows remain healthy: `6433` S ep252, `6435` AB ep221-222, `6441` AF ep168-170, `6448` AH ep126, `6450` CE sanity ep92, `6453` Z ep86. No fresh fatal/error/OOM lines in active logs.
- New eval this half-hour:
  - `6435` AB K80 low-aug mixed mix/cut ep219: `74.726%` single / `76.526%` iterative, loss `0.255938`, invalid `0.298%`. This is the new best pure-diffusion result and AB is still improving after ep199 `74.064/75.886`.
- No new eval yet for:
  - AF: running ep170; latest ep159 `72.946/74.926`; ep179 next.
  - AH: running ep126; latest ep119 `72.216/74.424`; ep139 next.
  - S: running ep252; latest ep239 `66.574/70.134`; ep259 next.
  - CE sanity: running ep92; latest ep79 `54.122`; ep99 next.
  - Z: running ep86; latest ep79 `52.408/58.076`; ep99 next.
- Interpretation:
  - AB remains the best mature branch and is not fully saturated; keep at least to ep239/259.
  - AH remains the main candidate to challenge AB because it was better at matched ep119; wait ep139/159.
  - CE sanity is still top diagnostic concern but needs ep99 before action.
- Action/decision: no kill, no resume, no new launch. Continue current six jobs. Next checks: AH ep139, AF ep179, S ep259, CE ep99, Z ep99, AB ep239.

## Manual Loop Update (2026-05-12 05:06-05:09 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 05:06:39 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_050649.txt` (`4190` lines, sha256 `48225a2cd12b472f882d78ef75d30ab8171d14793ee352ca1ad71bd76d4b5601`). Re-read `agents/notes.md` before status/log checks.
- Manager/log status: effective active DeiT windows remain healthy: `6433` S ep257, `6435` AB ep227-228, `6441` AF ep175-176, `6448` AH ep132, `6450` CE sanity ep97, `6453` Z ep89-90. No fresh fatal/error/OOM lines in active logs.
- New useful evals this half-hour: none.
- Latest useful evals remain:
  - AB ep219 `74.726/76.526`.
  - AF ep159 `72.946/74.926`.
  - AH ep119 `72.216/74.424`.
  - S ep239 `66.574/70.134`.
  - CE sanity ep79 `54.122`.
  - Z ep79 `52.408/58.076`.
- Near-term expected evals:
  - S ep259 should arrive soon.
  - AF ep179 should arrive soon.
  - AH ep139 should arrive by next loop.
  - CE sanity ep99 should arrive by next loop; this is important for deciding whether to replace it with a v3+current-fixes sanity ablation.
  - Z ep99 will likely take longer because v5p throughput is about 2.0-2.2 steps/s.
- Action/decision: no kill, no resume, no new launch. Continue current six jobs. Next checks: S ep259, AF ep179, AH ep139, CE ep99, AB ep239, Z progress.

## Manual Loop Update (2026-05-12 05:38-05:45 UTC)
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_053840.txt` (`4223` lines, sha256 `8e24f9c2b133dd743bb569d275bc1dda7504cb7c4b6f64af06eeec105823a3c3`). Re-read `agents/notes.md` before status/log checks.
- Manager/log status before intervention: effective DeiT windows were `6433` S, `6435` AB, `6441` AF, `6448` AH, `6450` CE sanity, and `6453` Z; no fresh fatal/OOM/Traceback in active logs.
- New evals this loop:
  - `6433` S K10 full baseline aug mixed mix/cut ep259: `67.742%` single / `70.980%` iterative, loss `0.292185`, invalid `0.424%`. S is still slowly improving, so keep.
  - `6441` AF K80 low-aug cutmix-only ep179: `73.300%` single / `74.798%` iterative, loss `0.259000`, invalid `0.288%`. AF remains below AB at matched/nearby mature epochs and may be flattening, but keep to ep199 unless slot pressure changes.
  - `6450` CE sanity v2/no-LayerScale ep99: `56.084%`, loss `2.065007`. This is clearly not reproduced: old `6380` v3 sanity was ep99 `62.754%`, and current CE is also behind old v3 at ep19/39/59/79.
- Decision on sanity: retire `6450` after ep99 rather than spending more on a clearly inferior v2/no-LayerScale trajectory. The next most diagnostic sanity is to keep the old `ViT_base_v3`/LearnedScale architecture while retaining current reference-alignment fixes (exact GELU, AdamW `b2=0.999`, CE label smoothing wiring, input-pipeline fix, WD mask). This tests whether the v2/no-LayerScale change caused the regression.
- Config/action:
  - Added `configs/remote_run_CF_sanity_v3_currentfixes_config.yml` and copied it to `configs/remote_run_config.yml` for launch.
  - Verified with `get_config('remote_run')`: `model=ViT_base_v3`, `adamw_b2=0.999`, top-level `label_smoothing=0.0`, dataset `label_smoothing=0.1`, `dataset.use_mixup_cutmix=True`.
  - Explicitly no warm start, no `load_backbone_from`, no diffusion shortcut, no aux CE. This is a fresh CE sanity run.
  - First launch attempt used the wrong manager syntax `--config=...` and failed locally with `Unknown config key --config` after killing remote CE processes; correct syntax is `config=configs/load_config.py:remote_run`.
  - Killed/stopped `6450`, cleaned remote processes on `kmh-tpuvm-v6e-8-spot-gzy-06q7u9`, then launched CF as window `6457` on the same TPU with `config=configs/load_config.py:remote_run`.
- `6457` launch details:
  - Window: `6457`.
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260512_054135_4vf63h_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`.
  - Stage dir: `/kmh-nfs-ssd-us-mount/staging/sqa/260512054134-oqt6bx-921e605-code`.
  - Status shortly after launch: manager `Unknown`/staging, log reached parameter print, `Initial compilation`, and `epoch 0...`.
  - Log contains repeated `/tmp/tpu_logs/... Permission denied` TPU driver-log messages, but no Python traceback/fatal/OOM in the checked tail; treat as permission noise unless it escalates.
- Other current state after intervention:
  - `6433` S running ep263.
  - `6435` AB running ep235; latest eval remains ep219 `74.726/76.526`.
  - `6441` AF running ep182; latest eval ep179 `73.300/74.798`.
  - `6448` AH running ep139; latest eval remains ep119 `72.216/74.424`, ep139 eval should arrive soon.
  - `6453` Z running ep93; latest eval remains ep79 `52.408/58.076`.
  - `6457` CF sanity compiling/starting epoch 0.
- Next checks: CF eval0/ep19 and any startup error; AH ep139; AB ep239; AF ep199; S ep279; Z ep99. Do not auto-resume stale `6450`; effective sanity is now `6457`.

## Manual Loop Update (2026-05-12 06:14-06:18 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 06:14:01 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_061408.txt` (`4266` lines, sha256 `5df5eb7c06b48bfd4335d9c639f4ad5b236f9f285952e33ac2639d1d21156525`). Re-read `agents/notes.md` before status/log checks.
- Manager status: effective active DeiT windows are `6433` S ep268, `6435` AB ep240, `6441` AF ep188-189, `6448` AH ep145, `6453` Z ep96, and `6457` CF sanity ep4.6. Stale/failed `6450` should stay ignored/killed; effective sanity is `6457`.
- New evals/results this loop:
  - `6435` AB K80 low-aug mixed mix/cut ep239: `75.504%` single / `76.992%` iterative, loss `0.255311`, invalid `0.218%`. This is the new best pure-diffusion result and AB is still climbing after ep219 `74.726/76.526`.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep139: `73.224%` single / `75.284%` iterative, loss `0.247610`, invalid `0.342%`. AH remains better than AB at matched ep139 (`72.550/74.542`) and AF ep139 (`72.502/74.584`), but it is still behind AB's mature ep239 curve.
  - `6457` CF sanity v3/current-fixes eval0: `0.096%`, loss `6.927275`; training alive at ep4.635 with train accuracy `0.035273`, train loss `6.3514`, ~`3.4` steps/s. No Python fatal/Traceback/OOM in checked tail. The repeated `/tmp/tpu_logs/... Permission denied` messages remain driver-log permission noise.
- No new eval yet for:
  - S: latest ep259 `67.742/70.980`, running ep268; ep279 next.
  - AF: latest ep179 `73.300/74.798`, running ep189; ep199 next.
  - Z: latest ep79 `52.408/58.076`, running ep96; ep99 next.
- Interpretation:
  - AB is the best deployable pure-diffusion config so far. It is still improving late, so keep it to ep259+.
  - AH confirms that cutmix-heavy p=0.75 is stronger than AB at matched early/mid epochs (ep119/139), so do not retire it before ep159/179 even though AB's mature curve is ahead.
  - AF is likely dominated by AB/AH, but keep to ep199 for a clean mature cutmix-only datapoint unless slot pressure becomes urgent.
  - CF sanity is now the correct sanity branch to watch; eval0 alone is not informative, but startup is healthy.
- Action/decision: no kill, no resume, no launch. Continue S/AB/AF/AH/Z/CF. Next checks: Z ep99, AB ep259, AH ep159, AF ep199, S ep279, CF ep19.

## Manual Loop Update (2026-05-12 06:44-06:49 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 06:44:51 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_064459.txt` (`4298` lines, sha256 `487fda2d6ad286375dc21a758148bbdc758cc169b53ba3520ede318be3755ed6`). Re-read `agents/notes.md` before status/log checks.
- Manager status: effective active DeiT windows are `6433` S ep273, `6435` AB ep247, `6441` AF ep194-196, `6448` AH ep151, `6453` Z ep99-100, and `6457` CF sanity ep9.6. All six effective logs are live; no fresh fatal/OOM/Traceback in effective active logs.
- Stale auto-rerun cleanup:
  - Old failed no-LayerScale sanity `6449` was auto-rerun by manager as `6458` on `v6e-64-spot-sqa-mbc06d` and then failed. This is not an intended/effective sanity branch; effective sanity is CF `6457`.
  - Ran `ignore-error 6458 sqa` to prevent the stale no-LayerScale branch from driving future decisions. `6458` log reached train ep~5 at high throughput before thread Tracebacks, then failed; do not resume unless explicitly requested.
- New eval/result this loop:
  - `6453` Z K80 full baseline aug mixed mix/cut manual resume ep99: `57.724%` single / `62.532%` iterative, loss `0.335394`, invalid `0.574%`. Z is still catching up from ep79 `52.408/58.076`, but remains far below low-aug K80 at matched ep99 (`AB 70.380/73.076`, `AH 70.798/73.598`). Keep to ep119/139 because it is the only K80 full-aug control and full augmentation may have delayed convergence.
- No new eval yet for:
  - S: latest ep259 `67.742/70.980`, running ep273; ep279 next.
  - AB: latest ep239 `75.504/76.992`, running ep247; ep259 next.
  - AF: latest ep179 `73.300/74.798`, running ep196; ep199 next.
  - AH: latest ep139 `73.224/75.284`, running ep151; ep159 next.
  - CF sanity: latest eval0 `0.096%`, running ep9.6; ep19 next.
- Interpretation:
  - AB remains best overall and should continue.
  - AH remains an important matched-epoch challenger; continue to ep159/179.
  - Z shows K80 full augmentation is not dead, but the gap to low-aug K80 is still very large at ep99. Do not kill yet because this branch directly tests the user's augmentation concern.
  - AF is still likely dominated; wait ep199 before deciding whether it can be replaced by another augmentation/control branch.
  - CF sanity startup remains healthy; the meaningful sanity comparison is ep19 versus old v3 sanity `38.770%` and failed v2/no-LayerScale `29.264%`.
- Action/decision: ignore stale `6458`; no kill/resume/launch for effective jobs. Continue S/AB/AF/AH/Z/CF. Next checks: AF ep199, AH ep159, AB ep259, S ep279, Z ep119, CF ep19.

## Manual Loop Update (2026-05-12 07:16-07:33 UTC)
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_072001.txt` (`4335` lines, sha256 `88bede4a0ec655db14e0f93f7465b14d550d8419acfe7835ff0b989b864c644a`). Re-read `agents/notes.md` before status/log checks.
- Manager status at start: effective active DeiT windows were `6433` S, `6435` AB, `6441` AF, `6448` AH, `6453` Z, and `6457` CF sanity. Stale parent `6381` still reports `resumed in 6459`, but no active `6459` window appeared in focused checks after cleanup; keep verifying absence because `6459` was the forbidden aux-CE Run H auto-resume.
- New evals/results this loop:
  - `6441` AF K80 low-aug cutmix-only ep199: `73.162%` single / `74.960%` iterative, loss `0.268570`, invalid `0.284%`. This is dominated by AB at matched ep199 (`74.064/75.886`) and is not improving versus AF ep179 (`73.300/74.798`).
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep159: `73.402%` single / `75.440%` iterative, loss `0.249140`, invalid `0.306%`. AH was better than AB at ep119/139, but by ep159 it is slightly below AB ep159 (`73.496/75.538`). Early cutmix-heavy acceleration did not translate into a mature advantage.
  - `6433` S K10 full baseline augmentation ep279: `68.450%` single / `71.582%` iterative, loss `0.293842`, invalid `0.472%`. S is still improving slowly; keep because full augmentation delayed convergence was expected.
- No new useful eval yet for:
  - `6435` AB: latest ep239 `75.504/76.992`, running ep256; ep259 next.
  - `6453` Z: latest ep99 `57.724/62.532`, running ep104; keep to ep119/139 as K80 full-aug control.
  - `6457` CF sanity: eval0 `0.096%`, running ep17; ep19 imminent and critical for sanity reproduction.
- Decision on AF slot: retired/killed `6441` after ep199 because pure cutmix-only is clearly dominated. Do not resume AF unless explicitly requested.
- Launched new interpolation branch `6460` AI on `kmh-tpuvm-v6e-8-spot-gzy-z169mq`:
  - Config: `configs/remote_run_AI_mlp_k80_lowaug_cutmix625_config.yml`; copied to `configs/remote_run_config.yml` for launch.
  - Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260512_073102_nkjdxz_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval`.
  - Stage dir: `/kmh-nfs-ssd-us-mount/staging/sqa/260512073101-ziooni-921e605-code`.
  - Purpose: interpolate between AB `switch_prob=0.5` and AH `switch_prob=0.75` after AH's early advantage faded by ep159; AI uses `switch_prob=0.625` while keeping K80, low-aug, mixup alpha `0.8`, cutmix alpha `1.0`, AdamW b2 `0.95`, WD `0.02`, SD `0.05`, MLP head and `soft_top2` labels fixed.
  - Safety: explicit `load_from: ''`, `load_backbone_from: ''`, `head_aux_ce: false`, `aux_ce_loss_weight: 0.0`; no warm-start, no backbone load, no aux CE, no CE shortcut.
  - Startup: manager window `6460`; status moved from `Staging` to `Initializing`; log reached `Initial compilation` and `epoch 0...`; no startup traceback/OOM in checked tail.
- Action/decision summary: killed dominated AF, launched AI p=0.625. Continue S/AB/AH/Z/CF/AI. Next checks: AI startup/eval0, CF ep19, AB ep259, AH ep179, S ep299, Z ep119, and verify no active `6459` resurrection.

## Manual Loop Update (2026-05-12 08:03-08:06 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 08:03:40 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_080340.txt` (`4371` lines, sha256 `6fb50f060be6198ad386e3ebcd513b685c2fa6f66b5c1f63ff66648cac9ce005`). Re-read `agents/notes.md` before status/log checks.
- Manager status: effective active DeiT windows are `6433` S ep285, `6435` AB ep262, `6448` AH ep166, `6453` Z ep107-108, `6457` CF sanity ep22, and `6460` AI ep4-5. No effective active log has fresh fatal/OOM/Traceback in checked tails.
- Forbidden aux-CE status: old parent `6381` still says `resumed in 6459`, but focused manager check shows no active `6459` window. Continue verifying absence; do not resume Run H or any aux-CE branch.
- New evals/results this loop:
  - `6435` AB K80 low-aug mixed mix/cut ep259: `75.758%` single / `77.414%` iterative, loss `0.256073`, invalid `0.250%`. This is the new best pure-diffusion result; AB is still improving after ep239 `75.504/76.992`.
  - `6457` CF sanity v3/current-fixes ep19: `37.472%`, loss `3.081337`. This is close to old v3 sanity ep19 `38.770%` and much better than failed v2/no-LayerScale ep19 `29.264%`; early evidence points to the v2/no-LayerScale architecture as the main sanity regression source, not the current exact-GELU/b2/input fixes. Keep to ep39/59.
  - `6460` AI K80 low-aug p=0.625 startup: eval0 `0.094%` single / `0.132%` iterative, loss `0.691232`; training healthy at ep5.83 with train accuracy `0.061972`, loss `0.63936`, ~`4.3` steps/s; no startup fatal/OOM.
- No new useful eval yet for:
  - `6433` S: latest ep279 `68.450/71.582`, running ep285; keep to ep299.
  - `6448` AH: latest ep159 `73.402/75.440`, running ep166; keep to ep179 for mature p=0.75 curve.
  - `6453` Z: latest ep99 `57.724/62.532`, running ep108; keep to ep119/139 as full-aug K80 control.
- Interpretation:
  - AB remains the best mature deployable configuration and should keep running.
  - CF now largely validates the old v3 sanity path at ep19; wait ep39 before declaring reproduced, but do not kill.
  - AI is alive and will provide the p=0.625 interpolation; first meaningful comparison is ep19/39 against AB/AH/AF.
  - S/Z full-aug controls are slow but still required for the augmentation question; do not kill.
- Action/decision: no kill, no resume, no launch. Continue S/AB/AH/Z/CF/AI. Next checks: AI progress/eval19, CF ep39, AH ep179, AB ep279, S ep299, Z ep119, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 08:35-08:38 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 08:35:28 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_083528.txt` (`4405` lines, sha256 `2d6c76ca7e7d7a7ad09106d695d5140116025501a3ade9c6f1c2020fe0303534`). Re-read `agents/notes.md` before status/log checks.
- Manager status: effective active DeiT windows are `6433` S ep290-291, `6435` AB ep268, `6448` AH ep172-173, `6453` Z ep111, `6457` CF sanity ep27, and `6460` AI ep10-12. No fresh fatal/OOM/Traceback in effective active logs.
- Stale auto-rerun cleanup:
  - Old failed no-LayerScale sanity `6458` was auto-resumed again as `6463` on `v5p-128-dmy-e28b` and failed. This is not an intended/effective sanity branch; effective sanity remains CF `6457`.
  - Ran `ignore-error 6463 sqa`; manager acknowledged monitor. `6458` still shows parent state `resumed in 6463`, but `6463` is stale/ignored and must not be resumed unless explicitly requested.
- Forbidden aux-CE status: old parent `6381` still says `resumed in 6459`, but no active `6459` window appears. Continue verifying absence; do not resume Run H or any aux-CE branch.
- New useful evals this loop: none.
- Current effective latest/status:
  - `6435` AB: latest ep259 `75.758/77.414`, running ep268.8; remains best pure-diffusion branch.
  - `6460` AI p=0.625: eval0 `0.094/0.132`, running ep12.1 with train acc ~`0.244`, loss ~`0.572`, healthy; ep19 expected in ~35-40 min.
  - `6457` CF sanity: latest ep19 `37.472`, running ep27.3 with train acc ~`0.279`, loss ~`4.87`, healthy; ep39 expected later.
  - `6448` AH p=0.75: latest ep159 `73.402/75.440`, running ep173.3; ep179 expected near next loop.
  - `6433` S full-aug K10: latest ep279 `68.450/71.582`, running ep290.9; ep299 expected after next loop or near it.
  - `6453` Z full-aug K80: latest ep99 `57.724/62.532`, running ep111.3; ep119 still later because v5p throughput is ~2.1 steps/s.
- Interpretation:
  - No effective job requires intervention. Keep all six effective branches.
  - The only action this loop was administrative cleanup of stale `6463`; it should not influence DeiT decisions.
  - Next loop is likely to catch AH ep179 and possibly AI approaching ep19; CF ep39/Z ep119/AB ep279/S ep299 likely need another loop.
- Action/decision: ignored stale `6463`; no kill/resume/launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AH ep179, AI ep19 progress, AB ep279, S ep299, Z ep119, CF ep39, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 09:07-09:19 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 09:07:13 UTC`.
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_090713.txt` (`4442` lines, sha256 `3fecdf82d54cef7897be7b445d70494b164d0ad19b8e93fa29f91f4fc87a133d`). Re-read `agents/notes.md` before status/log checks.
- Manager status at wake: effective active DeiT windows were `6433` S ep295, `6435` AB ep274, `6448` AH ep179, `6453` Z ep114, `6457` CF sanity ep32, and `6460` AI ep17-18. CF manager status briefly showed `Unknown` due `/tmp/tpu_logs` permission noise, but the real train log is live and healthy.
- Short-wait rationale: AH was at ep179.4 and AI at ep18.5, so waited ~10 minutes to catch imminent evals before writing the loop record.
- New evals/results:
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep179: `74.010%` single / `75.796%` iterative, loss `0.251355`, invalid `0.342%`. This is better than matched AB ep179 (`73.716/75.582`) and AF ep179 (`73.300/74.798`). AH's ep159 dip versus AB was not a clear death signal; keep to ep199/219.
  - `6460` AI K80 low-aug p=0.625 ep19: `33.830%` single / `41.432%` iterative, loss `0.466993`, invalid `2.096%`. AI is better than AB ep19 (`32.838/39.854`) and AH ep19 (`32.018/39.488`), but below AF pure cutmix ep19 (`36.586/43.614`). This is consistent with interpolation: some cutmix early acceleration, not as aggressive as pure cutmix. Keep to ep39/59 before judging.
- No new useful eval yet for:
  - `6435` AB: latest ep259 `75.758/77.414`, running ep274.9; ep279 next.
  - `6433` S: latest ep279 `68.450/71.582`, running ep296; ep299 next.
  - `6453` Z: latest ep99 `57.724/62.532`, running ep114.5; ep119 later.
  - `6457` CF sanity: latest ep19 `37.472`, running ep34.1 after short wait; ep39 later.
- Stale/forbidden status:
  - `6463` old no-LayerScale sanity auto-resume still appears as error from `6458`; ran `ignore-error 6463 sqa` again. Treat as stale/ignored. Effective sanity is `6457` only.
  - Old aux-CE parent `6381` still references `resumed in 6459`, but no active `6459` window appears. Continue verifying absence; do not resume Run H or any aux-CE branch.
- Interpretation:
  - AB remains best overall at mature epochs, but AH p=0.75 is competitive at matched ep179 and should continue.
  - AI p=0.625 has a better ep19 than AB/AH but not AF; this is not enough to judge final quality. Continue.
  - S/Z full-aug controls continue; no evidence justifies killing them.
  - CF sanity remains on-track enough to continue to ep39/59.
- Action/decision: no kill/resume/launch for effective jobs. Administrative `ignore-error 6463` only. Continue S/AB/AH/Z/CF/AI. Next checks: AB ep279, S ep299, Z ep119, CF ep39, AI ep39 progress, AH ep199, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 10:01-10:06 UTC)
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_100134.txt` (`4480` lines, sha256 `8a3cf4e7bcb24f22fc56cbc71bf8811d7ba364b8f810f0010da08b2f8598aa62`). Re-read `agents/notes.md` before status/log checks.
- Manager status: effective active DeiT windows are `6433` S ep304, `6435` AB ep285, `6448` AH ep189, `6453` Z ep119, `6457` CF sanity ep41, and `6460` AI ep27-29. All six effective logs are live; no fresh fatal/OOM/Traceback in checked tails.
- New evals/results captured this loop:
  - `6435` AB K80 low-aug mixed mix/cut ep279: `76.190%` single / `77.848%` iterative, loss `0.255026`, invalid `0.228%`. This is the new best pure-diffusion result. AB is still improving after ep239 `75.504/76.992` and ep259 `75.758/77.414`; continue.
  - `6433` S K10 full baseline augmentation ep299: `68.774%` single / `72.158%` iterative, loss `0.297835`, invalid `0.442%`. S is slow but still improving versus ep279 `68.450/71.582`; keep as the K10 full-augmentation control.
  - `6453` Z K80 full baseline augmentation ep119: `60.146%` single / `64.446%` iterative, loss `0.323961`, invalid `0.622%`. Z improved from ep99 `57.724/62.532`, but remains far below low-aug K80 at matched ep119 (`AB 71.886/74.052`, `AH 72.216/74.424`). Keep because it directly tests whether full augmentation only delays late convergence; next meaningful point is ep139.
  - `6457` CF sanity v3/current-fixes ep39: `49.134%`, loss `2.422293`. This is healthy and much better than the failed v2/no-LayerScale branch, but it has not yet convincingly reproduced the old v3 sanity trajectory; continue to ep59/79 before deciding.
- Other active latest/status:
  - `6448` AH p=0.75 latest ep179 `74.010/75.796`, running ep189. Keep to ep199/219 because it remains competitive at matched epochs.
  - `6460` AI p=0.625 latest ep19 `33.830/41.432`, running ep29. Keep to ep39/59 before judging the interpolation.
- Stale/forbidden status:
  - `6463` old no-LayerScale sanity auto-resume still shows Error from stale `6458`; treat as stale/ignored. Effective sanity is only `6457` CF.
  - Old aux-CE parent `6381` still reports `resumed in 6459`, but no active `6459` window appears in the manager listing. Continue verifying absence; do not resume Run H or any aux-CE branch.
- Interpretation:
  - AB remains the best deployable pure-diffusion configuration so far and should run onward.
  - Full augmentation (`S`, `Z`) is still significantly behind low-aug at matched epochs, but both continue improving, so killing them now would leave the augmentation question underdetermined.
  - CF sanity is no longer a startup/code-path bug; the important next check is whether it reaches the old v3 sanity neighborhood by ep59/79.
  - No effective job requires resume or relaunch.
- Action/decision: no kill, no resume, no launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AH ep199, AI ep39, Z ep139, CF ep59, AB ep299, S ep319, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 10:33-10:50 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 10:32:46 UTC` and started the loop by re-reading the full `agents/` directory.
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_103328.txt` (`4510` lines, sha256 `b0079f19a7951316a420eb4d79c489737d68244befd4917a0ce1d83ac437586a`). Re-read `agents/notes.md` before status/log checks.
- Manager/log status before intervention:
  - Effective healthy jobs: `6433` S ep309, `6435` AB ep291, `6448` AH ep195-196, `6453` Z ep123, `6457` CF ep46. No fresh fatal/OOM/Traceback in these five effective logs.
  - `6460` AI had stopped at ep31.411 after SSH/spot failure (`Connection refused/timed out`, gcloud ssh return code 255). Manager had auto-resumed it to `6467`, but `6467` failed immediately.
- New evals this loop: none beyond the 10:01 loop. Latest effective evals remain S ep299 `68.774/72.158`, AB ep279 `76.190/77.848`, AH ep179 `74.010/75.796`, Z ep119 `60.146/64.446`, CF ep39 `49.134`, AI ep19 `33.830/41.432`.
- AI failure diagnosis:
  - Original `6460` failure was infrastructure/spot SSH loss, not a model/code failure. It had reached train step `39300` / ep `31.411` after checkpointing at ep19.
  - Auto-resume `6467` on `oulo2v` failed with `FATAL Flags parsing error: The flag 'config.load_from' is defined twice`.
  - Cause: manager resume path passed both the stored `--config=configs/load_config.py:remote_run` and a new `--config.load_from=...`; because `run_remote.sh` already hard-codes `--config=configs/load_config.py:remote_run`, this caused duplicate config/override flag registration during resume. This is the same class of resume-flag bug previously seen on Z, but through a slightly different path.
- AI recovery action:
  - Killed stale/wrong attempts: `6468` was accidentally launched once with the default manager directory `jax_gpt` because `tpu.py run` defaults to dir id `1`; killed it immediately. Correct DeiT dir id is `7`.
  - Relaunched AI manually as `6469` on `kmh-tpuvm-v6e-8-spot-gzy-oulo2v` with `dir=7` and only one explicit override: `config.load_from=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260512_073102_nkjdxz_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval`.
  - `6469` logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260512_104015_jt7a0v_kmh-tpuvm-v6e-8-spot-gzy-oulo2v_asia-northeast1-b__b_lr_ep_eval`.
  - Verified config in `6469`: `model=ViT_base_mdh_mlp`, `n_masks_per_image=80`, `switch_prob=0.625`, `load_from` is the original AI own logdir, `load_backbone_from=''`, `head_aux_ce=false`, `aux_ce_loss_weight=0.0`. This is same-run checkpoint resume only; no warm-start, no backbone loading, no aux CE, no CE shortcut.
  - Verified restore: `6469` restored from `gs://kmh-gcp-asia-northeast1-b/.../20260512_073102.../checkpoint_25020`, printed `Initial compilation`, and resumed at `epoch 20...`.
- Administrative cleanup: ran `ignore-error 6467 sqa` and `ignore-error 6463 sqa`. Keep treating `6463` as stale no-LayerScale sanity and `6467` as stale failed AI auto-resume. Effective AI is now `6469`.
- Interpretation:
  - The p=0.625 AI branch is still worth continuing, but because only `checkpoint_25020` existed, the manual resume restarts from ep20 and loses the ep20-31 compute after the spot failure.
  - Do not use manager auto-resume blindly for this branch; if another resume is needed, either remove duplicate `--config` from the resume path or use the manual `dir=7 ... config.load_from=<own logdir>` pattern.
  - Other branches remain healthy and should continue.
- Action/decision: AI manually recovered as `6469`; no new scientific branch launched. Continue effective jobs S/AB/AH/Z/CF/AI(`6469`). Next checks: `6469` first train lines after compile, AH ep199, AI ep39 after resumed progress, Z ep139, CF ep59, AB ep299, S ep319, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 11:19-11:23 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 11:18:49 UTC`.
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_111923.txt` (`4540` lines, sha256 `f8a85cec3043b127f50078d1a8b5b9e59b19e649f74e7cd2ce0c8adff00d48c0`). Re-read `agents/notes.md` before status/log checks.
- Manager/effective status:
  - `6433` S running ep316-317, log healthy.
  - `6435` AB running ep300 after new ep299 eval, log healthy.
  - `6448` AH running ep204-205 after new ep199 eval, log healthy.
  - `6453` Z running ep127-128, log healthy.
  - `6457` CF sanity manager still says Unknown due `/tmp/tpu_logs` permission noise, but real train log is healthy at ep53.
  - `6469` AI manual resume is running ep24-26; first train lines after restore are present, so the manual resume is successful.
- New evals/results this loop:
  - `6435` AB K80 low-aug mixed mix/cut ep299: `76.266%` single / `77.988%` iterative, loss `0.252067`, invalid `0.260%`. This is the new best pure-diffusion result. Improvement over ep279 is small but positive (`76.190/77.848` -> `76.266/77.988`); keep running.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep199: `74.454%` single / `76.218%` iterative, loss `0.253656`, invalid `0.264%`. AH remains better than AB at matched ep199 (`AB ep199 74.064/75.886`), but behind AB's mature ep299 result. Keep to ep219 because p=0.75 remains a matched-epoch challenger.
- AI recovery verification:
  - `6469` restored from original AI `checkpoint_25020`, compiled, and is training normally. Latest checked train line: ep26.135, train accuracy about `0.489`, loss `0.446`, ~`4.25` steps/s.
  - No duplicate `config.load_from` fatal, no Traceback/OOM in `6469` checked tail.
  - Effective AI is now `6469`; stale `6460`/`6467` should not drive decisions. `6467` may still display Error in manager despite `ignore-error`; keep treating it as stale failed auto-resume.
- Other latest/status:
  - `6433` S latest eval remains ep299 `68.774/72.158`, running ep317; ep319 next.
  - `6453` Z latest eval remains ep119 `60.146/64.446`, running ep128; ep139 later.
  - `6457` CF latest eval remains ep39 `49.134`, running ep53; ep59 likely next loop.
- Forbidden/stale status:
  - Old aux-CE parent `6381` still says `resumed in 6459`, but no active `6459` window appears. Do not resume Run H or any aux-CE branch.
  - `6463` no-LayerScale sanity and `6467` AI auto-resume are stale errors; ignore.
- Interpretation:
  - AB remains the deployable best, but its slope is flattening; it should continue at least to ep319/329 unless it finishes first.
  - AH still has a matched-epoch advantage over AB, so do not retire it before ep219.
  - AI p=0.625 lost compute from ep20-31 due spot failure but has been recovered cleanly; wait for resumed ep39 before judging.
  - Full augmentation controls S/Z are still lagging but alive; keep for the delayed-augmentation question.
- Action/decision: no kill, no new launch. Continue S/AB/AH/Z/CF/AI(`6469`). Next checks: S ep319, CF ep59, AI resumed ep39 progress, AH ep219, AB ep319/finish, Z ep139, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 12:02-12:04 UTC)
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_120218.txt` (`4577` lines, sha256 `4fa6a24610ed810874d5b28caf3e8d397ed3f64c5e4e563a14473adf34c663ca`). Re-read `agents/notes.md` before status/log checks.
- Manager/effective status:
  - `6433` S running ep323, log healthy.
  - `6435` AB running ep308, log healthy.
  - `6448` AH running ep213, log healthy.
  - `6453` Z running ep132, log healthy.
  - `6457` CF sanity manager still says Unknown due `/tmp/tpu_logs` permission noise, but train/eval log is healthy at ep60.
  - `6469` AI manual resume running ep34-35, log healthy; no new eval yet after resume.
- New evals/results captured or newly recorded this loop:
  - `6433` S K10 full baseline augmentation ep319: `69.324%` single / `72.296%` iterative, loss `0.296627`, invalid `0.430%`. Single-acc is still improving versus ep299 `68.774`, but iterative gain is nearly flat versus ep299 `72.158`.
  - `6457` CF sanity v3/current-fixes ep59: `53.316%`, loss `2.231922`. This is clearly alive and above the failed v2/no-LayerScale sanity, but it is still below the old v3 sanity trajectory at ep59 (`59.132%` from earlier notes), so do not declare full reproduction yet. Continue to ep79/99.
- Other effective latest/status:
  - `6435` AB latest remains ep299 `76.266%` / `77.988%`, loss `0.252067`, invalid `0.260%`; running ep308. AB remains best mature pure-diffusion config.
  - `6448` AH latest remains ep199 `74.454%` / `76.218%`, loss `0.253656`, invalid `0.264%`; running ep213. Keep to ep219 because p=0.75 is still a matched-epoch challenger.
  - `6453` Z latest remains ep119 `60.146%` / `64.446%`, loss `0.323961`, invalid `0.622%`; running ep132. Keep to ep139/159 as K80 full-aug delayed-convergence control.
  - `6469` AI latest eval remains original/resumed branch ep19 `33.830%` / `41.432%`; running ep34.9 at ~4.3 sps. Next meaningful eval is resumed ep39.
- Stale/forbidden status:
  - Ran `ignore-error 6467 sqa` and `ignore-error 6463 sqa` again. `6467` is stale failed AI auto-resume with duplicate `config.load_from`; `6463` is stale old no-LayerScale sanity auto-resume. Do not resume either.
  - Old aux-CE parent `6381` still reports `resumed in 6459`, but manager listing still shows no active `6459` window. Continue verifying absence; do not resume Run H or any aux-CE branch.
- Interpretation:
  - AB remains the current deployable best and should continue through at least ep319 if it stays alive.
  - S confirms full augmentation is slow but still moving; do not kill because it is the K10 full-aug control.
  - CF sanity has not fully reproduced old v3 by ep59; keep collecting ep79/99 before changing code again.
  - AI manual resume is healthy; wait for ep39/59 before comparing p=0.625 against AB/AH.
  - No effective job requires kill, resume, or a new launch this loop.
- Action/decision: no kill, no resume, no new launch for effective jobs. Administrative stale-error ignores only. Continue S/AB/AH/Z/CF/AI(`6469`). Next checks: AI ep39, AH ep219, Z ep139, AB ep319, CF ep79, S ep339, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 12:34-12:39 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 12:33:59 UTC`.
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_123408.txt` (`4616` lines, sha256 `8cd36a420049c9c69bfef497cbb9747be3f6554cf6082fa7ab247f1537cdb343`). Re-read `agents/notes.md` before status/log checks.
- Manager/effective status at wake:
  - `6433` S running ep328, log healthy.
  - `6435` AB running ep314-315, log healthy.
  - `6448` AH running ep219 and then ep220 after short wait/eval, log healthy.
  - `6453` Z running ep135, log healthy.
  - `6457` CF sanity running ep65; manager still Unknown from `/tmp/tpu_logs` permission noise, but real train log is healthy.
  - `6469` AI manual resume running ep39 and then ep40-41 after eval, log healthy.
- New evals/results this loop:
  - `6469` AI K80 low-aug cutmix/mixup p=0.625 ep39: `57.646%` single / `62.260%` iterative, loss `0.334645`, invalid `0.832%`. This is slightly above AB ep39 (`57.148/61.642`) and AH ep39 (`57.566/62.152`), but the margin is tiny; it mainly says p=0.625 is healthy and not obviously worse early. Keep to ep59/79 before judging.
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep219: `74.944%` single / `76.552%` iterative, loss `0.253152`, invalid `0.290%`. This is a new AH best and beats matched AB ep219 (`74.726/76.526`) by a small margin, but still trails mature AB ep299 (`76.266/77.988`). Keep to ep239 if slots permit.
- Other effective latest/status:
  - `6433` S latest remains ep319 `69.324%` / `72.296%`; running ep328.9. Next eval ep339.
  - `6435` AB latest remains ep299 `76.266%` / `77.988%`; running ep315.1. Next eval ep319, likely next loop.
  - `6453` Z latest remains ep119 `60.146%` / `64.446%`; running ep135.6. Next eval ep139, likely next loop or shortly after.
  - `6457` CF latest remains ep59 `53.316%`; running ep65.5. Next eval ep79.
- Stale/forbidden status:
  - New stale `6472` appeared from old Z resume1 auto-rerun (`6438`) and failed. Ran `ignore-error 6472 sqa`; do not resume. This is not the effective Z branch; effective Z is manual `6453`.
  - Re-ran `ignore-error 6467 sqa` and `ignore-error 6463 sqa`. Both remain stale and should not drive decisions.
  - Old aux-CE parent `6381` still reports `resumed in 6459`, but no active `6459` window appears. Continue verifying absence; do not resume Run H or any aux-CE branch.
- Interpretation:
  - AH p=0.75 remains competitive at matched epochs through ep219, but AB remains the best mature/deployable branch because it is ~1.3 single / ~1.4 iter points ahead at ep299.
  - AI p=0.625 is healthy and slightly best at ep39 among AB/AH/AI, but the result is too early to decide; wait ep59/79.
  - Full-augmentation controls S/Z are still alive and still required for the augmentation question; do not kill.
  - No effective job requires resume or new launch this loop.
- Action/decision: no kill, no resume, no new launch for effective jobs. Administrative stale-error ignores only. Continue S/AB/AH/Z/CF/AI(`6469`). Next checks: AB ep319, Z ep139, AI ep59, CF ep79, AH ep239, S ep339, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 13:10-13:21 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 13:10:07 UTC`.
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_131016.txt` (`4656` lines, sha256 `8785dacacf185d52777eb3cef1512cfb64c3ecc5c494d892962b14cd72938a40`). Re-read `agents/notes.md` before status/log checks.
- Manager/effective status at wake:
  - `6433` S finished on `favaxa`. Final useful eval remains ep319 because the run ended around ep330 before the next scheduled eval.
  - `6435` AB running ep321-322, log healthy.
  - `6448` AH running ep226-227, log healthy.
  - `6453` Z running ep139-140, log healthy after ep139 eval.
  - `6457` CF sanity running ep71, log healthy despite recurring `/tmp/tpu_logs` permission noise.
  - `6469` AI manual resume running ep47-48, log healthy.
- New evals/results this loop:
  - `6435` AB K80 low-aug mixed mix/cut ep319: `76.600%` single / `78.294%` iterative, loss `0.248487`, invalid `0.274%`. This is the new best pure-diffusion result. AB is still improving late (`76.266/77.988` at ep299 -> `76.600/78.294` at ep319), so keep to finish.
  - `6453` Z K80 full baseline augmentation mixed mix/cut ep139: `62.192%` single / `66.086%` iterative, loss `0.315654`, invalid `0.492%`. Z continues delayed catch-up (`57.724/62.532` ep99 -> `60.146/64.446` ep119 -> `62.192/66.086` ep139), but remains far below low-aug K80 at matched ep139 (`AB 72.550/74.542`, `AH 73.224/75.284`). Keep as full-aug K80 control, but do not treat as best.
  - `6433` S K10 full baseline augmentation finished. Final useful eval: ep319 `69.324%` / `72.296%`, loss `0.296627`, invalid `0.430%`. S validates delayed full-aug catch-up but remains well below AB/AH; freeing this slot is appropriate.
- New launch using freed S slot:
  - Created `configs/remote_run_AJ_mlp_k80_randaug_only_mixlabel_config.yml` and copied it to `configs/remote_run_config.yml`.
  - Launched `6474` AJ on `kmh-tpuvm-v6e-8-spot-gzy-favaxa`; logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260512_131342_4g48mx_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval`.
  - AJ purpose: test a medium-augmentation compromise after AB's strong late performance and Z/S full-aug lag. It starts from AB and adds only RandAugment while keeping `reprob=0.0`, `repeated_aug=1`, `weight_decay=0.02`, and `stochastic_depth_rate=0.05`.
  - AJ safety/config verification: `model=ViT_base_mdh_mlp`, `n_masks_per_image=80`, `use_rand_augment=true`, `reprob=0.0`, `repeated_aug=1`, `use_mixup_cutmix=true`, `switch_prob=0.5`, `diffusion_label_mode=soft_top2`, `load_from=''`, `load_backbone_from=''`, `head_aux_ce=false`, `aux_ce_loss_weight=0.0`. No warm-start, no backbone loading, no aux CE, no CE shortcut.
  - AJ startup verified: initial compilation completed, train reached `[700]` / ep `0.55869` with loss `0.69284`; no Python traceback/OOM/fatal in checked tail. TPU driver-log permission messages are the same benign noise seen on CF.
- Other active latest/status:
  - `6448` AH latest remains ep219 `74.944%` / `76.552%`; running ep227. Keep to ep239 because it remains the strongest matched-epoch p=0.75 comparison, though below mature AB.
  - `6469` AI latest remains ep39 `57.646%` / `62.260%`; running ep48.4. Keep to ep59/79.
  - `6457` CF latest remains ep59 `53.316%`; running ep71.4. Keep to ep79/99 before sanity-code changes.
- Stale/forbidden status:
  - Stale `6472` old Z resume1 auto-rerun, `6467` failed AI auto-resume, and `6463` old no-LayerScale sanity remain errors in manager listing; continue ignoring, do not resume.
  - Old aux-CE parent `6381` still reports `resumed in 6459`, but no active `6459` window appears. Continue verifying absence; do not resume Run H or any aux-CE branch.
- Interpretation:
  - AB is now the clear best deployable pure-diffusion config. Its late improvement means the ImageNet-22K collaborator should follow AB unless a future medium-augmentation branch beats it.
  - Full augmentation is not dead, but Z/S still trail by a very large margin. The right next test is not another full-aug run; it is AJ's RandAug-only compromise.
  - No further launch/resume this loop. Let AJ reach eval0/ep19, Z reach ep159, AB finish, AI reach ep59, CF reach ep79, AH reach ep239.
- Action/decision: launched fresh AJ on freed S slot. No other kill/resume/launch. Continue AB/AH/Z/CF/AI/AJ. Next checks: AJ startup/eval0, AB finish/ep329 if any, Z ep159, AI ep59, CF ep79, AH ep239, and forbidden `6459` absence.

## Manual Report/Loop Update (2026-05-12 14:03-14:10 UTC)
- User requested a 24h progress report. Before writing it, I re-read the full `agents/` directory as loop memory: `/tmp/deit_agents_full_read_20260512_140316.txt` (`4701` lines, sha256 `3b8249268b43c56278d9599f1c4cd2d1a9ecf00311690faf574ce1dc7cfeb81a`).
- Manager status at `2026-05-12 14:03 UTC`:
  - `6435` AB is now `Finished`; final useful eval remains ep319 `76.600%` single / `78.294%` iterative, loss `0.248487`, invalid `0.274%`.
  - `6448` AH running ep236-237; latest eval remains ep219 `74.944%` / `76.552%`.
  - `6453` Z running ep144; latest eval ep139 `62.192%` / `66.086%`.
  - `6457` CF running past ep80 despite manager `Unknown` from `/tmp/tpu_logs` permission noise; new ep79 eval arrived: `55.926%`, loss `2.090736`.
  - `6469` AI running ep59; latest eval remains ep39 `57.646%` / `62.260%`; ep59 eval is not completed yet.
  - `6474` AJ running ep7; eval0 `0.104%` / `0.114%`, startup healthy.
  - `6475` AK startup/eval0 completed: ep0 `0.098%` / `0.116%`, loss `0.691946`, invalid `0.964%`; startup healthy.
- Wrote 24h progress report: `agents/report_24h_progress_2026-05-12.html`.
- Report headline conclusions:
  - AB is now the clear best no-shortcut pure diffusion config: K80 low-aug + corrected mixup/cutmix `soft_top2`, ep319 `76.600/78.294`.
  - Full augmentation is catching up slowly but remains far behind: Z ep139 `62.192/66.086` vs AB matched ep139 `72.550/74.542`; S final useful ep319 `69.324/72.296`.
  - AH p=0.75 is matched-epoch competitive but not better than mature AB; AI p=0.625 still awaiting ep59/79.
  - CF sanity v3/current-fixes is alive but not fully reproduced; continue to ep99 before more sanity-code changes.
  - AJ/AK are the next medium-augmentation diagnostics.
- No kill/resume/new launch was needed during report writing. Forbidden aux-CE `6459` still not active in manager listing; stale `6472`, `6467`, and `6463` remain non-effective errors.
- Next checks: AH ep239, AI ep59, AJ/AH progress, AK ep19 later, Z ep159, CF ep99, and forbidden `6459` absence.

## Manual Loop Update (2026-05-12 14:40-14:43 UTC)
- Woke from in-turn `sleep 1800` at `2026-05-12 14:39:58 UTC`.
- Loop-start full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260512_144009.txt` (`4962` lines, sha256 `1798991600a9379193b33fd8d028877b81c5a6cb1dd86f54dd1da138192be9f0`). Re-read `agents/notes.md` before status/log checks.
- Manager/effective status:
  - `6448` AH running ep243-244; log healthy.
  - `6453` Z running ep148; log healthy.
  - `6457` CF sanity running ep85; log healthy and manager now says Running.
  - `6469` AI manual resume running ep65-66; log healthy.
  - `6474` AJ running ep13; log healthy.
  - `6475` AK running ep6; log healthy.
  - `6435` AB and `6433` S remain Finished.
- New evals/results this loop:
  - `6448` AH K80 low-aug cutmix-heavy p=0.75 ep239: `75.364%` single / `77.016%` iterative, loss `0.254000`, invalid `0.234%`. This is essentially tied with matched AB ep239 (`75.504/76.992`; AH slightly lower single, slightly higher iterative), but still below mature AB ep319 `76.600/78.294`.
  - `6469` AI K80 low-aug p=0.625 ep59: `63.920%` single / `67.830%` iterative, loss `0.299868`, invalid `0.606%`. This is below matched AB ep59 `64.890/68.616`, AH ep59 `65.572/69.300`, and AF ep59 `66.298/69.818`; p=0.625 is not an early win. Keep to ep79 only because the branch is healthy and already running.
- No new eval yet for:
  - `6453` Z: latest ep139 `62.192/66.086`; running ep148; ep159 next.
  - `6457` CF: latest ep79 `55.926`; running ep85; ep99 next.
  - `6474` AJ: latest eval0 `0.104/0.114`; running ep13; ep19 next.
  - `6475` AK: latest eval0 `0.098/0.116`; running ep6; ep19 later.
- Updated `agents/report_24h_progress_2026-05-12.html` to include AH ep239 and AI ep59, so the report stays current with this wake.
- Stale cleanup: reran `ignore-error 6472`, `6467`, and `6463`. These remain non-effective. Old aux-CE parent `6381` still references `resumed in 6459`, but no active `6459` window appears; do not resume Run H or any aux-CE branch.
- Interpretation/decision:
  - AB remains the best deployable pure-diffusion config.
  - AH remains a useful matched-epoch challenger, but not a replacement for AB.
  - AI p=0.625 looks weaker at ep59; continue to ep79, then likely retire if it does not recover.
  - Continue Z as full-aug K80 control, CF sanity to ep99, and AJ/AK to first meaningful ep19 reads.
- Action/decision: no kill, no resume, no new launch. Continue AH/Z/CF/AI/AJ/AK. Next checks: AJ ep19, AI ep79 progress, Z ep159, CF ep99, AH ep259, AK progress, and forbidden `6459` absence.
