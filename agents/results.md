# DeiT Research Results

## Phase 1: Baseline Reproduction (目标: 81.8% top-1 @ ep330)

---

### Run 1 — ViT_base (original: no biases, LearnedScale=1e-4)
- **Window**: 6323
- **WandB notes**: `deit baseline (ViT-base, b1024, lr5e-4, ep330)`
- **TPU**: v6e-8-spot-gzy-axuxm0 (asia-northeast1-b), alias v6e-8-tmp202
- **Config**: b1024, lr5e-4, ep330, wd0.05, sdepth0.1, RandAugment, MixUp(0.8)+CutMix(1.0), label_smooth=0.1
- **Architecture**: qkv_bias=False, ln_bias=False, LearnedScale=True(1e-4)
- **Eval checkpoints**:
  | epoch | eval_accuracy | notes |
  |-------|--------------|-------|
  | 0     | 2.5%         |       |
  | 19    | 42.9%        |       |
  | 39    | 56.7%        |       |
  | 59    | 60.6%        |       |
  | 79    | 62.4%        |       |
  | 99    | 64.4%        |       |
  | 119   | 64.4%        | ⚠️ flat — plateau between ep99-119 |
  | 139   | 63.9%        | ⚠️ dip |
  | 159   | 65.41%       | ✅ RECOVERY — plateau was temporary |
  | 179   | 66.80%       | ✅ continued growth |
  | 199   | 68.07%       | ✅ steady growth (+1.27%) |
  | 219   | 68.40%       | ✅ continued growth (+0.33%) |
  | 239   | 68.98%       | ✅ steady growth (+0.58%) |
  | 259   | **69.726%**  | ✅ continued growth (+0.75%) |
  | 279   | **70.300%**  | ✅ continued growth (+0.57%) |
  | 299   | **71.470%**  | ✅ continued growth (+1.17%) |
  | 319   | **71.956%**  | ✅ continued growth (+0.49%) |
- **Status**: ✅ **FINISHED** — ep=319 final eval (ep=329 not reached in schedule); axuxm0 now IDLE+free
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260507_182637_7910jh_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`

---

### Run 2 — ViT_base_v2 (exact DeiT-B: biases=True, LearnedScale=False)
- **Window**: 6340
- **WandB notes**: `deit-v2-exact-biases-no-LS`
- **TPU**: v6e-8-spot-gzy-p1u4mx (asia-northeast1-b), alias v6e-8-tmp203
- **Config**: same as Run 1
- **Architecture**: qkv_bias=True, ln_bias=True, LearnedScale=False
- **Eval checkpoints**:
  | epoch | eval_accuracy |
  |-------|--------------|
  | 0     | 2.4%         |
  | 19    | 42.1%        |
  | 39    | 55.0%        |
  | 59    | 59.3%        |
  | 79    | 61.15%       |
  | 99    | 60.03%       | ⚠️ DROPPED — declining without LS |
  | 119   | 62.05%       | partial recovery but still lagging |
  | 139   | 63.01%       | ✅ continuing to recover (+0.96%) |
  | 159   | 64.08%       | ✅ solid jump (+1.07%) |
  | 179   | 63.032%      | ⚠️ DIP — dropped from ep=159 (no LS instability?) |
  | 199   | 63.614%      | partial recovery (+0.58%) but still below ep=159 peak |
  | 219   | **64.396%**  | ✅ recovery continuing (+0.78%) |
  | 239   | **65.360%**  | ✅ solid growth (+0.96%) |
  | 259   | **65.678%**  | ✅ slow growth (+0.32%) — decelerating |
  | 279   | **65.822%**  | ⚠️ near-flat (+0.14%) — approaching plateau; no-LS hurts late training |
- **Status**: Running (ep=280 as of 2026-05-09 08:23; ep=299 eval upcoming)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_023930_kq4hfm_kmh-tpuvm-v6e-8-spot-gzy-p1u4mx_asia-northeast1-b__b_lr_ep_eval`

---

### Run 3 — ViT_base_v3 (biases=True, LearnedScale=True)
- **Window**: 6342
- **WandB notes**: `deit-v3-biases-with-LS`
- **TPU**: v6e-8-spot-gzy-j3rqvs (asia-northeast1-b), alias v6e-8-tmp201
- **Config**: same as Run 1
- **Architecture**: qkv_bias=True, ln_bias=True, LearnedScale=True(1e-4)
- **Eval checkpoints**:
  | epoch | eval_accuracy |
  |-------|--------------|
  | 0     | 2.6%         |
  | 19    | 43.9%        |
  | 39    | 56.84%       |
  | 59    | 61.2%        |
  | 79    | 61.85%       |
  | 99    | 64.22%       | ✅ strong recovery — nearly matches v1 |
  | 119   | 64.69%       | ✅ now AHEAD of v1 (64.42%) at same epoch |
  | 139   | 64.68%       | ✅ essentially flat (v1 had dip to 63.9% here) |
  | 159   | **65.86%**   | ✅ +1.18% jump — now LEADING all runs at ep=159 |
  | 179   | 66.348%      | ✅ continued growth (+0.49%) |
  | 199   | 66.254%      | ⚠️ slight dip (-0.09%) — essentially flat; v1 gap widening |
  | 219   | **67.156%**  | ✅ recovery (+0.90%) |
  | 239   | **68.784%**  | ✅ big jump (+1.63%) |
  | 259   | **70.620%**  | ✅ BIG JUMP (+1.836%) — now AHEAD of Run 1 at ep=259 (69.726%)! biases+LS winning |
- **Status**: Running (ep=260 as of 2026-05-09 06:22; ep=279 eval upcoming)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_024938_sxvz3e_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`

---

## Ablation Matrix
| Run | qkv_bias | ln_bias | LearnedScale | ep19  | ep39   | ep59  | ep79  | ep99  | ep119 | ep139 | ep159 | ep179 | ep199 | ep219 | ep259 | ep279 | ep299 | ep330 |
|-----|----------|---------|--------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1 (ViT_base)   | False | False | True  | 42.9% | 56.7%  | 60.6% | 62.4% | 64.4% | 64.42% | 63.9%⚠️ | 65.41% | 66.80% | 68.07% | TBD | 69.73% | **70.30%** | **71.47%** | TBD |
| 2 (ViT_base_v2)| True  | True  | False | 42.1% | 55.0%  | 59.3% | 61.15%| 60.03%⚠️| 62.05% | 63.01% | 64.08% | 63.03%⚠️ | 63.61% | 64.40% | **65.68%** | **65.82%** | TBD | TBD |
| 3 (ViT_base_v3)| True  | True  | True  | 43.9% | 56.84% | 61.2% | 61.85%| 64.22% | 64.69% | 64.68% | 65.86% | 66.35% | 66.25%⚠️ | 67.16% | **70.62%** | TBD | TBD | TBD |

**Ranking at ep=259**: v3(**70.62%**) > v1(**69.73%**) >> v2(65.68%) — Run 3 (biases+LS) overtook Run 1 at ep=259!
- **Run 1 ep=319 = 71.96%**: FINISHED
- **Run 3 ep=259 = 70.62%**: still growing (+1.84%); on track to potentially surpass Run 1 final

## TODO / Next Steps (2026-05-09 05:41)
- Run 2: ep=260, ep=279 eval upcoming
- Run 3: ep=260, ep=279 eval upcoming — **70.62% at ep=259, LEADING Run 1!**
- **Phase 2 Run A** (ep=160): **30.56%** — ep=179 upcoming; growth decelerating (1.25x vs 1.36x vs 1.47x)
- **Phase 2 Run C** (ep=100): **26.15%/32.56%** — ep=119 upcoming; EXCEEDS Run A ep=139!
- **Phase 2 Run D** (ep=100): **22.84%/29.12%** — ep=119 upcoming; behind C by 3.3pp
- **Phase 2 Run E**: READY TO LAUNCH — axuxm0 already IDLE+MOUNTED (Run 1 finished)
  - User run: `tpu run kmh-tpuvm-v6e-8-spot-gzy-axuxm0 sqa dir=7 --config=configs/load_config.py:remote_run_E`
  - Config: configs/remote_run_config_E.yml (zero-init out_proj, uniform schedule)
- **Phase 2 Runs F/G/H**: user needs ftmd+tpu_run (safety classifier blocks autonomous):
  - Run F (MLP): `ftmd kmh-tpuvm-v6e-8-spot-gzy-3djlis v6e-8-tmp208 && tpu run kmh-tpuvm-v6e-8-spot-gzy-3djlis sqa dir=7 --config=configs/load_config.py:remote_run`
  - Run G (large): `ftmd kmh-tpuvm-v6e-8-spot-gzy-06q7u9 v6e-8-tmp209 && tpu run kmh-tpuvm-v6e-8-spot-gzy-06q7u9 sqa dir=7 --config=configs/load_config.py:remote_run_G`
  - Run H (aux CE): `ftmd kmh-tpuvm-v6e-8-spot-gzy-qxxa8y v6e-8-tmp210 && tpu run kmh-tpuvm-v6e-8-spot-gzy-qxxa8y sqa dir=7 --config=configs/load_config.py:remote_run_H`

---

## Code Fixes Applied (2026-05-08) — commit 37b5f2f
- ✅ Skip mixup/cutmix for diffusion training (P0 fix — was causing noisy supervision)
- ✅ Add iterative decode in eval_step_diffusion (4-step confidence-based unmasking via jax.lax.scan)
- ✅ Add mask_schedule config ('uniform'/'logit_normal') 
- ✅ Track invalid code rate (pred >= 1000) and iter decode accuracy in eval
- ⚠️ Runs A/B use OLD code (with mixup issue) — they may show suboptimal results

---

## Phase 2: Masked Diffusion Head

### Phase 2 Accuracy Metric Notes
- **Accuracy = sequence-level** (all 10 bits decoded correctly). Random baseline ≈ 0.5^10 ≈ **0.097%**.
- **Loss** = binary cross-entropy averaged over all bits (all-masked input). Random baseline = **0.693** (-log 0.5).
- The eval uses single-step greedy decoding from fully-masked input (`masked_bits=None` → defaults to all-masked).
- Expect very slow early improvement vs CE — diffusion has sparser gradient signal. EP=39 will be more informative.
- ✅ **Confirmed learning at ep=39**: Run A/B ~0.52% (5x from ep=0). Run A at ep=59: **2.318%** (4.4x from ep=39) — ACCELERATING.
- ✅ **Run C ep=19 = 0.282%/iter=0.478%**: outperforms Runs A/B at ep=19 (0.186%/0.134%) by 1.5-2x — NO-MIXUP FIX CONFIRMED BETTER.
- ✅ **Approach is NOT fundamentally broken**: trajectory is accelerating, warmup ~30 epochs before learning kicks in.
- ⚠️ **train_loss was not logged** (fix committed 2eb25cc): loss computed but dropped from metrics dict. Now fixed.
- **Training-eval mismatch**: train with partial masks (avg 50% masked), eval from all-masked. Warmup ~30 epochs before learning kicks in. Normal.
- **invalid_rate interpretation**: logged as percentage (0.286% for Run D at ep=0), NOT 28.6%. Both C/D well below random 2.34%.

### Run A — ViT_base_mdh (baseline diffusion head, b2=0.95)
- **Window**: 6352
- **WandB notes**: `phase2-mdh-baseline`
- **TPU**: v6e-8-spot-gzy-c8umw4 (us-east5-b), alias v6e-8-tmp51
- **Branch**: phase2-masked-diffusion (commit 557f6f4 — KeyError fix applied)
- **Config**: same as Phase 1, model=ViT_base_mdh (no biases backbone, LS, diffusion head)
- **Architecture**: backbone=ViT_base (no biases, LS), head=MaskedDiffusionHead(n_bits=10, inner_dim=256, n_layers=2, n_heads=4)
- **⚠️ Uses OLD code** (mixup issue not fixed; b2=0.95)
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | notes |
  |-------|--------------|-----------|-------|
  | 0     | 0.132%       | 0.686     | random baseline |
  | 19    | 0.186%       | 0.687     | ✅ +41% relative — small but positive signal |
  | 39    | **0.524%**   | **0.664** | ✅ LEARNING — 3.97x jump from ep=19; slightly beats Run B |
  | 59    | **2.318%**   | **0.636** | ✅ BIG JUMP — 4.4x from ep=39; accelerating learning |
  | 79    | **5.784%**   | **0.609** | ✅ ACCELERATING — 2.5x from ep=59; steep growth curve |
  | 99    | **12.212%**  | **0.566** | ✅ ACCELERATING — 2.1x from ep=79; best so far |
  | 119   | **17.948%**  | **0.537** | ✅ ACCELERATING — 1.47x from ep=99; still climbing |
  | 139   | **24.440%**  | **0.502** | ✅ ACCELERATING — 1.36x from ep=119; +6.49% jump |
  | 159   | **30.556%**  | **0.467** | ✅ 1.25x from ep=139; +6.1% — growth decelerating |
- **Status**: ✅ Running (ep=160 as of 2026-05-09 08:20; ep=179 eval upcoming)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_152809_369ujx_kmh-tpuvm-v6e-8-spot-gzy-c8umw4_us-east5-b__b_lr_ep_eval`
### Run B — ViT_base_mdh (b2=0.999)
- **Window**: 6349
- **WandB notes**: `phase2-mdh-b2-0.999`
- **TPU**: v6e-8-spot-gzy-cz2ivo (us-east5-b), alias v6e-8-tmp52
- **Branch**: phase2-masked-diffusion
- **Config**: same as Run A + adamw_b2=0.999 (reference DeiT value)
- **Architecture**: same as Run A
- **⚠️ Uses OLD code** (mixup issue not fixed; b2=0.999 — unnecessary hyperparam)
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | notes |
  |-------|--------------|-----------|-------|
  | 0     | 0.146%       | 0.685     | random baseline (0.5^10 ≈ 0.097%) |
  | 19    | 0.134%       | 0.689     | ⚠️ essentially flat — loss barely below random (0.693) |
  | 39    | **0.516%**   | **0.667** | ✅ LEARNING — 3.85x jump from ep=19; loss clearly dropping |
- **Status**: ⚠️ Preempted at ep~52.8 (log stopped 20:03); auto-resume pending >3hr — spot instance likely unavailable
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_142712_quc6xc_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval`

### Run C — ViT_base_mdh (FIXED: no mixup, uniform mask schedule)
- **Window**: 6363
- **WandB notes**: `phase2-mdh-no-mixup-uniform-C`
- **TPU**: v6e-8-spot-gzy-yq00yh (us-east5-b), alias v6e-8-tmp206
- **Branch**: phase2-masked-diffusion (commit d1e9883 — all fixes applied)
- **Config**: `configs/remote_run_config.yml` at commit d1e9883
- **Architecture**: ViT_base_mdh (biases+LS+diffusion head, n_bits=10, inner_dim=256, n_layers=2)
- **Key fixes**: no mixup/cutmix in training, iterative eval (4-step), uniform mask schedule, b2=0.95
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_211542_1ekprl_kmh-tpuvm-v6e-8-spot-gzy-yq00yh_us-east5-b__b_lr_ep_eval`
- **Early signal**: step=200 train_accuracy=**8.4%** (vs ~0.1% same point in Runs A/B) — NO-MIXUP FIX WORKING 🚀
- **Status**: ✅ Running (ep=81 as of 2026-05-09 05:41; ep=99 eval upcoming)
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.090%       | 0.681     | 0.186%            | 0.000%            | near random — expected at ep=0 |
  | 19    | **0.282%**   | 0.682     | **0.478%**        | 0.000%            | ✅ 3.1x single-step, 2.6x iter vs ep=0; outperforms Run A/B at ep=19 (0.186%, 0.134%) |
  | 39    | **2.122%**   | **0.640** | **3.958%**        | 2.242%            | ✅ 7.5x single, 8.3x iter vs ep=19; 4x ahead of Run A at same epoch! |
  | 59    | **9.180%**   | **0.586** | **14.042%**       | 2.808%            | ✅ 4.3x single, 3.5x iter vs ep=39; ACCELERATION |
  | 79    | **17.944%**  | **0.538** | **23.924%**       | 0.942%            | ✅ 1.95x single, 1.70x iter vs ep=59; matches Run A ep=119 (17.95%) in just 79ep!
  | 99    | **26.146%**  | **0.494** | **32.562%**       | 2.006%            | ✅ 1.46x single, 1.36x iter vs ep=79; EXCEEDS Run A ep=139 (24.44%) — ~40ep advantage |
- **Status**: ✅ Running (ep=100 as of 2026-05-09 07:35; ep=119 eval upcoming)

### Run D — ViT_base_mdh (FIXED: no mixup, logit-normal mask schedule)
- **Window**: 6364
- **WandB notes**: `phase2-mdh-no-mixup-logit-normal-D`
- **TPU**: v6e-8-spot-gzy-8507kk (us-east5-b), alias v6e-8-tmp53
- **Branch**: phase2-masked-diffusion (commit ffae847)
- **Config**: `configs/remote_run_config.yml` at commit ffae847
- **Architecture**: same as Run C
- **Key fixes**: no mixup/cutmix, iterative eval (4-step), logit_normal mask schedule, b2=0.95
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_212230_hc12e8_kmh-tpuvm-v6e-8-spot-gzy-8507kk_us-east5-b__b_lr_ep_eval`
- **Status**: ✅ Running (ep=80 as of 2026-05-09 05:41; ep=99 eval upcoming)
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.130%       | 0.682     | 0.192%            | 0.286%            | near random |
  | 19    | **0.326%**   | **0.680** | **0.532%**        | 5.120%            | ✅ 2.5x single, 2.8x iter vs ep=0; slightly AHEAD of C (0.282%/0.478%); high invalid rate from logit-normal OOD |
  | 39    | **2.258%**   | **0.638** | **4.074%**        | 1.632%            | ✅ 6.9x single, 7.7x iter vs ep=19; slightly ahead of C (2.12%/3.96%); invalid rate improved |
  | 59    | **8.456%**   | **0.591** | **12.456%**       | 2.702%            | ⚠️ now BEHIND C (9.18%/14.04%) — uniform schedule overtook logit-normal |
  | 79    | **15.706%**  | **0.548** | **21.532%**       | 0.858%            | ⚠️ still behind C (17.94%/23.92%) — gap widening; uniform schedule consistently better |
  | 99    | **22.842%**  | **0.512** | **29.124%**       | 2.104%            | ⚠️ behind C (26.15%/32.56%) by 3.3%/3.4% — gap growing vs ep=79 (2.2%/2.4%) |
- **Status**: ✅ Running (ep=100 as of 2026-05-09 07:45; ep=119 eval upcoming)

### Run E — ViT_base_mdh_zero_init (zero-init out_proj, uniform schedule)
- **Window**: TBD (not yet launched)
- **WandB notes**: `phase2-mdh-zero-init-E`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-i91hh1 (asia-northeast1-b) — IDLE, alias v6e-8-tmp207 needed
- **Branch**: phase2-masked-diffusion (commit 7cd83ac)
- **Config**: `configs/remote_run_config_E.yml`
- **Architecture**: ViT_base_mdh_zero_init (biases+LS+diffusion head, head_zero_init_proj=True)
- **Key change vs Run C**: zero-init kernel for out_proj (2-layer head); all other settings identical
- **Status**: ⏳ READY TO LAUNCH — alias v6e-8-tmp207 registered ✅; user needs to run ftmd+tpu_run
  - Run: `ftmd kmh-tpuvm-v6e-8-spot-gzy-i91hh1 v6e-8-tmp207 && tpu run kmh-tpuvm-v6e-8-spot-gzy-i91hh1 sqa dir=7 --config=configs/load_config.py:remote_run_E`

---

## Code Fixes Applied (2026-05-09) — commit 7cd83ac
- ✅ Fix: restore missing `class ViT(nn.Module):` declaration (accidentally dropped in e54ec1c when MLPDiffusionHead was added)
- ✅ Add `head_zero_init_proj` flag to MaskedDiffusionHead and MLPDiffusionHead
- ✅ Add `ViT_base_mdh_zero_init` partial (zero-init out_proj)
- ✅ Restore `ViT_base_mdh_mlp` partial (was accidentally removed)
- ✅ Fix `remote_run_config.yml`: use_mixup_cutmix=false (was incorrectly set to true for MLP baseline)
- ✅ Add `configs/remote_run_config_E.yml` for Run E
