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
  | 299   | **65.660%**  | ⚠️ DROPPED (-0.16%) — confirmed plateau/oscillation; no-LS run stagnated |
  | 319   | **65.958%**  | ⚠️ tiny tick (+0.30%) — still plateaued ~66%; no-LS run effectively converged |
  | 329   | **65.958%**  | ⚠️ FINAL — flat 319→329; no-LS run fully plateaued at 65.96% |
- **Status**: ✅ **FINISHED** — ep=329 final; p1u4mx now IDLE+MOUNTED
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
  | 279   | **71.404%**  | ✅ continued growth (+0.784%) — near Run 1's final 71.956% (ep=319); ~20ep ahead! |
  | 299   | **72.770%**  | ✅ BIG JUMP (+1.366%) — NOW SURPASSES Run 1 final (71.956%@ep=319)! biases+LS = clear winner |
  | 319   | **73.140%**  | ✅ continued growth (+0.370%) — LEADS Run 1 final by 1.184pp |
  | 329   | **73.140%**  | ✅ FINAL — plateau at 319→329; best = 73.14% @ep=319 (+1.18pp over Run 1 final) |
- **Status**: ✅ **FINISHED** — ep=329 final; j3rqvs now IDLE; Run I can start immediately
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_024938_sxvz3e_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`

---

## Ablation Matrix
| Run | qkv_bias | ln_bias | LearnedScale | ep19  | ep39   | ep59  | ep79  | ep99  | ep119 | ep139 | ep159 | ep179 | ep199 | ep219 | ep259 | ep279 | ep299 | ep330 |
|-----|----------|---------|--------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1 (ViT_base)   | False | False | True  | 42.9% | 56.7%  | 60.6% | 62.4% | 64.4% | 64.42% | 63.9%⚠️ | 65.41% | 66.80% | 68.07% | TBD | 69.73% | **70.30%** | **71.47%** | TBD |
| 2 (ViT_base_v2)| True  | True  | False | 42.1% | 55.0%  | 59.3% | 61.15%| 60.03%⚠️| 62.05% | 63.01% | 64.08% | 63.03%⚠️ | 63.61% | 64.40% | **65.68%** | **65.82%** | **65.66%**⚠️ | **65.96%** |
| 3 (ViT_base_v3)| True  | True  | True  | 43.9% | 56.84% | 61.2% | 61.85%| 64.22% | 64.69% | 64.68% | 65.86% | 66.35% | 66.25%⚠️ | 67.16% | **70.62%** | **71.40%** | **72.77%** | **73.14%** |

**PHASE 1 FINAL RESULTS**: v3(**73.14%**) >> v1(71.96%) >> v2(65.96%⚠️ plateaued)
- **Run 1 FINAL = 71.96%** (no biases, LS)
- **Run 2 FINAL = 65.96%** (biases, no LS) — LearnedScale is CRITICAL; without it run stagnates ~66%
- **Run 3 FINAL = 73.14%** (biases+LS) — **best Phase 1 result**, +1.18pp over Run 1; biases+LS = optimal config
- **Conclusion**: LearnedScale essential; biases help further; Run 3 backbone used for Phase 2 Run I warm-start

## TODO / Next Steps (2026-05-09 10:37)
- **Phase 1 Run 2**: ✅ **FINISHED** — 65.96% final (plateaued); p1u4mx now IDLE+MOUNTED
- **Phase 1 Run 3**: ✅ **FINISHED** — 73.14% final; j3rqvs IDLE+MOUNTED
- **Phase 2 Run A/C/D**: machines DELETED by preemption; MONITOR.py searching for new v6e-8 spot slots; progress will resume when new machines acquired
  - Run A last: **37.25%** @ep=179; Run C last: **33.28%/39.14%** @ep=119; Run D last: **28.72%/34.98%** @ep=119

### Queued Launches — 4 machines IDLE (user action required; only 1 active DeiT job = cz2ivo/RunB)
- **Run E** (zero-init, axuxm0 IDLE+MOUNTED): `tpu run kmh-tpuvm-v6e-8-spot-gzy-axuxm0 sqa dir=7 --config=configs/load_config.py:remote_run_E`
- **Run F** (MLP, 3djlis IDLE+MOUNTED): `ftmd kmh-tpuvm-v6e-8-spot-gzy-3djlis v6e-8-tmp208 && tpu run kmh-tpuvm-v6e-8-spot-gzy-3djlis sqa dir=7 --config=configs/load_config.py:remote_run`
- **Run G** (large head, 06q7u9 IDLE+MOUNTED): `ftmd kmh-tpuvm-v6e-8-spot-gzy-06q7u9 v6e-8-tmp209 && tpu run kmh-tpuvm-v6e-8-spot-gzy-06q7u9 sqa dir=7 --config=configs/load_config.py:remote_run_G`
- **Run I** (pretrained backbone, j3rqvs IDLE+MOUNTED): `tpu run kmh-tpuvm-v6e-8-spot-gzy-j3rqvs sqa dir=7 --config=configs/load_config.py:remote_run_I`
- **Run H** (aux CE, qxxa8y IDLE+MOUNTED): `ftmd kmh-tpuvm-v6e-8-spot-gzy-qxxa8y v6e-8-tmp210 && tpu run kmh-tpuvm-v6e-8-spot-gzy-qxxa8y sqa dir=7 --config=configs/load_config.py:remote_run_H`
- **p1u4mx** now free (Phase 1 Run 2 finished): available for future use or additional Phase 2 runs

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
  | 179   | **37.246%**  | **0.433** | ✅ 1.22x from ep=159; +6.69% — deceleration slowing; Run C ep=119 (33.28%) trails by ~60ep |
- **Status**: ⚠️ PREEMPTED at ep=186 (~11:07 2026-05-09); auto-resuming (c8umw4 us-east5-b spot preemption wave)
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
  | 119   | **33.276%**  | **0.461** | **39.138%**       | 1.430%            | ✅ 1.27x single, 1.20x iter vs ep=99; EXCEEDS Run A ep=159 (30.56%) — 40ep advantage holds |
- **Status**: ⚠️ PREEMPTED at ep=134.8 (~11:07 2026-05-09); auto-resuming (yq00yh us-east5-b spot preemption wave)

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
  | 119   | **28.722%**  | **0.482** | **34.980%**       | 2.654%            | ⚠️ behind C (33.28%/39.14%) by 4.6%/4.2% — gap continues to widen; uniform definitively better |
- **Status**: ⚠️ PREEMPTED at ep=132.5 (~11:08 2026-05-09); auto-resuming (8507kk us-east5-b spot preemption wave)

### Run E — ViT_base_mdh_zero_init (zero-init out_proj, uniform schedule)
- **Window**: 6382
- **WandB notes**: `phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-axuxm0 (asia-northeast1-b), alias v6e-8-tmp202
- **Branch**: phase2-masked-diffusion (commit 05264ce)
- **Config**: `configs/remote_run_E_config.yml`
- **Architecture**: ViT_base_mdh_zero_init (biases+LS+attention diffusion head, head_zero_init_proj=True)
- **Key change vs Run C**: zero-init kernel for out_proj; all other settings identical
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214004_nin5vh_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.100%       | 0.693     | 0.100%            | 0.000%            | near random |
  | 19    | **0.186%**   | 0.690     | **0.212%**        | 0.384%            | 1.86x improvement over random; zero-init helps stability |
- **Status**: ✅ Running (ep~20 as of 23:46)

### Run F — ViT_base_mdh_mlp (MLP head baseline)
- **Window**: 6375
- **WandB notes**: `phase2 Run F: MLP head baseline. Ablation: attention vs MLP diffusion head.`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-3djlis (asia-northeast1-b), alias v6e-8-tmp208
- **Branch**: phase2-masked-diffusion
- **Config**: `configs/remote_run_F_config.yml`
- **Architecture**: ViT_base_mdh_mlp (biases+LS+MLP diffusion head; no attention)
- **Key change vs Run C**: replaces 2-layer attention head with 2-layer MLP
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_200228_jwwv8y_kmh-tpuvm-v6e-8-spot-gzy-3djlis_asia-northeast1-b__b_lr_ep_eval`
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.206%       | 0.681     | 0.302%            | 0.002%            | slightly above random |
  | 19    | **5.584%**   | **0.626** | **9.666%**        | 1.870%            | ✅ strong early signal; compare Run C ep=19 = 0.282%/0.478% → F much stronger early |
  | 39    | **14.902%**  | **0.565** | **21.230%**       | 2.302%            | ✅ 2.67x jump from ep=19; MLP scales well early |
- **Status**: ✅ Running (ep~48 as of 01:07)

### Run G — ViT_base_mdh_large (512-dim, 4-layer head)
- **Window**: 6383
- **WandB notes**: `phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads)`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-06q7u9 (asia-northeast1-b), alias v6e-8-tmp209
- **Branch**: phase2-masked-diffusion (commit 05264ce)
- **Config**: `configs/remote_run_G_config.yml`
- **Architecture**: ViT_base_mdh_large (2x head width: 512 vs 256, 2x layers: 4 vs 2)
- **Key change vs Run C**: 4x head parameter count; ablates head capacity
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214224_qa2gek_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.108%       | 0.693     | 0.082%            | 3.244%            | near random; high invalid from large uninit head |
  | 19    | **0.162%**   | 0.690     | **0.238%**        | 0.000%            | 1.5x improvement over random; large head learns slower than baseline |
- **Status**: ✅ Running (ep~20 as of 23:49)

### Run H — ViT_base_mdh_aux_ce (attention head + aux CE loss λ=0.1)
- **Window**: 6381
- **WandB notes**: `phase2 Run H: attention head + aux CE loss (lambda=0.1)`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-qxxa8y (asia-northeast1-b), alias v6e-8-tmp210
- **Branch**: phase2-masked-diffusion (commit 537e610 — fc init fix)
- **Config**: `configs/remote_run_H_config.yml`
- **Architecture**: ViT_base_mdh_aux_ce (attention head + auxiliary FC on CLS, aux CE loss λ=0.1)
- **Key change vs Run C**: auxiliary CE loss helps backbone learn discriminative features
- **Note**: Bugfix required — `fc` wasn't initialized during model.init() (commit 537e610)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_213033_is69hk_kmh-tpuvm-v6e-8-spot-gzy-qxxa8y_asia-northeast1-b__b_lr_ep_eval`
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.100%       | 0.693     | 0.100%            | 0.000%            | near random |
  | 19    | **23.832%**  | 0.512     | **28.002%**       | 2.454%            | 🚀 BREAKTHROUGH — aux CE loss gives 238x improvement over random; 4.3x better than Run F MLP baseline (5.58%) |
- **Status**: ✅ Running (ep~19 as of 23:35)

### Run I — ViT_base_mdh (warm-started from Phase 1 Run 3 backbone)
- **Window**: 6388
- **WandB notes**: `phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone)`
- **TPU**: kmh-tpuvm-v6e-8-spot-gzy-j3rqvs (asia-northeast1-b), alias v6e-8-tmp201
- **Branch**: phase2-masked-diffusion (commit 05264ce)
- **Config**: `configs/remote_run_I_config.yml`
- **Architecture**: ViT_base_mdh (standard attention diffusion head), warm-started from Phase 1 Run 3
- **Key change vs Run C**: backbone pre-trained on CE for 329 epochs; diffusion head randomly initialized
- **Note**: Multiple bugfixes for load_backbone_params (commits 37daeb8, 2f57c6c, 48514a4, 05264ce)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_221135_gmkq7q_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`
- **Eval checkpoints**:
  | epoch | eval_accuracy | eval_loss | eval_accuracy_iter | eval_invalid_rate | notes |
  |-------|--------------|-----------|-------------------|-------------------|-------|
  | 0     | 0.184%       | 0.676     | 0.224%            | 49.816%           | ⚠️ high invalid_rate (warm backbone generates non-random codes, many OOB at init) |
  | 19    | **47.082%**  | **0.398** | **50.802%**       | 0.566%            | 🚀🚀 EXTRAORDINARY — 470x improvement over random; 2x better than Run H (23.8%); invalid_rate collapsed to near-zero |
- **Status**: ✅ Running (ep~28 as of 01:07; train_acc=34%, train_loss=0.377)
- **Key observation**: warm backbone provides ~50x faster training convergence vs cold-start runs (E/G/H at ep~8 with train_acc~0.5% vs I at ep~3 with train_acc~45%). ep=19 eval confirms this translates to eval: **47% vs 0.19% for Run E**.

---

## Code Fixes Applied (2026-05-09) — commits faa44ba, e4fd44b
- ✅ Fix `aux_ce_loss_weight` gating: was defaulting to 0.1 even when `head_aux_ce=False`, causing ValueError in loss_fn (commit faa44ba)
- ✅ Fix `load_backbone_params` (initial): preserve FrozenDict types, reinit opt_state from merged params (commit e4fd44b — superseded by later fixes)

## Code Fixes Applied (2026-05-09) — commit 7cd83ac
- ✅ Fix: restore missing `class ViT(nn.Module):` declaration (accidentally dropped in e54ec1c when MLPDiffusionHead was added)
- ✅ Add `head_zero_init_proj` flag to MaskedDiffusionHead and MLPDiffusionHead
- ✅ Add `ViT_base_mdh_zero_init` partial (zero-init out_proj)
- ✅ Restore `ViT_base_mdh_mlp` partial (was accidentally removed)
- ✅ Fix `remote_run_config.yml`: use_mixup_cutmix=false (was incorrectly set to true for MLP baseline)
- ✅ Add `configs/remote_run_config_E.yml` for Run E

## Code Fixes Applied (2026-05-09, this session) — commits 537e610, 37daeb8, 2f57c6c, 48514a4, 05264ce
- ✅ Fix: `fc` params not initialized when `head_aux_ce=True`. Flax only initializes submodule params when called during `model.init()`. Since init uses `return_aux_ce=False`, fc was never called → params missing. Fix: always call `self.fc(cls)` when `head_aux_ce=True`, gate return (not call) on `return_aux_ce` (commit 537e610)
- ✅ Fix `load_backbone_params`: FrozenDict/plain-dict type mismatch cascade. Root cause: Phase 1 ckpt has `final_ln: ['scale']` only (no bias), causing structure mismatch. Fixed with recursive copy that preserves Phase 2 tree structure (commit 2f57c6c). Also: `state.params` is plain dict; never use `freeze()` or it breaks optimizer (commit 05264ce).
- ✅ Fix: E/G/I were launched with wrong MLP config (previous agent used Run F config for all). Killed and relaunched with correct configs.

---

## Loop Update (2026-05-10 ~02:xx UTC)

- Window cleanup done (obsolete/stale): 6323, 6352, 6363, 6364, 6375, 6379.
- Active DeiT windows now: 6380, 6381, 6382, 6383, 6388.

Latest eval snapshots from logs:

- **6380 (sanity-run, CE)**:
  - ep39: `eval_accuracy=54.47%`, `eval_loss=2.1369`
  - status: running (~ep46.9)

- **6381 (Run H, aux CE)**:
  - ep39: `eval_accuracy=39.81%`, `eval_accuracy_iter=44.114%`, `eval_invalid_rate=1.618%`, `eval_loss=0.44364`
  - status: running (~ep46.3)

- **6382 (Run E, zero-init)**:
  - ep39: `eval_accuracy=0.178%`, `eval_accuracy_iter=0.364%`, `eval_invalid_rate=6.032%`, `eval_loss=0.68782`
  - status: running (~ep44)
  - note: currently much weaker than expected; likely deprioritize unless late catch-up appears.

- **6383 (Run G, large head 512/4L)**:
  - ep39: `eval_accuracy=0.162%`, `eval_accuracy_iter=0.398%`, `eval_invalid_rate=0.346%`, `eval_loss=0.68776`
  - status: running (~ep44)
  - note: large attention head still underperforming at this stage.

- **6388 (Run I, warm-start)**:
  - latest logged eval remains ep19: `eval_accuracy=47.082%`, `eval_accuracy_iter=50.802%`, `eval_invalid_rate=0.566%`, `eval_loss=0.39755`
  - status: running (~ep39), waiting for next eval checkpoint.

---

## Loop Update (2026-05-10 ~02:50 UTC) — Architecture Switch + TPU Fill

### Implemented architecture changes (before launching)
- ✅ `n_masks_per_image` implemented in training path (encode once, decode K masks).
- ✅ MLP head switched to proposed form:
  - tiny bit embedding (`head_bit_dim`),
  - large CLS decoder MLP (`head_mlp_hidden_dim`, default 3072).
- ✅ Attention head switched to proposed form:
  - full-token cross-attention (`encode_tokens`) instead of CLS-only input,
  - bit queries attend to CLS+patch tokens.

### New runs launched
- **Run L** (window 6389): cross-attention over full patch tokens, 4 layers, K=10.
  - TPU: `kmh-tpuvm-v6e-8-spot-gzy-p1u4mx`
  - Status: Running (early stage, ep~0.16)
- **Run M** (window 6390): tiny-bit MLP (`bit_dim=8`, hidden=3072), K=10.
  - TPU: `kmh-tpuvm-v6e-8-spot-gzy-cz2ivo`
  - Status: Running (early stage, ep~0.07)
- **Run N** (window 6392): tiny-bit MLP + K=10 + low-aug mild.
  - TPU: `kmh-tpuvm-v6e-8-spot-gzy-3djlis`
  - Status: Unknown in `tpu check` due log permission parser issue, but `output.log` is updating (`train step 100` seen), so training is active.

### Capacity status
- Active DeiT windows now = **8** (limit reached):
  - 6380, 6381, 6382, 6383, 6388, 6389, 6390, 6392
- Cleaned stale duplicate window 6391 (killed).

## Loop Update — 2026-05-10 11:37 UTC

- **Run L chain cleanup**:
  - Window `6389` failed due to SSH timeout to TPU `p1u4mx`.
  - Window `6398` failed due to duplicated config flag in resume command (`--config` duplicated -> absl flag definition conflict).
- **Relaunch**:
  - New clean Run L launched on TPU `z169mq`.
  - New window: **`6400`**
  - WandB run: `https://wandb.ai/sqa24-massachusetts-institute-of-technology/deit/runs/usk9rmbb`
- **Current DeiT set**: `6380, 6381, 6382, 6383, 6388, 6390, 6392, 6400` (8 windows total).
  - Running: first 7 windows above.
  - `6400`: startup pipeline still progressing, currently shown as `Unknown` by `tpu check` (not yet epoch-printing).

## Incident Update — 2026-05-10 12:00 UTC (z169mq)

- `6402` failed because SSD/NFS mount missing after reboot (`staging` path missing).
- Fixed with `mount-disk` on `kmh-tpuvm-v6e-8-spot-gzy-z169mq`.
- `6403` then failed with missing dependency `timm`.
- Installed on TPU: `python3 -m pip install --user timm`.
- Relaunched as **window `6404`** (Run L cross-attn + K10).
- Current state: startup/unknown (not yet epoch-printing), run URL: `https://wandb.ai/sqa24-massachusetts-institute-of-technology/deit/runs/wptfcqkt`.

---

## Loop Update — 2026-05-10 12:45 UTC (Takeover)

- Full `agents/` memory re-read completed before checking jobs.
- Current DeiT active set is exactly 8/8 occupied: `6380, 6381, 6382, 6383, 6388, 6390, 6392, 6405`. Latest `tpu check` shows 7 Running plus `6405` as Unknown, but `6405/output.log` is actively updating, so no resume is needed.
- No resume or launch action needed this loop; capacity is full.
- HTML takeover report refreshed: `agents/report_takeover_2026-05-10.html`.

Latest eval snapshot:

| Window | Run | Latest eval | Status note |
|--------|-----|-------------|-------------|
| 6380 | sanity CE full alignment | ep139: `63.356%`, loss `1.8344` | Running ep~145.8; still below target trajectory. |
| 6381 | Run H aux CE | ep139: `52.506%`, iter `54.816%`, loss `0.4380` | Running; useful diagnostic but not pure-diffusion mainline. |
| 6382 | Run E zero-init | ep139: `1.326%`, iter `3.172%`, loss `0.6592` | Running but low ROI; candidate to free if a slot is needed. |
| 6383 | Run G large attention head | ep139: `2.952%`, iter `6.636%`, loss `0.6445` | Running but low ROI; candidate to free if a slot is needed. |
| 6388 | Run I warm-start | ep139: `58.986%`, iter `62.470%`, loss `0.3343` | Strong diagnostic; warm backbone works. |
| 6390 | Run M tiny-bit MLP + K10 | ep79: `51.186%`, iter `56.400%`, loss `0.4110` | Running ep~98; wait for ep99/e119. |
| 6392 | Run N tiny-bit MLP + K10 + low-aug | ep119: `59.060%`, iter `62.800%`, loss `0.4201` | Best pure-diffusion mainline so far. |
| 6405 | Run L cross-attn full tokens + K10 | ep0: `0.100%`, iter `0.116%`, loss `0.6920` | Clean relaunch now running; wait for ep19/39. |

Current interpretation:
- Pure diffusion is viable: Run N has reached `59.06%` / iterative `62.80%` by ep119 without aux CE or warm-start.
- K10 multi-mask + tiny-bit MLP is the strongest pure-diffusion direction so far.
- Low augmentation/regularization appears beneficial: Run N is ahead of Run M at comparable early/mid training.
- Old attention-head variants E/G are weak even by ep139 and should be deprioritized.
- If a new slot is needed, prefer freeing E or G after confirming user approval / project policy.

---

## Manual Loop Update — 2026-05-10 14:35 UTC

### Wake mechanism incident
- Previous `sleep 1800` did complete at `2026-05-10 13:50:47 UTC`, but the agent had already sent a final response, so the session completion was not surfaced until the user sent a new message and the agent polled it.
- Correct procedure going forward: when starting a 30-minute loop, keep the assistant turn open and wait on the `sleep 1800` tool call; do not send final before the sleep returns.

### Full memory read
- Full `agents/` directory re-read completed and stored in `/tmp/deit_agents_full_read_20260510_143149.txt` for this loop.

### TPU state
- `6380` sanity CE: Running ep~163.6.
- `6381` Run H aux CE: Error in `tpu check`; log shows SSH connection closed/refused/timed out after active training at ep~164.5, no Python traceback. Treat as TPU/preemption/network failure; do not manually fix resume logic.
- `6382` Run E zero-init: Running ep~164.
- `6383` Run G large attention: Running ep~163.
- `6388` Run I warm-start: Running ep~159.
- `6390` Run M tiny-bit MLP K10: Running ep~116.
- `6392` Run N tiny-bit MLP K10 low-aug: Running ep~143.6.
- `6405` Run L cross-attn K10: Running ep~21.

### Latest eval snapshot

| Window | Run | Latest eval | Interpretation |
|--------|-----|-------------|----------------|
| 6380 | sanity CE | ep159 `65.530%`, loss `1.7123` | Still below DeiT paper target trajectory. |
| 6381 | Run H aux CE | latest before disconnect around ep~164 train only; ep139 `52.506%` / iter `54.816%` | Error is SSH/preemption-like, not code traceback. |
| 6382 | Run E zero-init | ep159 `2.384%` / iter `4.706%`, loss `0.6483` | Low ROI. |
| 6383 | Run G large attention | ep159 `4.614%` / iter `9.764%`, loss `0.6278` | Low ROI, but better than E. |
| 6388 | Run I warm-start | ep159 `61.020%` / iter `63.754%`, loss `0.3300` | Strong diagnostic, slightly above Run N iter at current snapshots. |
| 6390 | Run M tiny-bit MLP K10 | ep99 `52.132%` / iter `56.992%`, loss `0.4306` | Progress slowing vs Run N. |
| 6392 | Run N tiny-bit MLP K10 low-aug | ep139 `59.908%` / iter `63.410%`, loss `0.4189` | Best pure-diffusion mainline remains Run N. |
| 6405 | Run L cross-attn full tokens K10 | ep19 `19.220%` / iter `29.756%`, loss `0.5330` | Cross-attn works, but early curve trails Run M/N. |

### Decision
- No new launch this loop.
- Do not manually resume `6381`; let MONITOR/TPU manager handle preemption-like failure.
- Keep watching Run N and Run L. Run N remains the pure-diffusion mainline; Run L needs ep39 before deciding whether cross-attn is worth a slot.
- E/G remain candidates to free if a new high-ROI run is needed, but no automatic kill without explicit approval.

---

## Manual Loop Update — 2026-05-10 15:08 UTC

### Full memory read
- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260510_150315.txt`.

### TPU state
- Running DeiT windows: `6380, 6382, 6383, 6388, 6390, 6392, 6405`.
- `6381` remains Error. Latest log has no Python traceback; it ended with SSH connection closed/refused/timed out after ep~164.5. Treat as preemption/network failure. Since Run H is aux-CE and deprioritized, do not manually rescue it now.

### Latest eval snapshot

| Window | Run | Latest eval | Notes |
|--------|-----|-------------|-------|
| 6380 | sanity CE | ep159 `65.530%` | Still below target trajectory. |
| 6381 | Run H aux CE | ep159 `53.206%` / iter `55.122%` before SSH failure | Not mainline. |
| 6382 | Run E zero-init | ep159 `2.384%` / iter `4.706%` | Low ROI. |
| 6383 | Run G large attention | ep159 `4.614%` / iter `9.764%` | Low ROI. |
| 6388 | Run I warm-start | ep159 `61.020%` / iter `63.754%` | Strong diagnostic. |
| 6390 | Run M tiny-bit MLP K10 | ep119 `52.990%` / iter `57.468%` | Plateauing, clearly behind Run N. |
| 6392 | Run N tiny-bit MLP K10 low-aug | ep139 `59.908%` / iter `63.410%` | Best pure-diffusion mainline. |
| 6405 | Run L cross-attn K10 | ep19 `19.220%` / iter `29.756%` | Learns, but trails M/N early; wait for ep39. |

### New config prepared
- Added `configs/remote_run_O_mlp_k10_lowaug_aggressive_config.yml`.
- Copied Run O config to `configs/remote_run_config.yml` per launch convention.
- Run O design: `ViT_base_mdh_mlp`, `n_masks_per_image=10`, no RandAugment, `reprob=0`, `repeated_aug=1`, `weight_decay=0.0`, `stochastic_depth_rate=0.0`.

### Launch attempt
- Attempted Run O on `kmh-tpuvm-v6e-8-spot-gzy-vtcoc1`.
- First attempt used wrong wrapper syntax (`--config=...`) and failed locally with `Unknown config key --config`; no remote launch.
- Second attempt used correct syntax (`config=configs/load_config.py:remote_run --auto`) but failed because tpu manager reported `get_zone_pre: TPU kmh-tpuvm-v6e-8-spot-gzy-vtcoc1 not found`.
- Decision: do not force registration or keep trying random unregistered IDLE cards. Run O is queued locally and should launch when a registered/manager-visible slot is available.

---

## Manual Loop Update — 2026-05-10 15:38 UTC

- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260510_153613.txt`.
- Current DeiT state: `6380, 6382, 6383, 6388, 6390, 6392, 6405` Running; `6381` remains Error/preemption-like.
- No new eval checkpoint since the 15:08 loop for the key mainline runs.
- Current progress:
  - `6392` Run N is at ep~157; next eval should be ep159 soon.
  - `6405` Run L is at ep~32; next important eval is ep39.
  - `6390` Run M is at ep~127; next eval is ep139.
- No new launch this loop. Previous Run O launch failed because the candidate idle TPU was not recognized by tpu manager. Do not retry random unregistered cards.
- Next decision point: Run N ep159 and Run L ep39. If Run L ep39 remains far below M/N, prioritize Run O or another pure MLP/K10 variant over cross-attn.

---

## Manual Loop Update — 2026-05-10 16:10 UTC

- Full `agents/` directory re-read completed: `/tmp/deit_agents_full_read_20260510_160714.txt`.
- TPU status: `6380, 6382, 6383, 6388, 6390, 6392, 6405` Running; `6381` remains Error/preemption-like.
- Key new result: Run N ep159 is available.

Latest important evals:

| Window | Run | Latest eval | Readout |
|--------|-----|-------------|---------|
| 6392 | Run N tiny-bit MLP K10 low-aug | ep159 `59.974%` / iter `63.478%`, loss `0.4401` | Essentially plateaued vs ep139 (`59.908%` / `63.410%`). Low-aug MLP-K10 is strong but may be saturating around 60/63.5. |
| 6388 | Run I warm-start | ep159 `61.020%` / iter `63.754%`, loss `0.3300` | Warm-start still slightly ahead; diagnostic, not mainline. |
| 6390 | Run M MLP K10 normal aug | ep119 `52.990%` / iter `57.468%` | Clearly behind Run N; low aug mattered. |
| 6405 | Run L cross-attn K10 | ep19 `19.220%` / iter `29.756%`; currently ep~37 | Await ep39. Training accuracy at ep32 already ~53%, so ep39 eval will be informative. |

Decision:
- No new launch now; Run O remains locally queued but launch is blocked by lack of manager-visible available TPU.
- Main interpretation: Run N has likely saturated or slowed sharply by ep159. Next high-ROI options remain aggressive low-reg Run O, loss normalization ablation, or eval-only decoding/voting tricks.
- Need Run L ep39 before deciding whether cross-attn deserves continued slot pressure.

---

## User-Directed Update — 2026-05-10 16:45 UTC

### Report for ImageNet-22K collaborator
- Created/updated: `agents/report_imagenet22k_mdh_handoff_2026-05-10.html`.
- Report contains enough implementation detail to reconstruct the current best masked-diffusion classifier path:
  - tiny-bit MLP diffusion head,
  - K-mask training (`n_masks_per_image`),
  - low-augmentation config,
  - ImageNet-22K changes (`NUM_CLASSES=21841`, `n_bits=15`),
  - invalid-code handling,
  - K memory estimate and recommended 22K run matrix.

### E/G stopped
- E/G were low-ROI old attention-head runs and have been killed/freeing slots.
- E architecture: `ViT_base_mdh_zero_init`, old CLS+bit-token attention head, 256 dim, 2 layers, 4 heads, zero-init output projection, K=1, full aug/reg.
- G architecture: `ViT_base_mdh_large`, old CLS+bit-token attention head, 512 dim, 4 layers, 8 heads, K=1, full aug/reg.
- Latest useful results before stop:
  - E ep159: `2.384%` / iter `4.706%`.
  - G ep159: `4.614%` / iter `9.764%`.
- Interpretation: sparse K=1 supervision + full aug/reg + old attention-head design failed; MLP+K10/low-aug is the mainline.

### Run L ep39
| Run | Epoch | Accuracy | Iter accuracy | Loss | Invalid |
|---|---:|---:|---:|---:|---:|
| L cross-attn full tokens K10 | 39 | `39.676%` | `49.884%` | `0.43349` | `1.978%` |

- L learns, but trails Run N at ep39 (`49.738%` / `55.010%`). Cross-attn is therefore not the next priority unless it catches up later.

### New K40 runs
| Run | Window | TPU | Config | Status |
|---|---:|---|---|---|
| P | 6409 | `v6e-8-spot-gzy-axuxm0` | tiny-bit MLP, K40, low-aug mild (`wd=0.02`, `sd=0.05`) | first batch ready, compiling; no immediate OOM |
| Q | 6410 | `v6e-8-spot-gzy-06q7u9` | tiny-bit MLP, K40, low-aug aggressive (`wd=0`, `sd=0`) | process started, W&B/config loaded; awaiting first batch/compile confirmation |

Logdirs:
- P: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_163819_kdc043_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`
- Q: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_164147_hf92wd_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`

### Immediate K40 startup correction — 2026-05-10 16:46 UTC
- Run Q reached `First batch ready` at `16:43:26 UTC`; no immediate OOM before initial compilation.

---

## Analysis Update — 2026-05-10 16:58 UTC

### E/G architecture and failure analysis
- E/G are not the user-proposed MLP. They are old attention-head variants (`head_type: attention`).
- User-proposed MLP runs are M/N/P/Q (`head_type: mlp`).
- E: old attention head, 256 dim, 2 layers, 4 heads, zero-init output projection, K=1, full aug/reg. Result before kill: ep159 `2.384%` / iter `4.706%`.
- G: old attention head, 512 dim, 4 layers, 8 heads, K=1, full aug/reg. Result before kill: ep159 `4.614%` / iter `9.764%`.
- Main interpretation: E/G underperformance vs the earlier fixed K=1 MDH baseline is due to the combination of bad bootstrap/optimization (E zero-init), deeper old attention head not solving sparse supervision (G), full augmentation/regularization, and exact 10-bit sequence accuracy amplifying per-bit errors.

### REPA note
- Downloaded REPA paper to `../readings/repa_2410.06940.pdf`.
- Cloned/read official code at `/tmp/REPA`.
- REPA projector is a plain 3-linear MLP with SiLU activations and no LayerNorm; its alignment loss L2-normalizes features for cosine similarity.
- For our MLP head, the most faithful REPA-inspired architecture ablation is SiLU + 3 Linear layers + no internal LayerNorm, not adding LayerNorm by default.

### K40 startup
- Run P (`6409`) and Run Q (`6410`) are both Running, past first batch and eval0. K40 compiled/runs at least through startup.

### Report constraint update — 2026-05-10 17:03 UTC
- Updated `agents/report_imagenet22k_mdh_handoff_2026-05-10.html` to emphasize no CE shortcut / no "投敌".
- ImageNet-22K reproduction constraints now explicitly exclude CE warm-start, `load_backbone_from`, `head_aux_ce`, auxiliary CE loss, and standard class-logit training.
- Run I and H are documented as diagnostic-only, not recommended recipes.

---

## Manual Loop Update — 2026-05-10 17:39 UTC

| Run | Latest new result | Interpretation |
|---|---|---|
| M normal-aug MLP K10 | ep139 `52.826%` / iter `57.356%`, loss `0.46864` | Flat/declining; normal augmentation is worse than low-aug. |
| N low-aug MLP K10 | ep179 `60.576%` / iter `64.094%`, loss `0.45628` | Still improving slowly; best pure-diffusion result so far. |
| L cross-attn K10 | latest ep39 `39.676%` / iter `49.884%` | Learns but trails MLP. |
| P K40 mild | running ep~10, eval0 `0.078%` / iter `0.098%` | K40 compiles/runs; wait ep19. |
| Q K40 aggressive | running ep~9.7, eval0 `0.082%` / iter `0.096%` | K40 compiles/runs; wait ep19. |

### REPA-style MLP ablation
- Added `head_mlp_activation` config and support for `silu` in `MLPDiffusionHead`; default remains `gelu`.
- Created `configs/remote_run_R_mlp_k40_lowaug_silu_config.yml` for a pure no-CE Run R: K40 low-aug mild, SiLU activation, no LayerNorm.
- Local smoke test passed: SiLU MLP model init/apply returns `(1, 10, 2)`.
- Launch attempt on `qxxa8y` failed because TPU manager reports the TPU is deleted; no Run R window exists yet.

---

## Manual Loop Update — 2026-05-10 18:08 UTC

- Run P K40 mild (`6409`) is Running around ep17; no ep19 eval yet.
- Run Q K40 aggressive (`6410`) is Running around ep16.5; no ep19 eval yet.
- Run N latest remains ep179 `60.576%` / iter `64.094%`.
- Run L latest remains ep39 `39.676%` / iter `49.884%`.
- Run R SiLU/no-LN/K40 remains queued locally; no manager-visible TPU slot available. `qxxa8y` is deleted and cannot be used.

---

## Manual Loop Update — 2026-05-10 18:40 UTC

| Run | Epoch | Accuracy | Iter accuracy | Loss | Invalid | Readout |
|---|---:|---:|---:|---:|---:|---|
| P K40 mild | 19 | `41.366%` | `47.526%` | `0.43401` | `1.486%` | K40 beats K10 early. |
| Q K40 wd0/sd0 | 19 | `41.540%` | `47.992%` | `0.43054` | `1.542%` | Slightly ahead of P; aggressive low-reg not harmful early. |
| N K10 low-aug | 19 | `38.328%` | `44.502%` | `0.45261` | `1.682%` | Matched-epoch baseline. |
| L cross-attn K10 | 59 | `47.802%` | `55.968%` | `0.40572` | `0.766%` | Learns but still trails MLP low-aug path. |
| I warm-start diagnostic | 199 | `64.028%` | `66.462%` | `0.31402` | `0.480%` | Diagnostic only; CE warm-start not allowed for main recipe. |

Conclusion:
- K40 is validated as promising at ep19. Continue P/Q to ep39 before deciding whether to scale further or tune regularization.
- Run R SiLU/no-LN/K40 remains queued. Launch when a real manager-visible TPU is available, likely by retiring M after ep159 if still dominated.

---

## Manual Loop Action — 2026-05-10 19:12 UTC

### Retired M
- Run M normal-aug MLP K10 ep159: `54.266%` / iter `58.480%`, loss `0.46328`.
- Interpretation: normal augmentation/reg is dominated by low-aug MLP variants. Stopped M/window `6390` to free the `cz2ivo` slot.

### Launched R
| Run | Window | TPU | Config | Status |
|---|---:|---|---|---|
| R | 6411 | `v6e-8-spot-gzy-cz2ivo` | K40 low-aug mild + SiLU MLP activation, no LayerNorm, no CE shortcut | First batch ready at `19:10:26 UTC`; compiling |

Logdir:
- `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_190919_lm2dnh_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval`

---

## Manual Loop Update — 2026-05-10 19:43 UTC

| Run | Latest | Readout |
|---|---|---|
| R SiLU/no-LN/K40 | eval0 `0.080%` / iter `0.094%`, loss `0.69092`, invalid `1.676%`; running ep~6 | Startup healthy. Wait ep19. |
| N K10 low-aug | ep199 `60.792%` / iter `64.684%`, loss `0.46998` | Still slowly improving, but loss rising. |
| P K40 mild | running ep~37 | ep39 expected next loop. |
| Q K40 wd0/sd0 | running ep~36 | ep39 expected next loop. |

Decision: no launch now. Await P/Q ep39; they decide whether K40 becomes the main recommendation for 22K.

---

## User-Directed Mixup/CutMix Run S — 2026-05-10 20:05 UTC

### Code/config change
- Added `diffusion_label_mode: soft_top2` for diffusion-aware mixup/cutmix supervision.
- For each mixed image, the top-2 soft-label classes receive the K repeated diffusion targets in proportion to their weights. Example: K10 and 50/50 mix -> 5 label-A targets + 5 label-B targets.
- Default `diffusion_label_mode` remains `argmax`; previous diffusion configs are unchanged unless they explicitly opt into `soft_top2`.
- Diffusion training only applies `apply_mixup_cutmix_batch` when `diffusion_label_mode == 'soft_top2'`.
- Syntax/smoke checks passed locally.

### P/Q ep39 update
| Run | Epoch | Accuracy | Iter accuracy | Loss | Invalid | Readout |
|---|---:|---:|---:|---:|---:|---|
| P K40 mild | 39 | `51.626%` | `56.988%` | `0.38740` | `0.998%` | Best K40 branch so far at ep39. |
| Q K40 wd0/sd0 | 39 | `50.968%` | `56.540%` | `0.40941` | `0.888%` | Slightly behind P; aggressive low-reg no longer ahead. |
| N K10 low-aug | 39 | `49.738%` | `55.010%` | `0.40066` | `1.058%` | Matched-epoch K10 reference. |

### Run S launch
| Run | Window | TPU | Config | Status |
|---|---:|---|---|---|
| S | 6412 | `v6e-8-spot-gzy-j3rqvs` | `configs/remote_run_S_mlp_k10_fullaug_mixlabel_config.yml` | Launched; initial compilation |

Logdir:
- `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_200200_2v7ro7_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`

Slot action:
- Stopped Run I/window `6388` because it is CE warm-start diagnostic-only and conflicts with the no-投敌 main recipe.

Run S startup confirmation:
- First batch ready at `2026-05-10 20:03:41 UTC`; awaiting compilation/eval0.

---

## Manual Loop Update — 2026-05-10 20:37 UTC

| Run | Latest | Readout |
|---|---|---|
| S full-aug soft_top2 K10 | eval0 `0.074%` / iter `0.116%`, loss `0.69222`, invalid `0.056%`; running ep~4.3 | New mixup/cutmix label path works through startup and eval0. Wait ep19. |
| L cross-attn K10 | ep79 `50.770%` / iter `58.048%`, loss `0.42392` | Still trails MLP K10 low-aug N at matched ep79 (`56.896%` / `60.954%`). |
| P K40 mild | ep39 `51.626%` / iter `56.988%`, loss `0.38740` | Best K40 branch so far; wait ep59. |
| Q K40 wd0/sd0 | ep39 `50.968%` / iter `56.540%`, loss `0.40941` | Behind P by ep39; aggressive low-reg less attractive. |
| R K40 SiLU/no-LN | running ep~17, latest eval0 `0.080%` / iter `0.094%` | Await ep19. |
| N K10 low-aug | running ep~217, latest ep199 `60.792%` / iter `64.684%` | Slow late improvement continues. |

Decision: no new launch/kill. Await S ep19, R ep19, and P/Q ep59.

---

## Manual Loop Update — 2026-05-10 21:08 UTC

| Run | Latest | Readout |
|---|---|---|
| R K40 SiLU/no-LN | ep19 `42.032%` / iter `48.254%`, loss `0.42070`, invalid `1.206%` | Best K40 ep19 so far by a small margin; wait ep39. |
| N K10 low-aug | ep219 `61.480%` / iter `65.148%`, loss `0.45957`, invalid `0.492%` | Still slowly improving; keep as mature K10 reference. |
| S full-aug soft_top2 K10 | running ep~9.3, latest eval0 `0.074%` / iter `0.116%` | Await ep19. |
| P K40 mild | running ep~54, latest ep39 `51.626%` / iter `56.988%` | Await ep59. |
| Q K40 wd0/sd0 | running ep~53, latest ep39 `50.968%` / iter `56.540%` | Await ep59. |

Decision: no new launch/kill. Await P/Q ep59 and S ep19.

---

## Manual Loop Action — 2026-05-10 21:42 UTC

### P/Q ep59
| Run | Epoch | Accuracy | Iter accuracy | Loss | Invalid | Readout |
|---|---:|---:|---:|---:|---:|---|
| P K40 mild | 59 | `55.920%` | `60.550%` | `0.38654` | `0.916%` | Still ahead of K10 matched epoch, but margin shrinking. |
| Q K40 wd0/sd0 | 59 | `54.176%` | `58.888%` | `0.44758` | `0.688%` | Dominated by P and slightly behind N ep59; stopped. |
| N K10 low-aug | 59 | `54.634%` | `59.000%` | `0.39770` | `0.940%` | Matched K10 reference. |

### Run T launch
| Run | Window | TPU | Config | Status |
|---|---:|---|---|---|
| T | 6415 | `v6e-8-spot-gzy-06q7u9` | `configs/remote_run_T_mlp_k80_lowaug_mild_config.yml` | Launched; reached Python config logging |

Logdir:
- `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_213853_66dm3n_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval`

Decision: Q retired; T K80 tests whether more K helps beyond K40 without changing the low-aug mild recipe.

Run T startup confirmation:
- First batch ready at `2026-05-10 21:40:34 UTC`; awaiting initial compilation/eval0.

---

## Manual Loop Action — 2026-05-10 22:16 UTC

### New results
| Run | Latest | Readout |
|---|---|---|
| T K80 GELU low-aug | eval0 `0.090%` / iter `0.124%`, loss `0.69085`, invalid `2.406%` | K80 compiles/runs through eval0; no immediate OOM. |
| S full-aug soft_top2 K10 | ep19 `2.364%` / iter `5.054%`, loss `0.64609`, invalid `0.778%` | Strong negative result; mixup/cutmix remains harmful despite correct K-view label allocation. |
| M full-aug/no-mixup K10 | ep19 `21.244%` / iter `28.400%` | S is far worse than old no-mix full-aug. |
| N low-aug K10 | ep19 `38.328%` / iter `44.502%` | Low-aug remains the reference. |

### Run U launch
| Run | Window | TPU | Config | Status |
|---|---:|---|---|---|
| U | 6417 | `v6e-8-spot-gzy-j3rqvs` | `configs/remote_run_U_mlp_k80_lowaug_silu_config.yml` | Launched; reached Python config logging |

Logdir:
- `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_221310_8gp2lr_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`

Decision: S retired after ep19 failure. U tests K80 + SiLU as the current highest-ROI combined direction.

Run U startup confirmation:
- First batch ready at `2026-05-10 22:14:50 UTC`; awaiting initial compilation/eval0.

---

## Manual Loop Update — 2026-05-10 22:49 UTC

| Run | Latest | Readout |
|---|---|---|
| U K80 SiLU low-aug | eval0 `0.088%` / iter `0.108%`, loss `0.69092`, invalid `1.930%` | K80+SiLU compiles/runs through eval0; no immediate OOM. |
| R K40 SiLU low-aug | ep39 `53.514%` / iter `58.272%`, loss `0.38476`, invalid `0.840%` | Beats P/GELU K40 ep39 by +1.888/+1.284; SiLU is promising. |
| P K40 GELU low-aug | latest ep59 `55.920%` / iter `60.550%` | Await ep79. |
| T K80 GELU low-aug | running ep~13, latest eval0 `0.090%` / iter `0.124%` | Await ep19. |
| N K10 low-aug | ep239 `62.212%` / iter `65.986%`, loss `0.45503` | Mature K10 baseline still improving slowly. |
| L cross-attn K10 | ep99 `51.808%` / iter `58.396%`, loss `0.44191` | Still behind MLP N; low priority. |
| sanity CE | ep239 `69.900%` | CE baseline continues rising. |

Decision: no launch/kill. Next critical checkpoints are T/U ep19 and R ep59.

---

## Manual Loop Update — 2026-05-10 23:24 UTC

| Run | Latest | Readout |
|---|---|---|
| T K80 GELU low-aug | ep19 `42.050%` / iter `48.410%`, loss `0.42973`, invalid `1.670%` | K80 compiles/runs and modestly beats P K40 GELU ep19, but is basically tied with R K40 SiLU ep19. |
| P K40 GELU low-aug | ep79 `57.226%` / iter `61.538%`, loss `0.39962`, invalid `0.888%`; stopped | K40 GELU advantage over K10 GELU shrank by ep79; retired to test K10 SiLU. |
| V K10 SiLU low-aug | window `6419`, first batch `23:22:51 UTC`, compile complete `23:23:38 UTC` | Activation-isolation run vs N is healthy at startup; await eval0/ep19. |
| U K80 SiLU low-aug | running ep~13; latest eval0 `0.088%` / iter `0.108%` | Await ep19. |
| R K40 SiLU low-aug | running ep~51; latest ep39 `53.514%` / iter `58.272%` | Await ep59; SiLU remains strongest architectural signal. |
| N K10 GELU low-aug | running ep~251; latest ep239 `62.212%` / iter `65.986%` | Best mature pure-diffusion baseline. |
| L cross-attn K10 | running ep~107; latest ep99 `51.808%` / iter `58.396%` | Low ROI; retire first if a slot is needed. |

Decision: stopped P and launched V. The current experiment set now isolates activation (V vs N), K scaling (T vs P/N), and combined SiLU+K scaling (U/R/T). Full augmentation/mixup-cutmixed diffusion remains disfavored after S.

---

## Manual Loop Update — 2026-05-11 00:05 UTC

| Run | Latest | Readout |
|---|---|---|
| U K80 SiLU low-aug | ep19 `42.366%` / iter `48.870%`, loss `0.41920`, invalid `1.490%` | Best ep19 among K40/K80 variants, but only modestly ahead of T/R. |
| R K40 SiLU low-aug | ep59 `56.122%` / iter `60.246%`, loss `0.40242`, invalid `0.634%` | Still ahead of N matched ep59; versus P K40 GELU ep59 it is slightly better single-step but slightly worse iterative. |
| V K10 SiLU low-aug | eval0 `0.094%` / iter `0.110%`, loss `0.69097`, invalid `1.992%`; running ep~8 | Startup healthy. Await ep19 for activation isolation against N. |
| T K80 GELU low-aug | running ep~29; latest ep19 `42.050%` / iter `48.410%` | Await ep39. |
| N K10 GELU low-aug | running ep~259; latest ep239 `62.212%` / iter `65.986%` | Mature baseline; next logged eval pending. |

Decision: no launch/kill. Continue the current matrix until R/T/U/V hit the next key checkpoints.

---

## Manual Loop Update — 2026-05-11 00:37 UTC

| Run | Latest | Readout |
|---|---|---|
| N K10 GELU low-aug | ep259 `62.594%` / iter `66.404%`, loss `0.45679`, invalid `0.422%` | Best mature no-shortcut baseline; still slowly improving. |
| T K80 GELU low-aug | manager/parser `Unknown`, but fresh train log around ep35.8 and no error/OOM | Treat as parser noise; await ep39. |
| U K80 SiLU low-aug | running ep~28.4; latest ep19 `42.366%` / iter `48.870%` | Await ep39. |
| V K10 SiLU low-aug | running ep~14.3; latest eval0 `0.094%` / iter `0.110%` | Await ep19. |
| R K40 SiLU low-aug | running ep~66; latest ep59 `56.122%` / iter `60.246%` | Await ep79. |

Decision: no intervention. Keep the current matrix running.

---

## Manual Loop Update — 2026-05-11 01:08 UTC

| Run | Latest | Readout |
|---|---|---|
| T K80 GELU low-aug | ep39 `52.628%` / iter `57.866%`, loss `0.39504`, invalid `0.878%` | Beats K40 GELU ep39 but trails K40 SiLU ep39. |
| V K10 SiLU low-aug | ep19 `37.934%` / iter `44.542%`, loss `0.44154`, invalid `1.748%` | Does not beat K10 GELU/N ep19; SiLU is not universally better. |
| L cross-attn K10 | ep119 `53.488%` / iter `59.382%`, loss `0.44323`, invalid `0.674%` | Still below MLP N at matched epoch; low ROI. |
| sanity CE | ep259 `71.024%`, loss `1.48117` | CE sanity continues normally. |
| U K80 SiLU low-aug | running ep~35; latest ep19 `42.366%` / iter `48.870%` | Await ep39, the key K80+SiLU checkpoint. |
| R K40 SiLU low-aug | running ep~72; latest ep59 `56.122%` / iter `60.246%` | Await ep79. |

Decision: no intervention. Wait for U ep39 and R ep79 before recycling slots.

---

## Manual Loop Action — 2026-05-11 01:48 UTC

| Run | Latest | Readout |
|---|---|---|
| U K80 SiLU low-aug | ep39 `52.828%` / iter `57.594%`, loss `0.39629`, invalid `0.668%` | Similar to T/K80 GELU, worse than R/K40 SiLU ep39. No clear K80+SiLU compounding. |
| R K40 SiLU low-aug | ep79 `57.624%` / iter `61.282%`, loss `0.42108`, invalid `0.544%` | Slightly better single-step than P/K40 GELU ep79, slightly worse iterative. SiLU remains mixed. |
| W K160 GELU low-aug | window `6424`, first batch ready `2026-05-11 01:47:40 UTC` | New K scaling run. Await compile/eval0 for memory confirmation. |
| L cross-attn K10 | stopped after ep119 `53.488%` / iter `59.382%` | Retired as low ROI; MLP variants dominate. |

Decision: retired L and launched W K160 GELU low-aug. Continue existing T/U/V/R/N runs until next checkpoints.

Run W startup confirmation:
- Initial compilation completed at `2026-05-11 01:48:43 UTC`.
- eval0 `0.096%` / iter `0.132%`, loss `0.69086`, invalid `2.574%`.
- K160 passed startup/compile/eval0 without immediate OOM. Await ep19.

## User-Corrected Run S Resume — 2026-05-11 02:58 UTC

The earlier stop of Run S after ep19 was too aggressive for the user's intended test. Full augmentation/mixup/cutmix may deliberately slow early convergence while improving late generalization, so S has been resumed from checkpoint rather than restarted.

| Run | Window | TPU | State | Key detail |
|-----|--------|-----|-------|------------|
| S resume | 6427 | `v6e-8-tmp213` / `1dqe89` | Running/compiling in manager, train logs alive | Restored from original S `checkpoint_25020`, now training from ep20 |

Resume details:
- Active logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_024615_6lmien_kmh-tpuvm-v6e-8-spot-gzy-1dqe89_asia-northeast1-b__b_lr_ep_eval`
- Restore confirmed from original S checkpoint: `checkpoint_25020`.
- Current train logs: ep20.061 `train_accuracy=0.022567`, loss `0.65389`; ep20.62 `train_accuracy=0.024247`, loss `0.65324`.
- Keep S to at least ep39/59 before judging the full-augmentation late-growth hypothesis.

New checkpoint reads:

| Run | Config | Latest eval | Interpretation |
|-----|--------|-------------|----------------|
| T | K80 GELU low-aug | ep59 `56.584%` / iter `61.278%` | K80 now beats R K40 SiLU at matched ep59 on both metrics. |
| V | K10 SiLU low-aug | ep39 `50.654%` / iter `56.088%` | Worse than K10 GELU N at matched ep39; SiLU is not a universal K10 win. |
| W | K160 GELU low-aug | training ep14, no ep19 yet | Healthy; wait ep19. |
| S | K10 full aug + soft_top2 mixup/cutmix | resumed ep20.6 | Do not kill on early slowness; wait ep39/59. |

## Manual Loop Results — 2026-05-11 03:31 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| N K10 GELU low-aug | ep299 `63.408%` / iter `67.480%` | Best mature pure-diffusion baseline continues to improve. |
| R K40 SiLU low-aug | ep99 `59.146%` / iter `62.334%` | Improving, but below mature N; SiLU not decisive. |
| T K80 GELU low-aug | ep59 `56.584%` / iter `61.278%` | Stronger than U/K80 SiLU at matched ep59. |
| U K80 SiLU low-aug | ep59 `56.480%` / iter `60.426%` | Slightly worse single-step and much worse iterative than T. |
| V K10 SiLU low-aug | ep39 `50.654%` / iter `56.088%`, running ep50 | Behind N/GELU at K10. |
| W K160 GELU low-aug | ep19 `42.846%` / iter `49.468%` | Best early K-scaling result; keep. |
| S K10 full aug + soft_top2 | resumed, training ep25.9, no new eval | Keep to ep39/59 for late-growth test. |

Decision: no new launch/kill/resume. K160 GELU is now the main K-scaling candidate; S remains the user's full-augmentation test and must not be killed on early slowness alone.

## Manual Loop Status — 2026-05-11 04:03 UTC

No new eval checkpoint since the 03:31 loop. All main runs are alive.

| Run | Current progress | Next important eval |
|-----|------------------|---------------------|
| T K80 GELU | ep78 | ep79 imminent |
| V K10 SiLU | ep57 | ep59 soon |
| S full aug soft_top2 | ep31 | ep39 |
| W K160 GELU | ep27 | ep39 |
| N K10 GELU | ep307.8 | ep319 |
| R K40 SiLU | ep108 | ep119 |
| U K80 SiLU | ep71 | ep79 |

Decision: wait; no intervention.

## Manual Loop Results — 2026-05-11 04:34 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | ep79 `58.302%` / iter `62.476%` | Best mid-epoch K-scaling branch so far; beats K40 P/R at ep79. |
| V K10 SiLU low-aug | ep59 `55.462%` / iter `59.876%` | Beats N/GELU at matched ep59 after trailing at ep19/39; keep for later. |
| S full aug soft_top2 | running ep35.9, no new eval | Keep to ep39/59. |
| W K160 GELU low-aug | running ep33.7, no new eval | Await ep39; ep19 was strongest early K result. |
| U K80 SiLU low-aug | running ep77, latest ep59 `56.480%` / iter `60.426%` | Await ep79 before retiring. |

Decision: no intervention.

## 2026-05-11 05:13 UTC - S kept alive; U retired; X K320 launched
- Run S (K10 full baseline aug + corrected soft_top2 mixup/cutmix) is running, not killed. Manager: window 6427 on `v6e-8-spot-gzy-1dqe89`, around ep41. Latest eval remains ep39 single 11.834%, iterative 17.946%, invalid 0.974%. Keep S at least through ep59 before making conclusions about augmentation.
- Run U (K80 SiLU low-aug) was retired on `v6e-8-spot-gzy-j3rqvs` because it is dominated by Run T (K80 GELU) at matched checkpoints; this does not affect S.
- Launched Run X: K320 GELU low-aug mild, no warm-start, no aux CE, no CE shortcut. Manager window 6429 on `v6e-8-spot-gzy-j3rqvs`.
- Run X logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_050959_acnfxy_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`.
- Run X startup verified: config printed `n_masks_per_image: 320`, total params 97,927,036; compilation completed; first train log `[100] train_accuracy=0.0009902, train_loss=0.69315, ep=0.079128`. No OOM observed at launch.

## 2026-05-11 05:44 UTC loop check
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260511_054405.txt`.
- No resume needed. Active DeiT jobs: N ep328.18, R ep128.36, T ep98.55, V ep78, W ep48, S ep47, X ep6. U is killed; H remains failed diagnostic.
- S remains alive and must be kept: latest eval still ep39 11.834% / iter 17.946%; latest train around ep46.9 with train acc ~0.137-0.145 and loss ~0.588-0.594.
- X K320 launched successfully and is running: eval0 0.096% / iter 0.108%, train reached ep6.31 with train acc 0.29622 and loss 0.50396. No launch OOM.
- N K10 GELU is near completion: ep328.26, latest mature eval ep319 63.604% / iter 67.726%.
- Waiting for next decision checkpoints: V ep79, T ep99, W ep59, S ep59, X ep19. Do not kill S before ep59.

## 2026-05-11 06:20 UTC loop check and Run Y launch
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260511_061506.txt`.
- Run N finished. Best/latest eval is ep319: 63.604% single / 67.726% iterative, invalid 0.448%. No later final eval was printed before finish.
- Run T K80 GELU reached ep99: 59.146% single / 63.188% iterative, loss 0.43325, invalid 0.658%. It matches R/K40 SiLU single at ep99 (59.146%) but has higher iterative accuracy (63.188% vs 62.334%).
- Run V K10 SiLU reached ep79: 57.348% single / 61.210% iterative, loss 0.38456, invalid 0.630%. This is slightly below earlier P/K40 GELU ep79 and below T/K80 GELU ep79; SiLU at K10 is not an obvious win by ep79.
- Run W K160 GELU is running ep54.5; latest eval remains ep39 53.186% / iter 58.478%.
- Run S full aug + soft_top2 is running ep51.9; latest eval remains ep39 11.834% / iter 17.946%. Keep to ep59+ per user correction.
- Run X K320 GELU is running ep12.7; latest eval0 only. No OOM.
- Historical check: full augmentation without mixup/cutmix was already Run M; it reached ep159 54.266% / iter 58.480% and remained below low-aug Run N. Therefore do not waste the freed N slot on a duplicate no-mixup full-aug run.
- Launched Run Y: K640 GELU low-aug mild on `v6e-8-spot-gzy-3djlis`, window 6430, logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_061640_ak1rkw_kmh-tpuvm-v6e-8-spot-gzy-3djlis_asia-northeast1-b__b_lr_ep_eval`.
- Run Y startup verified: config printed `n_masks_per_image: 640`, params 97,927,036, first batch ready, compilation completed, train step reached `[400] train_accuracy=0.0012684, train_loss=0.69254, ep=0.31891`. No OOM observed at launch.

## 2026-05-11 07:06 UTC loop results after waiting for S ep59
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260511_065156.txt`.
- Run S full baseline aug + corrected soft_top2 mixup/cutmix reached ep59: 30.338% single / 37.420% iterative, loss 0.47554, invalid 0.934%.
- S interpretation: user was right that early full-aug convergence was misleading; S jumped from ep39 11.834% / 17.946% to ep59 30.338% / 37.420%. However it is still far behind low-aug N/K10 at ep59 (54.634% / 59.000%) and behind the old full-aug/no-mix Run M trajectory. Keep S to ep79 for late-growth evidence; do not call it best.
- Run W K160 GELU ep59: 56.710% / iter 61.160%, loss 0.40450, invalid 0.616%. This is roughly comparable to T/K80 ep59 (56.584% / iter 61.278%), not a clear K160 win.
- Run X K320 GELU ep19: 42.706% / iter 48.984%, loss 0.42800, invalid 1.330%. This is slightly below W/K160 ep19 (42.846% / iter 49.468%), so larger K is not monotonically helping at early eval.
- Run Y K640 is running and healthy: eval0 0.092% / iter 0.108%; train reached around ep6.55 with loss ~0.4966, no OOM so far.
- Run R K40 SiLU ep139: 60.020% / iter 63.098%. Mature but not compelling versus T/K80 iterative trajectory.
- No resume needed. Do not launch a new job this loop; wait for T ep119, V ep99, W ep79, X ep39, Y ep19, S ep79.

## 2026-05-11 07:42 UTC loop check and Run Z launch
- Full `agents/` read completed: `/tmp/deit_agents_full_read_20260511_073653.txt`.
- T K80 GELU ep119: 59.774% single / 63.574% iterative, loss 0.43353, invalid 0.746%. It is now slightly better than R K40 SiLU ep119 on single and clearly better on iterative accuracy.
- V K10 SiLU ep99: 58.030% / iter 61.646%, loss 0.41189, invalid 0.520%. This underperforms the K10 GELU trajectory and is not a useful branch to continue.
- R K40 SiLU ep139: 60.020% / iter 63.098%, still improving slowly but no longer higher priority than T/K80 or the K-scaling/full-aug branches.
- W K160 GELU running ep71; latest eval remains ep59 56.710% / iter 61.160%.
- S K10 full aug + soft_top2 running ep64.9; latest eval remains ep59 30.338% / iter 37.420%. Keep S to ep79.
- X K320 GELU running ep29.5; latest eval remains ep19 42.706% / iter 48.984%.
- Y K640 GELU running ep15.8; no OOM, eval0 only so far.
- Action: stopped Run V/window 6419 on `axuxm0` and launched Run Z on the same TPU.
- Run Z: K80 + full baseline augmentation + corrected `soft_top2` mixup/cutmix, GELU, no warm-start, no aux CE, no CE shortcut. Window 6431, logdir `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_073833_rzxudx_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval`.
- Z startup verified: printed `use_mixup_cutmix: true`, `diffusion_label_mode: soft_top2`, `n_masks_per_image: 80`; first batch ready; train reached `[200] train_accuracy=0.0012223, train_loss=0.69304, ep=0.15905`. No OOM at startup.

## 2026-05-11 08:27 UTC - Run S resume status
- S was not killed; it has been resumed after the original TPU/SSH failure.
- Failed resume attempt 6432 diagnosis: missing `model: ViT_base_mdh_mlp` in resume yml made the code instantiate default `ViT_base`; diffusion training then failed with missing `diffusion_head`.
- Fixed config and relaunched S resume2 as window 6433 on `favaxa`.
- Logdir: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260511_082259_aoimz8_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval`.
- Verified resumed from `checkpoint_75060`; after compilation, training continued from step 75100 / ep60.024. Latest checked: step 75300 / ep60.184, train_acc 0.27961, train_loss 0.55298.
- Await S ep79 eval before judging full augmentation; do not retire S on early slowness.

## Manual Loop Results — 2026-05-11 09:04 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| W K160 GELU low-aug | ep79 `58.320%` / iter `62.348%` | Tied with K80 at matched ep79; K160 is not a clear win. |
| X K320 GELU low-aug | ep39 `52.928%` / iter `58.350%` | Slightly below W/K160 ep39; high-K scaling not monotonic. |
| Y K640 GELU low-aug | ep19 `42.734%` / iter `49.136%` | Similar to X/K320 ep19; no high-K gain yet. |
| R K40 SiLU low-aug | ep159 `60.540%` / iter `63.454%` | Retired; iterative accuracy already below T/K80 ep119. |
| S K10 full aug soft_top2 | resumed, running ep65; latest eval ep59 `30.338%` / iter `37.420%` | Keep to ep79; do not kill due slow convergence. |
| Z K80 full aug soft_top2 | running ep12, eval0 only | Keep to ep19+. |
| AA K80 strong aug no mix/cut | launched window 6434; train step 200 / ep0.159 | Tests RA/reprob/repeated/WD/SD without mixup/cutmix. |

Action: replaced R with Run AA on `cz2ivo`. No warm-start, no aux CE, no CE shortcut.

## Manual Loop Results — 2026-05-11 09:41 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | ep139 `60.280%` / iter `64.014%` | Strongest active low-aug scaling branch; confirms retiring R was correct. |
| Y K640 GELU low-aug | ep39 `52.898%` / iter `58.352%` | Not better than X/K320 or W/K160 at matched ep39; retired. |
| AA K80 strong aug no mix/cut | eval0 `0.106%` / iter `0.078%`; running ep5 | Healthy; will test strong augmentation without mix/cut. |
| AB K80 low aug + mix/cut soft_top2 | launched window 6435; train step 200 / ep0.159 | Healthy; isolates mix/cut under otherwise low-aug T settings. |
| S K10 full aug soft_top2 | running ep71; latest eval ep59 `30.338%` / iter `37.420%` | Keep to ep79 per user correction. |
| Z K80 full aug soft_top2 | running ep18, eval0 only | Keep to ep19+. |

Action: replaced Y/K640 with AB/K80 low-aug + corrected mixup/cutmix soft_top2. No warm-start, no aux CE, no CE shortcut.

## Manual Loop Results — 2026-05-11 10:18 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| W K160 GELU low-aug | ep99 `58.822%` / iter `62.762%` | Below T/K80 ep99; K160 not useful. |
| X K320 GELU low-aug | ep59 `56.668%` / iter `61.136%` | No gain over K80/K160; retired. |
| Z K80 full aug soft_top2 | ep19 `3.694%` / iter `7.152%` | Full aug + mix/cut still very slow at K80; keep for later checkpoints. |
| AB K80 low aug + mix/cut soft_top2 | eval0 `0.074%` / iter `0.100%`; running ep6 | Healthy startup. |
| AA K80 strong aug no mix/cut | running ep11; eval0 `0.106%` / iter `0.078%` | Healthy; await ep19. |
| S K10 full aug soft_top2 | running ep77; latest eval ep59 `30.338%` / iter `37.420%` | Keep to ep79+. |
| AC K80 SiLU+LayerNorm low-aug | launched window 6436; train step 300 / ep0.239 | Tests REPA-style normalized MLP head. |

Action: added optional MLP LayerNorm support, verified py_compile, replaced X/K320 with AC/K80 SiLU+LayerNorm. No warm-start, no aux CE, no CE shortcut.

## Manual Loop Results — 2026-05-11 10:54 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | ep159 `60.578%` / iter `64.212%` | Strongest active low-aug branch; keep running. |
| W K160 GELU low-aug | ep99 `58.822%` / iter `62.762%` | Dominated by T/K80 ep99; retired. |
| S K10 full aug soft_top2 resume2 | ep79 `43.392%` / iter `49.778%` | Late growth is real; keep to ep99 despite slow early convergence. |
| Z K80 full aug soft_top2 | ep19 `3.694%` / iter `7.152%` | Very slow under full aug + mix/cut; keep to ep39. |
| AA K80 strong aug no mix/cut | running ep17; eval0 `0.106%` / iter `0.078%` | Healthy; await ep19. |
| AB K80 low aug + mix/cut soft_top2 | running ep13; eval0 `0.074%` / iter `0.100%` | Healthy; await ep19. |
| AC K80 SiLU+LayerNorm low-aug | running ep6; eval0 `0.112%` / iter `0.136%` | Healthy early LayerNorm diagnostic. |
| AD K80 GELU+LayerNorm low-aug | launched window 6437; train step 200 / ep0.159 | Paired LayerNorm diagnostic against AC/T; no warm-start, no aux CE. |

Action: replaced W/K160 with AD/K80 GELU+LayerNorm. High-K >80 is no longer a priority. S remains active and must not be killed before ep99 unless it fails.

## Manual Loop Results — 2026-05-11 11:25 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | running ep168; latest ep159 `60.578%` / iter `64.212%` | Keep; active low-aug reference. |
| S K10 full aug soft_top2 resume2 | running ep88; latest ep79 `43.392%` / iter `49.778%` | Keep to ep99 per user correction. |
| Z K80 full aug soft_top2 | running ep35; latest ep19 `3.694%` / iter `7.152%` | Weakest augmentation branch; keep to ep39 before decision. |
| AA K80 strong aug no mix/cut | ep19 `24.904%` / iter `33.344%` | Strong augmentation alone hurts early convergence but is not catastrophic. |
| AB K80 low aug + mix/cut soft_top2 | ep19 `32.838%` / iter `39.854%` | Mix/cut alone hurts early convergence less than strong augmentation alone in iterative gap, and far less than full strong+mix. |
| AC K80 SiLU+LayerNorm low-aug | running ep14; eval0 `0.112%` / iter `0.136%` | Await ep19. |
| AD K80 GELU+LayerNorm low-aug | running ep6; eval0 `0.124%` / iter `0.136%` | Await ep19. |

Decision: no intervention. The augmentation 2x2 now suggests the severe early failure comes from combining strong augmentation with mix/cut, not from either component alone.

## Manual Loop Results — 2026-05-11 11:56-12:10 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| Z K80 full aug soft_top2 | Error after train ep37.165; latest eval ep19 `3.694%` / iter `7.152%` | Infrastructure failure, not model failure. Resume from own `checkpoint_25020` when a v6e-8 slot/capacity exists. |
| AC K80 SiLU+LayerNorm low-aug | ep19 `40.864%` / iter `46.078%` | Viable, but below T/GELU-no-LN ep19 `42.050%` / iter `48.410%`. |
| AD K80 GELU+LayerNorm low-aug | running ep15.9; eval0 `0.124%` / iter `0.136%` | Await ep19. |
| S K10 full aug soft_top2 resume2 | running ep95.8; latest ep79 `43.392%` / iter `49.778%` | Keep to ep99. |
| T K80 GELU low-aug | running ep177; latest ep159 `60.578%` / iter `64.212%` | Await ep179. |
| AA K80 strong aug no mix/cut | running ep30.6; latest ep19 `24.904%` / iter `33.344%` | Await ep39. |
| AB K80 low aug + mix/cut soft_top2 | running ep29.8; latest ep19 `32.838%` / iter `39.854%` | Await ep39. |

Action: attempted Z resume. It failed because original TPU `axuxm0` is deleted and re-apply timed out due asia-northeast1-b capacity. No experiment slot was killed.

## Manual Loop Results — 2026-05-11 12:41 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| S K10 full aug soft_top2 resume2 | ep99 `50.646%` / iter `56.452%` | Strong late catch-up, but still below T/K80 low-aug at ep99. Keep. |
| T K80 GELU low-aug | ep179 `60.734%` / iter `64.308%`; log still running ep183 | Growth is slowing; keep to ep199 before retiring. |
| AD K80 GELU+LayerNorm low-aug | ep19 `42.010%` / iter `47.516%` | No early improvement over plain GELU T ep19. Keep to ep39. |
| AC K80 SiLU+LayerNorm low-aug | running ep29.8; latest ep19 `40.864%` / iter `46.078%` | Viable but behind T/AD at ep19. |
| AA K80 strong aug no mix/cut | running ep35.7; latest ep19 `24.904%` / iter `33.344%` | Await ep39. |
| AB K80 low aug + mix/cut soft_top2 | running ep36; latest ep19 `32.838%` / iter `39.854%` | Await ep39. |
| Z K80 full aug soft_top2 | Error; latest ep19 `3.694%` / iter `7.152%` | Resume blocked by deleted TPU and capacity. |

Decision: no intervention. Keep active diagnostics to their next matched checkpoints.

## Manual Loop Results — 2026-05-11 13:12 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| AB K80 low aug + mix/cut soft_top2 | ep39 `57.148%` / iter `61.642%` | Strong result; beats T low-aug/no-mix ep39 by +4.520 / +3.776. Keep. |
| AA K80 strong aug no mix/cut | ep39 `43.568%` / iter `50.788%` | Retired after ep39; strong augmentation alone is much weaker than T/AB. |
| Z K80 full aug soft_top2 resume2 | launched window 6439; restored own `checkpoint_25020`; train step 25100 / ep20.061 | Resumed successfully after copying checkpoint to us-east5 bucket. Await ep39. |
| Z resume1 | window 6438 failed | GCS 403: us-east5 service account could not read asia checkpoint. Fixed by copying checkpoint to us-east5. |
| S K10 full aug soft_top2 resume2 | running ep105; latest ep99 `50.646%` / iter `56.452%` | Keep to ep119. |
| T K80 GELU low-aug | running by log ep189; latest ep179 `60.734%` / iter `64.308%` | Manager says Unknown due temp log permission, but output log is active. |
| AC K80 SiLU+LayerNorm low-aug | running ep36; latest ep19 `40.864%` / iter `46.078%` | Await ep39. |
| AD K80 GELU+LayerNorm low-aug | running ep28; latest ep19 `42.010%` / iter `47.516%` | Await ep39. |

Action: retired AA and relaunched Z as resume2 on `cz2ivo` from Z's own checkpoint copied into us-east5. No warm-start, no aux CE, no `load_backbone_from`.

## Manual Loop Results — 2026-05-11 14:24-14:27 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | ep199 `61.544%` / iter `65.190%` | Still improving; keep to ep219 before retirement decision. |
| S K10 full aug soft_top2 resume2 | running ep117; latest ep99 `50.646%` / iter `56.452%` | Keep to ep119; do not kill for slow full-augmentation convergence. |
| AB K80 low-aug mix/cut soft_top2 | running ep56; latest ep39 `57.148%` / iter `61.642%` | Strong branch; wait ep59/79. |
| AE K80 low-aug mixup-only soft_top2 | launched window 6440; train step 200 / ep0.159 | Healthy; no warm-start, no aux CE, no `load_backbone_from`. |
| AF K80 low-aug cutmix-only soft_top2 | launched window 6441; train step 200 / ep0.159 | Healthy; no warm-start, no aux CE, no `load_backbone_from`. |
| Z K80 full aug soft_top2 | still failed/infrastructure-blocked | Resume later only from its own checkpoint when a v6e-8 slot/capacity exists. |

Action: killed retired LayerNorm diagnostic slots AC/AD and replaced them with AE/AF to split AB's promising low-augmentation mix/cut result into mixup-only vs cutmix-only controls.

## Manual Loop Results — 2026-05-11 14:58-15:00 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| AB K80 low-aug mix/cut soft_top2 | ep59 `64.890%` / iter `68.616%` | New best pure-diffusion result; already beats completed N/K10 final ep319 `63.604%` / `67.726%`. Preserve and keep running. |
| S K10 full aug soft_top2 resume2 | ep119 `55.440%` / iter `60.658%` | Continues delayed catch-up but remains below AB/T; keep per user correction. |
| T K80 GELU low-aug | running ep211; latest ep199 `61.544%` / iter `65.190%` | Keep to ep219 for next reference point. |
| AE K80 low-aug mixup-only soft_top2 | eval0 `0.080%` / iter `0.088%`; running ep6 | Healthy; wait ep19. |
| AF K80 low-aug cutmix-only soft_top2 | eval0 `0.100%` / iter `0.138%`; running ep6 | Healthy; wait ep19. |
| Z K80 full aug soft_top2 | stale failed reruns 6443/6444; no useful new eval | Infrastructure/manager artifact, not model verdict. Resume later only from own checkpoint on a real free v6e-8 slot. |

Decision: no intervention. AB is the current best config; next critical checkpoints are AE/AF ep19, AB ep79, S ep139, and T ep219.

## Manual Loop Results — 2026-05-11 15:29-15:31 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | running ep217; latest ep199 `61.544%` / iter `65.190%` | Ep219 imminent; keep. |
| AB K80 low-aug mix/cut soft_top2 | running ep68; latest ep59 `64.890%` / iter `68.616%` | Current best; keep to ep79+. |
| S K10 full aug soft_top2 resume2 | running ep127; latest ep119 `55.440%` / iter `60.658%` | Keep; delayed catch-up continues but still below AB. |
| AE K80 low-aug mixup-only soft_top2 | running ep12; latest eval0 only | Healthy; wait ep19. |
| AF K80 low-aug cutmix-only soft_top2 | running ep12; latest eval0 only | Healthy; wait ep19. |
| Z K80 full aug soft_top2 resume2 rerun | running ep22 on v5p-8 from own checkpoint_25020 | Valid same-run resume, not warm-start. Keep for missing full-aug K80 eval; wall-clock not comparable to v6e. |

Decision: no intervention. Next decisive checks remain T ep219, AE/AF ep19, AB ep79, S ep139, and Z ep39 if it survives long enough.

## Manual Loop Results — 2026-05-11 16:01-16:07 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| T K80 GELU low-aug | ep219 `62.162%` / iter `65.682%` | Retired after ep219; dominated by AB ep59. |
| AG K160 low-aug mix/cut soft_top2 | launched window 6447; train step 300 / ep0.239 | AB follow-up testing K160 under corrected mix/cut; no warm-start, no aux CE, no `load_backbone_from`. |
| AB K80 low-aug mix/cut soft_top2 | running ep75; latest ep59 `64.890%` / iter `68.616%` | Current best; wait ep79. |
| AE K80 low-aug mixup-only soft_top2 | running ep18.8; latest eval0 only | Ep19 imminent; wait. |
| AF K80 low-aug cutmix-only soft_top2 | running ep18.8; latest eval0 only | Ep19 imminent; wait. |
| S K10 full aug soft_top2 resume2 | running ep132; latest ep119 `55.440%` / iter `60.658%` | Keep; full-aug delayed catch-up but below AB. |
| Z K80 full aug soft_top2 resume2 rerun | running ep25 on v5p-8 from own checkpoint | Keep for missing full-aug K80 control. |

Action: replaced T with AG. Next decisive checks are AE/AF ep19, AB ep79, S ep139, AG eval0/ep19, and Z ep39 if it survives.

## Manual Loop Results — 2026-05-11 16:37-16:39 UTC

| Run | Latest eval/status | Read |
|-----|--------------------|------|
| AB K80 low-aug mix/cut soft_top2 | ep79 `68.464%` / iter `71.404%` | New best by a large margin; keep running. |
| AF K80 low-aug cutmix-only soft_top2 | ep19 `36.586%` / iter `43.614%` | Fastest early single-augment control; keep to ep39. |
| AE K80 low-aug mixup-only soft_top2 | ep19 `35.092%` / iter `41.990%` | Also faster than AB at ep19; keep to ep39. |
| S K10 full aug soft_top2 resume2 | running ep138; latest ep119 `55.440%` / iter `60.658%` | Ep139 imminent; keep. |
| AG K160 low-aug mix/cut soft_top2 | running ep6; eval0 `0.078%` / iter `0.096%` | Healthy K160 follow-up; wait ep19. |
| Z K80 full aug soft_top2 resume2 rerun | running ep29 on v5p-8; no new eval | Keep for missing full-aug K80 control. |

Decision: no intervention. AB is current best config; next checks are S ep139, AE/AF ep39, AB ep99, AG ep19, and Z ep39.

## Manual Loop Results — 2026-05-11 17:11 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| S K10 full aug soft_top2 resume2 | ep139 `58.748%` / iter `63.468%` | Delayed full-aug catch-up continues; keep, but still below AB. |
| AB K80 low-aug mix/cut soft_top2 | running ep89; latest ep79 `68.464%` / iter `71.404%` | Current best; wait ep99. |
| AE K80 low-aug mixup-only soft_top2 | running ep32; latest ep19 `35.092%` / iter `41.990%` | Keep to ep39. |
| AF K80 low-aug cutmix-only soft_top2 | running ep32; latest ep19 `36.586%` / iter `43.614%` | Keep to ep39. |
| AG K160 low-aug mix/cut soft_top2 | running ep13; eval0 `0.078%` / iter `0.096%` | Healthy K160 follow-up; wait ep19. |
| Z K80 full aug soft_top2 resume2 rerun | running ep32 on v5p-8; no new eval | Keep to ep39 if it survives. |

Decision: no intervention. Next decisive checks are AE/AF ep39, AB ep99, AG ep19, Z ep39, and S ep159.

## Manual Loop Results — 2026-05-11 17:42-18:09 UTC
| Run | Latest eval/status | Read |
|-----|--------------------|------|
| AB K80 low-aug mix/cut soft_top2 | ep99 `70.380%` / iter `73.076%` | New best; keep running. |
| AF K80 low-aug cutmix-only soft_top2 | ep39 `58.938%` / iter `63.560%` | Beats AB and AE at matched ep39; keep to ep59. |
| AE K80 low-aug mixup-only soft_top2 | ep39 `55.496%` / iter `59.818%`; retired | Dominated by AF and below AB at matched ep39. |
| AG K160 low-aug mix/cut soft_top2 | ep19 `32.340%` / iter `39.472%` | Not better than AB K80 mixed at ep19; keep to ep39 before final K160 verdict. |
| AH K80 low-aug cutmix-heavy p=0.75 soft_top2 | launched window 6448; eval0 `0.074%` / iter `0.100%` | Tests whether cutmix-heavy mixed augmentation keeps AF's early advantage with some mixup regularization. |
| S K10 full aug soft_top2 resume2 | running ep152; latest ep139 `58.748%` / iter `63.468%` | Keep; next ep159. |
| Z K80 full aug soft_top2 resume2 rerun | running ep38 on v5p-8; no new eval | Keep to ep39 if it survives. |

Action: retired AE and launched AH on the same TPU. No warm-start, no aux CE, no `load_backbone_from`. Next checks: Z ep39, AB ep119, AF ep59, S ep159, AG ep39, AH ep19.

## 2026-05-11 Reference-Aligned CE Sanity Relaunch

| Run | Window | TPU | Status / Result | Notes |
|---|---:|---|---|---|
| CE sanity reference-aligned retry | 6449 | v5p-8-tmp201 / `pzfrwj` | Failed before training | Infrastructure occupancy: unrelated `LAION400M_upload.py` and `/tmp/libtpu_lockfile` on TPU. Not a code/config result. |
| CE sanity reference-aligned retry | 6450 | v6e-8 `06q7u9` | Running; step 300 ep0.239 | Config confirmed: `ViT_base_v2`, no LayerScale, exact GELU, `adamw_b2=0.999`, CE extra smoothing 0.0, dataset mixup smoothing 0.1. |
| AG K160 low-aug mixed | 6447 | v6e-8 `06q7u9` | Killed at ep~33.6 | Killed to free TPU for sanity. Last useful eval: ep19 `32.340%` single / `39.472%` iterative. |

## 2026-05-11 20:35 Loop Results

| Run | Latest useful eval | Status | Decision |
|---|---:|---|---|
| CE sanity ref-aligned `6450` | eval0 `0.570%`, train ep15.3 loss `5.8746` | Running on `06q7u9` | Top priority; keep to ep19/39. |
| AB K80 low-aug mixed mix/cut | ep119 `71.886%` / iter `74.052%`, loss `0.248336` | Running ep129 | New best pure-diffusion result; keep. |
| AF K80 low-aug cutmix-only | ep59 `66.298%` / iter `69.818%`, loss `0.278813` | Running ep73.8 | Stronger than AB at matched ep59; keep to ep79/99. |
| S K10 full-aug mixed mix/cut | ep159 `61.124%` / iter `65.528%`, loss `0.316067` | Running ep176.3 | Slow but improving; keep per user correction. |
| AH K80 low-aug cutmix-heavy | ep19 `32.018%` / iter `39.488%`, loss `0.482421` | Running ep31.9 | Worse than AF early; keep only to ep39 verdict. |
| Z K80 full-aug mixed mix/cut | ep39 `19.730%` / iter `27.324%`, loss `0.530267` | Auto-resume `6452` failed | Retire / do not resume unless explicitly requested. |

## 2026-05-11 Z Resume Bug Fix

| Run | Window | Result | Notes |
|---|---:|---|---|
| Z auto-resume | 6452 | Failed before training | Duplicate `--config.load_from` generated by resume machinery caused `FATAL Flags parsing error: The flag 'config.load_from' is defined twice.` |
| Z manual resume3 | 6453 | Restored from `checkpoint_50040` and training ep40+ | Uses old Z stage code for fidelity; no duplicate flag; first confirmed train steps `[50100]` ep40.043 loss `0.57983`, `[50400]` ep40.282 loss `0.58955`, `[50600]` ep40.442 loss `0.58107`. |

## 2026-05-12 03:01 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | ep199 `74.064%` / iter `75.886%`, loss `0.257361` | Best mature pure-diffusion branch; keep to ep219+. |
| AF K80 low-aug cutmix-only | ep139 `72.502%` / iter `74.584%`, loss `0.250571` | Competitive with AB, but no longer clearly ahead; keep to ep159/179. |
| AH K80 low-aug cutmix-heavy p=0.75 | ep99 `70.798%` / iter `73.598%`, loss `0.252995` | Best matched ep99 among AB/AF/AH; keep, do not retire. |
| S K10 full-aug mixed mix/cut | ep219 `65.662%` / iter `69.198%`, loss `0.296665` | Still improving slowly; keep per full-aug delayed-convergence concern. |
| CE sanity ref-aligned `6450` | ep59 `50.392%`, train ep77 | Not reproduced; old v3 sanity was ep59 `59.132%`. Keep to ep99, then consider v3+current-fixes ablation when a slot opens. |
| Z K80 full-aug mixed mix/cut manual resume | ep59 `40.768%` / iter `47.954%`, train ep77.5 | Resume bug fixed and run is live; keep to ep79/99 for K80 full-aug control. |
| Z auto-resume `6455` | Failed | Ignore as stale/duplicate auto-resume; effective Z is `6453`. |

Action: no kill, no resume, no new launch this loop. Next checks: AB ep219, AF ep159, AH ep119, S ep239, CE ep79/99, Z ep79/99.

## 2026-05-12 03:34 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| S K10 full-aug mixed mix/cut | ep239 `66.574%` / iter `70.134%`, loss `0.297908` | Slow catch-up continues; keep for full-aug curve. |
| CE sanity ref-aligned `6450` | ep79 `54.122%`, loss `2.187716` | Still not reproduced; old v3 sanity ep79 was `59.342%`. Keep to ep99 before replacing/ablating. |
| Z K80 full-aug mixed mix/cut manual resume | ep79 `52.408%` / iter `58.076%`, loss `0.359067` | Catching up after resume; keep to ep99. |
| AB K80 low-aug mixed mix/cut | running ep210; latest ep199 `74.064%` / iter `75.886%` | Keep to ep219+. |
| AF K80 low-aug cutmix-only | running ep157-158; latest ep139 `72.502%` / iter `74.584%` | Ep159 imminent; keep. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep114; latest ep99 `70.798%` / iter `73.598%` | Ep119 next; keep because it was best matched ep99. |

Action: no kill, no resume, no launch. Next checks: AF ep159, AH ep119, AB ep219, CE ep99, Z ep99, S ep259.

## 2026-05-12 04:05 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AF K80 low-aug cutmix-only | ep159 `72.946%` / iter `74.926%`, loss `0.252867` | Below AB at matched ep159; keep to ep179. |
| AH K80 low-aug cutmix-heavy p=0.75 | ep119 `72.216%` / iter `74.424%`, loss `0.249503` | Best matched ep119 among AB/AF/AH; keep to ep139/159. |
| AB K80 low-aug mixed mix/cut | running ep216; latest ep199 `74.064%` / iter `75.886%` | Keep; ep219 soon. |
| S K10 full-aug mixed mix/cut | running ep247; latest ep239 `66.574%` / iter `70.134%` | Keep. |
| CE sanity ref-aligned `6450` | running ep87; latest ep79 `54.122%` | Not reproduced; keep to ep99. |
| Z K80 full-aug mixed mix/cut manual resume | running ep83; latest ep79 `52.408%` / iter `58.076%` | Keep to ep99. |

Action: no kill, no resume, no launch. Next checks: AB ep219, AH ep139, AF ep179, CE ep99, Z ep99, S ep259.

## 2026-05-12 04:35 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | ep219 `74.726%` / iter `76.526%`, loss `0.255938` | New best; still improving. Keep to ep239/259+. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep126; latest ep119 `72.216%` / iter `74.424%` | Main challenger to AB at matched checkpoints; keep to ep139/159. |
| AF K80 low-aug cutmix-only | running ep170; latest ep159 `72.946%` / iter `74.926%` | Keep to ep179. |
| S K10 full-aug mixed mix/cut | running ep252; latest ep239 `66.574%` / iter `70.134%` | Keep to ep259. |
| CE sanity ref-aligned `6450` | running ep92; latest ep79 `54.122%` | Keep to ep99 before deciding next sanity ablation. |
| Z K80 full-aug mixed mix/cut manual resume | running ep86; latest ep79 `52.408%` / iter `58.076%` | Keep to ep99. |

Action: no kill, no resume, no launch. Next checks: AH ep139, AF ep179, S ep259, CE ep99, Z ep99, AB ep239.

## 2026-05-12 05:06 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | running ep227-228; latest ep219 `74.726%` / iter `76.526%` | Keep. |
| AF K80 low-aug cutmix-only | running ep175-176; latest ep159 `72.946%` / iter `74.926%` | Keep; ep179 soon. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep132; latest ep119 `72.216%` / iter `74.424%` | Keep; ep139 next. |
| S K10 full-aug mixed mix/cut | running ep257; latest ep239 `66.574%` / iter `70.134%` | Keep; ep259 soon. |
| CE sanity ref-aligned `6450` | running ep97; latest ep79 `54.122%` | Keep to ep99; then decide sanity ablation. |
| Z K80 full-aug mixed mix/cut manual resume | running ep90; latest ep79 `52.408%` / iter `58.076%` | Keep; ep99 later. |

Action: no kill, no resume, no launch. Next checks: S ep259, AF ep179, AH ep139, CE ep99, AB ep239, Z progress.

## 2026-05-12 05:38 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| S K10 full-aug mixed mix/cut | ep259 `67.742%` / iter `70.980%`, loss `0.292185` | Slow full-aug catch-up continues; keep. |
| AB K80 low-aug mixed mix/cut | running ep235; latest ep219 `74.726%` / iter `76.526%` | Best mature pure-diffusion branch; keep to ep239/259. |
| AF K80 low-aug cutmix-only | ep179 `73.300%` / iter `74.798%`, loss `0.259000` | Below AB at mature checkpoints and flattening; keep to ep199 unless slot pressure. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep139; latest ep119 `72.216%` / iter `74.424%` | Ep139 imminent; keep because it was best matched ep119 challenger. |
| Z K80 full-aug mixed mix/cut manual resume | running ep93; latest ep79 `52.408%` / iter `58.076%` | Keep to ep99 for full-aug K80 control. |
| CE sanity v2/no-LayerScale `6450` | ep99 `56.084%`, loss `2.065007`; killed | Not reproduced; old v3 sanity ep99 was `62.754%`. Retired after ep99. |
| CF sanity v3/current-fixes `6457` | launched on `06q7u9`; logdir `20260512_054135_4vf63h...`; initial compilation/epoch0 | Fresh CE sanity with `ViT_base_v3`, exact GELU, b2=0.999, no warm-start. Watch eval0/ep19. |

Action: replaced failed CE v2 sanity with CF v3/current-fixes sanity. No intervention on S/AB/AF/AH/Z. Next checks: CF startup/eval0, AH ep139, AB ep239, AF ep199, S ep279, Z ep99.

## 2026-05-12 06:14 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | ep239 `75.504%` / iter `76.992%`, loss `0.255311` | New best; keep to ep259+. |
| AH K80 low-aug cutmix-heavy p=0.75 | ep139 `73.224%` / iter `75.284%`, loss `0.247610` | Best matched ep139 among AB/AF/AH; keep to ep159/179. |
| AF K80 low-aug cutmix-only | running ep189; latest ep179 `73.300%` / iter `74.798%` | Likely dominated but keep to ep199. |
| S K10 full-aug mixed mix/cut | running ep268; latest ep259 `67.742%` / iter `70.980%` | Keep for delayed full-aug curve. |
| Z K80 full-aug mixed mix/cut manual resume | running ep96; latest ep79 `52.408%` / iter `58.076%` | Keep to ep99. |
| CF sanity v3/current-fixes `6457` | eval0 `0.096%`, train ep4.6 alive, no fatal errors | Startup healthy; watch ep19 against old v3 sanity. |

Action: no kill, no resume, no launch. Next checks: Z ep99, AB ep259, AH ep159, AF ep199, S ep279, CF ep19.

## 2026-05-12 06:44 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| Z K80 full-aug mixed mix/cut manual resume | ep99 `57.724%` / iter `62.532%`, loss `0.335394` | Catching up but far below low-aug K80; keep to ep119/139 as full-aug control. |
| AB K80 low-aug mixed mix/cut | running ep247; latest ep239 `75.504%` / iter `76.992%` | Best overall; keep to ep259+. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep151; latest ep139 `73.224%` / iter `75.284%` | Keep to ep159/179. |
| AF K80 low-aug cutmix-only | running ep196; latest ep179 `73.300%` / iter `74.798%` | Likely dominated; wait ep199 before replacing. |
| S K10 full-aug mixed mix/cut | running ep273; latest ep259 `67.742%` / iter `70.980%` | Keep for delayed full-aug curve. |
| CF sanity v3/current-fixes `6457` | running ep9.6; eval0 `0.096%`, no active fatal errors | Watch ep19. |
| Stale sanity auto-rerun `6458` | failed after unintended auto-rerun of old no-LayerScale branch | Ran `ignore-error`; do not resume. |

Action: ignored stale `6458`. No intervention on effective S/AB/AF/AH/Z/CF. Next checks: AF ep199, AH ep159, AB ep259, S ep279, Z ep119, CF ep19.

## 2026-05-12 07:16 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| S K10 full-aug mixed mix/cut | ep279 `68.450%` / iter `71.582%`, loss `0.293842` | Still improving slowly; keep for full-aug delayed-convergence curve. |
| AB K80 low-aug mixed mix/cut | running ep256; latest ep239 `75.504%` / iter `76.992%` | Best mature pure-diffusion branch; keep to ep259+. |
| AF K80 low-aug cutmix-only | ep199 `73.162%` / iter `74.960%`, loss `0.268570` | Dominated by AB and not improving; killed/retired after ep199. |
| AH K80 low-aug cutmix-heavy p=0.75 | ep159 `73.402%` / iter `75.440%`, loss `0.249140` | Early advantage faded by ep159; keep to ep179 but no longer clearly better than AB. |
| Z K80 full-aug mixed mix/cut manual resume | running ep104; latest ep99 `57.724%` / iter `62.532%` | Keep to ep119/139 as full-aug K80 control. |
| CF sanity v3/current-fixes `6457` | running ep17; eval0 `0.096%`, no active fatal errors | Ep19 imminent; compare to old v3 sanity ep19 `38.770%`. |
| AI K80 low-aug cutmix/mixup p=0.625 `6460` | launched; logdir `20260512_073102_nkjdxz...`; initial compilation/epoch0 | Tests interpolation between AB p=0.5 and AH p=0.75; no warm-start/backbone load/aux CE. |
| Forbidden aux-CE auto-resume `6459` | no active `6459` window in focused check, but parent `6381` still references it | Keep verifying absence; do not resume Run H or any aux-CE branch. |

Action: killed dominated AF (`6441`) and launched AI (`6460`) on the freed `z169mq` slot. Next checks: AI startup/eval0, CF ep19, AB ep259, AH ep179, S ep299, Z ep119, and forbidden `6459` absence.

## 2026-05-12 08:03 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | ep259 `75.758%` / iter `77.414%`, loss `0.256073` | New best pure-diffusion result; keep to ep279+. |
| CF sanity v3/current-fixes `6457` | ep19 `37.472%`, loss `3.081337` | Close to old v3 sanity ep19 `38.770%`, much better than v2/no-LayerScale ep19; keep to ep39/59. |
| AI K80 low-aug cutmix/mixup p=0.625 `6460` | eval0 `0.094%` / iter `0.132%`, train ep5.83 healthy | Startup good; wait ep19/39 for comparison to AB/AH/AF. |
| S K10 full-aug mixed mix/cut | running ep285; latest ep279 `68.450%` / iter `71.582%` | Keep for full-aug delayed-convergence curve. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep166; latest ep159 `73.402%` / iter `75.440%` | Keep to ep179; early advantage over AB faded by ep159. |
| Z K80 full-aug mixed mix/cut manual resume | running ep108; latest ep99 `57.724%` / iter `62.532%` | Keep to ep119/139 as K80 full-aug control. |
| Forbidden aux-CE auto-resume `6459` | no active `6459` window in focused check; parent `6381` still references it | Continue verifying absence; do not resume aux-CE. |

Action: no kill/resume/launch. Continue S/AB/AH/Z/CF/AI. Next checks: AI progress/eval19, CF ep39, AH ep179, AB ep279, S ep299, Z ep119, and forbidden `6459` absence.

## 2026-05-12 08:35 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AB K80 low-aug mixed mix/cut | running ep268; latest ep259 `75.758%` / iter `77.414%` | Keep; best pure-diffusion branch. |
| AI K80 low-aug cutmix/mixup p=0.625 `6460` | running ep12; eval0 `0.094%` / iter `0.132%`, train healthy | Keep; ep19 expected soon. |
| CF sanity v3/current-fixes `6457` | running ep27; latest ep19 `37.472%` | Keep to ep39/59; this is the effective sanity branch. |
| AH K80 low-aug cutmix-heavy p=0.75 | running ep173; latest ep159 `73.402%` / iter `75.440%` | Keep to ep179 for mature p=0.75 curve. |
| S K10 full-aug mixed mix/cut | running ep291; latest ep279 `68.450%` / iter `71.582%` | Keep; full-aug delayed-convergence control. |
| Z K80 full-aug mixed mix/cut manual resume | running ep111; latest ep99 `57.724%` / iter `62.532%` | Keep to ep119/139 as K80 full-aug control. |
| Stale no-LayerScale sanity auto-rerun `6463` | failed auto-resume from `6458`; `ignore-error` acknowledged | Do not resume; effective sanity is `6457`. |
| Forbidden aux-CE auto-resume `6459` | no active `6459` window in focused check; parent `6381` still references it | Continue verifying absence; do not resume aux-CE. |

Action: ignored stale `6463`. No kill/resume/launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AH ep179, AI ep19 progress, AB ep279, S ep299, Z ep119, CF ep39, and forbidden `6459` absence.

## 2026-05-12 09:07 Loop Results

| Run | Latest useful eval/status | Decision |
|---|---:|---|
| AH K80 low-aug cutmix-heavy p=0.75 | ep179 `74.010%` / iter `75.796%`, loss `0.251355` | Better than matched AB/AF ep179; keep to ep199/219. |
| AI K80 low-aug cutmix/mixup p=0.625 `6460` | ep19 `33.830%` / iter `41.432%`, loss `0.466993` | Better than AB/AH ep19, below AF ep19; keep to ep39/59. |
| AB K80 low-aug mixed mix/cut | running ep275; latest ep259 `75.758%` / iter `77.414%` | Keep; best mature pure-diffusion branch. |
| S K10 full-aug mixed mix/cut | running ep296; latest ep279 `68.450%` / iter `71.582%` | Keep; full-aug delayed-convergence control. |
| Z K80 full-aug mixed mix/cut manual resume | running ep114; latest ep99 `57.724%` / iter `62.532%` | Keep to ep119/139 as K80 full-aug control. |
| CF sanity v3/current-fixes `6457` | running ep34; latest ep19 `37.472%` | Keep to ep39/59; manager Unknown was TPU log permission noise, train log is live. |
| Stale no-LayerScale sanity auto-rerun `6463` | still listed as failed/error from `6458`; `ignore-error` rerun | Do not resume; effective sanity is `6457`. |
| Forbidden aux-CE auto-resume `6459` | no active `6459` window in focused check; parent `6381` still references it | Continue verifying absence; do not resume aux-CE. |

Action: no kill/resume/launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AB ep279, S ep299, Z ep119, CF ep39, AI ep39 progress, AH ep199, and forbidden `6459` absence.

## 2026-05-12 10:01 UTC Loop Results
- `6435` AB K80 low-aug mixed mix/cut: ep279 `76.190%` / `77.848%` iter, loss `0.255026`, invalid `0.228%`. New best pure-diffusion result; continue.
- `6433` S K10 full-aug: ep299 `68.774%` / `72.158%` iter, loss `0.297835`, invalid `0.442%`. Still slowly improving; keep.
- `6453` Z K80 full-aug: ep119 `60.146%` / `64.446%` iter, loss `0.323961`, invalid `0.622%`. Improving but far behind matched low-aug K80; keep to ep139+ for delayed-augmentation test.
- `6457` CF sanity v3/current-fixes: ep39 `49.134%`, loss `2.422293`. Healthy but not yet a clean reproduction; continue to ep59/79.
- `6448` AH latest remains ep179 `74.010%` / `75.796%` iter; running ep189.
- `6460` AI latest remains ep19 `33.830%` / `41.432%` iter; running ep29.
- Effective jobs healthy. Stale `6463` no-LayerScale auto-resume remains ignored; no active forbidden `6459` observed.

## 2026-05-12 10:33 UTC Loop Results
- No new evals beyond the previous loop. Effective latest results remain: AB ep279 `76.190/77.848`, S ep299 `68.774/72.158`, Z ep119 `60.146/64.446`, CF ep39 `49.134`, AH ep179 `74.010/75.796`, AI ep19 `33.830/41.432`.
- AI `6460` stopped at ep31.411 due spot/SSH failure. Auto-resume `6467` failed with duplicate `config.load_from` flag.
- Manual AI recovery launched as `6469` on `oulo2v`, correct DeiT dir id `7`, from AI's own original logdir/checkpoint. Verified restore from `checkpoint_25020` and resumed at epoch 20. No warm-start/backbone load/aux CE.
- Effective AI is now `6469`; stale `6467` and stale no-LayerScale `6463` are ignored.

## 2026-05-12 11:19 UTC Loop Results
- `6435` AB K80 low-aug mixed mix/cut: ep299 `76.266%` / `77.988%` iter, loss `0.252067`, invalid `0.260%`. New best pure-diffusion result; still improving slightly.
- `6448` AH K80 low-aug p=0.75: ep199 `74.454%` / `76.218%` iter, loss `0.253656`, invalid `0.264%`. Better than AB at matched ep199, but below mature AB ep299; keep.
- `6469` AI manual resume: verified restored from original AI `checkpoint_25020` and training normally through ep26. No duplicate load_from fatal. Effective AI is now `6469`.
- S latest remains ep299 `68.774/72.158`; Z latest remains ep119 `60.146/64.446`; CF latest remains ep39 `49.134`. All active logs healthy enough to continue.

## 2026-05-12 12:02 UTC Loop Results
- `6433` S K10 full-aug: ep319 `69.324%` / `72.296%` iter, loss `0.296627`, invalid `0.430%`. Still improving single-acc; iterative curve nearly flat. Keep as K10 full-aug control.
- `6457` CF sanity v3/current-fixes: ep59 `53.316%`, loss `2.231922`. Healthy and above failed v2/no-LayerScale, but still below old v3 sanity ep59 `59.132%`; continue to ep79/99.
- `6435` AB latest remains ep299 `76.266%` / `77.988%` iter; running ep308. Best mature pure-diffusion config; keep.
- `6448` AH latest remains ep199 `74.454%` / `76.218%` iter; running ep213. Keep to ep219.
- `6453` Z latest remains ep119 `60.146%` / `64.446%` iter; running ep132. Keep to ep139/159 as K80 full-aug control.
- `6469` AI manual resume running ep34.9, no post-resume eval yet; next meaningful eval is ep39. Log healthy, no duplicate load_from fatal.
- Stale `6467` and `6463` ignored again. No active forbidden aux-CE `6459` observed.

Action: no kill/resume/launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AI ep39, AH ep219, Z ep139, AB ep319, CF ep79, S ep339, and forbidden `6459` absence.

## 2026-05-12 12:34 UTC Loop Results
- `6469` AI K80 low-aug p=0.625: ep39 `57.646%` / `62.260%` iter, loss `0.334645`, invalid `0.832%`. Slightly above AB ep39 `57.148/61.642` and AH ep39 `57.566/62.152`; keep to ep59/79.
- `6448` AH K80 low-aug p=0.75: ep219 `74.944%` / `76.552%` iter, loss `0.253152`, invalid `0.290%`. Beats matched AB ep219 `74.726/76.526` by a small margin, but remains below mature AB ep299 `76.266/77.988`; keep to ep239 if slots permit.
- `6433` S latest remains ep319 `69.324%` / `72.296%` iter; running ep328.9. Keep as K10 full-aug control.
- `6435` AB latest remains ep299 `76.266%` / `77.988%` iter; running ep315.1. Best mature pure-diffusion config; next eval ep319.
- `6453` Z latest remains ep119 `60.146%` / `64.446%` iter; running ep135.6. Keep to ep139/159 as K80 full-aug control.
- `6457` CF latest remains ep59 `53.316%`; running ep65.5. Continue to ep79/99 before changing sanity code again.
- Stale `6472` (old Z resume1), `6467`, and `6463` ignored. No active forbidden aux-CE `6459` observed.

Action: no kill/resume/launch for effective jobs. Continue S/AB/AH/Z/CF/AI. Next checks: AB ep319, Z ep139, AI ep59, CF ep79, AH ep239, S ep339, and forbidden `6459` absence.

## 2026-05-12 13:10 UTC Loop Results
- `6435` AB K80 low-aug mixed mix/cut: ep319 `76.600%` / `78.294%` iter, loss `0.248487`, invalid `0.274%`. New best pure-diffusion result; keep to finish.
- `6453` Z K80 full-aug mixed mix/cut: ep139 `62.192%` / `66.086%` iter, loss `0.315654`, invalid `0.492%`. Still catching up but far below matched low-aug K80; keep as full-aug control.
- `6433` S K10 full-aug finished. Final useful eval ep319 `69.324%` / `72.296%` iter. Freed `favaxa` slot.
- `6474` AJ launched on `favaxa`: K80 AB-style mix/cut `soft_top2` plus RandAugment only (`reprob=0`, `repeated_aug=1`, WD `0.02`, SD `0.05`). No warm-start/backbone load/aux CE. Startup reached ep0.56 after compilation; log healthy.
- `6448` AH latest remains ep219 `74.944%` / `76.552%` iter; running ep227. Keep to ep239.
- `6469` AI latest remains ep39 `57.646%` / `62.260%` iter; running ep48. Keep to ep59/79.
- `6457` CF latest remains ep59 `53.316%`; running ep71. Keep to ep79/99.
- Stale `6472`, `6467`, and `6463` remain ignored. No active forbidden aux-CE `6459` observed.

Action: launched AJ medium-augmentation diagnostic; no other kill/resume/launch. Continue AB/AH/Z/CF/AI/AJ. Next checks: AJ eval0/startup, AB finish, Z ep159, AI ep59, CF ep79, AH ep239, and forbidden `6459` absence.

## 2026-05-12 14:03 UTC Report/Loop Results
- Wrote 24h progress report: `agents/report_24h_progress_2026-05-12.html`.
- `6435` AB finished; final useful eval ep319 `76.600%` / `78.294%` iter, loss `0.248487`, invalid `0.274%`. This remains the best deployable pure-diffusion config.
- `6457` CF sanity v3/current-fixes reached ep79 `55.926%`, loss `2.090736`. Healthy but still below old v3 sanity trajectory; continue to ep99.
- `6475` AK K80 RandAug+reprob medium-aug startup reached eval0 `0.098%` / `0.116%` iter, loss `0.691946`, invalid `0.964%`; no startup fatal/OOM.
- `6474` AJ latest eval0 remains `0.104%` / `0.114%` iter; running ep7.
- `6469` AI running ep59; latest eval still ep39 `57.646%` / `62.260%` iter; ep59 not yet complete.
- `6448` AH running ep237; latest eval ep219 `74.944%` / `76.552%` iter.
- `6453` Z running ep144; latest eval ep139 `62.192%` / `66.086%` iter.
- No active forbidden aux-CE `6459` observed. No new launch/resume/kill required.

## 2026-05-12 14:40 UTC Loop Results
- `6448` AH K80 low-aug cutmix-heavy p=0.75: ep239 `75.364%` / `77.016%` iter, loss `0.254000`, invalid `0.234%`. Essentially tied with AB at matched ep239 (`75.504/76.992`) but below mature AB ep319 `76.600/78.294`; keep as matched-epoch challenger.
- `6469` AI K80 low-aug p=0.625: ep59 `63.920%` / `67.830%` iter, loss `0.299868`, invalid `0.606%`. Below AB/AH/AF at matched ep59; keep to ep79 only because it is healthy and already running.
- `6457` CF latest remains ep79 `55.926%`; running ep85, ep99 next.
- `6453` Z latest remains ep139 `62.192%` / `66.086%` iter; running ep148, ep159 next.
- `6474` AJ latest remains eval0 `0.104%` / `0.114%` iter; running ep13, ep19 next.
- `6475` AK latest remains eval0 `0.098%` / `0.116%` iter; running ep6.
- Updated `agents/report_24h_progress_2026-05-12.html` with AH ep239 and AI ep59.
- No active forbidden aux-CE `6459` observed. No kill/resume/launch required.
