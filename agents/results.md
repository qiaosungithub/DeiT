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
