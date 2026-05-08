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
  | epoch | eval_accuracy |
  |-------|--------------|
  | 0     | 2.5%         |
  | 19    | 42.9%        |
  | 39    | 56.7%        |
  | 59    | 60.6%        |
  | 79    | 62.4%        |
  | 99    | 64.4%        |
- **Status**: Running (ep=101.7 as of 2026-05-08 05:10)
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
- **Status**: Running (ep=21.6 as of 2026-05-08 05:10)
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
- **Status**: Running (ep=20.2 as of 2026-05-08 05:10)
- **LogDir**: `/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260508_024938_sxvz3e_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval`

---

## Ablation Matrix
| Run | qkv_bias | ln_bias | LearnedScale | ep19 eval | ep99 eval | ep330 eval |
|-----|----------|---------|--------------|-----------|-----------|-----------|
| 1 (ViT_base)   | False | False | True  | 42.9% | 64.4% | TBD |
| 2 (ViT_base_v2)| True  | True  | False | 42.1% | TBD   | TBD |
| 3 (ViT_base_v3)| True  | True  | True  | 43.9% | TBD   | TBD |

**ep=19 ranking**: v3 (43.9%) > v1 (42.9%) > v2 (42.1%)
- LayerScale appears helpful even at ep=19 (v3 > v2 by 1.8%)
- Biases alone (v2 vs v1): slightly worse without LS; slightly better with LS

## TODO / Next Steps
- Run 1: next eval at ep=119 (~1.8h from 05:10)
- Runs 2 & 3: next eval at ep=39 (~1.8–1.9h from 05:10)
- ep=39 comparison will clarify whether the v3 advantage persists
- After ep=39: if v2 (exact DeiT-B) is competitive, it validates our reproduction
- Run 1 at ep=99=64.4%; reference DeiT-B trajectory ~67% at ep=100 → Run 1 is slightly below (may be because of missing biases)
- Compare trajectories; if v2/v3 significantly better → original architecture was wrong
- After baseline confirmed → Phase 2: masked diffusion head
