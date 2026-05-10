# DeiT Research TODO

_Last updated: 2026-05-09 22:20_

---

## Priority 1: Sanity Run — Reproduce 81.8%

**Status**: ✅ RUNNING on favaxa (window 6380). Logdir: `20260509_212142_mxfpdz_...favaxa...`
- ep=0 eval: 0.096% CE accuracy (random baseline — expected)
- ep=7.99: train_loss=6.156, train_acc=5.99% — normal early CE training

Alignment changes applied:
- [x] Weight decay mask (no WD on cls/pos_emb/bias/scale)
- [x] b2=0.999 (standard AdamW)
- [x] LR schedule: tang(1e-6, ep0) → linear warmup → cosine → const(1e-5)
- [x] CLS token init: normal(1e-6) instead of truncated_normal(0.02)
- [x] Stochastic depth: linear per-layer schedule (not uniform)
- [x] Final LayerNorm: use_bias=True (follows use_ln_bias flag)
- [ ] Patch embedding init (EmbedLinear style): NOT YET done
- [ ] randperm before mixup: NOT YET done

**Next**: Watch for ep=19 eval. Target >60%.

---

## Priority 2: Phase 2 Experiments — All Running

| Run | Config | Machine | Window | Status | Notes |
|-----|--------|---------|--------|--------|-------|
| E | zero-init out_proj, attention | axuxm0 | 6382 | ✅ RUNNING | ep~3, ep=0 eval=0.1% |
| F | MLP head baseline | 3djlis | 6375 | ✅ RUNNING | ep~20, ep=19 eval=5.58%/9.67%iter |
| G | large head (512-dim, 4L) | 06q7u9 | 6383 | ✅ RUNNING | ep~3, ep=0 eval=0.108% |
| H | attention + aux CE (λ=0.1) | qxxa8y | 6381 | ✅ RUNNING | ep~5, ep=0 eval=0.1% |
| I | warm-start backbone | j3rqvs | 6388 | ✅ RUNNING | ep~2; ep=0 eval=0.184%/0.224%iter; ⚠️ invalid_rate=49.816% |

**All 5 Phase 2 slots running correctly (E/F/G/H/I).**

---

## Priority 3: Monitor & Record

- [x] Run I ep=0 eval confirmed: 0.184%/0.224%iter, invalid_rate=49.816% (monitor if normalizes)
- [ ] Record ep=19 evals for E/G/H when they arrive
- [ ] Watch sanity run for ep=19 eval (target >60%)
- [ ] Update `agents/results.md` as evals arrive
- [ ] If sanity run hits ~81.8% at ep=330: update baseline config in main branch

---

## Known Issues & Gotchas

- **Previous session launched E/G/I with wrong MLP config** — fixed and relaunched.
- **Run H fc bug**: When `head_aux_ce=True`, fc must be called unconditionally in `__call__` so params are initialized. Fixed in commit 537e610. 
- **load_backbone_params**: Phase 1 ckpt has `final_ln: ['scale']` only (no bias). Use recursive copy to preserve Phase 2 structure. state.params is plain dict — never freeze(). Fixed in commits 2f57c6c, 05264ce.
- **MONITOR.py may claim idle TPUs for PaliGemma jobs** before DeiT can launch — happened with qxxa8y and favaxa. Monitor and relaunch if preempted.
- **wandb_notes must contain 'deit'** for MONITOR.py to route to v6e-8 TPUs.
- **Always use `configs/remote_run_config.yml`** for launches (copy from experiment config).
- **`staging.sh` blocks until training completes** — never queue multiple commands in same window.
