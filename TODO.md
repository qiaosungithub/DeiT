# DeiT Research TODO

_Last updated: 2026-05-09 20:45_

---

## Priority 1: Sanity Run — Reproduce 81.8%

**Status**: ✅ RUNNING on favaxa (window 6372). Logdir: `20260509_204009_1oz5ex_...favaxa...`

Alignment changes applied in commit `173d1bb`:
- [x] Weight decay mask (no WD on cls/pos_emb/bias/scale)
- [x] b2=0.999 (standard AdamW)
- [x] LR schedule: tang(1e-6, ep0) → linear warmup → cosine → const(1e-5)
- [x] CLS token init: normal(1e-6) instead of truncated_normal(0.02)
- [x] Stochastic depth: linear per-layer schedule (not uniform)
- [x] Final LayerNorm: use_bias=True (follows use_ln_bias flag)
- [ ] Patch embedding init (EmbedLinear style): NOT YET done
- [ ] randperm before mixup: NOT YET done

**Next**: Wait for ep=19 eval. Expect >60% if alignment is correct.

---

## Priority 2: Phase 2 Experiments — All Launched

| Run | Config | Machine | Window | Status | Notes |
|-----|--------|---------|--------|--------|-------|
| E | zero-init out_proj, uniform | axuxm0 | 6377 | ✅ RUNNING | ep~5.6, fresh start |
| F | MLP head baseline | 3djlis | 6375 | ✅ RUNNING | ep~5, step 6300 |
| F-dup | MLP head (accident) | p1u4mx | (stream done) | ✅ RUNNING (remote) | Duplicate of F, will complete naturally |
| G | large head (512-dim, 4L) | 06q7u9 | 6376 | ✅ RUNNING | ep~4.9, step 6100 |
| H | attention + aux CE (λ=0.1) | qxxa8y | 6372 | ❌ FAILED | "not found in sheet" — user must register qxxa8y in spreadsheet |
| I | warm-start backbone | j3rqvs | 6378 | ✅ RUNNING | ep~0, step 300 |

**Blocker**: qxxa8y is not registered in the spreadsheet. User action needed.

---

## Priority 3: Resume Preempted Runs A/C/D

Runs A/C/D were on us-east5-b spot TPUs that were deleted. These are abandoned — new runs (E/F/G/H/I) replaced them.

---

## Priority 4: Monitor & Record

- [ ] Record ep=0 evals for E/F/G/I when they arrive
- [ ] Watch sanity run ep=19 — target >60%
- [ ] Update `agents/results.md` with all evals
- [ ] If sanity run hits ~81.8%: update baseline config in main branch

---

## Known Issues & Gotchas

- `aux_ce_loss_weight` must be gated on `head_aux_ce`. Already fixed in commit `faa44ba`.
- `load_backbone_params`: must use `jnp.array()` to convert checkpoint values. Fixed in `e4fd44b`.
- `tpu.py resume` appends `--config.load_from=<logdir>` to extra_configs — but if the original config already parses `load_from` as a field, this duplicates the flag. Workaround: launch fresh without resume.
- **wandb_notes must contain 'deit'** for MONITOR.py auto-resume to classify job correctly.
- **Always use `configs/remote_run_config.yml`** for launches so `tpu.py check` shows correct tags.
- **`staging.sh` blocks until remote training completes** — queued shell commands run only after ~330 epochs.
- **qxxa8y not in spreadsheet**: requires user to add it before Run H can launch successfully.
- favaxa (v6e-8-tmp211) registered in data.json only; not in spreadsheet — if Run H needs to auto-resume it might also fail. Monitor.
