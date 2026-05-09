# DeiT Research TODO

## Priority 1: Reproduce 81.8% Baseline (BLOCKING)

Our best Phase 1 result is only 73.14% (vs target 81.8%). Need to align with the working reference implementation at `../papers/DeiT/`.

### Identified Gaps (from code diff, 2026-05-09)

- [ ] **Weight decay mask** (HIGH): Reference applies `get_no_weight_decay_dict` — no WD on `cls`, `pos_emb`, `bias`, `scale` params. Our current code has NO masking (WD applies everywhere). This is a standard ViT training trick and likely a significant contributor to the gap.
- [ ] **b2=0.999** (HIGH): Reference AdamW uses `b2=0.999` (standard). Ours defaults to `b2=0.95`. Need to set `adamw_b2=0.999` in the new baseline config.
- [ ] **LR schedule** (MEDIUM): Reference uses custom schedule: const `1e-6` for epoch 0, linear warmup epochs 1-5, then cosine to 1e-5 over epochs 5-320, then const 1e-5 for last 10. Ours uses `warmup_cosine_decay_schedule` starting from 0. Should align.
- [ ] **Patch embedding init** (MEDIUM): Reference uses torch-style `Uniform(-1/sqrt(in), 1/sqrt(in))` via `EmbedLinear`. Ours uses `truncated_normal(0.02)`. The patch embedding is a key learned component.
- [ ] **CLS token init** (LOW): Reference `normal(1e-6)`, ours `truncated_normal(0.02)`.
- [ ] **randperm before mixup** (LOW): Reference shuffles the batch before applying Mixup/CutMix.
- [ ] **WandB notes**: Fill in meaningful notes for all run configs before launch.

### Action Plan

1. Create a new `ViT_base_v4` architecture in `models.py` that matches the reference exactly:
   - `EmbedLinear`-style patch embedding init
   - `cls` init: `normal(1e-6)`
   - All other params: same as `ViT_base_v3` (biases=True, LearnedScale=True)
2. Add `get_no_weight_decay_dict` to `train.py` and wire into AdamW.
3. Fix LR schedule to match reference.
4. Set `b2=0.999` in the sanity run config.
5. Add randperm in `input_pipeline.py`.
6. Create `configs/remote_run_v4_config.yml` and launch sanity run.
7. Monitor ep=19/39 — expect >55% if aligned (reference hit 81.8%).

---

## Priority 2: Phase 2 Masked Diffusion Head — Remaining Experiments

Run I is launched. Runs E/F/G/H still need to be started. Machines reportedly IDLE+MOUNTED.

| Run | Config | Machine | Status |
|-----|--------|---------|--------|
| E | zero-init out_proj, uniform | axuxm0 | IDLE+MOUNTED — needs `tpu run` |
| F | MLP head baseline | 3djlis | IDLE+MOUNTED — needs `ftmd`+`tpu run` |
| G | large head (512-dim, 4L) | 06q7u9 | IDLE+MOUNTED — needs `ftmd`+`tpu run` |
| H | attention + aux CE (λ=0.1) | qxxa8y | IDLE+MOUNTED — needs `ftmd`+`tpu run` |
| I | warm-start backbone | j3rqvs | ✅ Running (window 6375) |

- [ ] Verify machine states, then launch E/F/G/H.
- [ ] Record ep=0 evals for all running Phase 2 jobs.

---

## Priority 3: Monitor & Record

- [ ] Monitor Run I for ep=0 eval (should show faster convergence than Run C due to warm-start).
- [ ] Monitor Run A/C/D auto-resume from preemption (us-east5-b wave).
- [ ] Update `agents/results.md` with all new evals as they come in.

---

## Gotchas Learned (update notes.md with these)

- `aux_ce_loss_weight` defaults to 0.1 even when `head_aux_ce=False`. Always gate it: `aux_ce_loss_weight = config.get('aux_ce_loss_weight', 0.0) if config.get('head_aux_ce', False) else 0.0`.
- `load_backbone_params` with `restore_checkpoint(target=None)` returns plain numpy dicts, not FrozenDicts. Must use `jnp.array()` conversion + reinit opt_state from new params.
- `tpu` is a shell alias, not a binary. In scripts, use the full Python path from `~/.bash_aliases`.
- WandB notes must be set — do not leave empty.
- The reference DeiT uses weight decay masking (no WD on cls/pos_emb/bias/scale). Missing this likely costs multiple accuracy points.
