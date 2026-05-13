import argparse
import functools
import json
import os
import time
from pathlib import Path

from flax import jax_utils
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax

import input_pipeline
import models
import train
from configs.load_config import get_config
from input_pipeline import prepare_batch_data_sqa
from utils import ckpt_util


NUM_CLASSES = 1000


def _copy_config(config):
    return ConfigDict(config.to_dict())


def _make_state(config, load_from):
    image_size = 224
    rng = jax.random.PRNGKey(config.seed)
    base_lr = config.learning_rate * config.batch_size / 512.0
    model_cls = getattr(models, config.model)
    model = train.create_model(
        model_cls=model_cls,
        half_precision=config.half_precision,
        dropout_rate=config.dropout_rate,
        stochastic_depth_rate=config.stochastic_depth_rate,
        head_inner_dim=config.get("head_inner_dim", 256),
        head_bit_dim=config.get("head_bit_dim", 8),
        head_mlp_hidden_dim=config.get("head_mlp_hidden_dim", 3072),
        head_mlp_activation=config.get("head_mlp_activation", "gelu"),
        head_mlp_layer_norm=config.get("head_mlp_layer_norm", False),
        gelu_approximate=config.get("gelu_approximate", False),
        head_n_layers=config.get("head_n_layers", 2),
        head_n_heads=config.get("head_n_heads", 4),
        head_type=config.get("head_type", "attention"),
        head_zero_init_proj=config.get("head_zero_init_proj", False),
        head_aux_ce=config.get("head_aux_ce", False),
    )
    lr_fn = train.create_learning_rate_fn(config, base_lr, steps_per_epoch=1)
    state = train.create_train_state(rng, config, model, image_size, lr_fn)
    state = ckpt_util.restore_checkpoint(state, load_from, load_from)
    return model, jax_utils.replicate(state)


def _chunks_for_method(method, n_bits):
    lsb = list(range(n_bits))
    msb = list(range(n_bits - 1, -1, -1))

    def split(order, n_chunks):
        out = []
        for i in range(n_chunks):
            lo = (len(order) * i) // n_chunks
            hi = (len(order) * (i + 1)) // n_chunks
            out.append(tuple(order[lo:hi]))
        return tuple(x for x in out if x)

    if method == "all":
        return (tuple(lsb),)
    if method == "lsb10":
        return tuple((i,) for i in lsb)
    if method == "msb10":
        return tuple((i,) for i in msb)
    if method == "lsb4":
        return split(lsb, 4)
    if method == "msb4":
        return split(msb, 4)
    if method == "lsb2":
        return split(lsb, 2)
    if method == "msb2":
        return split(msb, 2)
    if method.startswith("rand"):
        seed = int(method.replace("rand", ""))
        order = list(jax.random.permutation(jax.random.PRNGKey(seed), jnp.arange(n_bits)))
        return tuple((int(i),) for i in order)
    raise ValueError(f"Unknown scoring method: {method}")


def _parse_ensembles(ensembles):
    parsed = []
    for spec in ensembles:
        if not spec:
            continue
        parts = tuple(x.strip() for x in spec.split("+") if x.strip())
        if len(parts) < 2:
            raise ValueError(f"Ensemble must contain at least two methods: {spec}")
        parsed.append((spec.replace("+", "_plus_"), parts))
    return tuple(parsed)


def _make_eval_step(model, methods, ensembles):
    n_bits = models.NUM_BITS
    class_bits = models.class_to_bits(jnp.arange(NUM_CLASSES, dtype=jnp.int32), n_bits)
    method_chunks = {m: _chunks_for_method(m, n_bits) for m in methods}

    def score_chunks(variables, cls, chunks):
        bsz = cls.shape[0]
        context = jnp.full((bsz, NUM_CLASSES, n_bits), 2, dtype=jnp.int32)
        scores = jnp.zeros((bsz, NUM_CLASSES), dtype=jnp.float32)
        flat_cls = jnp.repeat(cls, NUM_CLASSES, axis=0)

        for chunk in chunks:
            idx = jnp.array(chunk, dtype=jnp.int32)
            logits = model.apply(
                variables,
                flat_cls,
                masked_bits=context.reshape((bsz * NUM_CLASSES, n_bits)),
                method=model.decode_diffusion_from_cls,
            )
            logits = logits.reshape((bsz, NUM_CLASSES, n_bits, 2))
            log_probs = jax.nn.log_softmax(logits[:, :, idx, :], axis=-1)
            target_bits = class_bits[None, :, idx]
            picked = jnp.take_along_axis(log_probs, target_bits[..., None], axis=-1)[..., 0]
            scores = scores + jnp.sum(picked, axis=-1)
            context = context.at[:, :, idx].set(target_bits)
        return scores

    def step(state, batch):
        variables = {"params": state.params, "batch_stats": state.batch_stats}
        images = batch["image"]
        labels = batch["label"]
        if labels.shape[-1] != NUM_CLASSES:
            labels = jax.nn.one_hot(labels, NUM_CLASSES)
        valid = (labels[..., 0] >= 0).astype(jnp.float32)
        true = jnp.argmax(labels, axis=-1).astype(jnp.int32)
        n_valid = jnp.sum(valid)

        rng = jax.random.PRNGKey(0)
        cls = model.apply(variables, images, rng=rng, train=False, method=model.encode)

        out = {"n_valid": n_valid}

        # Current all-masked greedy baseline, including invalid-code accounting.
        logits_single = model.apply(
            variables,
            cls,
            masked_bits=jnp.full((images.shape[0], n_bits), 2, dtype=jnp.int32),
            method=model.decode_diffusion_from_cls,
        )
        bits_single = jnp.argmax(logits_single, axis=-1)
        pred_raw = models.bits_to_class(bits_single)
        pred = jnp.clip(pred_raw, 0, NUM_CLASSES - 1)
        out["greedy_single_top1"] = jnp.sum((pred == true).astype(jnp.float32) * valid)
        out["greedy_single_invalid"] = jnp.sum((pred_raw >= NUM_CLASSES).astype(jnp.float32) * valid)

        # Current 4-step confidence greedy top-1 baseline.
        k_per_step = max(1, -((-n_bits) // 4))
        masked_bits = jnp.full((images.shape[0], n_bits), 2, dtype=jnp.int32)
        for _ in range(4):
            logits = model.apply(variables, cls, masked_bits=masked_bits, method=model.decode_diffusion_from_cls)
            probs = jax.nn.softmax(logits, axis=-1)
            pred_bits = jnp.argmax(logits, axis=-1)
            conf = jnp.max(probs, axis=-1)
            still_masked = masked_bits == 2
            eff_conf = jnp.where(still_masked, conf, jnp.full_like(conf, -jnp.inf))
            sorted_conf = jnp.sort(eff_conf, axis=-1)[:, ::-1]
            threshold = sorted_conf[:, k_per_step - 1:k_per_step]
            to_unmask = still_masked & (eff_conf >= threshold)
            masked_bits = jnp.where(to_unmask, pred_bits, masked_bits)
        pred_iter_raw = models.bits_to_class(masked_bits)
        pred_iter = jnp.clip(pred_iter_raw, 0, NUM_CLASSES - 1)
        out["greedy_iter4_top1"] = jnp.sum((pred_iter == true).astype(jnp.float32) * valid)
        out["greedy_iter4_invalid"] = jnp.sum((pred_iter_raw >= NUM_CLASSES).astype(jnp.float32) * valid)

        score_cache = {}
        for method in methods:
            scores = score_chunks(variables, cls, method_chunks[method])
            score_cache[method] = scores
            top5 = jax.lax.top_k(scores, 5)[1]
            top1 = top5[:, 0]
            out[f"{method}_top1"] = jnp.sum((top1 == true).astype(jnp.float32) * valid)
            out[f"{method}_top5"] = jnp.sum(jnp.any(top5 == true[:, None], axis=-1).astype(jnp.float32) * valid)
        for name, parts in ensembles:
            stack = jnp.stack([score_cache[p] for p in parts], axis=0)

            # Geometric path average: mean log p(y | path).
            scores = jnp.mean(stack, axis=0)
            top5 = jax.lax.top_k(scores, 5)[1]
            top1 = top5[:, 0]
            out[f"{name}_top1"] = jnp.sum((top1 == true).astype(jnp.float32) * valid)
            out[f"{name}_top5"] = jnp.sum(jnp.any(top5 == true[:, None], axis=-1).astype(jnp.float32) * valid)

            # Arithmetic path mixture: log mean_path p(y | path).
            scores_lme = jax.nn.logsumexp(stack, axis=0) - jnp.log(float(len(parts)))
            top5_lme = jax.lax.top_k(scores_lme, 5)[1]
            top1_lme = top5_lme[:, 0]
            out[f"{name}_logmeanexp_top1"] = jnp.sum((top1_lme == true).astype(jnp.float32) * valid)
            out[f"{name}_logmeanexp_top5"] = jnp.sum(jnp.any(top5_lme == true[:, None], axis=-1).astype(jnp.float32) * valid)

            # Best-path optimistic aggregation, useful as an upper-bound diagnostic.
            scores_max = jnp.max(stack, axis=0)
            top5_max = jax.lax.top_k(scores_max, 5)[1]
            top1_max = top5_max[:, 0]
            out[f"{name}_maxpath_top1"] = jnp.sum((top1_max == true).astype(jnp.float32) * valid)
            out[f"{name}_maxpath_top5"] = jnp.sum(jnp.any(top5_max == true[:, None], axis=-1).astype(jnp.float32) * valid)

            # Calibration-light rank aggregation diagnostics.
            ranks = jnp.argsort(jnp.argsort(-stack, axis=-1), axis=-1).astype(jnp.float32)
            scores_rankmean = -jnp.mean(ranks, axis=0)
            top5_rankmean = jax.lax.top_k(scores_rankmean, 5)[1]
            top1_rankmean = top5_rankmean[:, 0]
            out[f"{name}_rankmean_top1"] = jnp.sum((top1_rankmean == true).astype(jnp.float32) * valid)
            out[f"{name}_rankmean_top5"] = jnp.sum(jnp.any(top5_rankmean == true[:, None], axis=-1).astype(jnp.float32) * valid)

            scores_rrf = jnp.mean(1.0 / (60.0 + ranks), axis=0)
            top5_rrf = jax.lax.top_k(scores_rrf, 5)[1]
            top1_rrf = top5_rrf[:, 0]
            out[f"{name}_rrf_top1"] = jnp.sum((top1_rrf == true).astype(jnp.float32) * valid)
            out[f"{name}_rrf_top5"] = jnp.sum(jnp.any(top5_rrf == true[:, None], axis=-1).astype(jnp.float32) * valid)
        return jax.lax.psum(out, axis_name="batch")

    return jax.pmap(step, axis_name="batch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-mode", default="remote_run_AB_mlp_k80_lowaug_mixlabel")
    parser.add_argument("--load-from", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--methods", default="all,lsb2,msb2,lsb4,msb4,lsb10,msb10,rand0,rand1")
    parser.add_argument("--ensembles", default="")
    parser.add_argument("--out", default="agents/inference_score_ablation_ab_in1k.json")
    args = parser.parse_args()

    if args.batch_size % jax.local_device_count() != 0:
        raise ValueError("--batch-size must be divisible by jax.local_device_count()")

    config = _copy_config(get_config(args.config_mode))
    config.batch_size = args.batch_size
    config.dataset.num_workers = args.num_workers
    config.dataset.prefetch_factor = 2
    config.dataset.pin_memory = False
    config.dataset.repeated_aug = 1
    config.dataset.debug = False

    if config.get("head_type", "attention") != "mlp":
        raise ValueError("This inference scorer currently supports the MLP diffusion head used by AB.")

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    ensembles = _parse_ensembles(tuple(x.strip() for x in args.ensembles.split(",") if x.strip()))
    for _, parts in ensembles:
        missing = [p for p in parts if p not in methods]
        if missing:
            raise ValueError(f"Ensemble references methods not in --methods: {missing}")
    model, state = _make_state(config, args.load_from)
    eval_step = _make_eval_step(model, methods, ensembles)

    eval_loader, _ = input_pipeline.create_split(
        config.dataset,
        args.batch_size,
        split="val",
    )

    totals = None
    seen = 0
    t0 = time.time()
    for batch_idx, eval_batch in enumerate(eval_loader):
        batch = prepare_batch_data_sqa(eval_batch, args.batch_size)
        stats = eval_step(state, batch)
        stats = {k: float(jax.device_get(v)[0]) for k, v in stats.items()}
        if totals is None:
            totals = {k: 0.0 for k in stats}
        for k, v in stats.items():
            totals[k] += v
        seen += int(stats["n_valid"])
        if seen >= args.max_examples:
            break
        if (batch_idx + 1) % 10 == 0:
            print(json.dumps({"seen": seen, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    n = max(1.0, totals.pop("n_valid"))
    metrics = {
        "config_mode": args.config_mode,
        "load_from": args.load_from,
        "batch_size": args.batch_size,
        "max_examples": args.max_examples,
        "n_valid": int(n),
        "elapsed_sec": time.time() - t0,
        "methods": methods,
        "ensembles": ensembles,
        "metrics": {k: v / n for k, v in sorted(totals.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
