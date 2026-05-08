# Copied from Kaiming He's resnet_jax repository

import functools
import time
from typing import Any

from flax import jax_utils
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax

import input_pipeline
from input_pipeline import prepare_batch_data_sqa, apply_mixup_cutmix_batch, pre_process_batch
import models

from utils.info_util import print_params
from utils import ckpt_util
from utils.dataset_util import resolve_and_prepare_dataset_root
from utils.logging_util import Writer, log_for_0


NUM_CLASSES = 1000


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args, rng=key)

  log_for_0('Initializing params...')
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  if 'batch_stats' not in variables:
    variables['batch_stats'] = {}
  log_for_0('Initializing params done.')
  return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels, label_smoothing=0.1):
  labels = labels.astype(jnp.float32)
  # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  # apply label smoothing
  smooth_labels = optax.smooth_labels(labels, alpha=label_smoothing)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=smooth_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  # this is the version for both one-hot labels and not one-hot labels
  # compute per-sample loss
  # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  # print("labels.shape:", labels.shape)
  if labels.shape[-1] != NUM_CLASSES:
    labels = jax.nn.one_hot(labels, NUM_CLASSES)

  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  loss = xentropy  # (local_batch_size,)

  accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1))  # (local_batch_size, )
  # here we modify, but not very well defined
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'labels': labels,
  }
  metrics = lax.all_gather(metrics, axis_name='batch')
  labels = metrics['labels']
  metrics = jax.tree.map(lambda x: x.flatten(), metrics)  # (batch_size,)
  metrics['labels'] = labels
  return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
  """
  Create learning rate schedule.

  first warmup (increase to base_learning_rate) for config.warmup_epochs
  then cosine decay to 0 for the rest of the epochs
  """
  # warmup_fn = optax.linear_schedule(
  #     init_value=0.0,
  #     end_value=base_learning_rate,
  #     transition_steps=config.warmup_epochs * steps_per_epoch,
  # )
  cosine_epochs = max((config.num_epochs-10) - config.warmup_epochs, 1)
  # cosine_fn = optax.cosine_decay_schedule(
  #     init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
  # )
  # schedule_fn = optax.join_schedules(
  #     schedules=[warmup_fn, cosine_fn],
  #     boundaries=[config.warmup_epochs * steps_per_epoch],
  # )
  # print('warmup_epochs:', config.warmup_epochs)
  # print('cosine_epochs:', cosine_epochs)
  # print('steps_per_epoch:', steps_per_epoch)
  first_schedule = optax.schedules.warmup_cosine_decay_schedule(init_value=0.0, peak_value=base_learning_rate, warmup_steps=config.warmup_epochs*steps_per_epoch, decay_steps=(config.warmup_epochs + cosine_epochs)*steps_per_epoch, end_value=1e-5) 
  second_schedule = optax.schedules.constant_schedule(value=1e-5)
  return optax.join_schedules(schedules=[first_schedule, second_schedule], boundaries=[(config.warmup_epochs + cosine_epochs)*steps_per_epoch])



def train_step_sqa(state, batch, rng_init, learning_rate_fn,label_smoothing=0.1):
  """Perform a single training step."""

  # ResNet has no dropout; but maintain rng_dropout for future usage
  rng_step = random.fold_in(rng_init, state.step)
  rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  rng_dropout, _ = random.split(rng_device)

  def categorical_cross_entropy_loss(logits, labels,label_smoothing=label_smoothing):
    """计算分类交叉熵损失"""
    # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    # xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    # return jnp.mean(xentropy)
    return cross_entropy_loss(logits, labels,label_smoothing=label_smoothing)

  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      batch['image'],
      mutable=['batch_stats'],
      # rngs=dict(dropout=rng_dropout),
      rng=rng_dropout,
    )
    loss = categorical_cross_entropy_loss(logits, batch['label'])
    return loss, (new_model_state, logits)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['lr'] = lr

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_model_state['batch_stats']
  )

  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False, rng=jax.random.PRNGKey(0))
  return compute_metrics(logits, batch['label'])


def eval_step_agg(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False, rng=jax.random.PRNGKey(0))

  labels = batch['label']
  if labels.shape[-1] != NUM_CLASSES:
    labels = jax.nn.one_hot(labels, NUM_CLASSES)
  labels = labels.astype(jnp.float32)

  per_example_loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
  per_example_acc = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1)).astype(jnp.float32)
  valid = (labels[..., 0] >= 0).astype(jnp.float32)

  stats = {
    'loss_sum': jnp.sum(per_example_loss * valid),
    'acc_sum': jnp.sum(per_example_acc * valid),
    'n_valid': jnp.sum(valid),
  }
  return lax.psum(stats, axis_name='batch')


class TrainState(train_state.TrainState):
  batch_stats: Any


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if not hasattr(state, 'batch_stats'):
    return state
  if not state.batch_stats:
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")


  params, batch_stats = initialized(rng, image_size, model)
  
  print_params(params)

  if config.optimizer == 'sgd':
    if config.weight_decay != 0.0:
      print("Warning from sqa: weight decay is not supported in SGD")
    if config.grad_norm_clip != "None":
      print("Warning from sqa: grad norm clipping is not supported in SGD")
    tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
    )
  elif config.optimizer == 'adamw':
    grad_norm_clip = config.grad_norm_clip
    assert grad_norm_clip is None, "grad_norm_clip is not supported in AdamW"
    tx = optax.adamw(
      learning_rate=learning_rate_fn,
      b1=0.9,
      b2=config.get('adamw_b2', 0.95),
      eps=1e-8,
      weight_decay=config.weight_decay,
    )
  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer}, choose from "sgd" or "adamw"')
  state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    batch_stats=batch_stats,
  )
  return state


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = Writer(
      config,
      workdir,
      use_wandb=config.logging.use_wandb,
      use_tb=False,
  )

  rng = random.key(config.seed)

  image_size = 224

  log_for_0('config.batch_size: {}'.format(config.batch_size))
  resolve_and_prepare_dataset_root(config, workdir)

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  eval_loader, steps_per_eval = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='val',
  )
  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  log_for_0('steps_per_eval: {}'.format(steps_per_eval))

  base_learning_rate = config.learning_rate * config.batch_size / 512.0 # note that here the input config.learning_rate is 0.0005 in the paper

  model_cls = getattr(models, config.model)
  model = create_model(
    model_cls=model_cls, half_precision=config.half_precision,
    dropout_rate=config.dropout_rate,
    stochastic_depth_rate=config.stochastic_depth_rate,
  )

  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)

  if config.load_from != "":
    state = ckpt_util.restore_checkpoint(state, config.load_from, workdir)
  
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
  assert epoch_offset * steps_per_epoch == step_offset, (epoch_offset, steps_per_epoch, step_offset)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
    functools.partial(train_step_sqa, rng_init=rng, learning_rate_fn=learning_rate_fn, label_smoothing=0.0),
    axis_name='batch',
    donate_argnums=(0, 1),
  )
  p_eval_step = jax.pmap(eval_step_agg, axis_name='batch')

  train_metrics = []
  train_metrics_last_t = time.time()

  log_for_0('Initial compilation, this might take some minutes...')
  for epoch in range(epoch_offset, config.num_epochs):
    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))

    for n_batch, batch in enumerate(train_loader):
      batch = pre_process_batch(batch)
      batch = apply_mixup_cutmix_batch(config.dataset, batch)
      step = epoch * steps_per_epoch + n_batch
      batch = prepare_batch_data_sqa(batch)

      if step == 0: log_for_0('First batch ready')

      assert batch['label'].shape[-1] == NUM_CLASSES
      state, metrics = p_train_step(state, batch) # here is the training step
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')
        train_metrics_last_t = time.time()
      
      # normalize to IN1K epoch anyway
      ep = step * config.batch_size / 1281167

      if config.get('log_per_step'):
        train_metrics.append(metrics)
        if (step + 1) % config.log_per_step == 0:
          train_metrics = common_utils.get_metrics(train_metrics)
          train_metrics.pop('labels')  # used in val only
          summary = {
            f'train_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: float(x.mean()), train_metrics
            ).items()
          }
          summary['steps_per_second'] = config.log_per_step / (time.time() - train_metrics_last_t)
          # summary['seconds_per_step'] = (time.time() - train_metrics_last_t) / config.log_per_step

          # step for tensorboard
          summary["ep"] = ep

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

    # logging per epoch
    if (epoch + 1) % config.eval_per_epoch == 0 or epoch == 0:
      log_for_0('Eval epoch {}...'.format(epoch))
      loss_sum = 0.0
      acc_sum = 0.0
      n_valid = 0.0
      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for n_eval_batch, eval_batch in enumerate(eval_loader):
        if (n_eval_batch + 1) % config.log_per_step == 0:
          log_for_0('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
        eval_batch = prepare_batch_data_sqa(eval_batch, local_batch_size)

        stats = p_eval_step(state, eval_batch)
        loss_sum += float(jax.device_get(stats['loss_sum'])[0])
        acc_sum += float(jax.device_get(stats['acc_sum'])[0])
        n_valid += float(jax.device_get(stats['n_valid'])[0])

      if n_valid <= 0:
        raise ValueError('No valid samples during evaluation')
      log_for_0('valid samples: {}'.format(int(n_valid)))

      summary = {
        'loss': loss_sum / n_valid,
        'accuracy': acc_sum / n_valid,
      }
      log_for_0(
        'eval epoch: %d, loss: %.6f, accuracy: %.6f',
        epoch,
        summary['loss'],
        summary['accuracy'] * 100,
      )
      summary = {f'eval_{key}': val for key, val in summary.items()}
      summary["ep"] = ep
      writer.write_scalars(step + 1, summary)
      writer.flush()

    if (
      (epoch + 1) % config.checkpoint_per_epoch == 0
      or epoch == config.num_epochs
      or epoch == 0  # saving at the first epoch for sanity check
    ):
      state = sync_batch_stats(state)
      state_to_save = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
      log_for_0('Saving checkpoint step %d.', int(state_to_save.step))
      ckpt_util.save_checkpoint(state_to_save, workdir, keep=2)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  return state


def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
  writer = Writer(
      config,
      workdir,
      use_wandb=config.logging.use_wandb,
      use_tb=False,
  )

  rng = random.key(config.seed)
  image_size = 224
  resolve_and_prepare_dataset_root(config, workdir)

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  _, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  eval_loader, steps_per_eval = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='val',
  )

  base_learning_rate = config.learning_rate * config.batch_size / 512.0
  model_cls = getattr(models, config.model)
  model = create_model(
    model_cls=model_cls,
    half_precision=config.half_precision,
    dropout_rate=config.dropout_rate,
    stochastic_depth_rate=config.stochastic_depth_rate,
  )
  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)
  state = create_train_state(rng, config, model, image_size, learning_rate_fn)

  load_path = config.load_from if config.load_from != '' else workdir
  state = ckpt_util.restore_checkpoint(state, load_path, workdir)
  state = jax_utils.replicate(state)

  p_eval_step = jax.pmap(eval_step_agg, axis_name='batch')
  loss_sum = 0.0
  acc_sum = 0.0
  n_valid = 0.0
  state = sync_batch_stats(state)
  for n_eval_batch, eval_batch in enumerate(eval_loader):
    if (n_eval_batch + 1) % config.log_per_step == 0:
      log_for_0('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
    eval_batch = prepare_batch_data_sqa(eval_batch, local_batch_size)
    stats = p_eval_step(state, eval_batch)
    loss_sum += float(jax.device_get(stats['loss_sum'])[0])
    acc_sum += float(jax.device_get(stats['acc_sum'])[0])
    n_valid += float(jax.device_get(stats['n_valid'])[0])

  if n_valid <= 0:
    raise ValueError('No valid samples during evaluation')
  summary = {
    'loss': loss_sum / n_valid,
    'accuracy': acc_sum / n_valid,
  }
  summary = {f'eval_{key}': val for key, val in summary.items()}
  writer.write_scalars(int(jax.device_get(state.step)[0]), summary)
  writer.flush()

  return state
