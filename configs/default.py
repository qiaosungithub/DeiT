# Copied from Kaiming He's resnet_jax repository

import ml_collections

# sqa warning: this file may not work now, see fake_data_benchmark.py

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model
  config.model = 'ResNet50'

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  config.debug = False
  config.dataset.debug = config.debug
  dataset.name = 'imagenet'
  dataset.root = '/kmh-nfs-us-mount/data/imagenet'
  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False

  # Training
  config.learning_rate = 0.1
  config.warmup_epochs = 5
  config.momentum = 0.9
  config.batch_size = 128
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 100
  config.log_per_step = 100
  config.log_per_epoch = -1
  config.eval_per_epoch = 1
  config.checkpoint_per_epoch = 20

  config.steps_per_eval = -1

  
  config.half_precision = False

  config.seed = 0  # init random seed

  # added by sqa
  config.grad_norm_clip = None
  config.label_smoothing = 0.0
  config.dropout_rate = 0.0
  config.stochastic_depth_rate = 0.0
  config.weight_decay = 0.0

  return config


def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]