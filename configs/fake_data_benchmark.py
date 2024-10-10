# Copied from Kaiming He's resnet_jax repository


import jax

from configs import default as default_lib


def get_config():
  """Get the hyperparameter configuration for Fake data benchmark."""
  # Override default configuration to avoid duplication of field definition.
  config = default_lib.get_config()
  config.batch_size = 256 * jax.device_count()
  config.half_precision = True
  config.num_epochs = 5

  # Run for a single step:
  config.num_train_steps = 1
  config.steps_per_eval = 1
  config.grad_norm_clip = "None"
  config.optimizer = "None"

  # configs for data transforms
  # we can use `get()` for config.dataset to avoid KeyError
  if config.dataset.get('root'):
    print(f"Dataset root is set to: {config.dataset.get('root')}")
  
  # rand_augment:
  config.dataset.use_rand_augment = False
  config.dataset.rand_augment = 'rand-m9-mstd0.5-inc1'
  config.dataset.reprob = 0.0

  # mixup & cutmix
  # i have not figured out how to set the alphas, and whether i set the probs correct or not
  config.dataset.use_mixup_cutmix = False
  config.dataset.mixup_alpha = 0.2
  config.dataset.cutmix_alpha = 0.2
  config.dataset.mixup_prob = 1.0
  config.dataset.switch_prob = 0.5 # probability of switching to cutmix
  config.dataset.mixup_mode = 'batch'
  config.dataset.label_smoothing = 0.0 # a regularization technique for training
  config.dataset.repeated_aug = 3
  config.dataset.num_tpus = 32
  # you can add these things but I don't want to:
  # color_fitter: default 0.3

  return config