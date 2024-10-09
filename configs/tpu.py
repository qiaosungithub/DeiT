# Copied from Kaiming He's resnet_jax repository

from configs import default as default_lib


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = default_lib.get_config()

  # Dataset
  dataset = config.dataset
  dataset.cache = True

  # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
  # train on a larger pod slice.
  config.batch_size = 1024
  config.shuffle_buffer_size = 16 * 1024
  config.half_precision = True

  config.grad_norm_clip = "None"
  config.optimizer = "None"

  # configs for data transforms
  # we can use `get()` for config.dataset to avoid KeyError
  # if config.dataset.get('root'):
  #   print(f"Dataset root is set to: {config.dataset.get('root')}")
  
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

  return config


metrics = default_lib.metrics