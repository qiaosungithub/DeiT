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

  return config


metrics = default_lib.metrics
