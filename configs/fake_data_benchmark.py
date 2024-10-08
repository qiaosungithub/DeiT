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

  return config
