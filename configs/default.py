# Copied from Kaiming He's resnet_jax repository

import ml_collections

# sqa warning: this file may not work now, see fake_data_benchmark.py

def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model
  config.model = 'ViT_base'

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  config.debug = False
  config.dataset.debug = config.debug
  dataset.name = 'imagenet'
  dataset.root = '/kmh-nfs-ssd-us-mount/data/imagenet'
  dataset.num_workers = 64
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = True
  dataset.use_rand_augment = False
  dataset.rand_augment = 'rand-m9-mstd0.5-inc1'
  dataset.reprob = 0.0
  dataset.use_mixup_cutmix = False
  dataset.mixup_alpha = 0.2
  dataset.cutmix_alpha = 0.2
  dataset.mixup_prob = 1.0
  dataset.switch_prob = 0.5
  dataset.mixup_mode = 'batch'
  dataset.label_smoothing = 0.0
  dataset.repeated_aug = 3

  # Training
  config.learning_rate = 0.0005
  config.warmup_epochs = 5
  config.momentum = 0.9
  config.batch_size = 1024
  config.shuffle_buffer_size = 16 * 1024
  config.prefetch = 10

  config.num_epochs = 330
  config.log_per_step = 100
  config.log_per_epoch = -1
  config.eval_per_epoch = 20
  config.checkpoint_per_epoch = 20
  
  config.half_precision = False

  config.seed = 0  # init random seed
  config.load_from = ''

  # added by sqa
  config.grad_norm_clip = None
  config.label_smoothing = 0.0
  config.dropout_rate = 0.0
  config.stochastic_depth_rate = 0.0
  config.weight_decay = 0.0
  config.optimizer = 'adamw'
  config.adamw_b2 = 0.95
  config.eval_only = False

  # Phase 2: masked diffusion head
  config.use_diffusion_head = False
  config.head_inner_dim = 256
  config.head_n_layers = 2
  config.head_n_heads = 4
  config.mask_schedule = 'uniform'   # 'uniform' or 'logit_normal'
  config.eval_iter_steps = 4         # iterative decode steps in eval
  config.head_type = 'attention'     # 'attention' | 'mlp'
  config.head_zero_init_proj = False  # zero-init final projection layer of diffusion head

  config.logging = logging = ml_collections.ConfigDict()
  logging.wandb_project = ''
  logging.use_wandb = False
  logging.wandb_entity = ''
  logging.wandb_notes = ''
  logging.wandb_tags = []
  config.wandb_resume_id = ''

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
