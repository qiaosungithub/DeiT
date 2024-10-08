# Copied from Kaiming He's resnet_jax repository

# TODO: We should do our modifications here.

import numpy as np
import os
import random
import jax
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from absl import logging
from functools import partial
import timm.data.transforms as timm_transforms
from timm.data.mixup import Mixup
from timm.data import create_transform

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def prepare_batch_data(batch, batch_size=None):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  image, label = batch  

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    image = torch.cat([image, torch.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
    label = torch.cat([label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape(local_device_count, -1)

  image = image.numpy()
  label = label.numpy()

  return_dict = {
    'image': image,
    'label': label,
  }

  return return_dict

def prepare_batch_data_sqa(batch, batch_size=None):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size, n_classes=1000)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  image, label = batch  
  # print("label.shape:", label.shape)

  if label.ndim == 1: # make the label to be one-hot
    label = torch.nn.functional.one_hot(label, num_classes=1000)

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    image = torch.cat([image, torch.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
    label = torch.cat([label, -torch.ones((batch_size - label.shape[0], 1000), dtype=label.dtype)], axis=0)

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape((local_device_count, -1) + label.shape[1:])
  # print("label.shape:", label.shape)

  image = image.numpy()
  label = label.numpy()

  return_dict = {
    'image': image,
    'label': label,
  }

  return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from torchvision.datasets.folder import pil_loader
def loader(path: str):
    return pil_loader(path)

class ToTensorIfNeeded:
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return transforms.functional.to_tensor(pic)

def get_augmentations(dataset_cfg):
  """
  dataset_cfg: dataset configuration
  dataset_cfg.rand_augment: (str) if exist, the string of rand augment policy
  dataset_cfg.reprob: (float = 0.0) the probability of random erasing
  TODO: implement auto_augment
  """
  augmentations = []
  if dataset_cfg.use_rand_augment:
    augmentations.append(create_transform(
      input_size=IMAGE_SIZE,
      is_training=True,
      auto_augment=dataset_cfg.get('rand_augment', 'rand-m9-mstd0.5-inc1'),
      re_prob=dataset_cfg.get('reprob', 0.0),
      re_mode="pixel",
      re_count=1,
    ))
  elif dataset_cfg.reprob > 0.0:
    augmentations.append(create_transform(
      input_size=IMAGE_SIZE,
      is_training=True,
      re_prob=dataset_cfg.get('reprob', 0.0),
      re_mode="pixel",
      re_count=1,
    ))
  # if dataset_cfg.get('auto_augment', False):
  #     augmentations.append(create_transform(
  #       input_size=IMAGE_SIZE,
  #       is_training=True,
  #       auto_augment='v0',
  #     ))

  return augmentations

def apply_mixup_cutmix(dataset_cfg, dataloader):
  """apply Mixup, CutMix"""
  
  mixup_fn = None
  
  # set Mixup; for Cutmix, just set cutmix_alpha > 0
  if dataset_cfg.use_mixup_cutmix:
    mixup_fn = Mixup(
      mixup_alpha=dataset_cfg.get('mixup_alpha', 0.2),
      cutmix_alpha=dataset_cfg.get('cutmix_alpha', 0.2),
      prob=dataset_cfg.get('mixup_prob', 1.0),
      mode=dataset_cfg.get('mixup_mode', 'batch'),
      label_smoothing=dataset_cfg.get('label_smoothing', 0.0),
      switch_prob=dataset_cfg.get('switch_prob', 0.5),
      num_classes=1000,
    )

  # 逐批处理数据
  for batch in dataloader:
    inputs, targets = batch
    # print("inputs.shape:", inputs.shape)
    # print("targets.shape:", targets.shape)
    
    # apply Mixup
    if mixup_fn is not None:
      inputs, targets = mixup_fn(inputs, targets) # the labels will be turned into [bsz, num_classes]

    # print("inputs.shape:", inputs.shape)
    # print("targets.shape:", targets.shape)
    
    # TODO: ??? 在这里应用 StochasticDepth（假设有相关函数）by copilot
    # not sure if correct or not
    # model.apply_stochastic_depth(inputs)
    
    # 返回处理后的 batch
    yield inputs, targets

def create_split(
    dataset_cfg,
    batch_size,
    split,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    dataset_cfg: Configurations for the dataset.
    batch_size: Batch size for the dataloader.
    split: 'train' or 'val'.
  Returns:
    it: A PyTorch Dataloader.
    steps_per_epoch: Number of steps to loop through the DataLoader.
  """
  rank = jax.process_index()
  if split == 'train':
    # ds = datasets.ImageFolder(
    #   os.path.join(dataset_cfg.root, split),
    #   transform=transforms.Compose([
    #     transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
    #   ]),
    #   loader=loader,
    # )
    augmentations = get_augmentations(dataset_cfg)
    transform_list = augmentations + [
        # transforms.ToTensor(),
        ToTensorIfNeeded(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
    ]
    # TODO: whether we should keep the RandomResizedCrop and RandomHorizontalFlip
    ds = datasets.ImageFolder(
        os.path.join(dataset_cfg.root, split),
        transform=transforms.Compose(transform_list),
        loader=loader,
    )
    logging.info(ds)
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=True,
    )
    it = DataLoader(
      ds, batch_size=batch_size, drop_last=True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)

    # Apply Mixup, CutMix
    it = apply_mixup_cutmix(dataset_cfg, it)

  elif split == 'val':
    ds = datasets.ImageFolder(
      os.path.join(dataset_cfg.root, split),
      transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE + CROP_PADDING, interpolation=3),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
      ]),
      loader=loader,
    )
    logging.info(ds)
    '''
    The val has 50000 images. We want to eval exactly 50000 images. When the
    batch is too big (>16), this number is not divisible by the batch size. We
    set drop_last=False and we will have a tailing batch smaller than the batch
    size, which requires modifying some eval code.
    '''
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=False,  # don't shuffle for val
    )
    it = DataLoader(
      ds, batch_size=batch_size,
      drop_last=False if not dataset_cfg.debug else True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
  else:
    raise NotImplementedError

  return it, steps_per_epoch
