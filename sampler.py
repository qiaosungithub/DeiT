# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler

    comments by sqa:
    num_replicas: number of devices
    rank: device id
    The total dataset is repeated by num_repeats times
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # num_selected_samples: number of samples selected for each device (don't know why 256)
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        # self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)) * self.num_repeats
        # print("num_selected_samples: ",self.num_selected_samples)
        # exit()
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        # print("indices: ",indices.shape)
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        # print("indices: ",len(indices))
        # assert False
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # print("length of indices: ",len(indices))
        # print("final length: ",self.num_selected_samples)
        # assert False

        return iter(indices[:self.num_selected_samples]) # iter 把list转换成迭代器

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
