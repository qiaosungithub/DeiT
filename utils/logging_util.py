# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging as _logging
import shutil
from absl import logging

import jax
from jax.experimental import multihost_utils
from clu import metric_writers


def log_for_0(*args, stacklevel=1):
    if jax.process_index() == 0:
        logging.info(*args, stacklevel=stacklevel)

class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True

# Suppress orbax/flax checkpoint INFO logs: CommitFuture blocking, "No metadata found", etc.
exclude_files = [
    'orbax/checkpoint/async_checkpointer.py',
    'orbax/checkpoint/abstract_checkpointer.py',
    'orbax/checkpoint/multihost/utils.py',
    'orbax/checkpoint/future.py',
    'orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py',
    'orbax/checkpoint/type_handlers.py',
    'orbax/checkpoint/metadata/checkpoint.py',
    'orbax/checkpoint/metadata/sharding.py',
    'orbax/checkpoint/metadata/array_metadata_store.py',
    'array_metadata_store.py',
    'orbax/checkpoint/',  # catch any other checkpoint INFO under orbax (e.g. future.py path variants)
] + [
    'orbax/checkpoint/checkpointer.py',
    'flax/training/checkpoints.py',
] * jax.process_index()
file_filter = ExcludeInfo(exclude_files)

def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)

from termcolor import colored
import time


def set_time_logging(logger):
    pid = jax.process_index()
    prefix = "[p{:02d} %(asctime)s.%(msecs)03d %(levelname)s:%(filename)s:%(lineno)d] ".format(
        pid
    )
    str = colored(prefix, "green") + "%(message)s"
    logger.get_absl_handler().setFormatter(
        _logging.Formatter(str, datefmt="%m%d %H:%M:%S")
    )


def set_time_logging_short(logger):
    pid = jax.process_index()
    prefix = "[p{:02d} %(asctime)s] ".format(pid)
    str = colored(prefix, "green") + "%(message)s"
    logger.get_absl_handler().setFormatter(
        _logging.Formatter(str, datefmt="%m%d %H:%M:%S")
    )


def verbose_on():
    logging.set_verbosity(logging.INFO)  # show all processes


def verbose_off():
    if not (jax.process_index() == 0):  # not first process
        logging.set_verbosity(logging.ERROR)  # disable info/warning


def sync_and_delay(delay=None):
    # Block all hosts until directory is ready.
    multihost_utils.sync_global_devices(f"logging")
    if delay is None:
        delay = jax.process_index() * 0.1
    time.sleep(delay)


class Writer:
    def __init__(self, config, workdir, use_wandb=False, use_tb=False):
        self.use_wandb = False
        self.use_tb = False
        if jax.process_index() != 0:
            return
        if use_wandb:
            import wandb

            kwargs = {}
            if getattr(config, "wandb_resume_id", ""):
                kwargs["id"] = config.wandb_resume_id
                kwargs["resume"] = "must"

            wandb.init(
                project=config.logging.wandb_project + "_eval" * config.eval_only,
                entity=config.logging.wandb_entity or None,
                notes=config.logging.wandb_notes or None,
                tags=config.logging.wandb_tags or None,
                dir="/tmp",
                settings=wandb.Settings(_service_wait=60),
                **kwargs,
            )
            wandb.config.update(config.to_dict(), allow_val_change=True)
            self.wandb = wandb
            self.use_wandb = True

        if use_tb:
            self.writer = metric_writers.create_default_writer(logdir=workdir, just_logging=False)
            self.use_tb = True

    def write_scalars(self, step, scalar_dict):
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            if isinstance(v, float):
                log_str += f" {k}={v:.5g},"
            else:
                log_str += f" {k}={v},"
        logging.info(log_str.strip(","))
        if self.use_wandb:
            self.wandb.log(scalar_dict, step=step)
        if self.use_tb:
            self.writer.write_scalars(step, scalar_dict)

    def flush(self):
        if jax.process_index() != 0:
            return
        if self.use_tb:
            self.writer.flush()

    def __del__(self):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            self.wandb.finish()
            shutil.rmtree('/tmp/wandb', ignore_errors=True)
        if self.use_tb:
            self.writer.flush()
            self.writer.close()
