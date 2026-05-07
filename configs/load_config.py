import os

import yaml

from configs.default import get_config as get_default_config


def _resolve_imagenet_root():
    return "/kmh-nfs-ssd-us-mount/data/imagenet"


def _resolve_dataset_paths(config):
    if "dataset" not in config:
        return

    dataset = config.dataset
    if "root" not in dataset:
        return

    root = dataset.root
    if isinstance(root, str):
        root = os.path.expandvars(root)
        if root.endswith("/data/imagenet"):
            dataset.root = _resolve_imagenet_root()
        else:
            dataset.root = root


def get_config(mode_string):
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        f"{mode_string}_config.yml",
    )
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    default_config = get_default_config()

    def update_config(dst, src):
        for key, value in src.items():
            if isinstance(value, dict):
                if key not in dst:
                    dst[key] = {}
                update_config(dst[key], value)
            else:
                dst[key] = value

    update_config(default_config, config_dict)
    _resolve_dataset_paths(default_config)
    return default_config
