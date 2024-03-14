import os
import random
import yaml
from typing import Tuple
from pathlib import Path
import torch
import numpy as np
from argparse import Namespace

from .fixed_gymutil import parse_arguments


def build_args() -> Namespace:
    custom_params = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training. If not, policy will be trained."},
        {"name": "--config_train_path", "type": str, "required": True,
         "help": "Path to the training(like hyperparameter) config file"},
        {"name": "--config_env_path", "type": str, "required": True,
         "help": "Path to the environment(like task and physics) config file"},
    ]

    args = parse_arguments(custom_parameters=custom_params)
    return args


def load_config(args: Namespace) -> Tuple[dict, dict]:
    config_train_path = Path(args.config_train_path)
    assert config_train_path.exists(), f"Config path {config_train_path} does not exist"
    with open(config_train_path.as_posix(), 'r') as f:
        config_train = yaml.load(f, Loader=yaml.SafeLoader)

    config_env_path = Path(args.config_env_path)
    assert config_env_path.exists(), f"Config path {config_env_path} does not exist"
    with open(config_env_path.as_posix(), 'r') as f:
        config_env = yaml.load(f, Loader=yaml.SafeLoader)

    # TODO: Overriding parameters passed on the cli
    config_run = {
        "play": config_train["test"],
        "train": not config_train["test"],
        "checkpoint": config_train["checkpoint"] if "checkpoint" in config_train else None
    }
    config_env["sim"]["device_id"] = args.compute_device_id

    if config_train["test"]:
        config_env["sim"]["headless"] = False
        config_env["env"]["num_envs"] = 1

    # Overriding config_env to config_train
    config_train["config"]["name"] = config_env["env"]["name"]
    config_train["config"]["env_name"] = config_env["env"]["name"]
    config_train["config"]["num_actors"] = config_env["env"]["num_envs"]
    config_train["config"]["env_config"] = config_env

    return config_run, config_train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
