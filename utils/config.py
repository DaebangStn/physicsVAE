import os
import random
import yaml
from typing import Tuple
from datetime import datetime

from pathlib import Path
import torch
import numpy as np
from argparse import Namespace

from .fixed_gymutil import parse_arguments


def build_args() -> Namespace:
    custom_params = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training. If not, policy will be trained."},
        {"name": "--cfg_train", "type": str, "required": True,
         "help": "Path to the training(like hyperparameter) config file"},
        {"name": "--cfg_env", "type": str, "required": True,
         "help": "Path to the environment(like task and physics) config file"},
        {"name": "--num_envs", "type": int, "required": False,
         "help": "Number of environments to run in parallel"},
    ]

    args = parse_arguments(custom_parameters=custom_params)
    return args


def load_config(args: Namespace) -> Tuple[dict, dict]:
    config_train_path = Path(args.cfg_train)
    assert config_train_path.exists(), f"Config path {config_train_path} does not exist"
    with open(config_train_path.as_posix(), 'r') as f:
        config_train = yaml.load(f, Loader=yaml.SafeLoader)

    config_env_path = Path(args.cfg_env)
    assert config_env_path.exists(), f"Config path {config_env_path} does not exist"
    with open(config_env_path.as_posix(), 'r') as f:
        config_env = yaml.load(f, Loader=yaml.SafeLoader)

    # Compute discriminator related values
    if config_train["algo"]["name"] in ["styleAlgo", "skillAlgo"]:
        assert "disc" in config_train["network"], "Inconsistent config"
        config_train["network"]["disc"]["num_inputs"] = (config_train["hparam"]["style"]["disc"]["obs_traj_len"] *
                                                         config_train["hparam"]["style"]["disc"]["num_obs"])

        # Preprocess MotionLib config
        assert "joint_information_path" in config_env["env"], "Joint information not found in config"
        joint_info_path = Path(config_env["env"]['joint_information_path'])
        assert joint_info_path.exists(), f"Config path {joint_info_path} does not exist"
        with open(joint_info_path.as_posix(), 'r') as f:
            joint_info = yaml.load(f, Loader=yaml.SafeLoader)
            asset_filename = config_env["env"]["humanoid_asset_filename"]
            asset_filename = Path(asset_filename).stem
            assert asset_filename in joint_info, f"Asset {asset_filename} not found in joint information"
            config_train["algo"]["joint_information"] = joint_info[asset_filename]

    # Compute latent related values
    if config_train["algo"]["name"] in ["skillAlgo"]:
        latent_dim = config_train["network"]["space"]["latent_dim"]
        config_env["env"]["num_obs"] += latent_dim

    # Overriding parameters passed on the cli
    if args.test:
        config_train["test"] = args.test
    if args.num_envs:
        config_env["env"]["num_envs"] = args.num_envs

    full_experiment_name = (config_env["env"]["name"] + "_" + config_train["algo"]["name"] + "_" +
                            datetime.now().strftime("%d-%H-%M-%S") + "_" + str(config_train["algo"].get("memo", "")))

    config_train["config"] = {}

    # Overriding test mode
    if config_train["test"]:
        assert config_train["checkpoint"] is not None, "Checkpoint path not found in config"
        config_env["sim"]["headless"] = False
        config_env["env"]["num_envs"] = 2 if args.num_envs is None else args.num_envs
        full_experiment_name = "test_" + full_experiment_name

    # Overriding debug mode
    if config_train["debug"]:
        config_train["hparam"]["minibatch_size"] = 16
        config_env["sim"]["headless"] = False
        config_env["env"]["num_envs"] = 2 if args.num_envs is None else args.num_envs
        full_experiment_name = "debug_" + full_experiment_name

    # Overriding config_env to config_train
    config_train["config"]["full_experiment_name"] = full_experiment_name
    config_train["config"]["name"] = config_env["env"]["name"]
    config_train["config"]["env_name"] = config_env["env"]["name"]
    config_train["config"]["num_actors"] = config_env["env"]["num_envs"]
    config_train["config"]["env_config"] = config_env
    config_train["config"].update(config_train["hparam"])

    assert not (config_train["test"] and config_train["debug"]), "Test and debug mode cannot be enabled together"

    config_run = {
        "play": config_train["test"],
        "train": not config_train["test"],
        "checkpoint": config_train.get("checkpoint", None),
        "checkpoint_disc": config_train.get("checkpoint_disc", None),
    }

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
