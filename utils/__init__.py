import isaacgym
from isaacgym import gymapi, gymtorch

import os
import sys
import time
import yaml
import math
import random
from abc import abstractmethod
from typing import Tuple, Optional, List, Dict, Any, Callable, Union

import torch
import numpy as np
from gym.spaces import Box
from matplotlib import pyplot as plt
from rl_games.algos_torch import torch_ext


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def model_device(model):
    return next(model.parameters()).device


def load_checkpoint_to_network(model, ckpt_path):
    ckpt = torch_ext.load_checkpoint(ckpt_path)
    try:
        model.load_state_dict(ckpt['model'])
    except RuntimeError as e:
        print(f"Error: {e}")


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    return y


def reshape_gym_box(box, shape: tuple):
    low = box.low[0]
    high = box.high[0]
    return Box(low, high, dtype=box.dtype, shape=shape)


def mean_list(lst: List[float]) -> float:
    return sum(lst) / len(lst)


def std_tensor_list(lst: List[torch.Tensor]) -> torch.Tensor:
    return torch.std(torch.stack(lst), dim=0)
