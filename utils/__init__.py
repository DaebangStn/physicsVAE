import os
import sys
import math
import random
from typing import Tuple, Optional, List, Dict

import numpy as np
from gym.spaces import Box
from rl_games.algos_torch import torch_ext

from isaacgym import gymapi, gymtorch

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def model_device(model):
    return next(model.parameters()).device


def load_checkpoint_to_network(model, ckpt_path):
    ckpt = torch_ext.load_checkpoint(ckpt_path)
    try:
        model.load_state_dict(ckpt['model'])
    except RuntimeError as e:
        print(f"Error: {e}")


def reshape_gym_box(box, shape: tuple):
    low = box.low[0]
    high = box.high[0]
    return Box(low, high, dtype=box.dtype, shape=shape)
