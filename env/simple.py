import math
import sys
from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np
from isaacgym import gymapi

from env.vectask import VecTask
from utils import PROJECT_ROOT
from utils.env import *


class SimpleTask(VecTask):
    def __init__(self, **kwargs):
        self._env_spacing = None
        self._humanoid_asset_filename = None

        self._envs = []
        self._humanoids = []

        super().__init__(**kwargs)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)

        self._env_spacing = env_cfg['spacing']
        self._humanoid_asset_filename = env_cfg['humanoid_asset_filename']

    def _create_envs(self):
        # humanoid
        humanoid_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, self._humanoid_asset_filename,
                                              default_asset_option())

        # variable for aggregate
        num_rigid_body = self._gym.get_asset_rigid_body_count(humanoid_asset)
        num_shape = self._gym.get_asset_rigid_shape_count(humanoid_asset)
        self_collision = False

        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, *env_create_parameters(self._num_envs, self._env_spacing))
            self._envs.append(env)

            self._gym.begin_aggregate(env, num_rigid_body, num_shape, self_collision)
            humanoid = self._gym.create_actor(env, humanoid_asset, drop_transform(2), "humanoid", i, 0, 0)
            self._humanoids.append(humanoid)
            self._gym.end_aggregate(env)

    def _pre_physics(self, actions: torch.Tensor):
        pass

    def _post_physics(self, actions: torch.Tensor):
        pass
