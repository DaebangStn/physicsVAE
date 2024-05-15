from abc import ABC, abstractmethod

import torch

from env.keypointMaxObs import KeypointMaxObsTask
from utils import *
from utils.env import marker_asset_option


class HumanoidTask(KeypointMaxObsTask, ABC):
    def __init__(self, **kwargs):
        self._task_up_freq_max = None
        self._task_up_freq_min = None
        self._task_asset = None
        super().__init__(**kwargs)
        self._update_target(skip_draw=True)

    @abstractmethod
    def _build_env(self, env, env_id):
        super()._build_env(env, env_id)

    @abstractmethod
    def _build_tensors(self):
        super()._build_tensors()
        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        self._buf["humanoidPose"] = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 0]
        self._buf["humanoidPos"] = self._buf["humanoidPose"][..., 0:3]
        self._buf["taskRemain"] = torch.zeros(self._num_envs, device=self._compute_device, dtype=torch.int32)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        task_cfg = env_cfg['task']
        self._task_up_freq_max = task_cfg['update_freq_max']
        self._task_up_freq_min = task_cfg['update_freq_min']
        self._task_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, task_cfg['asset_filename'],
                                                marker_asset_option())
        return env_cfg

    def _post_physics(self, actions: torch.Tensor):
        super()._post_physics(actions)
        self._update_target()

    def _compute_aggregate_option(self) -> Tuple[int, int]:
        num_rigid_body, num_shape = super()._compute_aggregate_option()
        num_rigid_body += self._gym.get_asset_rigid_body_count(self._task_asset)
        num_shape += self._gym.get_asset_rigid_shape_count(self._task_asset)
        return num_rigid_body, num_shape

    @abstractmethod
    def _compute_reward(self):
        pass

    @abstractmethod
    def _draw_task(self, env_ids: Optional[torch.Tensor] = None):
        pass

    @abstractmethod
    def _update_target(self, skip_draw: bool = False):
        pass
