from typing import Dict, Optional, Tuple

import torch
from isaacgym import gymtorch, gymapi

from env.keypointMaxObs import KeypointMaxObsTask
from utils import *
from utils.env import *



class HeadingTask(KeypointMaxObsTask):
    def __init__(self, **kwargs):
        self._tar_update_freq_max = None
        self._tar_update_freq_min = None
        self._tar_asset = None

        # Visualize the target and actor heading direction
        self._targets_id_env = []
        self._targets_id_sim = []
        self._heads_id_env = []
        self._heads_id_sim = []
        self._vis_id_env = []
        self._vis_id_sim = []

        super().__init__(**kwargs)

    def _build_env(self, env, env_id):
        super()._build_env(env, env_id)

        if not self._headless:
            target = self._gym.create_actor(env, self._tar_asset, drop_transform(0), "target", env_id, 1, 0)
            self._gym.set_rigid_body_color(env, target, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))
            self._targets_id_env.append(target)
            self._vis_id_env.append(target)
            target_sim_id = self._gym.get_actor_index(env, target, gymapi.DOMAIN_SIM)
            self._targets_id_sim.append(target_sim_id)
            self._vis_id_sim.append(target_sim_id)

            heads = self._gym.create_actor(env, self._tar_asset, drop_transform(0), "head", env_id, 2, 0)
            self._gym.set_rigid_body_color(env, target, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))
            self._heads_id_env.append(heads)
            self._vis_id_env.append(heads)
            self._heads_id_sim.append(self._gym.get_actor_index(env, heads, gymapi.DOMAIN_SIM))


    def _build_tensors(self):
        # TODO: write actor view like location
        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        target_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 1]
        self._buf["tarQuat"] = target_root_reshaped[..., 3:7]
        head_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 2]
        self._buf["headQuat"] = target_root_reshaped[..., 3:7]

        self._buf["tarRemain"] = torch.zeros([self._num_envs], device=self._compute_device)
        self._update_target(skip_draw=True)

    def _compute_aggregate_option(self) -> Tuple[int, int]:
        num_rigid_body, num_shape = super()._compute_aggregate_option()
        num_rigid_body += self._gym.get_asset_rigid_body_count(self._tar_asset) * 2
        num_shape += self._gym.get_asset_rigid_shape_count(self._tar_asset) * 2
        return num_rigid_body, num_shape

    def _compute_reward(self):
        # self._buf["rew"] = heading_reward(self._buf["rPos"], self._buf["tarPos"])
        pass

    def _create_envs(self):
        super()._create_envs()

        if not self._headless:
            self._targets_id_env = torch.tensor(self._targets_id_env, device=self._compute_device, dtype=torch.int32)
            self._targets_id_sim = torch.tensor(self._targets_id_sim, device=self._compute_device, dtype=torch.int32)
            self._heads_id_env = torch.tensor(self._heads_id_env, device=self._compute_device, dtype=torch.int32)
            self._heads_id_sim = torch.tensor(self._heads_id_sim, device=self._compute_device, dtype=torch.int32)

    def _draw_target(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._compute_device)

        num_update = len(env_ids)
        if num_update == 0:
            return

        _id = self._targets_id_sim[env_ids]

