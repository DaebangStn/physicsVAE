from typing import Dict, Optional, Tuple

import torch
from isaacgym import gymtorch, gymapi

from env.keypoint import KeypointTask
from utils import *
from utils.env import *


class KeypointLocationTask(KeypointTask):
    def __init__(self, **kwargs):
        self._tar_away_max = None
        self._tar_away_min = None
        self._tar_update_freq_max = None
        self._tar_update_freq_min = None
        self._tar_asset = None

        self._targets_id_env = []
        self._targets_id_sim = []

        super().__init__(**kwargs)

    def _build_env(self, env, env_id):
        super()._build_env(env, env_id)

        target = self._gym.create_actor(env, self._tar_asset, drop_transform(0), "target", env_id, 1, 0)
        self._gym.set_rigid_body_color(env, target, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))
        self._targets_id_env.append(target)
        self._targets_id_sim.append(self._gym.get_actor_index(env, target, gymapi.DOMAIN_SIM))

    def _build_tensors(self):
        super()._build_tensors()
        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        target_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 1]
        self._buf["tarPos"] = target_root_reshaped[..., 0:3]

        self._buf["tarRemain"] = torch.zeros([self._num_envs], device=self._compute_device)
        self._update_target(skip_draw=True)

    def _compute_aggregate_option(self) -> Tuple[int, int]:
        num_rigid_body, num_shape = super()._compute_aggregate_option()
        num_rigid_body += self._gym.get_asset_rigid_body_count(self._tar_asset)
        num_shape += self._gym.get_asset_rigid_shape_count(self._tar_asset)
        return num_rigid_body, num_shape

    def _compute_reward(self):
        # self._buf["rew"] = location_reward(self._buf["rPos"], self._buf["tarPos"])
        pass

    def _create_envs(self):
        super()._create_envs()

        if not self._headless:
            self._targets_id_env = torch.tensor(self._targets_id_env, device=self._compute_device, dtype=torch.int32)
            self._targets_id_sim = torch.tensor(self._targets_id_sim, device=self._compute_device, dtype=torch.int32)

    def _draw_target(self, env_ids: Optional[torch.Tensor] = None):
        col = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._gym.clear_lines(self._viewer)
        lines = torch.cat([self._buf["tarPos"], self._buf["aPos"]], dim=-1).cpu().numpy()

        for i in range(self.num):
            self._gym.add_lines(self._viewer, self._envs[i], 1, lines[i].reshape([1, 6]), col)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._compute_device)

        num_update = len(env_ids)
        if num_update == 0:
            return

        _id = self._targets_id_sim[env_ids]
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), gymtorch.unwrap_tensor(_id), num_update)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)

        target_cfg = env_cfg['target']
        self._tar_away_max = target_cfg['away_max']
        self._tar_away_min = target_cfg['away_min']
        self._tar_update_freq_max = target_cfg['update_freq_max']
        self._tar_update_freq_min = target_cfg['update_freq_min']

        self._tar_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, target_cfg['asset_filename'],
                                               marker_asset_option())

        return env_cfg

    def _post_physics(self, actions: torch.Tensor):
        super()._post_physics(actions)

        tar_update_id = torch.where(self._buf["elapsedStep"] >= self._buf["tarRemain"])[0]
        self._buf["elapsedStep"][tar_update_id] = 0
        print(tar_update_id)

        self._update_target(tar_update_id)

    def _update_target(self, env_ids: Optional[torch.Tensor] = None, skip_draw: bool = False):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._compute_device)

        num_update = len(env_ids)
        if num_update != 0:
            distance = ((self._tar_away_max - self._tar_away_min) *
                        torch.rand([num_update, 1], device=self._compute_device) +
                        self._tar_away_min)
            direction = torch.rand([num_update, 2], device=self._compute_device) * 2 - 1
            zero = torch.zeros([num_update, 1], device=self._compute_device)
            direction = torch.cat([direction, zero], dim=-1)

            self._buf["tarPos"][env_ids] = self._buf["aPos"][env_ids] + distance * direction
            self._buf["tarPos"][..., 2] = 0

            self._buf["tarRemain"] = torch.randint(low=self._tar_update_freq_min, high=self._tar_update_freq_max,
                                                   size=[self._num_envs], device=self._compute_device, dtype=torch.int32)

        if not self._headless and not skip_draw:
            self._draw_target(env_ids)

    def _reset_surroundings(self, env_ids: torch.Tensor):
        self._update_target(env_ids.squeeze())


@torch.jit.script
def location_reward(root_pos: torch.Tensor, tar_pos: torch.Tensor) -> torch.Tensor:
    root_pos_xy = root_pos[..., 0:2]
    tar_pos_xy = tar_pos[..., 0:2]
    return -torch.norm(root_pos_xy - tar_pos_xy, dim=-1)
