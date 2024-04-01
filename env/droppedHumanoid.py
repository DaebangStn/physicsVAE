from math import ceil
from typing import Dict

import torch
from isaacgym import gymtorch

from env.humanoid import HumanoidTask


class DroppedHumanoid(HumanoidTask):
    def __init__(self, **kwargs):
        self._drop_on_reset_prob = None
        super().__init__(**kwargs)
        # self._build_dropped_state_tensor()

    # def reset(self) -> Dict[str, torch.Tensor]:
    #     env_ids = torch.nonzero(self._buf["reset"])
    #     num_reset = len(env_ids)
    #
    #     if num_reset > 0:  # TODO, refactor it as separate make_reset_state and do_reset
    #         rand_indices = torch.randperm(num_reset)
    #         num_drop = ceil(num_reset * self._drop_on_reset_prob)
    #
    #         drop_ids = env_ids[rand_indices[:num_drop]]
    #         normal_reset_ids = env_ids[rand_indices[num_drop:]]
    #
    #         if len(drop_ids) > 0:
    #             self._buf["actor"][drop_ids] = self._buf["actorDropped"][drop_ids]
    #             self._buf["dof"][drop_ids] = self._buf["dofDropped"][drop_ids]
    #         if len(normal_reset_ids) > 0:
    #             self._buf["actor"][normal_reset_ids] = self._buf["actorInit"][normal_reset_ids]
    #             self._buf["dof"][normal_reset_ids] = self._buf["dofInit"][normal_reset_ids]
    #
    #         # to prevent gc, this line is required
    #         _id = self._humanoids_id_sim[env_ids]
    #         id_gym = gymtorch.unwrap_tensor(_id)
    #         # reset actors
    #         self._gym.set_actor_root_state_tensor_indexed(
    #             self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), id_gym, len(env_ids))
    #         self._gym.set_dof_state_tensor_indexed(
    #             self._sim, gymtorch.unwrap_tensor(self._buf["dof"]), id_gym, len(env_ids))
    #
    #         self._buf["reset"][env_ids] = False
    #         self._buf["elapsedStep"][env_ids] = 0
    #         self._reset_surroundings(env_ids)
    #
    #     self._refresh_tensors()
    #     self._compute_observations()
    #
    #     return self._buf['obs']

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        self._drop_on_reset_prob = env_cfg.get("drop_on_reset_prob", 0.0)
        return env_cfg

    def _build_dropped_state_tensor(self):
        RELAXATION_STEPS = 100

        self._buf["aPos"][..., 2] += torch.rand(self.num, device=self._compute_device)
        self._buf["aRot"] = torch.rand_like(self._buf["aRot"]) * 0.2

        id_gym = gymtorch.unwrap_tensor(self._humanoids_id_sim)
        # reset actors
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), id_gym, self.num)
        self._gym.set_dof_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["dofInit"]), id_gym, self.num)

        self._pre_physics(torch.rand(self.num, self._num_actions) - 0.5)

        for _ in range(RELAXATION_STEPS):
            self._run_physics()

        self._refresh_tensors()

        self._buf["actorDropped"] = self._buf["actor"].clone()
        self._buf["actorDropped"][:, 7:] = 0
        self._buf["dofDropped"] = self._buf["dof"].clone()
        self._buf["dofDropped"][:, 1:] = 0
