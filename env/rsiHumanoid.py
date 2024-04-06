from math import ceil

import torch
from env.humanoid import HumanoidTask
from utils.buffer import MotionLibFetcher


class RSIHumanoidTask(HumanoidTask):
    """Humanoid task with reference state initialization(RSI)."""
    def __init__(self, **kwargs):
        self._rsi_prob = None
        self._motion_fetcher = None
        super().__init__(**kwargs)

    def _assign_reset_state(self, env_ids: torch.Tensor):
        assert self._motion_fetcher is not None, ("Motion library is not set by algorithm. "
                                                  "Call set_motion_lib() in the algorithm routine first.")
        num_reset = len(env_ids)

        rand_indices = torch.randperm(num_reset)
        num_reference_reset = ceil(num_reset * self._rsi_prob)

        rsi_ids = env_ids[rand_indices[:num_reference_reset]].squeeze()
        normal_reset_ids = env_ids[rand_indices[num_reference_reset:]].squeeze()

        if rsi_ids.ndim > 0 and len(rsi_ids) > 0:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
                self._motion_fetcher.fetch_snapshot(len(rsi_ids)))
            self._buf["actor"][rsi_ids, 0:3] = root_pos
            self._buf["actor"][rsi_ids, 3:7] = root_rot
            self._buf["actor"][rsi_ids, 7:10] = root_vel
            self._buf["actor"][rsi_ids, 10:13] = root_ang_vel
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[rsi_ids, :, 0] = dof_pos
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[rsi_ids, :, 1] = dof_vel
        if normal_reset_ids.ndim > 0 and len(normal_reset_ids) > 0:
            self._buf["actor"][normal_reset_ids] = self._buf["actorInit"][normal_reset_ids]
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[normal_reset_ids] = (
                self._buf["dofInit"][normal_reset_ids].clone())

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        self._rsi_prob = env_cfg.get("reference_state_init_prob", 0.0)
        return env_cfg

    def set_motion_fetcher(self, motion_fetcher: MotionLibFetcher):
        self._motion_fetcher = motion_fetcher
