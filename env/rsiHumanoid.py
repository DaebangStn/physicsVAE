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
                                                  "Call set_motion_fetcher() in the algorithm routine first.")
        super()._assign_reset_state(env_ids)

        num_reset = len(env_ids)

        rand_indices = torch.randperm(num_reset)
        num_reference_reset = round(num_reset * self._rsi_prob)

        rsi_ids = env_ids[rand_indices[:num_reference_reset]].squeeze()

        if rsi_ids.ndim > 0 and len(rsi_ids) > 0:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = (
                self._motion_fetcher.fetch_snapshot(len(rsi_ids)))
            self._buf["aPos"][rsi_ids] = root_pos
            self._buf["aRot"][rsi_ids] = root_rot
            self._buf["aVel"][rsi_ids] = root_vel
            self._buf["aAnVel"][rsi_ids] = root_ang_vel
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[rsi_ids, :, 0] = dof_pos
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[rsi_ids, :, 1] = dof_vel

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        self._rsi_prob = env_cfg.get("reference_state_init_prob", 0.0)
        return env_cfg

    def set_motion_fetcher(self, motion_fetcher: MotionLibFetcher):
        self._motion_fetcher = motion_fetcher
