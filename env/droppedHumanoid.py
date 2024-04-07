import torch
from isaacgym import gymtorch

from env.rsiHumanoid import RSIHumanoidTask


class DroppedHumanoidTask(RSIHumanoidTask):
    def __init__(self, **kwargs):
        self._drop_on_reset_prob = None
        super().__init__(**kwargs)
        self._build_dropped_state_tensor()

    def _assign_reset_state(self, env_ids: torch.Tensor):
        super()._assign_reset_state(env_ids)

        num_reset = len(env_ids)

        rand_indices = torch.randperm(num_reset)
        num_drop = round(num_reset * self._drop_on_reset_prob)

        drop_ids = env_ids[rand_indices[:num_drop]]

        if drop_ids.ndim > 0 and len(drop_ids) > 0:
            self._buf["actor"][drop_ids] = self._buf["actorDropped"][drop_ids]
            self._buf["dof"].view(self.num, self._dof_per_env, 2)[drop_ids] = self._buf["dofDropped"][drop_ids].clone()

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        self._drop_on_reset_prob = env_cfg.get("drop_on_reset_prob", 0.0)
        return env_cfg

    def _build_dropped_state_tensor(self):
        RELAXATION_STEPS = 150

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
        self._buf["dofDropped"] = self._buf["dof"].clone().view(self.num, self._dof_per_env, 2)
        self._buf["dofDropped"][:, :, 1:] = 0
