import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz

from env.keypointTask import KeypointTask


class NoPhyKeypointTask(KeypointTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_sim_param(self, **kwargs):
        cfg = super()._parse_sim_param(**kwargs)
        self._sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        return cfg

    def _pre_physics(self, actions: torch.Tensor):
        dof = actions[:, :28]
        root_state = actions[:, 28:]
        self._buf["actor"].copy_(self._process_r_body_state(root_state))
        actions = dof * self._action_scale + self._action_ofs
        self._buf["dPos"].copy_(actions)
        self._buf["dVel"].copy_(torch.zeros_like(actions))
        _action = gymtorch.unwrap_tensor(self._buf["dof"])
        actor_ids = torch.arange(self._num_envs, device=self._compute_device, dtype=torch.int32)
        _actor_ids = gymtorch.unwrap_tensor(actor_ids)

        self._gym.set_actor_root_state_tensor_indexed(self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), _actor_ids, len(actor_ids))
        self._gym.set_dof_state_tensor_indexed(self._sim, _action, _actor_ids, len(actor_ids))

    @staticmethod
    def _process_r_body_state(root_state):
        pos = root_state[:, :3] * 10
        pos[:, 2] = abs(pos[:, 2])
        rot = root_state[:, 3:6]
        rot = quat_from_euler_xyz(rot[:, 0], rot[:, 1], rot[:, 2])
        vel = root_state[:, 6:9]
        ang_vel = root_state[:, 9:]
        return torch.cat([pos, rot, vel, ang_vel], dim=-1)
