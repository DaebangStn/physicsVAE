import torch
from isaacgym import gymtorch

from env.vectask import VecTask
from utils import PROJECT_ROOT
from utils.env import *


class CartTask(VecTask):
    def __init__(self, **kwargs):
        self._env_spacing = None
        self._cart_asset_filename = None
        self._action_ofs = None
        self._action_scale = None
        self._max_episode_steps = None

        self._envs = []
        self._carts = []
        super().__init__(**kwargs)

        self.progress_buf = torch.zeros(self._num_envs, device=self._compute_device, dtype=torch.int32)
        self.num_envs = self._num_envs
        self.num_dof = 2
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        self._buf["dof"] = gymtorch.wrap_tensor(dof_state_tensor)
        self._buf["dofInit"] = torch.zeros_like(self._buf["dof"])
        self._buf["dPos"] = self._buf["dof"].view(self._num_envs, 2, 2)[..., 0]
        self._buf["dVel"] = self._buf["dof"].view(self._num_envs, 2, 2)[..., 1]

    def reset(self):
        env_ids = torch.nonzero(self._buf["reset"]).squeeze(-1)
        num_reset = len(env_ids)
        if num_reset > 0:
            positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self._compute_device) - 0.5)
            velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self._compute_device) - 0.5)

            self._buf["dPos"][env_ids, :] = positions[:]
            self._buf["dVel"][env_ids, :] = velocities[:]

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self._gym.set_dof_state_tensor_indexed(self._sim,
                                                   gymtorch.unwrap_tensor(self._buf["dof"]),
                                                   gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            self._buf["reset"][env_ids] = 0
            self.progress_buf[env_ids] = 0
        self._compute_observation()
        return self._buf["obs"]

    def _pre_physics(self, actions: torch.Tensor):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self._compute_device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self._compute_device).squeeze() * 400
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self._gym.set_dof_actuation_force_tensor(self._sim, forces)

    def _post_physics(self, actions: torch.Tensor):
        self.progress_buf += 1

        # self.reset()

        # compute rewards
        pole_angle = self._buf["obs"][:, 2]
        pole_vel = self._buf["obs"][:, 3]
        cart_vel = self._buf["obs"][:, 1]
        cart_pos = self._buf["obs"][:, 0]

        self._buf["rew"][:], self._buf["reset"][:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            3, self._buf["reset"], self.progress_buf, self._max_episode_steps
        )

    def _compute_observation(self):
        # compute observations
        self._gym.refresh_dof_state_tensor(self._sim)
        self._buf["obs"][:, 0] = self._buf["dPos"][:, 0].squeeze()
        self._buf["obs"][:, 1] = self._buf["dVel"][:, 0].squeeze()
        self._buf["obs"][:, 2] = self._buf["dPos"][:, 1].squeeze()
        self._buf["obs"][:, 3] = self._buf["dVel"][:, 1].squeeze()

    def _create_envs(self):
        cart_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, self._cart_asset_filename, cart_asset_option())

        num_rigid_body = self._gym.get_asset_rigid_body_count(cart_asset)
        num_shape = self._gym.get_asset_rigid_shape_count(cart_asset)
        self_collision = False

        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, *env_create_parameters(self._num_envs, self._env_spacing))
            self._envs.append(env)

            self._gym.begin_aggregate(env, num_rigid_body, num_shape, self_collision)
            cart = self._gym.create_actor(env, cart_asset, drop_transform(2), "cart", i, 0, 0)

            dof_props = self._gym.get_actor_dof_properties(env, cart)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self._gym.set_actor_dof_properties(env, cart, dof_props)

            self._carts.append(cart)
            self._gym.end_aggregate(env)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)

        self._env_spacing = env_cfg['spacing']
        self._max_episode_steps = env_cfg['max_episode_steps']
        self._cart_asset_filename = env_cfg['cart_asset_filename']

        return env_cfg


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
