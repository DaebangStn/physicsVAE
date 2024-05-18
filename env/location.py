from env.humanoid import compute_height_humanoid_redset
from env.humanoidTask import HumanoidTask
from utils import *
from utils.angle import *
from utils.env import drop_transform


class LocationTask(HumanoidTask):
    def __init__(self, **kwargs):
        self._task_id_env = []
        self._task_id_sim = []
        self._tar_away_scale = None
        self._tar_away_ofs = None
        super().__init__(**kwargs)

    def _build_env(self, env, env_id):
        super()._build_env(env, env_id)

        target = self._gym.create_actor(env, self._task_asset, drop_transform(0), "target", env_id, 1, 0)
        self._gym.set_rigid_body_color(env, target, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))
        self._task_id_env.append(target)
        self._task_id_sim.append(self._gym.get_actor_index(env, target, gymapi.DOMAIN_SIM))

    def _build_tensors(self):
        super()._build_tensors()
        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        target_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 1]
        self._buf["taskPos"] = target_root_reshaped[..., 0:3]
        self._buf["taskQuat"] = target_root_reshaped[..., 3:7]

    def _compute_observations(self):
        super()._compute_observations()
        (self._buf["obs"])["goal"] = location_goal(self._buf["aPos"], self._buf["aRot"], self._buf["taskPos"])

    def _compute_reward(self):
        self._buf["rew"] = location_reward(self._buf["aPos"], self._buf["taskPos"], self._buf["terminate"])

    def _create_env(self):
        super()._create_env()
        if not self._headless:
            self._task_id_env = torch.tensor(self._task_id_env, device=self._compute_device, dtype=torch.long)
            self._task_id_sim = torch.tensor(self._task_id_sim, device=self._compute_device, dtype=torch.long)


    def _draw_task(self, env_ids: Optional[torch.Tensor] = None):
        col = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._gym.clear_lines(self._viewer)
        lines = torch.cat([self._buf["taskPos"], self._buf["aPos"]], dim=-1).cpu().numpy()

        for i in range(self.num):
            self._gym.add_lines(self._viewer, self._envs[i], 1, lines[i].reshape([1, 6]), col)

        if env_ids is None:
            env_ids = torch.arange(self.num, device=self._compute_device)

        num_update = len(env_ids)
        if num_update == 0:
            return

        _id = self._task_id_sim[env_ids].to(torch.int32)
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), gymtorch.unwrap_tensor(_id), num_update)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)
        task_cfg = env_cfg['task']
        away_max = task_cfg['away_max']
        away_min = task_cfg['away_min']
        self._tar_away_scale = (away_max - away_min) / 2
        self._tar_away_ofs = (away_max + away_min) / 2
        return env_cfg

    def _compute_reset(self):
        height_min = 0.8
        height_max = 1.5

        actor_height = self._buf["rPos"][:, self._humanoid_head_rBody_id, 2]
        self._buf['reset'][:], self._buf['terminate'][:] = compute_height_humanoid_redset(
            self._buf['elapsedStep'], actor_height, height_max, height_min, self._max_episode_steps)

    def _update_target(self, skip_draw: bool = False):
        self._buf["taskRemain"] -= 1
        env_ids = (self._buf["taskRemain"] <= 0).nonzero(as_tuple=False).flatten()

        num_update = len(env_ids)
        if num_update != 0:
            tar_length = 2 * torch.rand(num_update, device=self._compute_device) - 1
            tar_length = tar_length * self._tar_away_scale + self._tar_away_ofs
            tar_theta = 2 * np.pi * torch.rand(num_update, device=self._compute_device)
            distance = torch.stack([tar_length * torch.cos(tar_theta), tar_length * torch.sin(tar_theta),
                                    torch.zeros(num_update, device=self._compute_device)], dim=-1)

            self._buf["taskPos"][env_ids] = self._buf["aPos"][env_ids] + distance
            self._buf["taskPos"][env_ids, 2] = 0
            self._buf["taskRemain"][env_ids] = (
                torch.randint(self._task_up_freq_min, self._task_up_freq_max, (num_update,),
                              device=self._compute_device, dtype=torch.int32))

        if not (skip_draw or self._headless):
            self._draw_task(env_ids)


@torch.jit.script
def location_reward(humanoid_position: torch.Tensor, marker_position: torch.Tensor, terminated: torch.Tensor
                    ) -> torch.Tensor:
    a = 0.7
    b = 0.5
    distance = torch.square(humanoid_position[..., :2] - marker_position[..., :2]).sum(dim=-1)
    reward = a * torch.exp(-b * (distance ** 2))
    return torch.where(terminated, -500.0, reward)


@torch.jit.script
def location_goal(humanoid_position: torch.Tensor, humanoid_rotation: torch.Tensor, marker_position: torch.Tensor
                  ) -> torch.Tensor:
    heading_rot_inv = calc_heading_quat_inv(humanoid_rotation)
    position_dif = marker_position - humanoid_position
    local_dif = quat_rotate(heading_rot_inv, position_dif)
    return local_dif[:, :2]
