import torch

from env.humanoidTask import HumanoidTask
from utils import *
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

    def _compute_reward(self):
        self._buf["rew"] = location_reward(self._buf["humanoidPos"], self._buf["taskPos"])

    def _create_env(self):
        super()._create_env()
        if not self._headless:
            self._task_id_env = torch.tensor(self._task_id_env, device=self._compute_device, dtype=torch.int32)
            self._task_id_sim = torch.tensor(self._task_id_sim, device=self._compute_device, dtype=torch.int32)

    def _draw_task(self, env_ids: Optional[torch.Tensor] = None):
        col = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._gym.clear_lines(self._viewer)
        lines = torch.cat([self._buf["taskPos"], self._buf["humanoidPose"]], dim=-1).cpu().numpy()

        for i in range(self.num):
            self._gym.add_lines(self._viewer, self._envs[i], 1, lines[i].reshape([1, 6]), col)

        if env_ids is None:
            env_ids = torch.arange(self.num, device=self._compute_device)

        num_update = len(env_ids)
        if num_update == 0:
            return

        _id = self._task_id_sim[env_ids]
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

    def _update_target(self, skip_draw: bool = False):
        env_ids = (self._buf["taskRemain"] == 0).nonzero(as_tuple=False).squeeze()

        num_update = len(env_ids)
        if num_update == 0:
            return

        distance = torch.rand(num_update, 3, device=self._compute_device) * 2 - 1
        distance[:, 2] = 0
        distance = distance / torch.norm(distance, dim=-1, keepdim=True)
        distance = distance * self._tar_away_scale + self._tar_away_ofs

        # self._buf["taskPos"][env_ids] = self._buf["humanoidPos"][env_ids] + distance
        self._buf["taskRemain"][env_ids] = torch.randint(self._task_up_freq_min, self._task_up_freq_max, (num_update,),
                                                         device=self._compute_device, dtype=torch.int32)

        if not skip_draw:
            self._draw_task(env_ids)


@torch.jit.script
def location_reward(humanoid_position: torch.Tensor, marker_position: torch.Tensor) -> torch.Tensor:
    a = 3.0
    b = 2.0
    distance = torch.square(humanoid_position[..., :2] - marker_position[..., :2]).sum(dim=-1)
    return a / (1 + b * distance)
