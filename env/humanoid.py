from typing import Dict, Tuple
from isaacgym import gymtorch


from env.vectask import VecTask
from utils import PROJECT_ROOT
from utils.env import *


class HumanoidTask(VecTask):
    def __init__(self, **kwargs):
        self._env_spacing = None
        self._humanoid_asset = None
        self._humanoid_head_rBody_id = None
        self._action_ofs = None
        self._action_scale = None
        self._max_episode_steps = None
        self._num_sensors = None

        self._envs = []
        self._humanoids_id_env = []
        self._humanoids_id_sim = []
        self._num_humanoid_dof = None
        self._num_humanoid_rigid_body = None

        super().__init__(**kwargs)

        self._build_tensors()

    def render(self):
        super().render()

    def _build_tensors(self):
        # get gym GPU state tensors
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        sensor_tensor = self._gym.acquire_force_sensor_tensor(self._sim)
        contact_force_tensor = self._gym.acquire_net_contact_force_tensor(self._sim)

        self._refresh_tensors()

        # Torch version for GPU state tensors.
        self._buf["dof"] = gymtorch.wrap_tensor(dof_state_tensor)
        self._buf["actor"] = gymtorch.wrap_tensor(actor_root_state)
        self._buf["rBody"] = gymtorch.wrap_tensor(rigid_body_state)
        if gymtorch.wrap_tensor(sensor_tensor) is not None:
            self._buf["sensor"] = gymtorch.wrap_tensor(sensor_tensor).view(self._num_envs, 6 * self._num_sensors)
        self._buf["contact"] = gymtorch.wrap_tensor(contact_force_tensor)

        # Since there could be other actor in the environment, we need to truncate tensors to match the humanoid size
        self._buf["dofInit"] = torch.zeros_like(self._buf["dof"])
        self._buf["actorInit"] = self._buf["actor"].clone()

        dof_per_env = self._buf["dof"].shape[0] // self._num_envs
        dof_state_tensor_reshaped = self._buf["dof"].view(self._num_envs, dof_per_env, 2)[:, :self._num_humanoid_dof]
        self._buf["dPos"] = dof_state_tensor_reshaped[..., 0]
        self._buf["dVel"] = dof_state_tensor_reshaped[..., 1]

        bodies_per_env = self._buf["rBody"].shape[0] // self._num_envs
        rigid_body_state_reshaped = (self._buf["rBody"].view(self._num_envs, bodies_per_env, 13)
                                     [:, :self._num_humanoid_rigid_body])
        self._buf["rPos"] = rigid_body_state_reshaped[..., 0:3]
        self._buf["rRot"] = rigid_body_state_reshaped[..., 3:7]
        self._buf["rVel"] = rigid_body_state_reshaped[..., 7:10]
        self._buf["rAnVel"] = rigid_body_state_reshaped[..., 10:13]

        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        actor_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 0]
        self._buf["aPos"] = actor_root_reshaped[..., 0:3]
        self._buf["aRot"] = actor_root_reshaped[..., 3:7]
        self._buf["aVel"] = actor_root_reshaped[..., 7:10]
        self._buf["aAnVel"] = actor_root_reshaped[..., 10:13]

        self._buf["elapsedStep"] = torch.zeros(self._num_envs, dtype=torch.int16, device=self._compute_device)
        # stands for reset by constraints (height, foot contact off...)
        self._buf["terminated"] = torch.zeros(self._num_envs, dtype=torch.bool, device=self._compute_device)

    def _create_envs(self):
        """VecTask didn't create any environments (just created ground in _parse_sim_param)
        So that child class must implement this method.
        Then this method is called by _create_sim

        :return:
            None
        """
        num_rigid_body, num_shape = self._compute_aggregate_option()
        self_collision = True

        sensor_install_sight = ["right_foot", "left_foot"]
        create_sensors(self._gym, self._humanoid_asset, sensor_install_sight)
        self._num_sensors = len(sensor_install_sight)

        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, *env_create_parameters(self._num_envs, self._env_spacing))
            self._envs.append(env)

            self._gym.begin_aggregate(env, num_rigid_body, num_shape, self_collision)
            self._build_env(env, i)
            self._gym.end_aggregate(env)

        self._humanoids_id_env = torch.tensor(self._humanoids_id_env, device=self._compute_device, dtype=torch.int32)
        self._humanoids_id_sim = torch.tensor(self._humanoids_id_sim, device=self._compute_device, dtype=torch.int32)
        self._humanoid_head_rBody_id = (
            self._gym.find_actor_rigid_body_index(self._envs[0], self._humanoids_id_env[0], "head", gymapi.DOMAIN_ENV))

        dof_prop = self._gym.get_actor_dof_properties(self._envs[0], self._humanoids_id_env[0])
        self._action_ofs = to_torch(0.5 * (dof_prop['upper'] + dof_prop['lower']), device=self._compute_device)
        self._action_scale = to_torch(0.5 * (dof_prop['upper'] - dof_prop['lower']), device=self._compute_device)

    def _build_env(self, env, env_id):
        humanoid = self._gym.create_actor(env, self._humanoid_asset, drop_transform(0.89), "humanoid", env_id, 0, 0)
        self._humanoids_id_env.append(humanoid)
        self._humanoids_id_sim.append(self._gym.get_actor_index(env, humanoid, gymapi.DOMAIN_SIM))

    def _compute_aggregate_option(self) -> Tuple[int, int]:
        return self._num_humanoid_rigid_body, self._gym.get_asset_rigid_shape_count(self._humanoid_asset)

    def _compute_observations(self):
        #  dof and actor root state is used for observation
        self._buf["obs"] = torch.cat([self._buf["dPos"], self._buf["dVel"], self._buf["actor"]], dim=-1)

    def _compute_reset(self):
        height_criteria = 0.5
        force_criteria = 1.0

        actor_height = self._buf["rPos"][:, self._humanoid_head_rBody_id, 2]
        actor_down = actor_height < height_criteria
        # contact_off = (self._buf["sensor"] ** 2).sum(dim=1) < force_criteria
        self._buf["terminate"] = actor_down  # & contact_off
        tooLongEpisode = self._buf["elapsedStep"] > self._max_episode_steps
        self._buf["reset"] = tooLongEpisode | self._buf["terminate"]

    def _compute_reward(self):
        reset_reward = -100.0
        actor_height = self._buf["rPos"][:, self._humanoid_head_rBody_id, 2]
        self._buf["rew"] = torch.where(self._buf["terminate"],
                                       reset_reward,
                                       self._buf["elapsedStep"] * 0.01)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)

        self._env_spacing = env_cfg['spacing']
        self._max_episode_steps = env_cfg['max_episode_steps']
        self._humanoid_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, env_cfg['humanoid_asset_filename'],
                                                    humanoid_asset_option())
        self._num_humanoid_rigid_body = self._gym.get_asset_rigid_body_count(self._humanoid_asset)
        self._num_humanoid_dof = self._gym.get_asset_dof_count(self._humanoid_asset)

        return env_cfg

    def _pre_physics(self, actions: torch.Tensor):
        actions = actions.to(self._compute_device).clone()
        pd_target = gymtorch.unwrap_tensor(actions * self._action_scale + self._action_ofs)
        self._gym.set_dof_position_target_tensor(self._sim, pd_target)

    def _post_physics(self, actions: torch.Tensor):
        """Order of the function is critical for the correct computation

        :param actions:
        :return:
        """
        self._buf["elapsedStep"] += 1

        self._refresh_tensors()
        self._compute_observations()
        self._compute_reset()
        self._compute_reward()

    def reset(self) -> Dict[str, torch.Tensor]:
        env_ids = torch.nonzero(self._buf["reset"]).long()

        if len(env_ids) > 0:
            # to prevent gc, this line is required
            _id = self._humanoids_id_sim[env_ids]
            id_gym = gymtorch.unwrap_tensor(_id)
            # reset actors
            self._gym.set_actor_root_state_tensor_indexed(
                self._sim, gymtorch.unwrap_tensor(self._buf["actorInit"]), id_gym, len(env_ids))
            self._gym.set_dof_state_tensor_indexed(
                self._sim, gymtorch.unwrap_tensor(self._buf["dofInit"]), id_gym, len(env_ids))

            self._buf["reset"][env_ids] = False
            self._buf["elapsedStep"][env_ids] = 0
            self._reset_surroundings(env_ids)

        self._refresh_tensors()
        self._compute_observations()

        return self._buf['obs']

    def _reset_surroundings(self, env_ids: torch.Tensor):
        pass
