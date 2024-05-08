from typing import Dict, Optional, Tuple

import torch
from isaacgym import gymtorch, gymapi

from env.vectask import VecTask
from utils import PROJECT_ROOT
from utils.env import *


class HumanoidTask(VecTask):
    def __init__(self, **kwargs):
        self._joint_info = None
        self._env_spacing = None
        self._humanoid_asset = None
        self._humanoid_head_rBody_id = None
        self._action_ofs = None
        self._action_scale = None
        self._max_episode_steps = None
        self._num_sensors = None
        self._recovery_limit = None
        self._viewer_follow_env0 = None
        self._contact_body_ids = None

        self._envs = []
        self._humanoids_id_env = []
        self._humanoids_id_sim = []
        self._num_humanoid_dof = None
        self._num_humanoid_rigid_body = None

        super().__init__(**kwargs)

        self._dof_per_env = None
        self._build_tensors()

    def render(self):
        if self._viewer_follow_env0:
            self._update_viewer_env0()
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
        self._buf["actorInit"] = self._buf["actor"].clone()

        self._dof_per_env = self._buf["dof"].shape[0] // self._num_envs
        dof_state_tensor_reshaped = ((self._buf["dof"].view(self._num_envs, self._dof_per_env, 2))
        [:, :self._num_humanoid_dof])
        # Caution! shape (num_envs, num_humanoid_dof, 2)
        self._buf["dofInit"] = torch.zeros_like(dof_state_tensor_reshaped)
        self._buf["dPos"] = dof_state_tensor_reshaped[..., 0]
        self._buf["dVel"] = dof_state_tensor_reshaped[..., 1]

        bodies_per_env = self._buf["rBody"].shape[0] // self._num_envs
        rigid_body_state_reshaped = (self._buf["rBody"].view(self._num_envs, bodies_per_env, 13)
        [:, :self._num_humanoid_rigid_body])
        self._buf["rPos"] = rigid_body_state_reshaped[..., 0:3]
        self._buf["rRot"] = rigid_body_state_reshaped[..., 3:7]
        self._buf["rVel"] = rigid_body_state_reshaped[..., 7:10]
        self._buf["rAnVel"] = rigid_body_state_reshaped[..., 10:13]
        self._buf["contact"] = self._buf["contact"].view(self.num, bodies_per_env, 3)[:, :self._num_humanoid_rigid_body]

        actors_per_env = self._buf["actor"].shape[0] // self._num_envs
        actor_root_reshaped = self._buf["actor"].view(self._num_envs, actors_per_env, 13)[:, 0]
        self._buf["aPos"] = actor_root_reshaped[..., 0:3]
        self._buf["aRot"] = actor_root_reshaped[..., 3:7]
        self._buf["aVel"] = actor_root_reshaped[..., 7:10]
        self._buf["aAnVel"] = actor_root_reshaped[..., 10:13]

        self._buf["elapsedStep"] = torch.zeros(self._num_envs, dtype=torch.int16, device=self._compute_device)
        self._buf["recoveryCounter"] = torch.zeros(self._num_envs, dtype=torch.int16, device=self._compute_device)
        # stands for reset by constraints (height, foot contact off...)
        self._buf["terminate"] = torch.zeros(self._num_envs, dtype=torch.bool, device=self._compute_device)

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
        self._contact_body_ids = torch.tensor(create_sensors(self._gym, self._humanoid_asset, sensor_install_sight),
                                              device=self._compute_device)
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

        self.num_dof = self._gym.get_asset_dof_count(self._humanoid_asset)
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self._dof_offsets = self._joint_info["dof_offsets"]
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self._compute_device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self._compute_device)

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()
        num_joints = len(self._dof_offsets) - 1

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if dof_size == 3:
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._action_ofs = to_torch(0.5 * (lim_high + lim_low), device=self._compute_device)
        self._action_scale = to_torch(0.5 * (lim_high - lim_low), device=self._compute_device)

    def _compute_aggregate_option(self) -> Tuple[int, int]:
        return self._num_humanoid_rigid_body, self._gym.get_asset_rigid_shape_count(self._humanoid_asset)

    def _build_env(self, env, env_id):
        humanoid = self._gym.create_actor(env, self._humanoid_asset, drop_transform(0.89), "humanoid", env_id, 0, 0)
        self._humanoids_id_env.append(humanoid)
        self._humanoids_id_sim.append(self._gym.get_actor_index(env, humanoid, gymapi.DOMAIN_SIM))

        dof_prop = self._gym.get_asset_dof_properties(self._humanoid_asset)
        dof_prop["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_prop["stiffness"] = dof_prop["stiffness"] * self._stiffness_coef
        dof_prop["damping"] = dof_prop["damping"] * self._damping_coef
        self._gym.set_actor_dof_properties(env, humanoid, dof_prop)

    def _compute_observations(self):
        #  dof and actor root state is used for observation
        self._buf["obs"] = torch.cat([self._buf["dPos"], self._buf["dVel"], self._buf["actor"]], dim=-1)

    # def _compute_reset(self):
    #     height_criteria = 0.8
    #     force_criteria = 1.0
    #
    #     actor_height = self._buf["rPos"][:, self._humanoid_head_rBody_id, 2]
    #     actor_down = actor_height < height_criteria
    #     self._buf["recoveryCounter"] = torch.where(actor_down, self._buf["recoveryCounter"] + 1, 0)
    #     self._buf["terminate"] = self._buf["recoveryCounter"] > self._recovery_limit
    #     # contact_off = (self._buf["sensor"] ** 2).sum(dim=1) < force_criteria
    #     # self._buf["terminate"] = actor_down  # & contact_off
    #     tooLongEpisode = self._buf["elapsedStep"] > self._max_episode_steps
    #     self._buf["reset"] = tooLongEpisode | self._buf["terminate"]

    # # AMP version reset
    def _compute_reset(self):
        self._buf["reset"][:], self._buf["terminate"][:] = \
            compute_amp_humanoid_reset(self._buf["reset"], self._buf["elapsedStep"], self._buf["contact"],
                                       self._contact_body_ids, self._buf["rPos"], self._max_episode_steps, 0.15)
        self._buf["info"]["terminate"] = self._buf["terminate"]

    def _compute_reward(self):
        reset_reward = -100.0
        # actor_height = self._buf["rPos"][:, self._humanoid_head_rBody_id, 2]
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
        self._joint_info = env_cfg["joint_information"]
        self._recovery_limit = env_cfg.get("recovery_limit", 0)
        self._viewer_follow_env0 = env_cfg.get("viewer_follow_env0", False)

        self._stiffness_coef = env_cfg.get("stiffness_coef", 1.0)
        self._damping_coef = env_cfg.get("damping_coef", 1.0)

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

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.nonzero(self._buf["reset"]).long()

        if env_ids.ndim > 0 and len(env_ids) > 0:
            self._assign_reset_state(env_ids)
            self._apply_reset_state(env_ids)
            self._buf["reset"][env_ids] = False
            self._buf["elapsedStep"][env_ids] = 0
            self._buf["recoveryCounter"][env_ids] = 0

        self._refresh_tensors()
        self._compute_observations()

        return self._buf['obs']

    def _assign_reset_state(self, env_ids: torch.Tensor):
        self._buf["dof"].view(self.num, self._dof_per_env, 2)[env_ids] = self._buf["dofInit"][env_ids].clone()
        self._buf["actor"][env_ids] = self._buf["actorInit"][env_ids].clone()

    def _apply_reset_state(self, env_ids: torch.Tensor):
        # to prevent gc, this line is required
        _id = self._humanoids_id_sim[env_ids]
        id_gym = gymtorch.unwrap_tensor(_id)
        # reset actors
        self._gym.set_actor_root_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["actor"]), id_gym, len(env_ids))
        self._gym.set_dof_state_tensor_indexed(
            self._sim, gymtorch.unwrap_tensor(self._buf["dof"]), id_gym, len(env_ids))

    def _reset_surroundings(self, env_ids: torch.Tensor):
        pass

    def _update_viewer_env0(self):
        if self._viewer is None:
            return

        cam_trans = self._gym.get_viewer_camera_transform(self._viewer, None)
        env0_root_pos = self._buf["aPos"][0, 0:3]
        cam_target = gymapi.Vec3(env0_root_pos[0], env0_root_pos[1], env0_root_pos[2])
        displace = cam_trans.transform_vector(gymapi.Vec3(0, 0, 5))
        displace = gymapi.Vec3(abs(displace.x), abs(displace.y), abs(displace.z))
        cam_pos = cam_target + displace
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)


@torch.jit.script
def compute_amp_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                               max_episode_length, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    masked_contact_buf = contact_buf.clone()
    masked_contact_buf[:, contact_body_ids, :] = 0
    fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
    fall_contact = torch.any(fall_contact, dim=-1)

    body_height = rigid_body_pos[..., 2]
    fall_height = body_height < termination_heights
    fall_height[:, contact_body_ids] = False
    fall_height = torch.any(fall_height, dim=-1)

    has_fallen = torch.logical_and(fall_contact, fall_height)

    # first timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    has_fallen *= (progress_buf > 1)
    terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
