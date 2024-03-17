from typing import Dict
import torch
from isaacgym import gymtorch

from env.vectask import VecTask
from utils import *
from utils.env import *


class BalanceTask(VecTask):
    def __init__(self, **kwargs):
        self._env_spacing = None
        self._humanoid_asset_filename = None
        self._balance_asset_filename = None
        self._action_ofs = None
        self._action_scale = None
        self._max_episode_steps = None

        self._envs = []
        self._humanoids = []
        self._balances = []

        super().__init__(**kwargs)

        self._capture_init_state()

    def _capture_init_state(self):
        self._buf["rBodyInit"] = get_tensor_like_r_body_state(self._gym, self._sim, self._num_envs)
        self._buf["elapsedStep"] = np.zeros(self._num_envs, dtype=np.int16)
        self._buf["terminated"] = np.zeros(self._num_envs, dtype=np.bool_)

    def _create_envs(self):
        """VecTask didn't create any environments (just created ground in _parse_sim_param)
        So that child class must implement this method.
        Then this method is called by _create_sim

        :return:
            None
        """
        humanoid_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, self._humanoid_asset_filename,
                                              humanoid_asset_option())
        balance_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, self._balance_asset_filename,
                                             soft_asset_option())

        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, *env_create_parameters(self._num_envs, self._env_spacing))
            self._envs.append(env)

            humanoid = self._gym.create_actor(env, humanoid_asset, drop_transform(3.3), "humanoid", i, 0, 0)
            self._humanoids.append(humanoid)

            props = self._gym.get_actor_dof_properties(env, humanoid)
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)
            self._gym.set_actor_dof_properties(env, humanoid, props)

            balance = self._gym.create_actor(env, balance_asset, drop_transform(0.2), "balance", i, 0, 0)
            self._balances.append(balance)

        dof_prop = self._gym.get_actor_dof_properties(self._envs[0], self._humanoids[0])
        self._action_ofs = 0.5 * (dof_prop['upper'] + dof_prop['lower'])
        self._action_scale = 0.5 * (dof_prop['upper'] - dof_prop['lower'])

    def _compute_observations(self):
        self._buf['height'] = np.empty(self._num_envs)
        for i in range(self._num_envs):
            humanoid_r_body_z = self._gym.get_actor_rigid_body_states(
                self._envs[i], self._humanoids[i], gymapi.STATE_POS)['pose']['p']['z']
            mean_height = np.mean(humanoid_r_body_z)
            self._buf['height'][i] = mean_height

        #  dof and rigid body state is used for observation
        rBody = get_tensor_like_r_body_state(self._gym, self._sim)
        rBody = rBody.reshape(self._num_envs, -1)
        self._buf["obs"] = to_torch(rBody, device=self._compute_device)

    def _compute_reset(self):
        height_criteria = 1.8
        actor_down = self._buf['height'] < height_criteria
        self._buf["terminate"] = actor_down
        tooLongEpisode = self._buf["elapsedStep"] > self._max_episode_steps
        self._buf["reset"] = to_torch(tooLongEpisode | self._buf["terminate"], device=self._compute_device)

    def _compute_reward(self):
        reset_reward = -10000.0
        self._buf["rew"] = to_torch(np.where(self._buf["terminated"], reset_reward, self._buf["elapsedStep"]),
                                    device=self._compute_device)

    def _parse_env_param(self, **kwargs):
        env_cfg = super()._parse_env_param(**kwargs)

        self._env_spacing = env_cfg['spacing']
        self._humanoid_asset_filename = env_cfg['humanoid_asset_filename']
        self._balance_asset_filename = env_cfg['balance_asset_filename']
        self._max_episode_steps = env_cfg['max_episode_steps']

        return env_cfg

    def _pre_physics(self, actions: torch.Tensor):
        actions = actions.cpu().numpy()
        pd_target = actions * self._action_scale + self._action_ofs
        set_tensor_like_dof_pose_state(self._gym, self._envs, self._humanoids, pd_target)

    def _post_physics(self, actions: torch.Tensor):
        self._buf["elapsedStep"] += 1

        self._compute_observations()
        self._compute_reset()
        self._compute_reward()

    def reset(self) -> Dict[str, torch.Tensor]:
        env_ids = self._buf["reset"].cpu().numpy().astype(np.int32)

        if np.any(env_ids):
            current_state = get_tensor_like_r_body_state(self._gym, self._sim, self._num_envs)
            reset_applied = np.where(env_ids[:, None, None], self._buf["rBodyInit"], current_state)
            set_tensor_like_r_body_state(self._gym, self._sim, reset_applied, self._num_envs)

            self._buf["reset"][env_ids == 1] = False
            self._buf["elapsedStep"][env_ids == 1] = 0
            self._buf["rew"][env_ids == 1] = 0

        self._compute_observations()
        return self._buf['obs']
