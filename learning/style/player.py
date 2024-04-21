from typing import Tuple, List

import h5py
import numpy as np
import torch
from rl_games.algos_torch import torch_ext

from learning.core.player import CorePlayer
from learning.style.algorithm import keyp_task_obs_angle_transform, StyleAlgorithm, disc_reward, obs_transform
from learning.logger.action import ActionLogger
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher


class StylePlayer(CorePlayer):
    def __init__(self, **kwargs):
        # env related
        self._key_body_ids = None
        self._dof_offsets = None
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None

        self._action_logger = None
        self._show_reward = None

        self._checkpoint_disc = None
        super().__init__(**kwargs)

    def env_step(self, env, actions):
        if self._action_logger is not None:
            self._action_logger.log(actions)
        obs, rew, done, info = super().env_step(env, actions)
        if self._show_reward:
            obs, disc_obs = keyp_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
            return {'obs': obs, 'disc_obs': disc_obs}, rew, done, info
        else:
            return {'obs': keyp_task_concat_obs(obs['obs'])}, rew, done, info

    def env_reset(self, env):
        obs = super().env_reset(env)
        if self._show_reward:
            obs, disc_obs = keyp_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
            return {'obs': obs, 'disc_obs': disc_obs}
        else:
            return {'obs': keyp_task_concat_obs(obs['obs'])}

    def restore(self, fn):
        super().restore(fn)
        if self._checkpoint_disc is not None:
            self._checkpoint_disc = torch_ext.load_checkpoint(self._checkpoint_disc)
            self.model.disc_load_state_dict(self._checkpoint_disc['model'])
            if self.normalize_input:
                self.model.disc_running_mean_load_state_dict(self._checkpoint_disc['model'])

    def _init_variables(self, **kwargs):
        super()._init_variables(**kwargs)

        self._checkpoint_disc = kwargs['params'].get('checkpoint_disc', None)

        algo_conf = kwargs['params']['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(
            self.env, algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']

        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self._demo_fetcher = MotionLibFetcher(self._disc_obs_traj_len, self.env.dt, self.device,
                                                  algo_conf['motion_file'],
                                                  algo_conf['joint_information']['dof_body_ids'],
                                                  self._dof_offsets, self._key_body_ids)
            self.env.set_motion_fetcher(self._demo_fetcher)

        self._show_reward = self.config['reward'].get('show_on_player', False)

        style_conf = self.config['style']
        self._disc_obs_traj_len = style_conf['disc']['obs_traj_len']
        if self._show_reward:
            self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            log_action = logger_config.get('action', False)
            if log_action:
                full_experiment_name = kwargs['params']['config']['full_experiment_name']
                self._action_logger = ActionLogger(logger_config['filename'], full_experiment_name, self.actions_num)

    def _disc_debug(self, disc_obs):
        reward = disc_reward(self.model, disc_obs, self.normalize_input, self.device).mean().item()
        print(f"disc_reward {reward:.3f}")
        if self._games_played == 0:
            self._writer.add_scalar("player/reward_disc", reward, self._n_step)

    def _pre_step(self):
        if self._show_reward:
            self._disc_obs_buf.push_on_reset(self.obses['disc_obs'], self.dones)

    def _post_step(self):
        if self._show_reward:
            self._disc_obs_buf.push(self.obses['disc_obs'])
            self._disc_debug(self._disc_obs_buf.history)


@torch.jit.script
def keyp_task_concat_obs(
        obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs
    obs = obs_transform(rPos, rRot, rVel, rAnVel)
    return obs
