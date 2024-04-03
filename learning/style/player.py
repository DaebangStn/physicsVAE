import h5py
import numpy as np
import torch
from rl_games.algos_torch import torch_ext

from learning.core.player import CorePlayer
from learning.style.algorithm import style_task_obs_angle_transform, StyleAlgorithm
from utils.buffer import TensorHistoryFIFO


class StylePlayer(CorePlayer):
    def __init__(self, **kwargs):
        # env related
        self._key_body_ids = None
        self._dof_offsets = None
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None

        self._log_file = None

        self._checkpoint_disc = None
        super().__init__(**kwargs)

    def env_step(self, env, actions):
        if self._log_file is not None:
            self._log_action(actions)
        obs, rew, done, info = super().env_step(env, actions)
        obs, disc_obs = style_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}, rew, done, info

    def env_reset(self, env):
        obs = super().env_reset(env)
        obs, disc_obs = style_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}

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

        style_conf = kwargs['params']['hparam']['style']
        self._disc_obs_traj_len = style_conf['disc']['obs_traj_len']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
        collect_action = self.config.get('collect_action', None)

        if collect_action is not None:
            collect_filename = collect_action.get('filename')
            full_experiment_name = kwargs['params']['config']['full_experiment_name']
            self._log_file = h5py.File(collect_filename, 'a')

            if full_experiment_name in self._log_file:
                self._log_file = self._log_file[full_experiment_name]
            else:
                self._log_file = self._log_file.create_dataset(
                    full_experiment_name, shape=(0, self.actions_num), maxshape=(None, self.actions_num),
                    data=np.array([]), dtype='f4', chunks=True)

            print("==> Action log file: {:s}".format(collect_filename) +
                  " with index {:s}".format(full_experiment_name))

    def _disc_debug(self, disc_obs):
        with torch.no_grad():
            if self.normalize_input:
                disc_obs = self.model.norm_disc_obs(disc_obs)
            disc = self.model.disc(disc_obs)
            prob = 1 / (1 + torch.exp(-disc))
            reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))

            disc = torch.mean(disc).item()
            reward = torch.mean(reward).item()
        print(f"rollout_disc {disc:.3f} disc_reward {reward:.3f}")
        if self._games_played == 0:
            self._writer.add_scalar("player/rollout_disc", disc, self._n_step)
            self._writer.add_scalar("player/reward_disc", reward, self._n_step)

    def _log_action(self, action):
        action = action.cpu().numpy()
        new_size = self._log_file.shape[0] + action.shape[0]
        self._log_file.resize(new_size, axis=0)
        self._log_file[-action.shape[0]:] = action

    def _pre_step(self):
        self._disc_obs_buf.push_on_reset(self.obses['disc_obs'], self.dones)

    def _post_step(self):
        self._disc_obs_buf.push(self.obses['disc_obs'])
        self._disc_debug(self._disc_obs_buf.history)
