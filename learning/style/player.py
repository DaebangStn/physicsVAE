import torch
from rl_games.algos_torch import torch_ext

from learning.core.player import CorePlayer
from learning.style.algorithm import style_task_obs_angle_transform
from utils.buffer import TensorHistoryFIFO


class StylePlayer(CorePlayer):
    def __init__(self, **kwargs):
        # env related
        self._key_body_ids = None
        self._dof_offsets = None
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None

        self._checkpoint_disc = None
        super().__init__(**kwargs)

    def env_step(self, env, actions):
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
        self._key_body_ids = self._find_key_body_ids(algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']

        style_conf = kwargs['params']['hparam']['style']
        self._disc_obs_traj_len = style_conf['disc']['obs_traj_len']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

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

    def _find_key_body_ids(self, key_body_names):
        return self.env.key_body_ids(key_body_names)

    def _pre_step(self):
        self._disc_obs_buf.push_on_reset(self.obses['disc_obs'], self.dones)

    def _post_step(self):
        self._disc_obs_buf.push(self.obses['disc_obs'])
        self._disc_debug(self._disc_obs_buf.history)
