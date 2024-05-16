from typing import Optional

import yaml
import torch
from rl_games.algos_torch import model_builder

from learning.core.algorithm import CoreAlgorithm
from learning.style.algorithm import StyleAlgorithm, keyp_task_obs_angle_transform, disc_reward
from utils import *
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher


class HighLevelAlgorithm(CoreAlgorithm):
    def __init__(self, **kwargs):
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._demo_fetcher = None

        # env related
        self._key_body_ids = None
        self._dof_offsets = None

        # low-level controller
        self._latent_dim = None
        self._llc = None
        self._llc_steps = None
        self._config_llc = None

        # reward
        self._disc_rew_scale = None

        # placeholders for the current episode
        self._disc_obs_exp_buffer = None
        self._

        super().__init__(**kwargs)

    def env_step(self, actions):
        # 1. parameter for env_reset makes hard reset,
        #    because if the reset is done immediately, idle loop makes error
        # 2. fix the done flow in the env_step loop
        # 3. calculate the disc_reward inside the loop

        z = torch.nn.functional.normalize(actions, dim=-1)

        rew_step = torch.zeros(self.vec_env.num, device=self.device).unsqueeze(1)
        disc_rew_step = torch.zeros(self.vec_env.num, device=self.device).unsqueeze(1)
        terminate = torch.zeros(self.vec_env.num, device=self.device, dtype=torch.bool)
        obs_step = None
        for i in range(self._llc_steps):
            obs_step = self.env_reset(self.dones.nonzero()[:, 0])

            self._disc_obs_buf.push_on_reset(obs_step['disc_obs'], self.dones)

            normed_obs = self._llc.norm_obs(obs_step['obs'])
            llc_action, _ = self._llc.actor_latent(normed_obs, z)
            obs, rew, done, info = super().env_step(llc_action)
            obs, disc_obs = keyp_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
            self._disc_obs_buf.push(disc_obs)
            obs_step = {'obs': obs}

            disc_rew = disc_reward(self._llc, self._disc_obs_buf.history)
            self._disc_obs_exp_buffer[i, :, :] = self._disc_obs_buf.history

            rew_step[~done] += rew[~done]
            disc_rew_step[~done] += disc_rew[~done]

            if i == 0:
                self.dones = done
            else:
                self.dones = self.dones | done
            terminate = terminate | info['terminate']

        rew = rew_step * self._task_rew_scale + disc_rew_step * self._disc_rew_scale

        return obs_step, rew, self.dones, {'terminate': terminate}

    def env_reset(self, env_ids: Optional[torch.Tensor] = None):
        obs = self.vec_env.reset(env_ids)
        obs, disc_obs = keyp_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}

    def _init_learning_variables(self, **kwargs):
        config_hparam = self.config
        self._llc, self._config_llc = self.build_llc(config_hparam, self.device)
        self._llc_steps = config_hparam['llc']['steps']

        self._latent_dim = self._config_llc['network']['space']['latent_dim']
        kwargs['params']['config']['env_config']['env']['num_act'] = self._latent_dim
        self.actions_num = self._latent_dim
        self.env_info['action_space'] = reshape_gym_box(self.env_info['action_space'], (self.actions_num,))
        super()._init_learning_variables(**kwargs)

        config_disc = self._config_llc['hparam']['style']['disc']
        self._disc_obs_traj_len = config_disc['obs_traj_len']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
        disc_obs_size = self._disc_obs_traj_len * config_disc['num_obs']
        self._disc_obs_exp_buffer = torch.empty(self._llc_steps, self.vec_env.num, disc_obs_size, device=self.device)

        self._disc_rew_scale = config_hparam['reward']['disc_scale']
        self._task_rew_w = 1.0

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)

        llc_algo_conf = self._config_llc['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(
            self.vec_env, llc_algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = llc_algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(
            self._disc_obs_traj_len, self.vec_env.dt, self.device, llc_algo_conf['motion_file'],
            llc_algo_conf['joint_information']['dof_body_ids'], self._dof_offsets, self._key_body_ids)
        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self.vec_env.set_motion_fetcher(self._demo_fetcher)

    @staticmethod
    def build_llc(config_hparam, device):
        # load config
        config_env = config_hparam['env_config']['env']
        with open(config_hparam['llc']['cfg'], 'r') as f:
            config_llc = yaml.safe_load(f)

        # check config(env) between HLC and LLC is consistent
        config_llc_env = config_llc['config']['env_config']['env']
        assert config_env['num_act'] == config_llc_env['num_act'], \
            f"Inconsistent action space. HLC: {config_env['num_act']} LLC: {config_llc_env['num_act']}"
        assert config_env['num_obs'] == config_llc_env['num_obs'], \
            f"Inconsistent observation space. HLC: {config_env['num_obs']} LLC: {config_llc_env['num_obs']}"

        # Override Env# with HLC config
        config_llc_env['num_envs'] = config_env['num_envs']
        config_llc['config']['num_actors'] = config_env['num_envs']

        # build model
        builder = model_builder.ModelBuilder()
        model = builder.load(config_llc)

        additional_config = {
            'actions_num': config_env['num_act'],
            'input_shape': (config_env['num_obs'],),
            'num_seqs': config_hparam['num_actors'],
            'value_size': config_hparam.get('value_size', 1),
            'normalize_value': config_hparam['normalize_value'],
            'normalize_input': config_hparam['normalize_input'],
            'obs_shape': config_env['num_obs'],
        }
        model = model.build(**config_llc['network'], **additional_config)
        model.to(device)

        # load checkpoint
        load_checkpoint_to_network(model, config_hparam['llc']['ckpt'])
        # if config_llc['hparam']['normalize_input'] and 'running_mean_std' in ckpt:
        #     model.running_mean_std.load_state_dict(ckpt['running_mean_std'])

        model.eval()
        return model, config_llc
