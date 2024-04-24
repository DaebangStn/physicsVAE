import time
from typing import Optional

import torch

from learning.style.algorithm import StyleAlgorithm, disc_reward, keyp_task_obs_angle_transform
from learning.logger.jitter import JitterLogger
from utils.angle import *
from utils.env import sample_color
from utils.buffer import TensorHistoryFIFO


class SkillAlgorithm(StyleAlgorithm):
    def __init__(self, **kwargs):
        self._disc_input_divisor = None

        # encoder related
        self._enc_loss_coef = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None

        # diversity related
        self._div_loss_coef = None

        # jitter related
        self._JITTER_SIZE = None
        self._jitter_obs_size = None
        self._jitter_obs_buf = None
        self._jitter_loss_coef = None
        self._jitter_input_divisor = None

        # reward related
        self._enc_rew_scale = None

        # Loggers
        self._action_jitter = None
        # self._dof_jitter = None

        # placeholders for the current episode
        self._z = None
        self._rollout_z = None
        self._mean_enc_reward = None
        self._std_enc_reward = None
        self._remain_latent_steps = None

        self._color_projector = None

        super().__init__(**kwargs)

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextnonterminal * mb_next_values[t] - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    """ keypointTask returns the Tuple of tensors so that we should post-process the observation.
    env_step and env_reset are overridden to post-process the observation.
    """

    def env_step(self, actions):
        if self._action_jitter is not None:
            self._action_jitter.log(actions, self.frame)

        obs, rewards, dones, infos = super(StyleAlgorithm, self).env_step(actions)
        obs_concat, disc_obs = keyp_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)

        # if self._dof_jitter is not None:
        #     aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs['obs']
        #     self._dof_jitter.log(dPos, self.frame)

        obs = {'obs': obs_concat, 'disc_obs': disc_obs}
        return obs, rewards, dones, infos

    def env_reset(self, env_ids: Optional[torch.Tensor] = None):
        self._update_latent()
        obs = super().env_reset(env_ids)
        return obs

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
            'rnn_states': self.rnn_states,
            'latent': self._z
        }
        with torch.no_grad():
            return self.model(input_dict)

    def get_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        with torch.no_grad():
            return self.model.critic(processed_obs, latent=self._z)

    def init_tensors(self):
        super().init_tensors()
        batch_size = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['rollout_z'] = torch.empty(batch_size + (self._latent_dim,),
                                                                      device=self.device)
        self.experience_buffer.tensor_dict['next_values'] = torch.empty(batch_size + (self.value_size,),
                                                                        device=self.device)
        if self._jitter_obs_buf is not None:
            self.experience_buffer.tensor_dict['jitter_obs'] = torch.empty(batch_size + (self._jitter_obs_size,),
                                                                           device=self.device)

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['rollout_z'] = batch_dict['rollout_z']
        if self._jitter_obs_buf is not None:
            dataset_dict['jitter_obs'] = batch_dict['jitter_obs']

    def _additional_loss(self, batch_dict, res_dict):
        loss = super()._additional_loss(batch_dict, res_dict)
        e_loss = self._enc_loss(res_dict['enc'], batch_dict['rollout_z'])
        loss += e_loss * self._enc_loss_coef
        div_loss = self._diversity_loss(batch_dict, res_dict['mus'])
        loss += div_loss * self._div_loss_coef
        if self._jitter_obs_buf is not None:
            jitter_loss = self._jitter_loss(batch_dict)
            loss += jitter_loss * self._jitter_loss_coef
        return loss

    def _diversity_loss(self, batch_dict, mu):
        rollout_z = batch_dict['latent']
        obs = batch_dict['obs']
        sampled_z = sample_latent(obs.shape[0], self._latent_dim, self.device)
        sampled_mu, sampled_sigma = self.model.actor(obs, latent=sampled_z)

        sampled_mu = torch.clamp(sampled_mu, -1.0, 1.0)
        mu = torch.clamp(mu, -1.0, 1.0)
        z_diff = (1 - (rollout_z * sampled_z).sum(dim=-1)) / 2

        # Original KL implementation (a)
        kl = torch.square(sampled_mu - mu).mean(dim=-1)

        # Right KL divergence (b)
        # kl = ((sampled_mu - mu) ** 2 /
        #       (2 * (sampled_sigma ** 2 + 1e-5))).sum(dim=-1)

        # Original loss implementation (1)
        loss = torch.square(kl / (z_diff + 1e-5) - 1).mean()

        # My loss suggestion (2)
        # loss = (kl / (z_diff + 1e-5)).mean()

        self._write_stat(amp_diversity_loss=loss.detach())
        return loss

    def _jitter_loss(self, batch_dict):
        jitter_obs = batch_dict['jitter_obs'].view(-1, self._JITTER_SIZE, self.model.input_size)
        mu, logstd = self.model.actor_module(jitter_obs)
        jitter_mu = mu[:, 0] - 2 * mu[:, 1] + mu[:, 2]
        jitter_loss = torch.abs(jitter_mu).mean()

        self._write_stat(jitter_loss=jitter_loss.detach())
        return jitter_loss

    def _enc_loss(self, enc, rollout_z):
        # encoding
        similarity = torch.sum(enc * rollout_z, dim=-1, keepdim=True)
        likelihood_loss = -similarity.mean()

        # TODO: original code ignores the gradient penalty and regularization

        self._write_stat(
            enc_loss=likelihood_loss.detach(),
            enc_reward_mean=self._mean_enc_reward,
            enc_reward_std=self._std_enc_reward,
        )
        return likelihood_loss

    def _enc_reward(self, disc_obs, z):
        with torch.no_grad():
            if self.normalize_input:
                disc_obs = self.model.norm_disc_obs(disc_obs)
            enc = self.model.enc(disc_obs)
            similarity = torch.sum(enc * z, dim=-1, keepdim=True)
        return torch.clamp_min(similarity, 0.0).view(self.horizon_length, self.num_actors, -1)

    def _init_learning_variables(self, **kwargs):
        super()._init_learning_variables(**kwargs)

        # encoder related
        config_hparam = self.config
        config_skill = config_hparam['skill']
        self._div_loss_coef = config_skill['div_loss_coef']
        self._enc_loss_coef = config_skill['enc']['loss_coef']
        self._latent_update_freq_max = config_skill['latent']['update_freq_max']
        self._latent_update_freq_min = config_skill['latent']['update_freq_min']
        self._remain_latent_steps = torch.zeros(self.vec_env.num, dtype=torch.int32)
        self._enc_rew_scale = config_hparam['reward']['enc_scale']

        # jitter related
        config_jitter = config_hparam.get('jitter', None)
        if config_jitter is not None:
            self._JITTER_SIZE = 3  # jitter = a(t) - 2 * a(t-1) + a(t-2)
            self._jitter_obs_size = self.model.input_size * self._JITTER_SIZE
            self._jitter_obs_buf = TensorHistoryFIFO(self._JITTER_SIZE)
            self._jitter_loss_coef = config_jitter['loss_coef']
            self._jitter_input_divisor = config_jitter['input_divisor']

        # latent related
        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)
        self._z = sample_latent(self.vec_env.num, self._latent_dim, self.device)

        # Loggers
        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            jitter = logger_config.get('jitter', False)
            if jitter:
                self._action_jitter = JitterLogger(self.writer, 'action')
                # self._dof_jitter = JitterLogger(self.writer, 'dof')

    def _post_rollout1(self):
        rollout_obs = self.experience_buffer.tensor_dict['rollout_obs']
        self._rollout_z = self.experience_buffer.tensor_dict['rollout_z']

        skill_reward = self._enc_reward(rollout_obs, self._rollout_z)
        style_reward = disc_reward(self.model, rollout_obs, self.normalize_input, self.device)
        task_reward = self.experience_buffer.tensor_dict['rewards']
        combined_reward = (self._task_rew_scale * task_reward +
                           self._disc_rew_scale * style_reward +
                           self._enc_rew_scale * skill_reward)
        self.experience_buffer.tensor_dict['rewards'] = combined_reward

        self._mean_task_reward = task_reward.mean()
        self._mean_style_reward = style_reward.mean()
        self._mean_enc_reward = skill_reward.mean()
        self._std_task_reward = task_reward.std()
        self._std_style_reward = style_reward.std()
        self._std_enc_reward = skill_reward.std()

    def _post_rollout2(self, batch_dict):
        batch_dict = super()._post_rollout2(batch_dict)
        batch_dict['rollout_z'] = self._rollout_z.view(-1, self._latent_dim)
        if self._jitter_obs_buf is not None:
            batch_dict['jitter_obs'] = self.experience_buffer.tensor_dict['jitter_obs'].view(-1, self._jitter_obs_size)
        return batch_dict

    def _pre_step(self, n: int):
        super()._pre_step(n)
        if self._jitter_obs_buf is not None:
            self._jitter_obs_buf.push_on_reset(self.model.attach_latent_to_obs(self.obs['obs'], self._z), self.dones)

    def _post_step(self, n):
        super()._post_step(n)
        if self._jitter_obs_buf is not None:
            self._jitter_obs_buf.push(self.model.attach_latent_to_obs(self.obs['obs'], self._z))
            self.experience_buffer.update_data('jitter_obs', n, self._jitter_obs_buf.history)
        self.experience_buffer.update_data('rollout_z', n, self._z)
        self.experience_buffer.update_data('next_values', n, self.get_values(self.obs))
        self._remain_latent_steps -= 1

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        enc_input_size = max(input_dict['rollout_obs'].shape[0] // self._disc_input_divisor, 2)
        batch_dict['rollout_z'] = input_dict['rollout_z'][0:enc_input_size]  # For encoder
        batch_dict['latent'] = input_dict['rollout_z']  # For diversity loss

        if self._jitter_obs_buf is not None:
            jitter_input_size = max(input_dict['jitter_obs'].shape[0] // self._jitter_input_divisor, 2)
            batch_dict['jitter_obs'] = input_dict['jitter_obs'][0:jitter_input_size]

        return (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
                return_batch, value_preds_batch)

    def _update_latent(self):
        update_env = torch.where(self._remain_latent_steps == 0)[0]
        if len(update_env) == 0:
            return
        self._remain_latent_steps[update_env] = torch.randint(self._latent_update_freq_min,
                                                              self._latent_update_freq_max, (len(update_env),),
                                                              dtype=self._remain_latent_steps.dtype)
        self._z[update_env] = sample_latent(len(update_env), self._latent_dim, self.device)
        self.vec_env.change_color(update_env, sample_color(self._color_projector, self._z[update_env]))


@torch.jit.script
def sample_latent(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    z = torch.normal(mean=0.0, std=1.0, size=(batch_size, latent_dim), device=device)
    z = torch.nn.functional.normalize(z, dim=1)
    return z
