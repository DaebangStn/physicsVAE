from typing import Optional

import torch

from learning.style.algorithm import StyleAlgorithm, disc_reward
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

        # reward related
        self._enc_rew_scale = None

        # placeholders for the current episode
        self._z = None
        self._rollout_z = None
        self._mean_enc_reward = None
        self._std_enc_reward = None
        self._remain_latent_steps = None

        self._color_projector = None

        super().__init__(**kwargs)

    """ keypointTask returns the Tuple of tensors so that we should post-process the observation.
    env_step and env_reset are overridden to post-process the observation.
    """

    def env_reset(self, env_ids: Optional[torch.Tensor] = None):
        self._update_latent()
        obs = super().env_reset(env_ids)
        return obs

    def get_action_values(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model({
                'is_train': False,
                'obs': self.model.norm_obs(obs['obs']),
                'latent': self._z,
            })

    def get_values(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model.critic_latent(self.model.norm_obs(obs['obs']), self._z)

    def init_tensors(self):
        super().init_tensors()
        batch_size = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['latent'] = torch.empty(batch_size + (self._latent_dim,), device=self.device)

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['latent'] = batch_dict['latent']

    def _additional_loss(self, batch_dict, res_dict):
        # loss = super()._additional_loss(batch_dict, res_dict)
        # e_loss = self._enc_loss(res_dict['enc'], batch_dict['latent_enc'])
        # loss += e_loss * self._enc_loss_coef
        div_loss = self._diversity_loss(batch_dict, res_dict['mus'])
        loss = div_loss
        return loss

    # def _diversity_loss(self, batch_dict, mu):
    #     rollout_z = batch_dict['latent']
    #     obs = batch_dict['obs']
    #     sampled_z = sample_latent(obs.shape[0], self._latent_dim, self.device)
    #     sampled_mu, _ = self.model.actor_latent(obs, sampled_z)
    #
    #     sampled_mu = torch.clamp(sampled_mu, -1.0, 1.0)
    #     mu = torch.clamp(mu, -1.0, 1.0)
    #     z_diff = (1 - (rollout_z * sampled_z).sum(dim=-1)) / 2
    #
    #     # Original KL implementation (a)
    #     kl = torch.square(sampled_mu - mu).mean(dim=-1)
    #
    #     # Right KL divergence (b)
    #     # kl = ((sampled_mu - mu) ** 2 /
    #     #       (2 * (sampled_sigma ** 2 + 1e-5))).sum(dim=-1)
    #
    #     # Original loss implementation (1)
    #     loss = torch.square(kl / (z_diff + 1e-5) - 1).mean()
    #
    #     # My loss suggestion (2)
    #     # loss = (kl / (z_diff + 1e-5)).mean()
    #
    #     self._write_stat(amp_diversity_loss=loss.detach())
    #     return loss

    def _diversity_loss(self, batch_dict, mu2):
        rollout_z = batch_dict['latent']
        normed_obs = batch_dict['obs']
        new_z = sample_latent(normed_obs.shape[0], self._latent_dim, self.device)
        mu, sigma = self.model.actor_latent(normed_obs, new_z)

        clipped_action_params = torch.clamp(mu2, -1.0, 1.0)
        clipped_mu = torch.clamp(mu, -1.0, 1.0)

        a_diff = clipped_action_params - clipped_mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * rollout_z
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(1 - diversity_bonus)

        loss = diversity_loss.mean()
        self._write_stat(amp_diversity_loss=loss.detach())
        return loss

    def _enc_loss(self, enc, rollout_z):
        # encoding
        similarity = torch.sum(enc * rollout_z, dim=-1, keepdim=True)
        likelihood_loss = -similarity.mean()

        # original code ignores the gradient penalty and regularization

        self._write_stat(
            enc_loss=likelihood_loss.detach(),
            enc_reward_mean=self._mean_enc_reward,
            enc_reward_std=self._std_enc_reward,
        )
        return likelihood_loss

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

        # latent related
        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)
        self._z = sample_latent(self.vec_env.num, self._latent_dim, self.device)

    def _post_rollout1(self):
        disc_obs = self.experience_buffer.tensor_dict['disc_obs']
        self._rollout_z = self.experience_buffer.tensor_dict['latent']

        skill_reward = enc_reward(self.model, disc_obs, self._rollout_z
                                  ).view(self.horizon_length, self.num_actors, -1)
        style_reward = disc_reward(self.model, disc_obs, self.device)
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
        batch_dict['latent'] = self._rollout_z.view(-1, self._latent_dim)
        return batch_dict

    def _pre_step(self, n: int):
        super()._pre_step(n)

    def _post_step(self, n):
        super()._post_step(n)
        self.experience_buffer.update_data('latent', n, self._z)
        self._remain_latent_steps -= 1

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        enc_input_size = max(input_dict['rollout_disc_obs'].shape[0] // self._disc_input_divisor, 2)
        batch_dict['latent_enc'] = input_dict['latent'][0:enc_input_size]  # For encoder
        batch_dict['latent'] = input_dict['latent']  # For diversity loss

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


def enc_reward(model, disc_obs, z):
    with torch.no_grad():
        enc = model.enc(model.norm_disc_obs(disc_obs))
        similarity = torch.sum(enc * z, dim=-1, keepdim=True)
        return torch.clamp_min(similarity, 0.0)
