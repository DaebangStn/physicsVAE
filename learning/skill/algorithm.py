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
        self._enc_rew_w = None
        self._enc_rew_scale = None

        # placeholders for the current episode
        self._z = None
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
        # Data for computing gradient should be passed as a tensor_list (post-processing uses tensor_list)
        self.tensor_list += ['latent']
        self.experience_buffer.tensor_dict['latent'] = torch.empty(batch_size + (self._latent_dim,), device=self.device)

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['latent'] = batch_dict['latent']

    def _additional_loss(self, batch_dict, res_dict):
        loss = super()._additional_loss(batch_dict, res_dict)
        e_loss = self._enc_loss(res_dict['enc'], batch_dict['latent_enc'])
        loss += e_loss * self._enc_loss_coef
        div_loss = self._diversity_loss(batch_dict, res_dict['mus'])
        loss += div_loss * self._div_loss_coef
        return loss

    def _diversity_loss(self, batch_dict, mu):
        rollout_z = batch_dict['latent']
        normed_obs = batch_dict['obs']
        sampled_z = sample_latent(normed_obs.shape[0], self._latent_dim, self.device)
        sampled_mu, _ = self.model.actor_latent(normed_obs, sampled_z)

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

    def _enc_loss(self, enc, rollout_z):
        # encoding
        similarity = torch.sum(enc * rollout_z, dim=-1, keepdim=True)
        likelihood_loss = -similarity.mean()

        # original code ignores the gradient penalty and regularization

        self._write_stat(enc_loss=likelihood_loss.detach())
        return likelihood_loss

    def _calc_rollout_reward(self):
        super()._calc_rollout_reward()
        skill_reward = enc_reward(self.model, self.experience_buffer.tensor_dict['disc_obs'],
                                  self.experience_buffer.tensor_dict['latent']) * self._enc_rew_w
        self.experience_buffer.tensor_dict['rewards'] += skill_reward * self._enc_rew_scale
        self._write_stat(
            enc_reward_mean=skill_reward.mean().item(),
            enc_reward_std=skill_reward.std().item(),
        )

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
        self._enc_rew_w = config_hparam['reward']['enc_weight']
        self._enc_rew_scale = config_hparam['reward']['enc_scale']

        # latent related
        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)
        self._z = sample_latent(self.vec_env.num, self._latent_dim, self.device)

    def _post_step(self, n):
        super()._post_step(n)
        self.experience_buffer.update_data('latent', n, self._z)
        self._remain_latent_steps -= 1

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        batch_dict['latent_enc'] = input_dict['latent'][0:self._disc_size_mb]  # For encoder
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
