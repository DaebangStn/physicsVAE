from learning.style.algorithm import StyleAlgorithm
from utils.angle import *
from utils.env import sample_color


class SkillAlgorithm(StyleAlgorithm):
    def __init__(self, **kwargs):
        # encoder related
        self._enc_loss_coef = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None

        self._div_loss_coef = None

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
    def env_step(self, actions):
        obs, rew, done, info = super().env_step(actions)
        obs['obs'] = torch.cat([obs['obs'], self._z], dim=1)
        return obs, rew, done, info

    def env_reset(self):
        self._update_latent()
        obs = super().env_reset()
        obs['obs'] = torch.cat([obs['obs'], self._z], dim=1)
        return obs

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['rollout_z'] = batch_dict['rollout_z']

    def _additional_loss(self, batch_dict, res_dict):
        loss = super()._additional_loss(batch_dict, res_dict)
        # e_loss = self._enc_loss(res_dict['enc'], batch_dict['rollout_z'])
        # div_loss = self._diversity_loss(batch_dict['obs'], res_dict['mus'])
        # return loss + e_loss * self._enc_loss_coef + div_loss * self._div_loss_coef
        return loss

    def _diversity_loss(self, obs, mu):
        rollout_z = obs[:, -self._latent_dim:]
        obs = obs[:, :-self._latent_dim]
        sampled_z = self.sample_latent(obs.shape[0], self._latent_dim, self.device)
        sampled_mu, sampled_sigma = self.model.actor(torch.cat([obs, sampled_z], dim=1))

        sampled_mu = torch.clamp(sampled_mu, -1.0, 1.0)
        mu = torch.clamp(mu, -1.0, 1.0)
        z_diff = (1 - (rollout_z * sampled_z).sum(dim=-1)) / 2

        # TODO, test (a-1), (b-1) and (b-2)

        # Original KL implementation (a)
        kl = torch.square(sampled_mu - mu).mean(dim=-1)

        # Right KL divergence (b)
        # kl = ((sampled_mu - mu) ** 2 /
        #       (2 * (sampled_sigma ** 2 + 1e-5))).sum(dim=-1)

        # Original loss implementation (1)
        loss = torch.square(kl / (z_diff + 1e-5) - 1).mean()

        # My loss suggestion (2)
        # loss = (kl / (z_diff + 1e-5)).mean()

        self._write_disc_stat(amp_diversity_loss=loss.detach())
        return loss

    def _enc_loss(self, enc, rollout_z):
        # encoding
        similarity = torch.sum(enc * rollout_z, dim=-1, keepdim=True)
        likelihood_loss = -similarity.mean()

        # TODO: original code ignores the gradient penalty and regularization

        self._write_disc_stat(
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

        config_hparam = self.config
        config_skill = config_hparam['skill']
        self._div_loss_coef = config_skill['div_loss_coef']
        self._enc_loss_coef = config_skill['enc']['loss_coef']
        self._latent_update_freq_max = config_skill['latent']['update_freq_max']
        self._latent_update_freq_min = config_skill['latent']['update_freq_min']
        self._remain_latent_steps = torch.zeros(self.vec_env.num, dtype=torch.int32)

        self._enc_rew_scale = config_hparam['reward']['enc_scale']

        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._z = self.sample_latent(self.vec_env.num, self._latent_dim, self.device)

        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)

    def _pre_rollout(self):
        super()._pre_rollout()
        self._rollout_zes = []

    def _post_rollout1(self):
        self._rollout_obs = torch.cat(self._rollout_obses, dim=0)
        self._rollout_z = torch.cat(self._rollout_zes, dim=0)

        # skill_reward = self._enc_reward(self._rollout_obs, self._rollout_z)
        style_reward = self._disc_reward(self._rollout_obs)
        task_reward = self.experience_buffer.tensor_dict['rewards']
        combined_reward = (self._task_rew_scale * task_reward +
                           self._disc_rew_scale * style_reward) # +
                           # self._enc_rew_scale * skill_reward)
        self.experience_buffer.tensor_dict['rewards'] = combined_reward

        self._mean_task_reward = task_reward.mean()
        self._mean_style_reward = style_reward.mean()
        # self._mean_enc_reward = skill_reward.mean()
        self._std_task_reward = task_reward.std()
        self._std_style_reward = style_reward.std()
        # self._std_enc_reward = skill_reward.std()

    def _post_rollout2(self, batch_dict):
        batch_dict = super()._post_rollout2(batch_dict)
        batch_dict['rollout_z'] = self._rollout_z
        return batch_dict

    def _post_step(self):
        super()._post_step()
        self._rollout_zes.append(self._z)
        self._remain_latent_steps -= 1

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        disc_input_size = max(input_dict['rollout_obs'].shape[0] // 512, 2)  # 512 is a magic number for performance
        batch_dict['rollout_z'] = input_dict['rollout_z'][0:disc_input_size]

        return (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
                return_batch, value_preds_batch)

    def _update_latent(self):
        update_env = torch.where(self._remain_latent_steps == 0)[0]
        if len(update_env) == 0:
            return
        self._remain_latent_steps[update_env] = torch.randint(self._latent_update_freq_min,
                                                              self._latent_update_freq_max, (len(update_env),),
                                                              dtype=self._remain_latent_steps.dtype)
        self._z[update_env] = SkillAlgorithm.sample_latent(len(update_env), self._latent_dim, self.device)
        self.vec_env.change_color(update_env, sample_color(self._color_projector, self._z[update_env]))

    @staticmethod
    def sample_latent(batch_size, latent_dim, device):
        z = torch.normal(mean=0.0, std=1.0, size=(batch_size, latent_dim), device=device)
        z = torch.nn.functional.normalize(z, dim=1)
        return z