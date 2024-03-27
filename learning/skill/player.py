import torch

from learning.style.player import StylePlayer
from learning.skill.algorithm import SkillAlgorithm
from utils.env import sample_color


class SkillPlayer(StylePlayer):
    def __init__(self, **kwargs):
        # env related
        self._latent_dim = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None
        self._color_projector = None

        # placeholders for the current episode
        self._z = None
        self._remain_latent_steps = None

        super().__init__(**kwargs)

    def env_step(self, env, actions):
        obs, rew, done, info = super().env_step(env, actions)
        obs['obs'] = torch.cat([obs['obs'], self._z], dim=1)
        return obs, rew, done, info

    def env_reset(self, env):
        self._update_latent()
        obs = super().env_reset(env)
        obs['obs'] = torch.cat([obs['obs'], self._z], dim=1)
        return obs

    def _init_variables(self, **kwargs):
        super()._init_variables(**kwargs)

        config_skill = kwargs['params']['hparam']['skill']
        self._latent_update_freq_max = config_skill['latent']['update_freq_max']
        self._latent_update_freq_min = config_skill['latent']['update_freq_min']
        self._remain_latent_steps = torch.zeros(self.env.num, dtype=torch.int32)

        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._z = SkillAlgorithm.sample_latent(self.env.num, self._latent_dim, self.device)

        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)

    def _enc_debug(self, disc_obs):
        with torch.no_grad():
            if self.normalize_input:
                disc_obs = self.model.norm_disc_obs(disc_obs)
            enc = self.model.enc(disc_obs)
            similarity = torch.sum(enc * self._z, dim=-1, keepdim=True)
            reward = torch.clamp_min(similarity, 0.0).mean().item()
        print(f"enc_reward {reward:.3f}")
        if self._games_played == 0:
            self._writer.add_scalar("player/reward_enc", reward, self._n_step)

    def _post_step(self):
        super()._post_step()
        self._enc_debug(self._disc_obs_buf.history)
        self._remain_latent_steps -= 1

    def _update_latent(self):
        update_env = torch.where(self._remain_latent_steps == 0)[0]
        if len(update_env) == 0:
            return
        self._remain_latent_steps[update_env] = torch.randint(self._latent_update_freq_min,
                                                              self._latent_update_freq_max, (len(update_env),),
                                                              dtype=self._remain_latent_steps.dtype)
        self._z[update_env] = SkillAlgorithm.sample_latent(len(update_env), self._latent_dim, self.device)
        self.env.change_color(update_env, sample_color(self._color_projector, self._z[update_env]))
