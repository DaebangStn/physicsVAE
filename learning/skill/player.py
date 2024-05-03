import torch

from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs

from learning.style.player import StylePlayer
from learning.skill.algorithm import sample_latent, enc_reward
from learning.logger.latentMotion import LatentMotionLogger
from learning.logger.motionTransition import MotionTransitionLogger
from utils.env import sample_color


class SkillPlayer(StylePlayer):
    def __init__(self, **kwargs):
        # env related
        self._latent_dim = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None
        self._color_projector = None

        # Loggers
        self._latent_logger = None
        self._transition_logger = None

        # placeholders for the current episode
        self._z = None
        self._remain_latent_steps = None

        super().__init__(**kwargs)

    def env_step(self, env, actions):
        obs_raw, rew, done, info = super(StylePlayer, self).env_step(env, actions)
        obs = self._post_process_obs(obs_raw)
        return obs, rew, done, info

    def env_reset(self, env):
        self._update_latent()
        obs_raw = super(StylePlayer, self).env_reset(env)
        obs = self._post_process_obs(obs_raw)
        return obs

    def get_action(self, obs, is_deterministic=False):
        with torch.no_grad():
            normed_obs = self.model.norm_obs(obs)
            mu, sigma = self.model.actor_latent(normed_obs, self._z)

        if is_deterministic:
            current_action = mu
        else:
            current_action = torch.normal(mu, sigma)

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def _init_variables(self, **kwargs):
        super()._init_variables(**kwargs)

        config_skill = kwargs['params']['hparam']['skill']
        self._latent_update_freq_max = config_skill['latent']['update_freq_max']
        self._latent_update_freq_min = config_skill['latent']['update_freq_min']
        self._remain_latent_steps = torch.zeros(self.env.num, dtype=torch.int32)

        config_network = kwargs['params']['network']
        self._latent_dim = config_network['space']['latent_dim']
        self._z = sample_latent(self.env.num, self._latent_dim, self.device)

        self._color_projector = torch.rand((self._latent_dim, 3), device=self.device)

        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            log_latent = logger_config.get('latent_motion_id', False)
            motion_transition = logger_config.get('motion_transition', False)

            if (log_latent or motion_transition) and self._matcher is None:
                self._build_matcher(logger_config)

            full_experiment_name = kwargs['params']['config']['full_experiment_name']
            if log_latent:
                self._latent_logger = LatentMotionLogger(logger_config['filename'], full_experiment_name,
                                                         self.env.num, self._latent_dim, self.config)
            if motion_transition:
                self._transition_logger = MotionTransitionLogger(logger_config['filename'], full_experiment_name,
                                                                 self.env.num, self.config)

    def _enc_debug(self, disc_obs):
        reward = enc_reward(self.model, disc_obs, self._z).mean().item()
        print(f"enc_reward {reward:.3f}")
        if self._games_played == 0:
            self._writer.add_scalar("player/reward_enc", reward, self._n_step)

    def _post_step(self):
        super()._post_step()
        self._remain_latent_steps -= 1

        if self._show_reward:
            self._enc_debug(self._disc_obs_buf.history)

        if self._matcher is not None:
            motion_id = self._matcher.match(self._matcher_obs_buf.history)

            if self._latent_logger:
                self._latent_logger.log(motion_id)
            if self._transition_logger:
                self._transition_logger.log(motion_id)

    def _update_latent(self):
        update_env = torch.where(self._remain_latent_steps == 0)[0]
        if len(update_env) == 0:
            return
        self._remain_latent_steps[update_env] = torch.randint(self._latent_update_freq_min,
                                                              self._latent_update_freq_max, (len(update_env),),
                                                              dtype=self._remain_latent_steps.dtype)
        self._z[update_env] = sample_latent(len(update_env), self._latent_dim, self.device)
        self.env.change_color(update_env, sample_color(self._color_projector, self._z[update_env]))

        if self._latent_logger:
            self._latent_logger.update_z(update_env, self._z[update_env])
        if self._transition_logger:
            self._transition_logger.update_z(update_env)
