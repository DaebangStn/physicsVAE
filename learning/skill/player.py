from typing import Tuple, List

import torch
import matplotlib.pyplot as plt

from rl_games.algos_torch.players import rescale_actions, unsqueeze_obs

from learning.style.player import StylePlayer, keyp_task_obs_angle_transform, keyp_task_concat_obs
from learning.skill.algorithm import sample_latent
from learning.logger.matcher import Matcher
from learning.logger.jitter import JitterLogger
from learning.logger.latentMotion import LatentMotionLogger
from learning.logger.motionTransition import MotionTransitionLogger
from utils.env import sample_color
from utils.buffer import TensorHistoryFIFO
from utils.angle import calc_heading_quat_inv, quat_rotate


class SkillPlayer(StylePlayer):
    def __init__(self, **kwargs):
        # env related
        self._latent_dim = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None
        self._color_projector = None

        # Loggers
        self._action_jitter = None
        self._dof_jitter = None
        self._matcher = None
        self._matcher_obs_buf = None
        self._latent_logger = None
        self._transition_logger = None

        # placeholders for the current episode
        self._z = None
        self._remain_latent_steps = None

        super().__init__(**kwargs)

    def env_step(self, env, actions):
        if self._action_logger is not None:
            self._action_logger.log(actions)
        if self._action_jitter is not None:
            self._action_jitter.log(actions, self._n_step)

        obs_raw, rew, done, info = super(StylePlayer, self).env_step(env, actions)
        obs = self._post_process_obs(obs_raw)

        if self._dof_jitter is not None:
            aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs_raw['obs']
            self._dof_jitter.log(dPos, self._n_step)

        return obs, rew, done, info

    def env_reset(self, env):
        self._update_latent()
        obs_raw = super(StylePlayer, self).env_reset(env)
        obs = self._post_process_obs(obs_raw)
        return obs

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        with torch.no_grad():
            mu, sigma = self.model.actor(obs, latent=self._z)
            sigma = torch.exp(sigma)

        if is_deterministic:
            current_action = mu
        else:
            current_action = torch.normal(mu, sigma)

        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def _post_process_obs(self, obs_raw):
        if self._show_reward:
            obs_concat, disc_obs = keyp_task_obs_angle_transform(obs_raw['obs'], self._key_body_ids, self._dof_offsets)
            obs = {'obs': obs_concat, 'disc_obs': disc_obs}
        else:
            obs_concat = keyp_task_concat_obs(obs_raw['obs'])
            obs = {'obs': obs_concat}

        if self._latent_logger or self._transition_logger:
            obs['matcher'] = keyp_obs_to_matcher(obs_raw['obs'], self._key_body_ids)
        return obs

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
            jitter = logger_config.get('jitter', False)
            log_latent = logger_config.get('latent_motion_id', False)
            motion_transition = logger_config.get('motion_transition', False)
            if jitter:
                self._action_jitter = JitterLogger(self._writer, 'action')
                self._dof_jitter = JitterLogger(self._writer, 'dof')
            if log_latent or motion_transition:
                full_experiment_name = kwargs['params']['config']['full_experiment_name']
                show_matcher_out = logger_config.get('show_matcher_out', False)
                motion_match_length = logger_config.get('motion_match_length', 4)

                self._build_matcher(show_matcher_out, motion_match_length)
                self._matcher_obs_buf = TensorHistoryFIFO(motion_match_length)

                if log_latent:
                    self._latent_logger = LatentMotionLogger(logger_config['filename'], full_experiment_name,
                                                             self.env.num, self._latent_dim)
                if motion_transition:
                    self._transition_logger = MotionTransitionLogger(logger_config['filename'], full_experiment_name,
                                                                     self.env.num, self.config)

    def _build_matcher(self, show_matcher_out: bool, motion_match_length: int):
        plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend
        self._matcher = Matcher(self._demo_fetcher.motion_lib, motion_match_length, self.env.dt, self.device,
                                show_matcher_out)

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

    def _pre_step(self):
        super()._pre_step()
        if self._latent_logger or self._transition_logger:
            self._matcher_obs_buf.push_on_reset(self.obses['matcher'], self.dones)

    def _post_step(self):
        super()._post_step()
        if self._show_reward:
            self._enc_debug(self._disc_obs_buf.history)
        if self._latent_logger or self._transition_logger:
            self._matcher_obs_buf.push(self.obses['matcher'])
            motion_id = self._matcher.match(self._matcher_obs_buf.history)
            if self._latent_logger:
                self._latent_logger.log(motion_id)
            if self._transition_logger:
                self._transition_logger.log(motion_id)
        self._remain_latent_steps -= 1

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


@torch.jit.script
def keyp_obs_to_matcher(
        obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        key_idx: List[int]) -> torch.Tensor:
    # returns: root_h, local_root_vel/anVel, dof_pos, dof_vel, local_keypoint_pos
    # dim:     1 +     3*2 +                 31[28] + 31[28] + 3 * 6[4]          = 87[75]
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs

    root_h = rPos[:, 0, 2:3]  # dim0: num_env, dim1: 1

    inv_heading_rot = calc_heading_quat_inv(rRot[:, 0])  # dim0: num_env, dim1: 4
    local_root_vel = quat_rotate(inv_heading_rot, rVel[:, 0])  # dim0: num_env, dim1: 3
    local_root_anVel = quat_rotate(inv_heading_rot, rAnVel[:, 0])  # dim0: num_env, dim1: 3

    dof_pos = dPos
    dof_vel = dVel

    local_rBody_pos = rPos - rPos[:, 0:1]  # dim0: num_env, dim1: number of rBody, dim2: 3

    # dim0: num_env, dim1: number of rBody, dim2: 4
    inv_heading_rot_exp = inv_heading_rot.unsqueeze(1).repeat(1, local_rBody_pos.shape[1], 1)

    # dim0: num_env * number of rBody, dim1: 4
    flat_inv_heading_rot_exp = inv_heading_rot_exp.view(-1, inv_heading_rot_exp.shape[-1])
    # dim0: num_env * number of rBody, dim1: 3
    flat_local_rBody_pos = local_rBody_pos.view(-1, local_rBody_pos.shape[-1])
    local_rBody_pos = quat_rotate(flat_inv_heading_rot_exp, flat_local_rBody_pos).view(local_rBody_pos.shape)
    local_keypoint_pos = local_rBody_pos[:, key_idx]

    return torch.cat([root_h, local_root_vel, local_root_anVel, dof_pos, dof_vel, local_keypoint_pos.flatten(1)],
                     dim=-1)
