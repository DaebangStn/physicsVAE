from typing import Tuple, List

import torch
import matplotlib.pyplot as plt

from learning.style.player import StylePlayer, keyp_task_obs_angle_transform, keyp_task_concat_obs
from learning.skill.algorithm import sample_latent
from learning.logger.latentMotion import LatentMotionLogger
from poselib.matcher import Matcher
from utils.buffer import TensorHistoryFIFO
from utils.angle import calc_heading_quat_inv, quat_rotate
from utils.env import sample_color


class SkillPlayer(StylePlayer):
    def __init__(self, **kwargs):
        # env related
        self._latent_dim = None
        self._latent_update_freq_max = None
        self._latent_update_freq_min = None
        self._color_projector = None

        self._matcher = None
        self._matcher_obs_buf = None
        self._latent_logger = None

        # placeholders for the current episode
        self._z = None
        self._remain_latent_steps = None

        super().__init__(**kwargs)

    def env_step(self, env, actions):
        if self._action_logger is not None:
            self._action_logger.log(actions)
        obs_raw, rew, done, info = super(StylePlayer, self).env_step(env, actions)
        obs = self._post_process_obs(obs_raw)
        return obs, rew, done, info

    def env_reset(self, env):
        self._update_latent()
        obs_raw = super(StylePlayer, self).env_reset(env)
        obs = self._post_process_obs(obs_raw)
        return obs

    def _post_process_obs(self, obs_raw):
        if self._show_reward:
            obs_concat, disc_obs = keyp_task_obs_angle_transform(obs_raw['obs'], self._key_body_ids, self._dof_offsets)
            obs_latent = torch.cat([obs_concat, self._z], dim=1)
            obs = {'obs': obs_latent, 'disc_obs': disc_obs}
        else:
            obs_concat = keyp_task_concat_obs(obs_raw['obs'])
            obs_latent = torch.cat([obs_concat, self._z], dim=1)
            obs = {'obs': obs_latent}

        if self._latent_logger:
            obs['matcher'] = keyp_obs_to_matcher(obs_raw['obs'], self._key_body_ids, self._dof_offsets)
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
            log_latent = logger_config.get('log_latent_motion_id', False)
            if log_latent:
                full_experiment_name = kwargs['params']['config']['full_experiment_name']

                self._build_matcher()
                self._matcher_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
                self._latent_logger = LatentMotionLogger(logger_config['filename'], full_experiment_name,
                                                         self.env.num, self._latent_dim)

    def _build_matcher(self):
        plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend
        self._matcher = Matcher(self._demo_fetcher.motion_lib, self._disc_obs_traj_len, self.env.dt, self.device)

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
        if self._latent_logger:
            self._matcher_obs_buf.push_on_reset(self.obses['matcher'], self.dones)

    def _post_step(self):
        super()._post_step()
        if self._show_reward:
            self._enc_debug(self._disc_obs_buf.history)
        if self._latent_logger:
            self._matcher_obs_buf.push(self.obses['matcher'])
            motion_id = self._matcher.match(self._matcher_obs_buf.history)
            self._latent_logger.log(motion_id)
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


@torch.jit.script
def keyp_obs_to_matcher(
        obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        key_idx: List[int], dof_offsets: List[int]) -> torch.Tensor:
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
