import torch

from learning.core.player import CorePlayer
from learning.style.algorithm import keyp_task_obs_angle_transform, StyleAlgorithm, disc_reward, obs_transform
from learning.logger.motion import MotionLogger
from learning.logger.matcher import Matcher
from utils import *
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher
from utils.angle import calc_heading_quat_inv, quat_rotate


class StylePlayer(CorePlayer):
    def __init__(self, **kwargs):
        # env related
        self._key_body_ids = None
        self._dof_offsets = None
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None

        self._matcher = None
        self._matcher_obs_buf = None
        self._motion_logger = None
        self._show_reward = None

        self._checkpoint_disc = None
        super().__init__(**kwargs)

    def env_step(self, env, actions):
        obs_raw, rew, done, info = super().env_step(env, actions)
        obs = self._post_process_obs(obs_raw)

        return obs, rew, done, info

    def env_reset(self, env):
        obs_raw = super().env_reset(env)
        obs = self._post_process_obs(obs_raw)
        return obs

    def restore(self, fn):
        try:
            super().restore(fn)
        except Exception as e:
            print(f"There was an error while restoring the player: {e}")
        if self._checkpoint_disc is not None:
            ckpt = torch_ext.load_checkpoint(self._checkpoint_disc)
            self.model.disc_load_state_dict(ckpt['model'])
            if self.normalize_input:
                self.model.disc_running_mean_load_state_dict(ckpt['model'])

    def _post_process_obs(self, obs_raw):
        if self._show_reward:
            obs_concat, disc_obs = keyp_task_obs_angle_transform(obs_raw['obs'], self._key_body_ids, self._dof_offsets)
            obs = {'obs': obs_concat, 'disc_obs': disc_obs}
        else:
            obs_concat = keyp_task_concat_obs(obs_raw['obs'])
            obs = {'obs': obs_concat}

        if self._matcher is not None:
            _obs = obs_raw['obs']
            obs['matcher'] = keyp_obs_to_matcher(_obs['rPos'], _obs['rRot'], _obs['rVel'], _obs['rAnVel'],
                                                 _obs['dPos'], _obs['dVel'], self._key_body_ids)
        return obs

    def _init_variables(self, **kwargs):
        super()._init_variables(**kwargs)

        self._checkpoint_disc = kwargs['params'].get('checkpoint_disc', None)

        algo_conf = kwargs['params']['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(
            self.env, algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']

        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self._demo_fetcher = MotionLibFetcher(self._disc_obs_traj_len, self.env.dt, self.device,
                                                  algo_conf['motion_file'],
                                                  algo_conf['joint_information']['dof_body_ids'],
                                                  self._dof_offsets, self._key_body_ids)
            self.env.set_motion_fetcher(self._demo_fetcher)

        self._show_reward = self.config['reward'].get('show_on_player', False)

        style_conf = self.config['style']
        self._disc_obs_traj_len = style_conf['disc']['obs_traj_len']

        if self._show_reward:
            self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            log_motion = logger_config.get('motion', False)
            full_experiment_name = kwargs['params']['config']['full_experiment_name']
            if log_motion:
                self._motion_logger = MotionLogger(logger_config['filename'], full_experiment_name, self.env.num,
                                                   self.config)
                self._build_matcher(logger_config)

    def _build_matcher(self, logger_config):
        show_matcher_out = logger_config.get('show_matcher_out', False)
        motion_match_length = logger_config['motion_match_length']
        if show_matcher_out:
            # Since pycharm IDE embeds matplotlib, it is necessary to switch backend
            plt.switch_backend('TkAgg')
        self._matcher = Matcher(self._demo_fetcher.motion_lib, motion_match_length, self.env.dt, self.device,
                                show_matcher_out)
        self._matcher_obs_buf = TensorHistoryFIFO(motion_match_length)

    def _calc_disc_rew(self, disc_obs):
        reward = disc_reward(self.model, disc_obs).mean().item()
        print(f"disc_reward {reward:.3f}")
        self._write_stat(reward_disc=reward)

    def _pre_step(self):
        super()._pre_step()
        if self._show_reward:
            self._disc_obs_buf.push_on_reset(self.obs['disc_obs'], self.dones)
        if self._matcher is not None:
            self._matcher_obs_buf.push_on_reset(self.obs['matcher'], self.dones)

    def _post_step(self):
        if self._show_reward:
            self._disc_obs_buf.push(self.obs['disc_obs'])
            self._calc_disc_rew(self._disc_obs_buf.history)

        if self._matcher is not None:
            self._matcher_obs_buf.push(self.obs['matcher'])

            if self._motion_logger is not None:
                motion_id = self._matcher.match(self._matcher_obs_buf.history)
                self._motion_logger.log(motion_id)


def keyp_task_concat_obs(obs: dict) -> torch.Tensor:
    transformed_obs = obs_transform(obs['rPos'], obs['rRot'], obs['rVel'], obs['rAnVel'])
    if 'goal' in obs.keys():
        transformed_obs = torch.cat([transformed_obs, obs['goal']], dim=-1)
    return transformed_obs


@torch.jit.script
def keyp_obs_to_matcher(r_pos: torch.Tensor, r_rot: torch.Tensor, r_vel: torch.Tensor, r_an_vel: torch.Tensor,
                        d_pos: torch.Tensor, d_vel: torch.Tensor, key_idx: List[int]) -> torch.Tensor:
    # returns: root_h, local_root_vel/anVel, dof_pos, dof_vel, local_keypoint_pos
    # dim:     1 +     3*2 +                 31[28] + 31[28] + 3 * 6[4]          = 87[75]
    # aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs

    root_h = r_pos[:, 0, 2:3]  # dim0: num_env, dim1: 1

    inv_heading_rot = calc_heading_quat_inv(r_rot[:, 0])  # dim0: num_env, dim1: 4
    local_root_vel = quat_rotate(inv_heading_rot, r_vel[:, 0])  # dim0: num_env, dim1: 3
    local_root_anVel = quat_rotate(inv_heading_rot, r_an_vel[:, 0])  # dim0: num_env, dim1: 3

    dof_pos = d_pos
    dof_vel = d_vel

    local_rBody_pos = r_pos - r_pos[:, 0:1]  # dim0: num_env, dim1: number of rBody, dim2: 3

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
