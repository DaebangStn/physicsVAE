from typing import Optional

import torch

from learning.core.algorithm import CoreAlgorithm
from utils.angle import *
from utils.buffer import MotionLibFetcher, TensorHistoryFIFO, SingleTensorBuffer
from matplotlib import pyplot


class StyleAlgorithm(CoreAlgorithm):
    def __init__(self, **kwargs):
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._disc_loss_coef = None
        self._disc_logit_reg_scale = None
        self._disc_reg_scale = None
        self._disc_grad_penalty_scale = None
        self._disc_obs_size = None
        self._disc_input_divisor = None
        self._disc_log_hist = None
        self._disc_size_mb = None

        # reward related
        self._task_rew_scale = None
        self._disc_rew_scale = None

        # env related
        self._key_body_ids = None
        self._dof_offsets = None

        self._demo_fetcher = None
        self._replay_buffer = None
        self._replay_store_prob = None
        self._replay_num_demo_update = None

        super().__init__(**kwargs)

    """ keypointTask returns the Tuple of tensors so that we should post-process the observation.
    env_step and env_reset are overridden to post-process the observation.
    """

    def env_step(self, actions):
        obs, rew, done, info = super().env_step(actions)
        obs, disc_obs = keyp_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}, rew, done, info

    def env_reset(self, env_ids: Optional[torch.Tensor] = None):
        obs = super().env_reset(env_ids)['obs']
        obs, disc_obs = keyp_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}

    def get_stats_weights(self, model_stats=False):
        weights = super().get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                weights['disc_running_mean_std'] = self.model.disc_running_mean_std.state_dict()
        return weights

    def init_tensors(self):
        super().init_tensors()
        config_hparam = self.config
        self._disc_obs_size = config_hparam['style']['disc']['num_obs'] * self._disc_obs_traj_len

        # append experience buffer
        batch_size = self.experience_buffer.obs_base_shape
        # Data for computing gradient should be passed as a tensor_list (post-processing uses tensor_list)
        self.tensor_list += ['disc_obs']
        self.experience_buffer.tensor_dict['disc_obs'] = torch.empty(batch_size + (self._disc_obs_size,),
                                                                     device=self.device)

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self.normalize_input and 'disc_running_mean_std' in weights:
            self.model.disc_running_mean_std.set_weights(weights['disc_running_mean_std'])

    def prepare_dataset(self, batch_dict):
        """
            1. Normalize the observation
            2. Add or sample custom observation
        """
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict

        dataset_dict['normalized_rollout_disc_obs'] = self.model.norm_disc_obs(batch_dict['disc_obs'])
        dataset_dict['normalized_replay_disc_obs'] = self.model.norm_disc_obs(
            self._replay_buffer['rollout'].sample(self.batch_size)
            if self._replay_buffer['rollout'].count > 0 else batch_dict['disc_obs'])
        dataset_dict['normalized_demo_disc_obs'] = self.model.norm_disc_obs(
            self._replay_buffer['demo'].sample(self.batch_size))

        self._update_replay_buffer(batch_dict['disc_obs'])

    def _additional_loss(self, batch_dict, res_dict):
        loss = super()._additional_loss(batch_dict, res_dict)
        agent_disc_logit = torch.cat([res_dict['rollout_disc_logit'], res_dict['replay_disc_logit']], dim=0)
        d_loss = self._disc_loss(agent_disc_logit, res_dict['demo_disc_logit'], batch_dict)
        loss += d_loss * self._disc_loss_coef
        return loss

    def _disc_loss(self, agent_disc_logit, demo_disc_logit, batch_dict):
        # prediction
        bce = torch.nn.BCEWithLogitsLoss()
        agent_loss = bce(agent_disc_logit, torch.zeros_like(agent_disc_logit))
        demo_loss = bce(demo_disc_logit, torch.ones_like(demo_disc_logit))
        pred_loss = agent_loss + demo_loss

        # weights regularization
        # (logit)
        logit_weights = self.model.disc_logistics_weights
        logit_weights_loss = torch.sum(torch.square(logit_weights))
        # (whole)
        disc_weights = torch.cat(self.model.disc_weights, dim=-1)
        disc_weights_loss = torch.sum(torch.square(disc_weights))

        # gradients penalty
        demo_grad = torch.autograd.grad(demo_disc_logit, batch_dict['normalized_demo_disc_obs'], create_graph=True,
                                        retain_graph=True, only_inputs=True,
                                        grad_outputs=torch.ones_like(demo_disc_logit))[0]
        penalty_loss = torch.mean(torch.sum(torch.square(demo_grad), dim=-1))

        loss = (pred_loss +
                self._disc_logit_reg_scale * logit_weights_loss +
                self._disc_reg_scale * disc_weights_loss +
                self._disc_grad_penalty_scale * penalty_loss)

        # (for logging) discriminator accuracy
        agent_acc = torch.mean((agent_disc_logit < 0).float())
        demo_acc = torch.mean((demo_disc_logit > 0).float())
        agent_disc_logit = torch.mean(agent_disc_logit)
        demo_disc_logit = torch.mean(demo_disc_logit)

        self._write_stat(
            disc_loss=loss.detach(),
            pred_loss=pred_loss.detach(),
            disc_logit_loss=logit_weights_loss.detach(),
            disc_weight_loss=disc_weights_loss.detach(),
            disc_grad_penalty=penalty_loss.detach(),
            disc_agent_acc=agent_acc.detach(),
            disc_demo_acc=demo_acc.detach(),
            disc_agent_logit=agent_disc_logit.detach(),
            disc_demo_logit=demo_disc_logit.detach(),
        )

        return loss

    def _init_learning_variables(self, **kwargs):
        super()._init_learning_variables(**kwargs)

        # discriminator related
        config_hparam = self.config
        config_disc = config_hparam['style']['disc']
        self._disc_obs_traj_len = config_disc['obs_traj_len']
        self._disc_loss_coef = config_disc['loss_coef']
        self._disc_logit_reg_scale = config_disc['logit_reg_scale']
        self._disc_reg_scale = config_disc['reg_scale']
        self._disc_grad_penalty_scale = config_disc['grad_penalty_scale']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
        self._disc_input_divisor = int(config_disc['input_divisor'])
        self._disc_log_hist = config_disc['log_hist']
        self._disc_size_mb = max(self.minibatch_size // self._disc_input_divisor, 2)

        # reward related
        config_rew = config_hparam['reward']
        self._disc_rew_scale = config_rew['disc_scale']

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)

        algo_conf = kwargs['params']['algo']
        self._key_body_ids = self.find_key_body_ids(self.vec_env, algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(self._disc_obs_traj_len, self.vec_env.dt, self.device,
                                              algo_conf['motion_file'], algo_conf['joint_information']['dof_body_ids'],
                                              self._dof_offsets, self._key_body_ids)
        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self.vec_env.set_motion_fetcher(self._demo_fetcher)

        # build replay buffer
        config_buffer = self.config['style']['replay_buf']
        buf_size = config_buffer['size']
        self._replay_buffer = {
            'demo': SingleTensorBuffer(buf_size, self.device),
            'rollout': SingleTensorBuffer(buf_size, self.device),
        }
        demo_obs = self._demo_fetcher.fetch_traj(buf_size // self._disc_obs_traj_len)
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)
        self._replay_store_prob = config_buffer['store_prob']
        self._replay_num_demo_update = int(config_buffer['num_demo_update'])

    def _pre_step(self, n: int):
        super()._pre_step(n)
        self._disc_obs_buf.push_on_reset(self.obs['disc_obs'], self.dones)

    def _calc_rollout_reward(self):
        super()._calc_rollout_reward()
        style_reward = disc_reward(self.model, self.experience_buffer.tensor_dict['disc_obs'], self.device)
        self.experience_buffer.tensor_dict['rewards'] += style_reward * self._disc_rew_scale
        self._write_stat(
            disc_reward_mean=style_reward.mean().item(),
            disc_reward_std=style_reward.std().item(),
        )

    def _post_step(self, n: int):
        super()._post_step(n)
        self._disc_obs_buf.push(self.obs['disc_obs'])
        self.experience_buffer.update_data('disc_obs', n, self._disc_obs_buf.history)

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        batch_dict['normalized_rollout_disc_obs'] = input_dict['normalized_rollout_disc_obs'][0:self._disc_size_mb]
        batch_dict['normalized_replay_disc_obs'] = input_dict['normalized_replay_disc_obs'][0:self._disc_size_mb]
        batch_dict['normalized_demo_disc_obs'] = input_dict['normalized_demo_disc_obs'][0:self._disc_size_mb]
        batch_dict['normalized_demo_disc_obs'].requires_grad = True

        return (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
                return_batch, value_preds_batch)

    def _update_replay_buffer(self, rollout_obs):
        demo_obs = self._demo_fetcher.fetch_traj(self._replay_num_demo_update)
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)

        rollout_buf = self._replay_buffer['rollout']
        if rollout_buf.count == rollout_buf.size:
            mask = torch.rand(rollout_obs.shape[0]) < self._replay_store_prob
            rollout_obs = rollout_obs[mask]
        elif rollout_obs.shape[0] > rollout_buf.size:
            rollout_obs = rollout_obs[-rollout_buf.size:]

        rollout_buf.store(rollout_obs)

    @staticmethod
    def find_key_body_ids(env, key_body_names: List[str]) -> List[int]:
        return env.key_body_ids(key_body_names)


@torch.jit.script
def disc_obs_transform(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dof_offsets: List[int]) -> torch.Tensor:
    aPos, aRot, dPos, aVel, aAnVel, dVel, keyPos = state
    hPos = aPos[..., 2].unsqueeze(-1)
    inv_head_rot = calc_heading_quat_inv(aRot)
    aRot = quat_to_tan_norm(aRot)

    localVel = quat_rotate(inv_head_rot, aVel)
    localAnVel = quat_rotate(inv_head_rot, aAnVel)

    localKeyPos = keyPos - aPos.unsqueeze(-2)

    inv_head_rot_expend = inv_head_rot.unsqueeze(-2)
    inv_head_rot_expend = inv_head_rot_expend.repeat((1, localKeyPos.shape[1], 1))
    flatKeyPos = localKeyPos.view(-1, localKeyPos.shape[-1])
    inv_head_rot_flat = inv_head_rot_expend.view(-1, inv_head_rot_expend.shape[-1])

    flatLocalKeyPos = quat_rotate(inv_head_rot_flat, flatKeyPos).view(localKeyPos.shape[0], -1)
    dof = joint_tan_norm(dPos, dof_offsets)
    return torch.cat([hPos, aRot, localVel, localAnVel, dof, dVel, flatLocalKeyPos], dim=-1)


@torch.jit.script
def obs_transform(body_pos: torch.Tensor, body_rot: torch.Tensor, body_vel: torch.Tensor, body_ang_vel: torch.Tensor):
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    root_h_obs = root_h

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                  heading_rot_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
                                                 local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
                                                 local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
                                                         body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
                                                         body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def keyp_task_obs_angle_transform(
        obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        key_idx: List[int], dof_offsets: List[int]):
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos, rRot, rVel, rAnVel = obs
    obs = obs_transform(rPos, rRot, rVel, rAnVel)
    keyPos = rPos[:, key_idx, :]
    disc_obs = disc_obs_transform((aPos, aRot, dPos, aVel, aAnVel, dVel, keyPos), dof_offsets)
    return obs, disc_obs


@torch.jit.script
def motion_lib_angle_transform(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dof_offsets: List[int], traj_len: int) -> torch.Tensor:
    num_steps = int(state[0].shape[0] / traj_len)
    obs = disc_obs_transform(state, dof_offsets)
    return obs.view(num_steps, -1)


def disc_reward(model, disc_obs, device):
    with torch.no_grad():
        disc = model.disc(model.norm_disc_obs(disc_obs))
        prob = 1 / (1 + torch.exp(-disc))
        reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=device)))
        return reward
