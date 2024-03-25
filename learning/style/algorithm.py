import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.core.algorithm import CoreAlgorithm
from utils.angle import *
from utils.buffer import MotionLibFetcher, TensorHistoryFIFO, SingleTensorBuffer


class StyleAlgorithm(CoreAlgorithm):
    def __init__(self, **kwargs):
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._disc_loss_coef = None
        self._disc_logit_reg_scale = None
        self._disc_reg_scale = None
        self._disc_grad_penalty_scale = None
        # reward related
        self._task_rew_scale = None
        self._disc_rew_scale = None

        # env related
        self._key_body_ids = None
        self._dof_offsets = None

        self._demo_fetcher = None
        self._replay_buffer = None
        self._replay_store_prob = None

        # placeholders for the current episode
        self._rollout_obs = None
        self._rollout_obses = None
        self._mean_task_reward = None
        self._mean_style_reward = None
        self._std_task_reward = None
        self._std_style_reward = None

        super().__init__(**kwargs)

    """ keypointTask returns the Tuple of tensors so that we should post-process the observation.
    env_step and env_reset are overridden to post-process the observation.
    """

    def env_step(self, actions):
        obs, rew, done, info = super().env_step(actions)
        obs, disc_obs = style_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}, rew, done, info

    def env_reset(self):
        obs = super().env_reset()['obs']
        obs, disc_obs = style_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets)
        return {'obs': obs, 'disc_obs': disc_obs}

    def get_stats_weights(self, model_stats=False):
        weights = super().get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                weights['disc_running_mean_std'] = self.model.disc_running_mean_std.state_dict()
        return weights

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self.normalize_input and 'disc_running_mean_std' in weights:
            self.model.disc_running_mean_std.set_weights(weights['disc_running_mean_std'])

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict

        dataset_dict['rollout_obs'] = batch_dict['rollout_obs']
        dataset_dict['replay_obs'] = (self._replay_buffer['rollout'].sample(self.batch_size)
                                      if self._replay_buffer['rollout'].count > 0 else
                                      batch_dict['rollout_obs'])
        dataset_dict['demo_obs'] = self._replay_buffer['demo'].sample(self.batch_size)
        self._update_replay_buffer(batch_dict['rollout_obs'])

    def _additional_loss(self, batch_dict, res_dict):
        agent_disc_logit = torch.cat([res_dict['rollout_disc_logit'], res_dict['replay_disc_logit']], dim=0)
        d_loss = self._disc_loss(agent_disc_logit, res_dict['demo_disc_logit'], batch_dict)
        return d_loss * self._disc_loss_coef

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
        demo_grad = torch.autograd.grad(demo_disc_logit, batch_dict['demo_obs'], create_graph=True,
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

        self._write_disc_stat(
            disc_loss=loss.detach(),
            pred_loss=pred_loss.detach(),
            disc_logit_loss=logit_weights_loss.detach(),
            disc_weight_loss=disc_weights_loss.detach(),
            disc_grad_penalty=penalty_loss.detach(),
            disc_agent_acc=agent_acc.detach(),
            disc_demo_acc=demo_acc.detach(),
            disc_agent_logit=agent_disc_logit.detach(),
            disc_demo_logit=demo_disc_logit.detach(),
            task_reward_mean=self._mean_task_reward,
            disc_reward_mean=self._mean_style_reward,
            task_reward_std=self._std_task_reward,
            disc_reward_std=self._std_style_reward,
        )

        return loss

    def _disc_reward(self, disc_obs):
        with torch.no_grad():
            if self.normalize_input:
                disc_obs = self.model.norm_disc_obs(disc_obs)
            disc = self.model.disc(disc_obs)
            prob = 1 / (1 + torch.exp(-disc))
            reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
        return reward.view(self.horizon_length, self.num_actors, -1)

    def _demo_fetcher_config(self, algo_conf):
        return {
            # Demo dimension
            'traj_len': self._disc_obs_traj_len,
            'dt': self.vec_env.dt,

            # Motion Lib
            'motion_file': algo_conf['motion_file'],
            'dof_body_ids': algo_conf['joint_information']['dof_body_ids'],
            'dof_offsets': self._dof_offsets,
            'key_body_ids': self._key_body_ids,
            'device': self.device
        }

    def _find_key_body_ids(self, key_body_names: List[str]) -> List[int]:
        return self.vec_env.key_body_ids(key_body_names)

    def _init_learning_variables(self, **kwargs):
        super()._init_learning_variables(**kwargs)

        config_hparam = self.config
        config_disc = config_hparam['style']['disc']
        self._disc_obs_traj_len = config_disc['obs_traj_len']
        self._disc_loss_coef = config_disc['loss_coef']
        self._disc_logit_reg_scale = config_disc['logit_reg_scale']
        self._disc_reg_scale = config_disc['reg_scale']
        self._disc_grad_penalty_scale = config_disc['grad_penalty_scale']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

        config_rew = config_hparam['reward']
        self._task_rew_scale = config_rew['task_scale']
        self._disc_rew_scale = config_rew['disc_scale']

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)
        algo_conf = kwargs['params']['algo']
        self._key_body_ids = self._find_key_body_ids(algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(**self._demo_fetcher_config(algo_conf))

        config_buffer = self.config['style']['replay_buf']
        self._replay_buffer = {
            'demo': SingleTensorBuffer(config_buffer['size'], self.device),
            'rollout': SingleTensorBuffer(config_buffer['size'], self.device),
        }
        demo_obs = self._demo_fetcher.fetch(config_buffer['size'] // self._disc_obs_traj_len)
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)
        self._replay_store_prob = config_buffer['store_prob']

    def _pre_rollout(self):
        self._rollout_obses = []

    def _post_rollout1(self):
        self._rollout_obs = torch.cat(self._rollout_obses, dim=0)

        style_reward = self._disc_reward(self._rollout_obs)
        task_reward = self.experience_buffer.tensor_dict['rewards']
        combined_reward = self._task_rew_scale * task_reward + self._disc_rew_scale * style_reward
        self.experience_buffer.tensor_dict['rewards'] = combined_reward

        self._mean_task_reward = task_reward.mean()
        self._mean_style_reward = style_reward.mean()
        self._std_task_reward = task_reward.std()
        self._std_style_reward = style_reward.std()

    def _post_rollout2(self, batch_dict):
        batch_dict['rollout_obs'] = self._rollout_obs
        return batch_dict

    def _pre_step(self):
        self._disc_obs_buf.push_on_reset(self.obs['disc_obs'], self.dones)

    def _post_step(self):
        self._disc_obs_buf.push(self.obs['disc_obs'])
        self._rollout_obses.append(self._disc_obs_buf.history)

    def _unpack_input(self, input_dict):
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
         return_batch, value_preds_batch) = super()._unpack_input(input_dict)

        disc_input_size = max(input_dict['rollout_obs'].shape[0] // 512, 2)  # 512 is a magic number for performance
        rollout_obs = input_dict['rollout_obs'][0:disc_input_size]
        replay_obs = input_dict['replay_obs'][0:disc_input_size]
        demo_obs = input_dict['demo_obs'][0:disc_input_size]
        if self.normalize_input:
            batch_dict['rollout_obs'] = self.model.norm_disc_obs(rollout_obs)
            batch_dict['replay_obs'] = self.model.norm_disc_obs(replay_obs)
            batch_dict['demo_obs'] = self.model.norm_disc_obs(demo_obs)
            batch_dict['demo_obs'].requires_grad_(True)
        else:
            batch_dict['rollout_obs'] = rollout_obs
            batch_dict['replay_obs'] = replay_obs
            batch_dict['demo_obs'] = demo_obs
        return (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
                return_batch, value_preds_batch)

    def _update_replay_buffer(self, rollout_obs):
        demo_obs = self._demo_fetcher.fetch(max(self.batch_size // 2048, 1))  # 2048 is a magic number for performance
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)

        rollout_buf = self._replay_buffer['rollout']
        if rollout_buf.count < rollout_buf.size:
            mask = torch.rand(rollout_obs.shape[0]) < self._replay_store_prob
            rollout_obs = rollout_obs[mask]

        rollout_buf.store(rollout_obs)

    def _write_disc_stat(self, **kwargs):
        frame = self.frame // self.num_agents
        for k, v in kwargs.items():
            if k.endswith("loss"):
                self.writer.add_scalar(f'losses/{k}', v, frame)
            else:
                self.writer.add_scalar(f'info/{k}', v, frame)


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
def style_task_obs_angle_transform(
        obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        key_idx: List[int], dof_offsets: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos = obs
    keyPos = rPos[:, key_idx, :]
    disc_obs = disc_obs_transform((aPos, aRot, dPos, aVel, aAnVel, dVel, keyPos), dof_offsets)
    return disc_obs, disc_obs  # return the same observation for actor and discriminator


@torch.jit.script
def motion_lib_angle_transform(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dof_offsets: List[int], traj_len: int) -> torch.Tensor:
    num_steps = int(state[0].shape[0] / traj_len)
    obs = disc_obs_transform(state, dof_offsets)
    return obs.view(num_steps, -1)
