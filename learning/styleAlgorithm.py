import torch
from typing import List, Tuple

from learning.rlAlgorithm import RlAlgorithm
from utils.angle import *
from utils.buffer import MotionLibFetcher, TensorHistoryFIFO, SingleTensorBuffer


class StyleAlgorithm(RlAlgorithm):
    def __init__(self, **kwargs):
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._disc_loss_coef = None
        self._disc_weight_reg_scale = None
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
        obs = style_task_obs_angle_transform(obs['obs'], self._key_body_ids, self._dof_offsets)
        return {'obs': obs}, rew, done, info

    def env_reset(self):
        obs = super().env_reset()['obs']
        return {'obs': style_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets)}

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict

        dataset_dict['rollout_obs'] = batch_dict['rollout_obs']
        dataset_dict['replay_obs'] = (self._replay_buffer['rollout'].sample(self.batch_size)
                                      if self._replay_buffer['rollout'].count > 0 else batch_dict['rollout_obs'])
        dataset_dict['demo_obs'] = self._replay_buffer['demo'].sample(self.batch_size)
        self._update_replay_buffer(batch_dict['rollout_obs'])

    def _additional_loss(self, input_dict, res_dict):
        agent_disc = torch.cat([res_dict['rollout_disc'], res_dict['replay_disc']], dim=0)
        d_loss = self._disc_loss(agent_disc, res_dict['demo_disc'], input_dict)
        return d_loss * self._disc_loss_coef

    def _disc_loss(self, agent_disc, demo_disc, input_dict):
        obs_demo = input_dict['demo_obs']

        # prediction
        bce = torch.nn.BCEWithLogitsLoss()
        agent_loss = bce(agent_disc, torch.zeros_like(agent_disc))
        demo_loss = bce(demo_disc, torch.ones_like(demo_disc))
        pred_loss = agent_loss + demo_loss

        # weights regularization
        weights = self.model.disc_logistics_weights
        weights_loss = torch.sum(torch.square(weights))

        # gradients penalty
        demo_grad = torch.autograd.grad(demo_disc, obs_demo, create_graph=True,
                                        retain_graph=True, only_inputs=True,
                                        grad_outputs=torch.ones_like(demo_disc))[0]
        penalty_loss = torch.mean(torch.sum(torch.square(demo_grad), dim=-1))

        loss = (pred_loss +
                self._disc_weight_reg_scale * weights_loss +
                self._disc_grad_penalty_scale * penalty_loss)

        # (for logging) discriminator accuracy
        agent_acc = torch.mean((agent_disc < 0).float())
        demo_acc = torch.mean((demo_disc > 0).float())
        agent_disc = torch.mean(agent_disc)
        demo_disc = torch.mean(demo_disc)

        self._write_disc_stat(
            loss=loss.detach(),
            pred_loss=pred_loss.detach(),
            disc_logit_loss=weights_loss.detach(),
            disc_grad_penalty=penalty_loss.detach(),
            disc_agent_acc=agent_acc.detach(),
            disc_demo_acc=demo_acc.detach(),
            disc_agent_logit=agent_disc.detach(),
            disc_demo_logit=demo_disc.detach(),
            task_reward_mean=self._mean_task_reward,
            disc_reward_mean=self._mean_style_reward,
            task_reward_std=self._std_task_reward,
            disc_reward_std=self._std_style_reward,
        )

        return loss

    def _disc_reward(self, obs):
        with torch.no_grad():
            disc = self.model.disc(obs)
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
        self._disc_weight_reg_scale = config_disc['weight_reg_scale']
        self._disc_grad_penalty_scale = config_disc['grad_penalty_scale']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

        config_rew = config_hparam['style']
        self._task_rew_scale = config_rew['task_rew_scale']
        self._disc_rew_scale = config_rew['disc_rew_scale']

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)
        algo_conf = kwargs['params']['algo']["style"]
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
        rollout_obs = torch.cat(self._rollout_obses, dim=0)
        style_reward = self._disc_reward(rollout_obs)
        task_reward = self.experience_buffer.tensor_dict['rewards']
        combined_reward = self._task_rew_scale * task_reward + self._disc_rew_scale * style_reward
        self.experience_buffer.tensor_dict['rewards'] = combined_reward

        self._rollout_obs = rollout_obs
        self._mean_task_reward = task_reward.mean()
        self._mean_style_reward = style_reward.mean()
        self._std_task_reward = task_reward.std()
        self._std_style_reward = style_reward.std()

    def _post_rollout2(self, batch_dict):
        batch_dict['rollout_obs'] = self._rollout_obs
        return batch_dict

    def _pre_step(self):
        self._disc_obs_buf.push(self.obs['obs'], self.dones)
        self._rollout_obses.append(self._disc_obs_buf.history)

    def _update_replay_buffer(self, rollout_obs):
        demo_obs = self._demo_fetcher.fetch(self.batch_size // 2048)  # 2048 is a magic number for performance
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
            if k == "loss":
                self.writer.add_scalar(f'losses/disc_loss', v, frame)
            else:
                self.writer.add_scalar(f'info/{k}', v, frame)


@torch.jit.script
def local_angle_transform(
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
        key_idx: List[int], dof_offsets: List[int]) -> torch.Tensor:
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos = obs
    keyPos = rPos[:, key_idx, :]
    return local_angle_transform((aPos, aRot, dPos, aVel, aAnVel, dVel, keyPos), dof_offsets)


@torch.jit.script
def motion_lib_angle_transform(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dof_offsets: List[int], traj_len: int) -> torch.Tensor:
    num_steps = int(state[0].shape[0] / traj_len)
    obs = local_angle_transform(state, dof_offsets)
    return obs.view(num_steps, -1)

