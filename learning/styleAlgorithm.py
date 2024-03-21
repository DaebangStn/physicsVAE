import time
from typing import List
import torch
from torch.optim import Adam
from rl_games.common.datasets import PPODataset
from rl_games.common.a2c_common import ContinuousA2CBase, swap_and_flatten01
from rl_games.algos_torch import torch_ext

from learning.rlAlgorithm import RlAlgorithm
from utils.angle import *
from utils.rl_games import rl_games_net_build_param
from utils.buffer import MotionLibFetcher, TensorHistoryFIFO, SingleTensorBuffer


class StyleAlgorithm(RlAlgorithm):
    def __init__(self, **kwargs):
        ContinuousA2CBase.__init__(self, **kwargs)

        # rl-games default values
        self.model = None
        self.states = None
        self.has_value_loss = True
        self.value_mean_std = None
        self.bound_loss_type = None
        self.optimizer = None
        self.is_tensor_obses = True
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._disc_loss_coef = None
        self._disc_weight_reg_scale = None
        self._disc_grad_penalty_scale = None
        # reward related
        self._task_rew_scale = None
        self._disc_rew_scale = None
        self._init_learning_variables(**kwargs)

        self.init_rnn_from_model(self.model)
        self.algo_observer.after_init(self)

        # env related
        self._key_body_ids = None
        self._dof_offsets = None

        self.dataset = None
        self._demo_fetcher = None
        self._replay_buffer = None
        self._prepare_data(**kwargs)

        # placeholders for the current episode
        self.train_result = None
        self.dones = None
        self.current_rewards = None
        self.current_shaped_rewards = None
        self.current_lengths = None
        self._mean_task_reward = None
        self._mean_style_reward = None
        self._std_task_reward = None
        self._std_style_reward = None
        self._last_last_lr = self.last_lr

        self._save_config(**kwargs)

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0
        rollout_obses = []

        for n in range(self.horizon_length):
            self.obs = self.env_reset()
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            # TODO: does the disc needs reset after policy update?
            self._disc_obs_buf.push(self.obs['obs'], self.dones)
            rollout_obses.append(self._disc_obs_buf.history)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        rollout_obs = torch.cat(rollout_obses, dim=0)
        style_reward = self._disc_reward(rollout_obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        task_reward = self.experience_buffer.tensor_dict['rewards']
        mb_rewards = self._task_rew_scale * task_reward + self._disc_rew_scale * style_reward

        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        batch_dict['rollout_obs'] = rollout_obs
        self._mean_task_reward = task_reward.mean()
        self._mean_style_reward = style_reward.mean()
        self._std_task_reward = task_reward.std()
        self._std_style_reward = style_reward.std()

        return batch_dict

    def calc_gradients(self, input_dict):
        # 1. Unpack the input
        (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch,
         old_sigma_batch, return_batch, value_preds_batch) = self._unpack_input(input_dict)

        with ((torch.cuda.amp.autocast(enabled=self.mixed_precision))):
            # 2. Run the model
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            agent_disc = torch.cat([res_dict['rollout_disc'], res_dict['replay_disc']], dim=0)

            # 3. Calculate the loss
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs,
                                          advantage, self.ppo, curr_e_clip)
            c_loss = self._critic_loss(curr_e_clip, return_batch, value_preds_batch, values)
            b_loss = self._bound_loss(mu)

            losses, _ = torch_ext.apply_masks(  # vestige for RNN
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses

            d_loss = self._disc_loss(agent_disc, res_dict['demo_disc'], input_dict)
            loss = (a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + d_loss * self._disc_loss_coef)

            # 4. Zero the gradients
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # 5. Back propagate the loss
        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        # 6. Store the results
        self.diagnostics.mini_batch(
            self, {
                'values': value_preds_batch,
                'returns': return_batch,
                'new_neglogp': action_log_probs,
                'old_neglogp': old_action_log_probs_batch,
            }, curr_e_clip, 0)

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, True)
        self.train_result = (a_loss, c_loss, entropy, kl_dist, self.last_lr, lr_mul,
                             mu.detach(), sigma.detach(), b_loss)

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
        self.model = self.network.build(**kwargs['params']['network'], **rl_games_net_build_param(self))
        self.model.to(self.ppo_device)

        config_hparam = self.config

        if self.normalize_value:
            self.value_mean_std = self.model.value_mean_std
        self.last_lr = float(self.last_lr)
        self.optimizer = Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                              weight_decay=self.weight_decay)

        self.bound_loss_type = config_hparam.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'

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

        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                  self.ppo_device, self.seq_length)

        algo_conf = kwargs['params']['algo']["style"]
        self._key_body_ids = self._find_key_body_ids(algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(**self._demo_fetcher_config(algo_conf))

        config_style = self.config['style']
        self._replay_buffer = {
            'demo': SingleTensorBuffer(config_style['replay_buf_size'], self.device),
            'rollout': SingleTensorBuffer(config_style['replay_buf_size'], self.device),
        }
        demo_obs = self._demo_fetcher.fetch(config_style['replay_buf_size'] // self._disc_obs_traj_len)
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)

    def _update_replay_buffer(self, rollout_obs):
        demo_obs = self._demo_fetcher.fetch(self.batch_size // 2048)  # 2048 is a magic number for performance
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._replay_buffer['demo'].store(demo_obs)
        self._replay_buffer['rollout'].store(rollout_obs)

    def _write_disc_stat(self, **kwargs):
        frame = self.frame // self.num_agents
        for k, v in kwargs.items():
            if k == "loss":
                self.writer.add_scalar(f'losses/disc_loss', v, frame)
            else:
                self.writer.add_scalar(f'info/{k}', v, frame)


# @torch.jit.script
def style_task_obs_angle_transform(obs: torch.Tensor, key_idx: torch.Tensor, dof_offsets: List[int]) -> torch.Tensor:
    aPos, aRot, aVel, aAnVel, dPos, dVel, rPos = obs
    keyPos = rPos[:, key_idx, :]
    return local_angle_transform((aPos, aRot, dPos, aVel, aAnVel, dVel, keyPos), dof_offsets)


def motion_lib_angle_transform(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dof_offsets: List[int], traj_len: int) -> torch.Tensor:
    num_steps = int(state[0].shape[0] / traj_len)
    obs = local_angle_transform(state, dof_offsets)
    return obs.view(num_steps, -1)


# @torch.jit.script
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
