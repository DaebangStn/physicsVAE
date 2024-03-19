import time
from typing import List
import torch
from torch.optim import Adam
from rl_games.common import common_losses
from rl_games.common.datasets import PPODataset
from rl_games.common.a2c_common import ContinuousA2CBase, swap_and_flatten01
from rl_games.algos_torch import torch_ext

from utils import TensorHistoryFIFO, MotionLibFetcher
from learning.rlAlgorithm import RlAlgorithm


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

        self.dataset = None
        self._demo_fetcher = None
        self._prepare_data(**kwargs)

        # placeholders for the current episode
        self.train_result = None
        self.dones = None
        self.current_rewards = None
        self.current_shaped_rewards = None
        self.current_lengths = None

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
        batch_dict['mean_task_reward'] = task_reward.mean()
        batch_dict['mean_style_reward'] = style_reward.mean()

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
            # TODO, implement replay buffer
            # agent_disc = torch.cat([res_dict['rollout_disc'], res_dict['replay_disc']], dim=0)

            # 3. Calculate the loss
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs,
                                          advantage, self.ppo, curr_e_clip)
            c_loss = self._critic_loss(curr_e_clip, return_batch, value_preds_batch, values)
            b_loss = self._bound_loss(mu)

            losses, _ = torch_ext.apply_masks(  # vestige for RNN
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses

            # TODO, implement replay buffer
            # d_loss = self._disc_loss(agent_disc, res_dict['demo_disc'], input_dict['demo_obs'])
            d_loss = self._disc_loss(res_dict['rollout_disc'], res_dict['demo_disc'], input_dict)
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

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict
        dataset_dict['rollout_obs'] = batch_dict['rollout_obs']
        demo = self._demo_fetcher.fetch()
        dataset_dict['demo_obs'] = torch.cat([demo] * self.horizon_length, dim=0)
        dataset_length = dataset_dict['mu'].shape[0]
        dataset_dict['mean_task_reward'] = (
            torch.cat([batch_dict['mean_task_reward'].unsqueeze(-1)] * dataset_length))
        dataset_dict['mean_style_reward'] = (
            torch.cat([batch_dict['mean_style_reward'].unsqueeze(-1)] * dataset_length))

    def _unpack_input(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        obs_batch = self._preproc_obs(obs_batch)
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
            'rollout_obs': input_dict['rollout_obs'],
            # TODO, implement replay buffer
            'replay_obs': None,
            'demo_obs': input_dict['demo_obs'].requires_grad_(True),
        }
        return advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch, return_batch, value_preds_batch

    def _bound_loss(self, mu):
        if self.bound_loss_type == 'regularisation':
            b_loss = self.reg_loss(mu)
        elif self.bound_loss_type == 'bound':
            b_loss = self.bound_loss(mu)
        else:
            b_loss = torch.zeros(1, device=self.ppo_device)
        return b_loss

    def _critic_loss(self, curr_e_clip, return_batch, value_preds_batch, values):
        if self.has_value_loss:
            c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch,
                                               self.clip_value)
        else:
            c_loss = torch.zeros(1, device=self.ppo_device)
        return c_loss

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

        self._write_disc_stat(
            loss=loss.detach(),
            pred_loss=pred_loss.detach(),
            weights_loss=weights_loss.detach(),
            penalty_loss=penalty_loss.detach(),
            agent_acc=agent_acc.detach(),
            demo_acc=demo_acc.detach(),
            mean_task_reward=input_dict['mean_task_reward'][0],
            mean_style_reward=input_dict['mean_style_reward'][0],
        )

        return loss

    def _disc_reward(self, obs):
        with torch.no_grad():
            disc = self.model.disc(obs)
            prob = 1 / (1 + torch.exp(-disc))
            reward = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
        return reward.view(self.horizon_length, self.num_actors, -1)

    def _find_key_body_ids(self, key_body_names: List[str]) -> List[int]:
        return self.vec_env.key_body_ids(key_body_names)

    def _rl_games_compatible_keywords(self):
        return {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
            'obs_shape': self.obs_shape,
        }

    def _init_learning_variables(self, **kwargs):
        self.model = self.network.build({**kwargs['params']['network'], **self._rl_games_compatible_keywords()})
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
        algo_conf = kwargs['params']['algo']["style"]

        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                  self.ppo_device, self.seq_length)

        demo_fetcher_config = {
            # Demo dimension
            'num_envs': self.num_actors * self.num_agents,
            'traj_len': self._disc_obs_traj_len,
            'dt': self.vec_env.dt,

            # Motion Lib
            'motion_file': algo_conf['motion_file'],
            'dof_body_ids': algo_conf['joint_information']['dof_body_ids'],
            'dof_offsets': algo_conf['joint_information']['dof_offsets'],
            'key_body_ids': self._find_key_body_ids(algo_conf['joint_information']['key_body_names']),
            'device': self.device
        }
        self._demo_fetcher = MotionLibFetcher(**demo_fetcher_config)

    def _write_disc_stat(self, **kwargs):
        frame = self.frame // self.num_agents
        for k, v in kwargs.items():
            self.writer.add_scalar(f'style/disc/{k}', v, frame)
