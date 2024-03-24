import time
import yaml
import torch
from torch.optim import Adam
from rl_games.common.datasets import PPODataset
from rl_games.common import common_losses
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import swap_and_flatten01, ContinuousA2CBase

from utils.rl_games import rl_games_net_build_param


class CoreAlgorithm(A2CAgent):
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
        self._init_learning_variables(**kwargs)

        self.init_rnn_from_model(self.model)
        self.algo_observer.after_init(self)

        self.dataset = None
        self._prepare_data(**kwargs)

        # placeholders for the current episode
        self.train_result = None
        self.dones = None
        self.current_rewards = None
        self.current_shaped_rewards = None
        self.current_lengths = None
        self._last_last_lr = self.last_lr

        self._save_config(**kwargs)

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

            # 3. Calculate the loss
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs,
                                          advantage, self.ppo, curr_e_clip)
            c_loss = self._critic_loss(curr_e_clip, return_batch, value_preds_batch, values)
            b_loss = self._bound_loss(mu)

            losses, _ = torch_ext.apply_masks(  # vestige for RNN
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses

            loss = (a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef)

            loss += self._additional_loss(batch_dict, res_dict)

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

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0
        self.obs = self.env_reset()
        self._pre_rollout()

        for n in range(self.horizon_length):
            self.obs = self.env_reset()
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            self._pre_step()
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            self._post_step()

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

        self._post_rollout1()

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return self._post_rollout2(batch_dict)

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

    def _prepare_data(self, **kwargs):
        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                  self.ppo_device, self.seq_length)

    def _save_config(self, **kwargs):
        algo_config = kwargs['params']
        env_config = self.vec_env.config

        with open(self.experiment_dir + '/algo_config.yaml', 'w') as file:
            yaml.dump(self.remove_unserializable(algo_config), file)
        with open(self.experiment_dir + '/env_config.yaml', 'w') as file:
            yaml.dump(self.remove_unserializable(env_config), file)

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
        lr_mul = self._last_last_lr / self.last_lr
        self._last_last_lr = self.last_lr
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }
        return (advantage, batch_dict, curr_e_clip, lr_mul, old_action_log_probs_batch, old_mu_batch, old_sigma_batch,
                return_batch, value_preds_batch)

    @staticmethod
    def remove_unserializable(config):
        clean_config = {}
        for k, v in config.items():
            if isinstance(v, dict):
                clean_config[k] = CoreAlgorithm.remove_unserializable(v)
            elif not isinstance(v, (int, float, str, bool, list)):
                print(f'[config] Ignoring unserializable value: {k}')
            else:
                clean_config[k] = v
        return clean_config

    def _additional_loss(self, batch_dict, res_dict):
        return torch.zeros(1, device=self.ppo_device)[0]

    def _pre_rollout(self):
        pass

    def _post_rollout1(self):
        pass

    def _post_rollout2(self, batch_dict):
        return batch_dict

    def _pre_step(self):
        pass

    def _post_step(self):
        pass
