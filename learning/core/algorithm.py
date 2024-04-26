import os
import time
import yaml
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from rl_games.common.datasets import PPODataset
from rl_games.common import common_losses
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import swap_and_flatten01, ContinuousA2CBase, print_statistics

from learning.logger.jitter import JitterLogger
from utils.rl_games import rl_games_net_build_param


class CoreAlgorithm(A2CAgent):
    def __init__(self, **kwargs):
        ContinuousA2CBase.__init__(self, **kwargs)

        self._fixed_sigma = None

        # rl-games default values
        self.model = None
        self.states = None
        self.has_value_loss = True
        self.value_mean_std = None
        self.bound_loss_type = None
        self.optimizer = None
        self.is_tensor_obses = True
        self.int_save_freq = None
        self._prev_int_ckpt_path = None

        # Loggers
        self._action_jitter = None

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
            normalized_values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # 3. Calculate the loss
            a_loss = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            c_loss = self._critic_loss(curr_e_clip, return_batch, value_preds_batch, normalized_values)
            b_loss = self._bound_loss(mu)

            losses, _ = torch_ext.apply_masks(  # vestige of RNN
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)])
            a_loss, c_loss, entropy, b_loss = losses

            loss = (a_loss
                    + c_loss * self.critic_coef
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

        mu = mu.detach()
        sigma = sigma.detach()
        kl = self._policy_kl(mu, sigma, old_mu_batch, old_sigma_batch)
        self.train_result = (a_loss, c_loss, entropy, kl, self.last_lr, lr_mul, mu, sigma, b_loss)

    def env_step(self, actions):
        if self._action_jitter is not None:
            self._action_jitter.log(actions, self.frame)
        return super().env_step(actions)

    def env_reset(self, env_ids: Optional[torch.Tensor] = None):
        obs = self.vec_env.reset(env_ids)
        return self.obs_to_tensors(obs)

    def get_action_values(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model({
                'is_train': False,
                'obs': obs['obs'],
            })

    def get_values(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model.critic(obs['obs'])

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time,
                                 epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                 a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                 scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                mean_rewards = 0
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                        if epoch_num % self.int_save_freq == 0:
                            if self._prev_int_ckpt_path is not None:
                                os.remove(self._prev_int_ckpt_path + '.pth')
                            self._prev_int_ckpt_path = os.path.join(self.nn_dir, checkpoint_name + f'_{epoch_num}')
                            self.save(self._prev_int_ckpt_path)

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num)
                                           + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame)
                                           + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

            if should_exit:
                self._cleanup()
                return self.last_mean_rewards, epoch_num

    def play_steps(self):
        step_time = 0.0
        self._pre_rollout()

        for n in range(self.horizon_length):
            self.obs = self.env_reset()  # update latent
            res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            for k in self.update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            self._pre_step(n)
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()
            self._post_step(n)

            step_time += (step_time_end - step_time_start)

            self.experience_buffer.update_data('rewards', n, rewards)
            self.experience_buffer.update_data('dones', n, self.dones)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
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

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        self._write_stat(clip_frac=clipped.detach().float().mean().item())

        return a_loss

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

    def _cleanup(self):
        print('Cleaning up Isaacs')
        self.vec_env.cleanup()

    def _init_learning_variables(self, **kwargs):
        self.int_save_freq = self.config.get('intermediate_save_frequency', 500)

        self.model = self.network.build(**kwargs['params']['network'], **rl_games_net_build_param(self))
        self.model.to(self.ppo_device)

        config_hparam = self.config

        if self.normalize_value:
            self.value_mean_std = self.model.value_mean_std
        self.last_lr = float(self.last_lr)
        self.optimizer = Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                              weight_decay=self.weight_decay)

        self.bound_loss_type = config_hparam.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'

        self._fixed_sigma = self.model.a2c_network.fixed_sigma

        # Loggers
        logger_config = self.config.get('logger', None)
        if logger_config is not None:
            jitter = logger_config.get('jitter', False)
            if jitter:
                self._action_jitter = JitterLogger(self.writer, 'action')

    def _policy_kl(self, mu, sigma, old_mu, old_sigma):
        with torch.no_grad():
            if self._fixed_sigma:
                kl = ((mu - old_mu) ** 2) / (2 * old_sigma ** 2 + 1e-7)
                kl = kl.sum(dim=-1).mean()
            else:
                kl = torch_ext.policy_kl(mu, sigma, old_mu, old_sigma, True)
            return kl

    def _prepare_data(self, **kwargs):
        self.dataset = PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                  self.ppo_device, self.seq_length)
        self.dones = torch.zeros(self.num_actors, device=self.ppo_device)

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

    def _write_stat(self, **kwargs):
        frame = self.frame // self.num_agents
        for k, v in kwargs.items():
            if k.endswith("loss"):
                self.writer.add_scalar(f'losses/{k}', v, frame)
            else:
                self.writer.add_scalar(f'info/{k}', v, frame)

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

    def _pre_step(self, n: int):
        pass

    def _post_step(self, n: int):
        pass
