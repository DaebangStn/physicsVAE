from torch.optim import Adam
from rl_games.common.datasets import PPODataset
from rl_games.common import common_losses
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import swap_and_flatten01, ContinuousA2CBase

from learning.logger.jitter import JitterLogger
from utils import *
from utils.buffer import TensorHistoryFIFO
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

        # jitter related
        self._JITTER_SIZE = None
        self._jitter_obs_size = None
        self._jitter_obs_buf = None
        self._jitter_loss_coef = None
        self._jitter_input_divisor = None

        # Reward related
        self._task_rew_w = None
        self._task_rew_scale = None

        # Loggers
        self._action_jitter = None

        self._init_learning_variables(**kwargs)

        self.init_rnn_from_model(self.model)
        self.algo_observer.after_init(self)

        self.dataset = None
        self._prepare_data(**kwargs)

        # placeholders for the current episode
        self.play_info = {}
        self.train_result = None
        self.reward = None
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

        curr_train_result = {}

        with ((torch.cuda.amp.autocast(enabled=self.mixed_precision))):
            # 2. Run the model
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            normalized_values = res_dict['values']
            entropy = res_dict['entropy'].mean()
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # 3. Calculate the loss
            a_loss, clip_frac = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_loss.mean()
            c_loss = self._critic_loss(curr_e_clip, return_batch, value_preds_batch, normalized_values).mean()
            b_loss = self._bound_loss(mu).mean()

            loss = (a_loss
                    + c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef)

            more_loss, more_train_result = self._additional_loss(batch_dict, res_dict)
            loss += more_loss
            curr_train_result.update(more_train_result)
            curr_train_result['total_loss'] = loss.detach()

            # 4. Zero the gradients
            for param in self.model.parameters():
                param.grad = None

        # 5. Back propagate the loss
        self.scaler.scale(loss).backward()
        # self.trancate_gradients_and_step()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 6. Store the results
        self.diagnostics.mini_batch(
            self, {
                'values': value_preds_batch,
                'returns': return_batch,
                'new_neglogp': action_log_probs,
                'old_neglogp': old_action_log_probs_batch,
            }, curr_e_clip, 0)

        curr_train_result.update({
            'a_loss': a_loss,
            'clip_frac': clip_frac,
            'c_loss': c_loss,
            'b_loss': b_loss,
            'entropy': entropy,
            'kl': self._policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch),
            'lr': self.last_lr,
            'lr_mul': lr_mul,
        })
        return curr_train_result

    def _discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

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
                'obs': self.model.norm_obs(obs['obs']),
            })

    def get_values(self, obs):
        self.model.eval()
        with torch.no_grad():
            return self.model.critic(self.model.norm_obs(obs['obs']))

    def init_tensors(self):
        super().init_tensors()
        batch_size = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['next_values'] = torch.empty(batch_size + (self.value_size,),
                                                                        device=self.device)
        if self._jitter_obs_buf is not None:
            self.experience_buffer.tensor_dict['jitter_obs'] = torch.empty(batch_size + (self._jitter_obs_size,),
                                                                           device=self.device)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()
            total_time += train_info['total_time']
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)

            self.diagnostics.epoch(self, current_epoch=epoch_num)
            self.frame += self.curr_frames
            self.write_tensorboard(train_info)
            self.print_console(epoch_num, frame, train_info)

            if self.game_rewards.current_size > 0:
                self.print_game(frame, epoch_num, total_time)
                self.save_ckpt(epoch_num)

                if self.check_exit(epoch_num):
                    self._cleanup()
                    return self.last_mean_rewards, epoch_num

    def train_epoch(self):
        super(ContinuousA2CBase, self).train_epoch()

        train_info = {}

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict = self.play_steps()
        train_info.update(self.play_info)
        play_time_end = time.time()

        update_time_start = time.time()

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.calc_gradients(self.dataset[i])
                for k, v in curr_train_info.items():
                    if k in train_info:
                        train_info[k].append(v)
                    else:
                        train_info[k] = [v]

                if self.schedule_type == 'legacy':
                    kl = curr_train_info['kl']
                    ep_kls.append(kl)
                    self.last_lr, self.entropy_coef = (
                        self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, kl))
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = (
                    self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()))
                self.update_lr(self.last_lr)

            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()

        update_time_end = time.time()

        train_info.update({
            'play_time': play_time_end - play_time_start,
            'update_time': update_time_end - update_time_start,
            'total_time': update_time_end - play_time_start,
        })

        return train_info

    def play_steps(self):
        step_time = 0.0
        self._pre_rollout()

        for n in range(self.horizon_length):
            self.obs = self.env_reset(self.dones.nonzero()[:, 0])  # update latent
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

            next_vals = self.get_values(self.obs)
            next_vals *= ~infos['terminate'].unsqueeze(1)
            self.experience_buffer.update_data('next_values', n, next_vals)
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

        self._calc_rollout_reward()
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        self.play_info['total_reward'] = mb_rewards.mean()

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_advs = self._discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        self._post_rollout(batch_dict)

        return batch_dict

    def check_exit(self, epoch_num):
        mean_rewards = self.game_rewards.get_mean()[0]

        if epoch_num >= self.max_epochs != -1:
            if self.game_rewards.current_size == 0:
                print('WARNING: Max epochs reached before any env terminated at least once')
                mean_rewards = -np.inf

            self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num)
                                   + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_'))
                      )
            print('MAX EPOCHS NUM!')
            return True

        if self.frame >= self.max_frames != -1:
            if self.game_rewards.current_size == 0:
                print('WARNING: Max frames reached before any env terminated at least once')
                mean_rewards = -np.inf

            self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame)
                                   + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_'))
                      )
            print('MAX FRAMES NUM!')
            return True

        return False

    def save_ckpt(self, epoch_num):
        mean_rewards = self.game_rewards.get_mean()[0]
        checkpoint_name = 'ep_' + str(epoch_num) + '_rew_' + str(mean_rewards)

        if self.save_freq > 0:
            if epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
            if epoch_num % self.int_save_freq == 0:
                if self._prev_int_ckpt_path is not None:
                    os.remove(self._prev_int_ckpt_path + '.pth')
                self._prev_int_ckpt_path = os.path.join(self.nn_dir, checkpoint_name + f'_{epoch_num}')
                self.save(self._prev_int_ckpt_path)

        if mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
            print('saving next best rewards: ', mean_rewards)
            self.last_mean_rewards = mean_rewards
            self.save(os.path.join(self.nn_dir, self.config['name']))

    def print_game(self, frame, epoch_num, total_time):
        mean_rewards = self.game_rewards.get_mean()
        mean_lengths = self.game_lengths.get_mean()
        self.mean_rewards = mean_rewards[0]

        self.writer.add_scalar('rewards/step', mean_rewards[0], frame)
        self.writer.add_scalar('rewards/iter', mean_rewards[0], epoch_num)
        self.writer.add_scalar('rewards/time', mean_rewards[0], total_time)

        self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
        self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
        self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

    def print_console(self, epoch_num: int, frame: int, train_info: dict):
        step_time = max(train_info['total_time'], 1e-9)
        fps_step = self.curr_frames / step_time
        fps_step_inference = self.curr_frames / train_info['update_time']
        fps_total = self.curr_frames / train_info['total_time']

        if self.max_epochs == -1 and self.max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps policy inference: {fps_step_inference:.0f} '
                  f'fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif self.max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps policy inference: {fps_step_inference:.0f} fps total: '
                  f'{fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{self.max_frames:.0f}')
        elif self.max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps policy inference: {fps_step_inference:.0f} fps total: '
                  f'{fps_total:.0f} epoch: {epoch_num:.0f}/{self.max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps policy inference: {fps_step_inference:.0f} fps total: '
                  f'{fps_total:.0f} epoch: {epoch_num:.0f}/{self.max_epochs:.0f} '
                  f'frames: {frame:.0f}/{self.max_frames:.0f}')

    def write_tensorboard(self, train_info: dict):
        info = {
            'rl_update_fps': self.curr_frames / train_info['update_time'],
            'step_inference_fps': self.curr_frames / train_info['play_time'],
            'total_fps': self.curr_frames / train_info['total_time'],
            'rl_update_time': train_info['update_time'],
            'step_inference_time': train_info['play_time'],

            'total_reward': train_info['total_reward'].mean().item(),
            'task_reward_mean': train_info['task_reward'].mean().item(),
            'task_reward_std': train_info['task_reward'].std().item(),

            'total_loss': torch_ext.mean_list(train_info['total_loss']).item(),
            'a_loss': torch_ext.mean_list(train_info['a_loss']).item(),
            'clip_frac': torch_ext.mean_list(train_info['clip_frac']).item(),
            'bounds_loss': torch_ext.mean_list(train_info['b_loss']).item(),
            'c_loss': torch_ext.mean_list(train_info['c_loss']).item(),

            'kl': torch_ext.mean_list(train_info['kl']).item(),
            'entropy': torch_ext.mean_list(train_info['entropy']).item(),
            'lr': mean_list(train_info['lr']),
            'lr_mul': mean_list(train_info['lr_mul']),
        }

        self._write_stat(**info)

    def prepare_dataset(self, batch_dict):
        """
            1. Normalize the observation
            2. Add or sample custom observation
        """
        super().prepare_dataset(batch_dict)
        if self._jitter_obs_buf is not None:
            dataset_dict = self.dataset.values_dict
            jitter_input_size = max(batch_dict['jitter_obs'].shape[0] // self._jitter_input_divisor, 2)
            dataset_dict['jitter_obs'] = batch_dict['jitter_obs'][0:jitter_input_size]

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        return a_loss, clipped.float().mean()

    def _bound_loss(self, mu):
        if self.bound_loss_type == 'regularisation':
            b_loss = self.reg_loss(mu)
        elif self.bound_loss_type == 'bound':
            b_loss = self.bound_loss(mu)
        else:
            b_loss = torch.zeros(1, device=self.ppo_device)
        return b_loss

    def _calc_rollout_reward(self):
        task_reward = self.experience_buffer.tensor_dict['rewards'] * self._task_rew_scale
        self.experience_buffer.tensor_dict['rewards'] *= self._task_rew_w
        self.play_info['task_reward'] = task_reward

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

        # jitter related
        config_jitter = config_hparam.get('jitter', None)
        if config_jitter is not None:
            self._JITTER_SIZE = 3  # jitter = a(t) - 2 * a(t-1) + a(t-2)
            self._jitter_obs_size = self.model.input_size * self._JITTER_SIZE
            self._jitter_obs_buf = TensorHistoryFIFO(self._JITTER_SIZE)
            self._jitter_loss_coef = config_jitter['loss_coef']
            self._jitter_input_divisor = config_jitter['input_divisor']

        # reward related
        config_rew = config_hparam['reward']
        self._task_rew_w = config_rew['task_weight']
        self._task_rew_scale = config_rew['task_scale']

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
        obs_batch = self.model.norm_obs(input_dict['obs'])

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
            if (k.endswith("loss") or k == "entropy") and k != "disc_logit_loss":
                self.writer.add_scalar(f'losses/{k}', v, frame)
            elif k.endswith("fps") or k.endswith("time"):
                self.writer.add_scalar(f'performance/{k}', v, frame)
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
        loss = torch.zeros(1, device=self.ppo_device)[0]
        if self._jitter_obs_buf is not None:
            jitter_loss = self._jitter_loss(batch_dict)
            loss += jitter_loss * self._jitter_loss_coef
        return loss, {}

    def _jitter_loss(self, batch_dict):
        jitter_obs = batch_dict['jitter_obs'].view(-1, self._JITTER_SIZE, self.model.input_size)
        mu, _ = self.model.actor(jitter_obs)
        jitter_mu = mu[:, 0] - 2 * mu[:, 1] + mu[:, 2]
        jitter_loss = torch.abs(jitter_mu).mean()

        self._write_stat(jitter_loss=jitter_loss.detach())
        return jitter_loss

    def _pre_rollout(self):
        self.play_info = {}

    def _post_rollout(self, batch_dict):
        if self._jitter_obs_buf is not None:
            batch_dict['jitter_obs'] = self.experience_buffer.tensor_dict['jitter_obs'].view(-1, self._jitter_obs_size)

    def _pre_step(self, n: int):
        if self._jitter_obs_buf is not None:
            self._jitter_obs_buf.push_on_reset(self.obs['obs'], self.dones)

    def _post_step(self, n: int):
        """
            Update the custom observation
        """
        if self._jitter_obs_buf is not None:
            self._jitter_obs_buf.push(self.obs['obs'])
            self.experience_buffer.update_data('jitter_obs', n, self._jitter_obs_buf.history)

    # Helper functions
    def print_gradient(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                print(f"{name} grad mean: {param.grad.abs().mean().item()}")
            else:
                print(f"{name} has no gradients")
