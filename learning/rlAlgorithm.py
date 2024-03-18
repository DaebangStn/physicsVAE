import time
import yaml
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import swap_and_flatten01


class RlAlgorithm(A2CAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._save_config(**kwargs)

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

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

        return batch_dict

    def _save_config(self, **kwargs):
        algo_config = kwargs['params']
        env_config = self.vec_env.config

        with open(self.experiment_dir + '/algo_config.yaml', 'w') as file:
            yaml.dump(self.remove_unserializable(algo_config), file)
        with open(self.experiment_dir + '/env_config.yaml', 'w') as file:
            yaml.dump(self.remove_unserializable(env_config), file)

    @staticmethod
    def remove_unserializable(config):
        clean_config = {}
        for k, v in config.items():
            if isinstance(v, dict):
                clean_config[k] = RlAlgorithm.remove_unserializable(v)
            elif not isinstance(v, (int, float, str, bool, list)):
                print(f'[config] Ignoring unserializable value: {k}')
            else:
                clean_config[k] = v
        return clean_config
