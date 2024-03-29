import time
import torch
from torch.utils.tensorboard import SummaryWriter
from rl_games.common.player import BasePlayer
from rl_games.algos_torch.players import PpoPlayerContinuous

from utils.rl_games import rl_games_net_build_param


class CorePlayer(PpoPlayerContinuous):
    def __init__(self, **kwargs):
        BasePlayer.__init__(self, kwargs['params'])

        # env related
        self.is_tensor_obses = True

        self.network = None
        self.model = None
        self._writer = None

        self.num_actors = None
        self.actions_num = None
        self.actions_low = None
        self.actions_high = None

        self.normalize_input = None
        self.normalize_value = None
        self.is_rnn = None

        self.max_steps = int(1e30)
        self.games_num = int(1e30)
        self._games_played = None
        self._n_step = None

        # placeholder for rollout
        self.dones = None
        self.obses = None

        self._init_variables(**kwargs)

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        self._games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if self._games_played >= n_games:
                break

            self.obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(self.obses['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for self._n_step in range(self.max_steps):
                self.obses = self.env_reset(self.env)
                if self.evaluation and self._n_step % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        self.obses['obs'], masks, is_deterministic)
                else:
                    action = self.get_action(self.obses['obs'], is_deterministic)

                self._pre_step()
                self.obses, r, self.done, info = self.env_step(self.env, action)
                self._post_step()
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = self.done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                self._games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - self.done.float())
                    steps = steps * (1.0 - self.done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                            self._writer.add_scalar()
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or self._games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / self._games_played * n_game_life, 'av steps:', sum_steps /
                  self._games_played * n_game_life, 'winrate:', sum_game_res / self._games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / self._games_played * n_game_life,
                  'av steps:', sum_steps / self._games_played * n_game_life)

    def env_step(self, env, actions):
        obs, rew, done, info = super().env_step(env, actions)
        return {'obs': obs}, rew, done, info

    def env_reset(self, env):
        return {'obs': super().env_reset(env)}

    def _init_variables(self, **kwargs):
        self.network = self.config['network']

        self.num_actors = self.env.num
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        self.model = self.network.build(**kwargs['params']['network'], **rl_games_net_build_param(self))
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        self.dones = torch.zeros(self.num_actors, dtype=torch.bool, device=self.device)

        logdir = 'runs/' + kwargs['params']['config']['full_experiment_name']
        self._writer = SummaryWriter(log_dir=logdir)

    def _pre_step(self):
        pass

    def _post_step(self):
        pass
