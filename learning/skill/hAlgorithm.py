from rl_games.algos_torch import model_builder
from rl_games.common.a2c_common import swap_and_flatten01

from learning.core.algorithm import CoreAlgorithm
from learning.style.algorithm import StyleAlgorithm, keyp_task_obs_angle_transform, disc_reward, obs_angle_transform
from utils import *
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher


class HighLevelAlgorithm(CoreAlgorithm):
    def __init__(self, **kwargs):
        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None
        self._demo_fetcher = None

        # env related
        self._key_body_ids = None
        self._dof_offsets = None

        # low-level controller
        self._latent_dim = None
        self._llc = None
        self._llc_steps = None
        self._config_llc = None

        # reward
        self._disc_rew_scale = None

        # placeholders for the current episode
        self._disc_obs_exp_buffer = None
        self._task_rew_exp_buffer = None
        self._done_exp_buffer = None

        super().__init__(**kwargs)

    def init_tensors(self):
        super().init_tensors()
        batch_size = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.empty(batch_size + (1,), device=self.device)

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
            self.experience_buffer.update_data('rewards', n, rewards['task'])
            self.experience_buffer.update_data('disc_rewards', n, rewards['disc'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.current_rewards += rewards['task'] * self._task_rew_scale + rewards['disc'] * self._disc_rew_scale
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

    def _calc_rollout_reward(self):
        task_reward = self.experience_buffer.tensor_dict['rewards'] * self._task_rew_scale
        disc_reward = self.experience_buffer.tensor_dict['disc_rewards'] * self._disc_rew_scale
        self.experience_buffer.tensor_dict['rewards'] = task_reward + disc_reward
        self.play_info['task_reward'] = task_reward
        self.play_info['disc_reward'] = disc_reward

    def env_step(self, actions):
        # 1. parameter for env_reset makes hard reset,
        #    because if the reset is done immediately, idle loop makes error
        # 2. fix the done flow in the env_step loop
        # 3. calculate the disc_reward inside the loop

        z = torch.nn.functional.normalize(actions, dim=-1)

        terminate = torch.zeros(self.vec_env.num, device=self.device, dtype=torch.bool)
        obs_raw = None
        for i in range(self._llc_steps):
            obs_step = self.env_reset(self.dones.nonzero()[:, 0], ignore_goal=True)

            self._disc_obs_buf.push_on_reset(obs_step['disc_obs'], self.dones)

            normed_obs = self._llc.norm_obs(obs_step['obs'])
            llc_action, _ = self._llc.actor_latent(normed_obs, z)
            obs_raw, rew, done, info = super().env_step(llc_action)
            assert 'goal' in obs_raw['obs'].keys(), "High-level algorithm must have 'goal' in the observation."
            obs, disc_obs = keyp_task_obs_angle_transform(obs_raw['obs'], self._key_body_ids, self._dof_offsets,
                                                          ignore_goal=True)
            self._disc_obs_buf.push(disc_obs)

            self._task_rew_exp_buffer[:, i] = rew.squeeze()
            self._disc_obs_exp_buffer[:, i, :] = self._disc_obs_buf.history
            self._done_exp_buffer[:, i] = done

            if i == 0:
                self.dones = done
            else:
                self.dones = self.dones | done
            terminate = terminate | info['terminate']

        task_rew_step = (self._task_rew_exp_buffer * ~self._done_exp_buffer).sum(dim=-1, keepdim=True)
        disc_rew = disc_reward(self._llc, self._disc_obs_exp_buffer.view(-1, self._disc_obs_exp_buffer.shape[-1]))
        disc_rew = disc_rew.view(self.vec_env.num, self._llc_steps)
        disc_rew_step = (disc_rew * ~self._done_exp_buffer).sum(dim=-1, keepdim=True)

        return ({'obs': obs_angle_transform(obs_raw['obs'])}, {'task': task_rew_step, 'disc': disc_rew_step},
                self.dones, {'terminate': terminate})

    def env_reset(self, env_ids: Optional[torch.Tensor] = None, ignore_goal=False):
        obs = self.vec_env.reset(env_ids)
        obs, disc_obs = keyp_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets,
                                                      ignore_goal=ignore_goal)
        return {'obs': obs, 'disc_obs': disc_obs}

    def write_tensorboard(self, train_info: dict):
        super().write_tensorboard(train_info)
        info = {
            'disc_reward': train_info['disc_reward'].mean().item(),
        }
        self._write_stat(**info)

    def _init_learning_variables(self, **kwargs):
        config_hparam = self.config
        self._llc, self._config_llc = self.build_llc(config_hparam, self.device)
        self._llc_steps = config_hparam['llc']['steps']

        self._latent_dim = self._config_llc['network']['space']['latent_dim']
        kwargs['params']['config']['env_config']['env']['num_act'] = self._latent_dim
        self.actions_num = self._latent_dim
        self.env_info['action_space'] = reshape_gym_box(self.env_info['action_space'], (self.actions_num,))
        super()._init_learning_variables(**kwargs)

        config_disc = self._config_llc['hparam']['style']['disc']
        self._disc_obs_traj_len = config_disc['obs_traj_len']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
        disc_obs_size = self._disc_obs_traj_len * config_disc['num_obs']

        self._task_rew_exp_buffer = torch.empty(self.vec_env.num, self._llc_steps, device=self.device)
        self._disc_obs_exp_buffer = torch.empty(self.vec_env.num, self._llc_steps, disc_obs_size, device=self.device)
        self._done_exp_buffer = torch.empty(self.vec_env.num, self._llc_steps, dtype=torch.bool, device=self.device)

        self._disc_rew_scale = config_hparam['reward']['disc_scale']
        self._task_rew_w = 1.0

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)

        llc_algo_conf = self._config_llc['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(
            self.vec_env, llc_algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = llc_algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(
            self._disc_obs_traj_len, self.vec_env.dt, self.device, llc_algo_conf['motion_file'],
            llc_algo_conf['joint_information']['dof_body_ids'], self._dof_offsets, self._key_body_ids)
        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self.vec_env.set_motion_fetcher(self._demo_fetcher)

    @staticmethod
    def build_llc(config_hparam, device):
        # load config
        config_env = config_hparam['env_config']['env']
        with open(config_hparam['llc']['cfg'], 'r') as f:
            config_llc = yaml.safe_load(f)

        # check config(env) between HLC and LLC is consistent
        config_llc_env = config_llc['config']['env_config']['env']
        assert 'num_goal' in config_env, "HLC env must have 'num_goal' in the config."
        assert config_env['num_act'] == config_llc_env['num_act'], \
            f"Inconsistent action space. HLC: {config_env['num_act']} LLC: {config_llc_env['num_act']}"
        assert config_env['num_obs'] == config_llc_env['num_obs'] + config_env['num_goal'], \
            f"Inconsistent observation space. HLC: {config_env['num_obs']} LLC: {config_llc_env['num_obs']}, " \
            f"Goal: {config_env['num_goal']}"

        # Override Env# with HLC config
        config_llc_env['num_envs'] = config_env['num_envs']
        config_llc['config']['num_actors'] = config_env['num_envs']

        # build model
        builder = model_builder.ModelBuilder()
        model = builder.load(config_llc)

        additional_config = {
            'actions_num': config_llc_env['num_act'],
            'input_shape': (config_llc_env['num_obs'],),
            'num_seqs': config_hparam['num_actors'],
            'value_size': config_hparam.get('value_size', 1),
            'normalize_value': config_hparam['normalize_value'],
            'normalize_input': config_hparam['normalize_input'],
            'obs_shape': config_llc_env['num_obs'],
        }
        model = model.build(**config_llc['network'], **additional_config)
        model.to(device)

        load_checkpoint_to_network(model, config_hparam['llc']['ckpt'])

        model.eval()
        return model, config_llc
