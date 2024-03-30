import yaml
from rl_games.algos_torch import model_builder
from rl_games.algos_torch import torch_ext

from learning.core.algorithm import CoreAlgorithm
from learning.style.algorithm import StyleAlgorithm, motion_lib_angle_transform
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher, SingleTensorBuffer


class HighLevelAlgorithm(CoreAlgorithm):
    def __init__(self, **kwargs):
        self._latent_dim = None

        # discriminator related
        self._disc_obs_buf = None
        self._disc_obs_traj_len = None

        # low-level controller
        self._llc_actor = None
        self._llc_disc = None
        self._llc_steps = None

        # reward
        self._rew_task_scale = None
        self._rew_disc_scale = None

        # placeholders for the current episode
        self._rollout_obses = None
        self._mean_task_reward = None
        self._mean_style_reward = None
        self._std_task_reward = None
        self._std_style_reward = None

        super().__init__(**kwargs)

    def env_step(self, actions):

        for _ in range(self._llc_steps):

            # pre step in the style algorithm

            action = self._llc_action(self.obs)
            obs, rew, done, info = super().env_step(actions)

            # post step in the style algorithm

        raise NotImplementedError

    def env_reset(self):
        raise NotImplementedError

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        dataset_dict = self.dataset.values_dict

        dataset_dict['demo_obs'] = self._demo_replay_buffer.sample(self.batch_size)
        self._update_demo_buffer()

    def _init_learning_variables(self, **kwargs):
        config_hparam = self.config
        self._llc_actor, self._llc_disc, self._latent_dim = self.build_llc(config_hparam, self.device)
        self._llc_steps = config_hparam['llc']['steps']

        kwargs['params']['config']['env_config']['env']['num_act'] = self._latent_dim
        super()._init_learning_variables(**kwargs)

        config_disc = config_hparam['style']['disc']
        self._disc_obs_traj_len = config_disc['obs_traj_len']
        self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)

        config_rew = config_hparam['reward']
        self._rew_task_scale = config_rew['task_scale']
        self._rew_disc_scale = config_rew['disc_scale']

    def _llc_action(self, obs):
        raise NotImplementedError

    def _prepare_data(self, **kwargs):
        super()._prepare_data(**kwargs)
        algo_conf = kwargs['params']['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = algo_conf['joint_information']['dof_offsets']
        # TODO: very weird loader sorry...
        self._demo_fetcher = MotionLibFetcher(**MotionLibFetcher.demo_fetcher_config(self, algo_conf))

        # build replay buffer only for demo
        buf_size = self.config['hlc']['demo_buf_size']
        self._demo_replay_buffer = SingleTensorBuffer(buf_size, self.device)
        demo_obs = self._demo_fetcher.fetch(buf_size // self._disc_obs_traj_len)
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._demo_replay_buffer.store(demo_obs)

    def _update_demo_buffer(self):
        demo_obs = self._demo_fetcher.fetch(max(self.batch_size // 2048, 1))  # 2048 is a magic number for performance
        demo_obs = motion_lib_angle_transform(demo_obs, self._dof_offsets, self._disc_obs_traj_len)
        self._demo_replay_buffer.store(demo_obs)

    @staticmethod
    def build_llc(config_hparam, device):
        # load config
        config_env = config_hparam['env_config']['env']
        with open(config_hparam['llc']['cfg'], 'r') as f:
            config_llc = yaml.safe_load(f)
        latent_dim = config_llc['network']['space']['latent_dim']

        # check config(env) between HLC and LLC is consistent
        config_llc_env = config_llc['config']['env_config']['env']
        assert config_env['num_act'] == config_llc_env['num_act'], \
            f"Inconsistent action space. HLC: {config_env['num_act']} LLC: {config_llc_env['num_act']}"
        assert config_env['num_obs'] == (config_llc_env['num_obs'] - latent_dim), \
            (f"Inconsistent observation space. "
             f"HLC: {config_env['num_obs']} LLC: {config_llc_env['num_obs'] - latent_dim}")

        # build model
        builder = model_builder.ModelBuilder()
        model = builder.load(config_llc)

        additional_config = {
            'actions_num': config_env['num_act'],
            'input_shape': (config_env['num_obs'] + latent_dim,),
            'num_seqs': config_hparam['num_actors'],
            'value_size': config_hparam.get('value_size', 1),
            'normalize_value': config_hparam['normalize_value'],
            'normalize_input': config_hparam['normalize_input'],
            'obs_shape': config_env['num_obs'] + latent_dim,
        }
        model = model.build(**config_llc['network'], **additional_config)
        model.to(device)

        # load checkpoint
        ckpt = torch_ext.load_checkpoint(config_hparam['llc']['ckpt'])
        model.load_state_dict(ckpt['model'])
        if config_llc['hparam']['normalize_input'] and 'running_mean_std' in ckpt:
            model.running_mean_std.load_state_dict(ckpt['running_mean_std'])

        model.eval()
        return model.actor, model.disc, latent_dim