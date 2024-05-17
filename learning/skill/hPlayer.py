from learning.core.player import CorePlayer
from learning.skill.hAlgorithm import HighLevelAlgorithm
from learning.style.algorithm import StyleAlgorithm, keyp_task_obs_angle_transform, disc_reward, obs_angle_transform
from utils import *
from utils.buffer import TensorHistoryFIFO, MotionLibFetcher


class HighLevelPlayer(CorePlayer):
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
        self._llc_actor = None
        self._llc_disc = None
        self._llc_steps = None
        self._config_llc = None

        self._show_reward = None

        super().__init__(**kwargs)

    def env_step(self, env, actions):
        z = actions
        rew_step = torch.zeros(self.env.num, device=self.device)
        disc_rew_step = torch.zeros(self.env.num, device=self.device)
        obs_raw = None
        for i in range(self._llc_steps):
            obs_step = self.env_reset(self.dones, ignore_goal=True)

            if self._show_reward:
                self._disc_obs_buf.push_on_reset(obs_step['disc_obs'], self.dones)

            normed_obs = self._llc_actor.norm_obs(obs_step['obs'])
            llc_action, _ = self._llc_actor.actor_latent(normed_obs, z)
            obs_raw, rew, done, info = super().env_step(env, llc_action)
            obs, disc_obs = keyp_task_obs_angle_transform(obs_raw['obs'], self._key_body_ids, self._dof_offsets,
                                                          ignore_goal=True)
            rew_step[~done] += rew[~done]
            if self._show_reward:
                disc_rew = disc_reward(self._llc_disc, disc_obs)
                disc_rew_step[~done] += disc_rew[~done]

            if i == 0:
                self.dones = done
            else:
                self.dones = self.dones | done

        return {'obs': obs_angle_transform(obs_raw['obs'])}, rew_step, self.dones, {}

    def env_reset(self, env, ignore_goal=False):
        obs = self.env.reset()
        obs, disc_obs = keyp_task_obs_angle_transform(obs, self._key_body_ids, self._dof_offsets,
                                                      ignore_goal=ignore_goal)
        return {'obs': obs, 'disc_obs': disc_obs}

    def _init_variables(self, **kwargs):
        config_hparam = self.config
        self._llc_actor, self._config_llc = HighLevelAlgorithm.build_llc(config_hparam, self.device)
        self._llc_steps = config_hparam['llc']['steps']

        self._latent_dim = self._config_llc['network']['space']['latent_dim']
        kwargs['params']['config']['env_config']['env']['num_act'] = self._latent_dim
        self.actions_num = self._latent_dim
        self.action_space = reshape_gym_box(self.env_info['action_space'], (self.actions_num,))
        super()._init_variables(**kwargs)

        llc_algo_conf = self._config_llc['algo']
        self._key_body_ids = StyleAlgorithm.find_key_body_ids(
            self.env, llc_algo_conf['joint_information']['key_body_names'])
        self._dof_offsets = llc_algo_conf['joint_information']['dof_offsets']
        self._demo_fetcher = MotionLibFetcher(
            self._disc_obs_traj_len, self.env.dt, self.device, llc_algo_conf['motion_file'],
            llc_algo_conf['joint_information']['dof_body_ids'], self._dof_offsets, self._key_body_ids)
        env_conf = kwargs['params']['config']['env_config']['env']
        if "reference_state_init_prob" in env_conf:
            self.env.set_motion_fetcher(self._demo_fetcher)

        self._show_reward = self.config['reward'].get('show_on_player', False)

        if self._show_reward:
            config_disc = self._config_llc['hparam']['style']['disc']
            self._disc_obs_traj_len = config_disc['obs_traj_len']
            self._disc_obs_buf = TensorHistoryFIFO(self._disc_obs_traj_len)
