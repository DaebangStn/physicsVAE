import sys
from abc import abstractmethod
from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np
from gym import spaces

from isaacgym import gymapi


class VecTask:
    def __init__(self, **kwargs):
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self._gym = gymapi.acquire_gym()

        # Simulation related variables
        self._headless = None
        self._compute_device = None
        self._graphics_device = None
        self._sim = None

        self._parse_sim_param(**kwargs)

        # Environment related variable
        self._num_envs = None
        self._num_obs = None
        self._num_actions = None
        self._num_states = None
        self._buf = None

        self._parse_env_param(**kwargs)

        # rl-games related variables
        self._num_agents = 1  # used for multi-agent environments
        self.observation_space = None
        self.action_space = None
        self.state_space = None

        self._allocate_buffers()

        self._create_sim()

        self._viewer = None
        if not self._headless:
            self._install_viewer()

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        :arg:
            actions: actions to apply
        :return:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        self._pre_physics(actions)
        self._run_physics()
        self._post_physics(actions)

        return self._buf['obs'], self._buf['rew'], torch.ones_like(self._buf['resets']), self._buf['info']

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments having the provided indices. If env_ids is None, then reset all environments.

        :return:
            Observation dictionary
        """
        return self._buf['obs']

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self._viewer is None:
            print("Rendering without viewer. Invalid access")
            sys.exit()

        for evt in self._gym.query_viewer_action_events(self._viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()

        self._gym.fetch_results(self._sim, None)

        self._gym.step_graphics(self._sim)
        self._gym.draw_viewer(self._viewer, self._sim, True)

    def _allocate_buffers(self):
        self._buf = {
            'obs': torch.zeros((self._num_envs, self._num_obs), dtype=torch.float32, device=self._compute_device),
            'states': torch.zeros((self._num_envs, self._num_states), dtype=torch.float32, device=self._compute_device),
            'rew': torch.zeros(self._num_envs, dtype=torch.float32, device=self._compute_device),
            'resets': torch.zeros(self._num_envs, dtype=torch.float32, device=self._compute_device),
            'info': {},
        }

        # To make no precision warning, we should use np.full. Not np.ones * scale
        self.observation_space = spaces.Box(
            low=np.full(self._num_obs, -np.Inf, dtype=np.float32),
            high=np.full(self._num_obs, np.Inf, dtype=np.float32),
            dtype=np.float32)
        self.state_space = spaces.Box(
            low=np.full(self._num_states, -np.Inf, dtype=np.float32),
            high=np.full(self._num_states, np.Inf, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(self._num_actions, -1.0, dtype=np.float32),
            high=np.full(self._num_actions, 1.0, dtype=np.float32),
            dtype=np.float32)

    def _create_ground(self):
        param = gymapi.PlaneParams()
        param.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        param.dynamic_friction = 0.5
        param.static_friction = 0.5
        param.restitution = 0.5
        self._gym.add_ground(self._sim, param)

    def _create_sim(self):
        self._sim = self._gym.create_sim(self._compute_device, self._graphics_device, type=self._sim_engine,
                                         params=self._sim_params)
        self._create_ground()
        self._create_envs()
        self._gym.prepare_sim(self._sim)

    def _install_viewer(self):
        self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_Q, "QUIT")

        # Suppose Z axis upward
        cam_pos = gymapi.Vec3(3.0, 3.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

    def _parse_sim_param(self, **kwargs):
        """ Parse simulation parameters. (Under the "sim" key in the env_config)
        To add additional parameters, declare new variables in the __init__ and override this method.
        Do not create classes' own methods.

        :param kwargs:
        :return: sim_cfg (used by overriding methods)
        """
        sim_cfg = kwargs['sim']

        self._headless = sim_cfg['headless']

        self._compute_device = sim_cfg['device_id']
        self._graphics_device = self._compute_device if not self._headless else -1

        self._sim_params = gymapi.SimParams()
        engine = sim_cfg['engine']
        if engine == 'PHYSX':
            self._sim_engine = gymapi.SIM_PHYSX
            self._sim_params.physx.use_gpu = True
        else:
            raise ValueError('Unknown engine {}'.format(engine))
        self._sim_params.use_gpu_pipeline = True

        # set z axis to upward
        self._sim_params.up_axis = gymapi.UP_AXIS_Z
        # self._sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self._sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        return sim_cfg

    def _parse_env_param(self, **kwargs):
        """ Parse environment parameters. (Under the "env" key in the env_config)
        To add additional parameters, declare new variables in the __init__ and override this method.
        Do not create classes' own methods.

        :param kwargs:
        :return: env_cfg (used by overriding methods)
        """
        env_cfg = kwargs['env']

        self._num_envs = env_cfg['num_envs']
        self._num_obs = env_cfg['num_obs']
        self._num_actions = env_cfg['num_act']
        self._num_states = env_cfg['num_states']

        return env_cfg

    def _run_physics(self):
        self.render()
        self._gym.simulate(self._sim)

    # Called by rl-games
    def get_number_of_agents(self):
        return self._num_agents

    # Called by rl-games
    def get_env_info(self):
        info = {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
            'state_space': self.state_space,
            'use_global_observations': False,
            'agents': self._num_agents,
            'value_size': 1,
        }
        if hasattr(self, 'use_central_value'):
            info['use_global_observations'] = self.use_central_value
        if hasattr(self, 'value_size'):
            info['value_size'] = self.value_size
        return info

    # Called by rl-games
    def set_train_info(self, env_frames, *args, **kwargs):
        """Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process.
        This is useful for implementing curriculums and things such as that.
        """
        pass

    @abstractmethod
    def _create_envs(self):
        """VecTask didn't create any environments (just created ground in _parse_sim_param)
        So that child class must implement this method.
        Then this method is called by _create_sim

        :return:
            None
        """
        pass

    @abstractmethod
    def _pre_physics(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def _post_physics(self, actions: torch.Tensor):
        pass
