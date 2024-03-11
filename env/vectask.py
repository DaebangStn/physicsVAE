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
        self._gym.prepare_sim(self._sim)

        # Environment related variable
        self._num_envs = None
        self._num_obs = None
        self._num_actions = None
        self._num_states = None
        self._buf = None

        self._parse_env_param(**kwargs)
        self._allocate_buffers()

        # rl-games related variables
        self._num_agents = 1  # used for multi-agent environments
        self.observation_space = spaces.Box(low=np.ones(self._num_obs) * -np.Inf, high=np.ones(self._num_obs) * np.Inf)
        self.state_space = spaces.Box(low=np.ones(self._num_states) * -np.Inf, high=np.ones(self._num_states) * np.Inf)
        self.action_space = spaces.Box(low=np.ones(self._num_actions) * -1., high=np.ones(self._num_actions) * 1.)

        self._viewer = None
        if not self._headless:
            self._install_viewer()

    def _parse_sim_param(self, **kwargs):
        sim_cfg = kwargs['sim']

        self._headless = sim_cfg['headless']

        self._compute_device = sim_cfg['device_id']
        self._graphics_device = self._compute_device if not self._headless else -1

        engine = sim_cfg['engine']
        if engine == 'PHYSX':
            sim_engine = gymapi.SIM_PHYSX
        elif engine == 'FLEX':
            sim_engine = gymapi.FLEX
        else:
            raise ValueError('Unknown engine {}'.format(engine))

        sim_params = gymapi.SimParams()
        # set z axis to upward
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self._sim = self._gym.create_sim(self._compute_device, self._graphics_device, type=sim_engine,
                                         params=sim_params)

    def _parse_env_param(self, **kwargs):
        env_cfg = kwargs['env']

        self._num_envs = env_cfg['num_envs']
        self._num_obs = env_cfg['num_obs']
        self._num_actions = env_cfg['num_act']
        self._num_states = env_cfg['num_states']

    def _allocate_buffers(self):
        self._buf = {
            'obs': torch.zeros((self._num_envs, self._num_obs), dtype=torch.float32, device=self._compute_device),
            'states': torch.zeros((self._num_envs, self._num_states), dtype=torch.float32, device=self._compute_device),
            'rew': torch.zeros(self._num_envs, dtype=torch.float32, device=self._compute_device),
            'resets': torch.zeros(self._num_envs, dtype=torch.float32, device=self._compute_device),
            'info': {},
        }

    def _install_viewer(self):
        self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        self._gym.subscribe_viewer_keyboard_event(self._viewer, gymapi.KEY_ESCAPE, "QUIT")

        # Suppose Z axis upward
        self._gym.viewer_camera_look_at(self._viewer, None,
                                        cam_pos=gymapi.Vec3(10.0, 10.0, 10.0),
                                        cam_target=gymapi.Vec3(0.0, 0.0, 0.0))

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
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        pass

    @abstractmethod
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments having the provided indices.
            If env_ids is None, then reset all environments.
        Returns:
            Observation dictionary
        """
        pass
