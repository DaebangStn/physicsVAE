import math
import sys
from typing import Dict, Tuple, Optional, Any
import torch
import numpy as np
from isaacgym import gymapi

from env.vectask import VecTask
from utils import PROJECT_ROOT


class SimpleTask(VecTask):
    def __init__(self, **kwargs):
        self._env_spacing = None
        self._humanoid_asset_filename = None

        self._envs = []
        self._humanoids = []

        super(SimpleTask, self).__init__(**kwargs)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        :arg:
            actions: actions to apply
        :returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        self._pre_physics(actions)
        self._run_physics()
        self._post_physics(actions)

        return self._buf['obs'], self._buf['rew'], torch.ones_like(self._buf['resets']), self._buf['info']

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments having the provided indices.
            If env_ids is None, then reset all environments.

        :returns:
            Observation dictionary
        """
        return self._buf['obs']

    def render(self, mode='rgb_array'):
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

    def _parse_env_param(self, **kwargs):
        env_cfg = super(SimpleTask, self)._parse_env_param(**kwargs)

        self._env_spacing = env_cfg['spacing']
        self._humanoid_asset_filename = env_cfg['humanoid_asset_filename']

    def _create_envs(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        #  Simulator stability
        asset_options.armature = 0.01
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0

        # humanoid
        humanoid_asset = self._gym.load_asset(self._sim, PROJECT_ROOT, self._humanoid_asset_filename, asset_options)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # variable for aggregate
        num_rigid_body = self._gym.get_asset_rigid_body_count(humanoid_asset)
        num_shape = self._gym.get_asset_rigid_shape_count(humanoid_asset)
        self_collision = False

        # inter-env parameter
        num_per_row = math.ceil(np.sqrt(self._num_envs))
        lower = gymapi.Vec3(-self._env_spacing/2, -self._env_spacing/2, 0.0)
        upper = gymapi.Vec3(+self._env_spacing/2, +self._env_spacing/2, self._env_spacing)

        for i in range(self._num_envs):
            env = self._gym.create_env(self._sim, lower, upper, num_per_row)
            self._envs.append(env)

            self._gym.begin_aggregate(env, num_rigid_body, num_shape, self_collision)
            humanoid = self._gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 0, 0)
            self._humanoids.append(humanoid)
            self._gym.end_aggregate(env)

    def _pre_physics(self, actions: torch.Tensor):
        pass

    def _post_physics(self, actions: torch.Tensor):
        pass

    def _run_physics(self):
        self.render()
        self._gym.simulate(self._sim)
