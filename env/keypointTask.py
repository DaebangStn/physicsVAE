from typing import Optional

import torch.jit
from isaacgym import gymapi

from env.rlTask import RlTask
from utils.angle import *
from utils.env import sample_color


class KeypointTask(RlTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_color(self, env_ids: Optional[torch.Tensor] = None):
        if self._viewer is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self._num_envs)
        num_env = len(env_ids)
        if num_env == 0:
            return

        color = sample_color(num_env)
        for env_id in env_ids:
            for r_id in range(self._num_rigid_body):
                self._gym.set_rigid_body_color(self._envs[env_id], self._humanoids[env_id], r_id,
                                               gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))

    def key_body_ids(self, key_body_names: List[str]) -> List[int]:
        return [self._gym.find_actor_rigid_body_handle(self._envs[0], self._humanoids[0], name)
                for name in key_body_names]

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = (self._buf["aPos"], self._buf["aRot"], self._buf["aVel"], self._buf["aAnVel"],
                            self._buf["dPos"], self._buf["dVel"], self._buf["rPos"])
