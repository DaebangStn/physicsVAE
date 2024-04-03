import torch.jit

from env.humanoid import HumanoidTask
from utils.angle import *


class KeypointTask(HumanoidTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def key_body_ids(self, key_body_names: List[str]) -> List[int]:
        return [self._gym.find_actor_rigid_body_handle(self._envs[0], self._humanoids_id_env[0], name)
                for name in key_body_names]

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = (self._buf["aPos"], self._buf["aRot"], self._buf["aVel"], self._buf["aAnVel"],
                            self._buf["dPos"], self._buf["dVel"], self._buf["rPos"], self._buf["rRot"],
                            self._buf["rVel"], self._buf["rAnVel"],)
