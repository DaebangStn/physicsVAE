from isaacgym import gymapi

from env.droppedHumanoid import DroppedHumanoidTask
from env.rsiHumanoid import RSIHumanoidTask
from env.humanoid import HumanoidTask
from utils.angle import *


class KeypointTask(RSIHumanoidTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_color(self, env_ids: torch.Tensor, col: torch.Tensor):
        if self._viewer is None:
            return

        num_env = len(env_ids)
        if num_env == 0:
            return

        for i, env_id in enumerate(env_ids):
            for r_id in range(self._num_humanoid_rigid_body):
                self._gym.set_rigid_body_color(self._envs[env_id], self._humanoids_id_env[env_id], r_id,
                                               gymapi.MESH_VISUAL, gymapi.Vec3(col[i, 0], col[i, 1], col[i, 2]))

    def key_body_ids(self, key_body_names: List[str]) -> List[int]:
        return [self._gym.find_actor_rigid_body_handle(self._envs[0], self._humanoids_id_env[0], name)
                for name in key_body_names]

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = (self._buf["aPos"], self._buf["aRot"], self._buf["aVel"], self._buf["aAnVel"],
                            self._buf["dPos"], self._buf["dVel"], self._buf["rPos"])
