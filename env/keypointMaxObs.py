from typing import Tuple, Dict, Any

import torch

from env.keypoint import KeypointTask


class KeypointMaxObsTask(KeypointTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = {
            "aPos": self._buf["aPos"].clone(),
            "aRot": self._buf["aRot"].clone(),
            "aVel": self._buf["aVel"].clone(),
            "aAnVel": self._buf["aAnVel"].clone(),
            "dPos": self._buf["dPos"].clone(),
            "dVel": self._buf["dVel"].clone(),
            "rPos": self._buf["rPos"].clone(),
            "rRot": self._buf["rRot"].clone(),
            "rVel": self._buf["rVel"].clone(),
            "rAnVel": self._buf["rAnVel"].clone(),
        }

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._pre_physics(actions)
        self._run_physics()
        self._post_physics(actions)

        return self._buf['obs'], self._buf['rew'].clone(), self._buf['reset'].clone(), self._buf['info']
