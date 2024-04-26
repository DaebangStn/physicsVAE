from typing import Tuple, Dict, Any

import torch

from env.keypoint import KeypointTask


class KeypointMaxObsTask(KeypointTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = (self._buf["aPos"].clone(), self._buf["aRot"].clone(), self._buf["aVel"].clone(),
                            self._buf["aAnVel"].clone(), self._buf["dPos"].clone(), self._buf["dVel"].clone(),
                            self._buf["rPos"].clone(), self._buf["rRot"].clone(), self._buf["rVel"].clone(),
                            self._buf["rAnVel"].clone(),)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._pre_physics(actions)
        self._run_physics()
        self._post_physics(actions)

        return self._buf['obs'], self._buf['rew'].clone(), self._buf['reset'].clone(), self._buf['info']
