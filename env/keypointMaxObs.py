from env.keypoint import KeypointTask


class KeypointMaxObsTask(KeypointTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_observations(self):
        #  It must be processed before the model is called.
        #  Also, rPos should be picked to keypoint body ids.
        self._buf["obs"] = (self._buf["aPos"], self._buf["aRot"], self._buf["aVel"], self._buf["aAnVel"],
                            self._buf["dPos"], self._buf["dVel"], self._buf["rPos"], self._buf["rRot"],
                            self._buf["rVel"], self._buf["rAnVel"],)
