from typing import Dict, Tuple, Optional, Any
import torch

from env.vectask import VecTask


class SimpleTask(VecTask):
    def __init__(self, **kwargs):
        super(SimpleTask, self).__init__(**kwargs)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        # print("step")
        return self._buf['obs'], self._buf['rew'], torch.ones_like(self._buf['resets']), self._buf['info']

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environments having the provided indices.
            If env_ids is None, then reset all environments.
        Returns:
            Observation dictionary
        """
        return self._buf['obs']
