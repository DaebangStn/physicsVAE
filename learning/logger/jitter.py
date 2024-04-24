from typing import Optional

import torch
from tensorboardX import SummaryWriter

from learning.logger.tensorBoardBase import TensorBoardBaseLogger


class JitterLogger(TensorBoardBaseLogger):
    def __init__(self, writer: SummaryWriter, name: str, derivation_order: int = 2, category: Optional[str] = None):
        assert derivation_order in [1, 2], f"Derivation order must be either 1 or 2, but got {derivation_order}"
        category = category if category is not None else 'info'
        super().__init__(writer, f'jitter_{name}', category)

        self._dOrder = derivation_order
        if self._dOrder == 1:
            self._prev = None
        elif self._dOrder == 2:
            self._prev2 = None
            self._prev = None

    def _process_data(self, data: torch.Tensor) -> float:
        if self._dOrder == 1:
            if self._prev is not None:
                jitter = data - self._prev
            else:
                jitter = torch.zeros_like(data)
            self._prev = data
        elif self._dOrder == 2:
            if self._prev is not None and self._prev2 is not None:
                jitter = data - 2 * self._prev + self._prev2
            else:
                jitter = torch.zeros_like(data)
            self._prev2 = self._prev
            self._prev = data.clone()
        else:
            raise NotImplementedError(f"Derivation order {self._dOrder} is not supported")

        return torch.abs(jitter).mean().item()
