import torch
import pytest

from learning.logger.base import BaseLogger
from learning.logger.action import ActionLogger
from learning.logger.latentMotion import LatentMotionLogger


@pytest.mark.logger
def test_base_logger():
    logger = BaseLogger("test", "test")
    logger.log(torch.tensor([1, 2, 3]))


@pytest.mark.logger
def test_action_logger():
    logger = ActionLogger("test", "test", 3)
    logger.log(torch.tensor([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.logger
def test_latent_logger():
    logger = LatentMotionLogger("test", "test", 3)
    logger.update_z(torch.tensor([1, 2, 3], dtype=torch.float32))
    logger.log(1)
    logger.log(2)
    logger.update_z(torch.tensor([4, 5, 6], dtype=torch.float32))
    logger.log(3)
    logger.log(4)
    logger.update_z(torch.tensor([1, 2, 3], dtype=torch.float32))
    logger.log(5)
    logger.log(6)
