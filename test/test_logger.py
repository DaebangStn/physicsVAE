import torch
import pytest
from tensorboardX import SummaryWriter

from learning.logger.hdfBase import HdfBaseLogger
from learning.logger.action import ActionLogger
from learning.logger.latentMotion import LatentMotionLogger
from learning.logger.jitter import JitterLogger


@pytest.mark.logger
def test_base_logger():
    logger = HdfBaseLogger("test", "test", {})
    logger.log(torch.tensor([1, 2, 3]))


@pytest.mark.logger
def test_action_logger():
    logger = ActionLogger("test", "test", 3, {})
    logger.log(torch.tensor([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.logger
def test_latent_logger():
    logger = LatentMotionLogger("test", "test", 3, latent_dim=3, cfg={})
    logger.update_z(torch.tensor([0, 1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    logger.log(torch.Tensor([1, 2, 3]))
    logger.log(torch.Tensor([4, 5, 6]))
    logger.update_z(torch.tensor([2]), torch.tensor([[4, 5, 6]]))
    logger.log(torch.Tensor([7, 8, 9]))
    logger.log(torch.Tensor([10, 11, 12]))
    logger.update_z(torch.tensor([1, 2]), torch.tensor([[1, 2, 3], [4, 5, 6]]))
    logger.log(torch.Tensor([13, 14, 15]))
    logger.log(torch.Tensor([16, 17, 18]))


@pytest.mark.logger
def test_jitter_logger():
    writer = SummaryWriter('test')
    logger = JitterLogger(writer, 2)
    logger.log(0, 0)
    logger.log(1, 1)
    logger.log(4, 2)
    logger.log(9, 3)
    logger.log(16, 4)