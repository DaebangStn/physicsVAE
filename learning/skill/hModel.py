import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.core.model import CoreModel


class HighLevelModel(CoreModel):
    def __init__(self, network):
        super().__init__(network)

    class NetworkWrapper(CoreModel.NetworkWrapper):
        def __init__(self, net, **kwargs):
            super().__init__(net, **kwargs)
            raise NotImplementedError  # todo: build disc from skill
