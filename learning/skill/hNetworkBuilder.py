import torch
import torch.nn as nn

from learning.core.networkBuilder import CoreNetworkBuilder


class HighLevelNetworkBuilder(CoreNetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(CoreNetworkBuilder.Network):
        """torch.nn.Module which includes actual parameters.
        Forwards returns [mu, logstd, value, states]
        """
        def __init__(self, param, **kwargs):
            super().__init__(param, **kwargs)  # build action/value network
            raise NotImplementedError  # todo: load disc from skill
