import torch
from torch import nn

from rl_games.algos_torch.network_builder import A2CBuilder


class CoreNetworkBuilder(A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, placeholder, **kwargs):
        """Return the actual torch.nn.Module which includes parameters.

        :param placeholder: not used, for compatibility
        :param kwargs: network parameters
        :return: Network (torch.nn.Module)
        """
        return self.Network(self.params, **kwargs)

    class Network(A2CBuilder.Network):
        def __init__(self, param, **kwargs):
            actions_num = kwargs.get('actions_num')
            super().__init__(param, **kwargs)
            assert self.separate, 'CoreModel only supports separate network'
            if self.fixed_sigma:  # Overwrite sigma to exclude from computation graph
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32),
                                          requires_grad=False)
                sigma_init(self.sigma)
