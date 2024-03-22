from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class BaseModel(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)

    def build(self, **kwargs):
        net = self.network_builder.build('', **kwargs)
        return self.NetworkWrapper(net, **kwargs)

    class NetworkWrapper(ModelA2CContinuousLogStd.Network):
        """torch.nn.Module which post-processes the network(variable 'net') output
        to compatible with rl-games.
        """
        def __init__(self, net, **kwargs):
            super().__init__(net, **self._rl_games_compatible_keywords(**kwargs))

        @staticmethod
        def _rl_games_compatible_keywords(**kwargs):
            return {
                'obs_shape': kwargs['obs_shape'],
                'normalize_value': kwargs['normalize_value'],
                'normalize_input': kwargs['normalize_input'],
                'value_size': kwargs['value_size'],
            }
