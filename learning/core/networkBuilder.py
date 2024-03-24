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
