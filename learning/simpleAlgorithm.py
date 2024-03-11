from rl_games.algos_torch.a2c_continuous import A2CAgent


class SimpleAlgorithm(A2CAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
