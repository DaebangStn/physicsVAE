from rl_games.algos_torch.players import PpoPlayerContinuous


class SimplePlayer(PpoPlayerContinuous):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
