from rl_games.algos_torch.players import PpoPlayerContinuous


class SimplePlayer(PpoPlayerContinuous):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = int(1e30)
        self.games_num = int(1e30)
