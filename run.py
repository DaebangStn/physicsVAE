import isaacgym
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver

from env.simpleTask import SimpleTask
from learning.simplePlayer import SimplePlayer
from learning.simpleAlgorithm import SimpleAlgorithm
from utils.config import load_config, build_args


if __name__ == '__main__':
    args = build_args()
    cfg_run, cfg_train = load_config(args)

    # former one called on train, latter one called on play. rl-games is very weird.
    vecenv.register('simple', lambda config_name, num_actors, **kwargs: SimpleTask(**kwargs))
    env_configurations.register('simple', {
        'vecenv_type': 'simple', 'env_creator': lambda **kwargs: SimpleTask(**kwargs)})

    runner = Runner(algo_observer=IsaacAlgoObserver())
    runner.algo_factory.register_builder('simple', lambda **kwargs: SimpleAlgorithm(**kwargs))
    runner.player_factory.register_builder('simple', lambda **kwargs: SimplePlayer(**kwargs))
    runner.load({'params': cfg_train})

    runner.reset()
    runner.run(cfg_run)
