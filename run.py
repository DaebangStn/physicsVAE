import isaacgym
from utils.config import load_config, build_args, set_seed


def build_runner():
    from rl_games.torch_runner import Runner
    from rl_games.common.algo_observer import IsaacAlgoObserver

    from env.simpleTask import SimpleTask
    from env.rlTask import RlTask
    from env.balanceTask import BalanceTask
    from env.keypointTask import KeypointTask

    from learning.core.model import CoreModel
    from learning.core.networkBuilder import CoreNetworkBuilder
    from learning.style.model import StyleModel
    from learning.style.networkBuilder import StyleNetworkBuilder

    from learning.simple.player import SimplePlayer
    from learning.simple.algorithm import SimpleAlgorithm
    from learning.core.player import CorePlayer
    from learning.core.algorithm import CoreAlgorithm
    from learning.style.player import StylePlayer
    from learning.style.algorithm import StyleAlgorithm

    from utils.rl_games import register_env_rl_games, register_algo_n_player, register_net_n_model

    register_env_rl_games('simpleTask', SimpleTask)
    register_env_rl_games('rlTask', RlTask)
    register_env_rl_games('balanceTask', BalanceTask)
    register_env_rl_games('keypointTask', KeypointTask)

    register_net_n_model('core', CoreNetworkBuilder, CoreModel)
    register_net_n_model('style', StyleNetworkBuilder, StyleModel)

    _runner = Runner(algo_observer=IsaacAlgoObserver())
    register_algo_n_player('simpleAlgo', _runner, SimpleAlgorithm, SimplePlayer)
    register_algo_n_player('rlAlgo', _runner, CoreAlgorithm, CorePlayer)
    register_algo_n_player('styleAlgo', _runner, StyleAlgorithm, StylePlayer)

    return _runner


if __name__ == '__main__':
    args = build_args()
    cfg_run, cfg_train = load_config(args)
    set_seed(cfg_train['seed'])

    runner = build_runner()
    runner.load({'params': cfg_train})

    runner.reset()
    runner.run(cfg_run)
