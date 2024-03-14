import isaacgym
from isaacgym import gymtorch
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver

from env.simpleTask import SimpleTask
from env.rlTask import RlTask
from env.balanceTask import BalanceTask
from learning.simplePlayer import SimplePlayer
from learning.simpleAlgorithm import SimpleAlgorithm
from learning.rlPlayer import RlPlayer
from learning.rlAlgorithm import RlAlgorithm
from utils.rl_games import register_env_rl_games, register_builder_to_runner
from utils.config import load_config, build_args, set_seed


if __name__ == '__main__':
    args = build_args()
    cfg_run, cfg_train = load_config(args)
    set_seed(cfg_train['seed'])

    register_env_rl_games('simpleTask', SimpleTask)
    register_env_rl_games('rlTask', RlTask)
    register_env_rl_games('balanceTask', BalanceTask)

    runner = Runner(algo_observer=IsaacAlgoObserver())
    register_builder_to_runner('simpleAlgo', runner, SimpleAlgorithm, SimplePlayer)
    register_builder_to_runner('rlAlgo', runner, RlAlgorithm, RlPlayer)
    runner.load({'params': cfg_train})

    runner.reset()
    runner.run(cfg_run)
