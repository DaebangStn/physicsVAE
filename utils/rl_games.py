from typing import Type

from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner
from rl_games.common.player import BasePlayer
from rl_games.algos_torch.models import BaseModel
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.interfaces.base_algorithm import BaseAlgorithm

from env.vectask import VecTask


def return_vectask(**kwargs):
    return VecTask(**kwargs)


def register_env_rl_games(name: str, env: Type['VecTask']):
    # former one called on train, latter one called on play. rl-games is very weird.
    vecenv.register(name, lambda config_name, num_actors, **kwargs: env(**kwargs))
    env_configurations.register(name, {'vecenv_type': name, 'env_creator': lambda **kwargs: env(**kwargs)})


def register_algo_n_player(name: str, runner: Runner, algo: Type['BaseAlgorithm'], player: Type['BasePlayer']):
    runner.algo_factory.register_builder(name, lambda **kwargs: algo(**kwargs))
    runner.player_factory.register_builder(name, lambda **kwargs: player(**kwargs))


def register_net_n_model(name: str, network: Type['NetworkBuilder'], model: Type['BaseModel']):
    model_builder.register_model(name, model)
    model_builder.register_network(name, lambda **kwargs: network(**kwargs))


def rl_games_net_build_param(cls):
    return {
        'actions_num': cls.actions_num,
        'input_shape': cls.obs_shape,
        'num_seqs': cls.num_actors * cls.num_agents,
        'value_size': cls.env_info.get('value_size', 1),
        'normalize_value': cls.normalize_value,
        'normalize_input': cls.normalize_input,
        'obs_shape': cls.obs_shape,
    }
