import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class CoreModel(ModelA2CContinuousLogStd):
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

        def _actor_module(self, obs):
            with torch.no_grad():
                a_out = obs
                a_out = self.a2c_network.actor_mlp(a_out)

                mu = self.a2c_network.mu_act(self.a2c_network.mu(a_out))
                if self.a2c_network.fixed_sigma:
                    logstd = mu * 0.0 + self.a2c_network.sigma_act(self.a2c_network.sigma)
                else:
                    logstd = self.a2c_network.sigma_act(self.a2c_network.sigma(a_out))
                return mu, logstd

        def _critic_module(self, obs):
            with torch.no_grad():
                c_out = obs
                c_out = self.a2c_network.critic_mlp(c_out)
                value = self.a2c_network.value_act(self.a2c_network.value(c_out))
                return value

        def actor(self, obs, **kwargs):
            assert self.a2c_network.separate, 'actor is not supported for non-separate network'
            obs = self.norm_obs(obs)
            return self._actor_module(obs)

        def critic(self, obs, **kwargs):
            assert self.a2c_network.separate, 'critic is not supported for non-separate network'
            obs = self.norm_obs(obs)
            return self._critic_module(obs)

        @staticmethod
        def _rl_games_compatible_keywords(**kwargs):
            return {
                'obs_shape': kwargs['obs_shape'],
                'normalize_value': kwargs['normalize_value'],
                'normalize_input': kwargs['normalize_input'],
                'value_size': kwargs['value_size'],
            }
