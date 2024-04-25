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

        def actor_module(self, normalized_obs):
            a_out = normalized_obs
            a_out = self.a2c_network.actor_mlp(a_out)

            mu = self.a2c_network.mu_act(self.a2c_network.mu(a_out))
            if self.a2c_network.fixed_sigma:
                logstd = mu * 0.0 + self.a2c_network.sigma_act(self.a2c_network.sigma)
            else:
                logstd = self.a2c_network.sigma_act(self.a2c_network.sigma(a_out))
            return mu, logstd

        def critic_module(self, normalized_obs):
            c_out = normalized_obs
            c_out = self.a2c_network.critic_mlp(c_out)
            normalized_value = self.a2c_network.value_act(self.a2c_network.value(c_out))
            return normalized_value

        def actor(self, obs):
            normalized_obs = self.norm_obs(obs)
            return self.actor_module(normalized_obs)

        def critic(self, obs):
            obs = self.norm_obs(obs)
            normalized_value = self.critic_module(obs)
            return self.denorm_value(normalized_value)

        def forward(self, input_dict):
            normalized_obs = self._compute_obs(input_dict)
            mu, logstd = self.actor_module(normalized_obs)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            normalized_value = self.critic_module(normalized_obs)

            result = {
                'mus': mu,
                'sigmas': sigma,
                'rnn_states': None,
            }

            if input_dict['is_train']:
                prev_neglogp = self.neglogp(input_dict['prev_actions'], mu, sigma, logstd)
                result.update({
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'entropy': distr.entropy().sum(dim=-1),
                    'values': normalized_value,
                })
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result.update({
                    'neglogpacs': torch.squeeze(neglogp),
                    'actions': selected_action,
                    'values': self.denorm_value(normalized_value),
                })
            return result

        def _compute_obs(self, input_dict: dict) -> torch.Tensor:
            obs = input_dict['obs']
            normalized_obs = self.norm_obs(obs)
            return normalized_obs

        @staticmethod
        def _rl_games_compatible_keywords(**kwargs):
            return {
                'obs_shape': kwargs['obs_shape'],
                'normalize_value': kwargs['normalize_value'],
                'normalize_input': kwargs['normalize_input'],
                'value_size': kwargs['value_size'],
            }
