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
            net = self.a2c_network
            a_out = normalized_obs.contiguous().view(normalized_obs.size(0), -1)
            a_out = net.actor_mlp(a_out)

            mu = net.mu_act(net.mu(a_out))
            if net.fixed_sigma:
                logstd = mu * 0.0 + net.sigma_act(net.sigma)
            else:
                logstd = net.sigma_act(net.sigma(a_out))
            return mu, logstd

        def critic_module(self, normalized_obs):
            net = self.a2c_network
            c_out = normalized_obs
            c_out = net.critic_mlp(c_out)
            normalized_value = net.value_act(net.value(c_out))
            return normalized_value

        def actor(self, normed_obs: torch.Tensor):
            mu, logstd = self.actor_module(normed_obs)
            sigma = torch.exp(logstd)
            return mu, sigma

        def critic(self, normed_obs: torch.Tensor):
            normalized_value = self.critic_module(normed_obs)
            value = self.denorm_value(normalized_value)
            return value

        def forward(self, input_dict):
            normalized_obs = input_dict['obs']
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

        @staticmethod
        def _rl_games_compatible_keywords(**kwargs):
            return {
                'obs_shape': kwargs['obs_shape'],
                'normalize_value': kwargs['normalize_value'],
                'normalize_input': kwargs['normalize_input'],
                'value_size': kwargs['value_size'],
            }
