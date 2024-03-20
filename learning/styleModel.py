from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class StyleModel(ModelA2CContinuousLogStd):
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

        def forward(self, input_dict):
            output_dict = super().forward(input_dict)
            if input_dict.get('is_train', False):
                output_dict['rollout_disc'] = self.disc(input_dict['rollout_obs'])
                output_dict['replay_disc'] = self.a2c_network.disc(input_dict['replay_obs'])
                output_dict['demo_disc'] = self.disc(input_dict['demo_obs'])
            return output_dict

        def disc(self, obs):
            return self.a2c_network.disc(obs)

        @property
        def disc_logistics_weights(self):
            return self.a2c_network.disc_logistics_weights

        @property
        def disc_mlp_weights(self):
            return self.a2c_network.disc_mlp_weights

        @staticmethod
        def _rl_games_compatible_keywords(**kwargs):
            return {
                'obs_shape': kwargs['obs_shape'],
                'normalize_value': kwargs['normalize_value'],
                'normalize_input': kwargs['normalize_input'],
                'value_size': kwargs['value_size'],
            }
