import torch
from torch import nn

from learning.style.networkBuilder import StyleNetworkBuilder


class SkillNetworkBuilder(StyleNetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(StyleNetworkBuilder.Network):
        """torch.nn.Module which includes actual parameters.
        Forwards returns [mu, logstd, value, states]
        """
        def __init__(self, param, **kwargs):
            self._input_shape = None
            self._latent_dim = int(kwargs['space']['latent_dim'])
            super().__init__(param, **kwargs)  # build action/value/disc network

            self._build_enc()

            self._build_latent_net = kwargs['latent']['build']
            if self._build_latent_net:
                self._build_latent_enc(**kwargs)

        def _build_latent_enc(self, **kwargs):
            units = kwargs['latent']['units']
            assert len(units) == 2, 'latent_enc_mlp requires 2 hidden layers'

            self._latent_enc_mlp = nn.Sequential(
                nn.Linear(self._latent_dim, units[0]),
                nn.ReLU(),
                nn.Linear(units[0], units[1]),
                nn.ReLU(),
                nn.Linear(units[1], self._latent_dim),
                nn.ReLU()
            )
            for m in self._latent_enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                    nn.init.zeros_(m.bias)

        def _build_enc(self):
            self._enc_linear = nn.Linear(self._disc_logistics.in_features, self._latent_dim)
            nn.init.uniform_(self._enc_linear.weight, -0.1, 0.1)
            nn.init.zeros_(self._enc_linear.bias)
            self._enc = nn.Sequential(self._disc_mlp, self._enc_linear)

        def _calc_input_size(self, input_shape, cnn_layers=None):
            self._input_shape = input_shape
            return input_shape[0] + self._latent_dim

        def enc(self, normalized_obs):
            return torch.nn.functional.normalize(self._enc(normalized_obs), dim=-1)

        def latent_feature(self, latent):
            if self._build_latent_net:
                return self._latent_enc_mlp(latent)
            return latent

        @property
        def enc_weights(self):
            weights = []
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._enc_linear.weight))
            return weights

        @property
        def latent_dim(self):
            return self._latent_dim

        @property
        def input_size(self):
            return self._calc_input_size(self._input_shape)
