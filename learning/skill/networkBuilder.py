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
            self._latent_dim = int(kwargs['space']['latent_dim'])
            super().__init__(param, **kwargs)  # build action/value/disc network
            self._build_enc()

        def _build_enc(self):
            self._enc_mlp = self._disc_mlp
            self._enc_linear = nn.Linear(self._disc_logistics.in_features, self._latent_dim)
            nn.init.uniform_(self._enc_linear.weight, -0.1, 0.1)
            nn.init.zeros_(self._enc_linear.bias)
            self._enc = nn.Sequential(self._enc_mlp, self._enc_linear)

        def enc(self, obs):
            return torch.nn.functional.normalize(self._enc(obs), dim=-1)

        # def _calc_input_size(self, input_shape, cnn_layers=None):
        #     return input_shape[0] + self._latent_dim

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
