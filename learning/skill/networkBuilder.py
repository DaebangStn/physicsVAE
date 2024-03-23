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
            super().__init__(param, **kwargs)  # build action/value network
            self._build_enc(**kwargs)

        def _build_enc(self, **kwargs):
            self._enc_mlp = self._disc_mlp
            self._enc_linear = nn.Linear(self._disc_logistics.out_features, )  # TODO: parse from kwargs, latent_dim
            self._enc_norm = nn.BatchNorm1d(self._disc_logistics.out_features)
            nn.init.uniform_(self._enc_linear.weight, -0.1, 0.1)
            nn.init.zeros_(self._enc_linear.bias)

            self.enc = nn.Sequential(self._enc_mlp, self._enc_linear, self._enc_norm)

        @property
        def enc_weights(self):
            weights = []
            for m in self._enc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._enc_linear.weight))
            return weights
