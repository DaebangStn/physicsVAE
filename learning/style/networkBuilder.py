import torch
import torch.nn as nn

from learning.core.networkBuilder import CoreNetworkBuilder


class StyleNetworkBuilder(CoreNetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(CoreNetworkBuilder.Network):
        """torch.nn.Module which includes actual parameters.
        Forwards returns [mu, logstd, value, states]
        """
        def __init__(self, param, **kwargs):
            super().__init__(param, **kwargs)  # build action/value network
            self._build_disc(**kwargs)

        def _build_disc(self, **kwargs):
            conf = kwargs['disc']

            args = {
                'input_size': conf["num_inputs"],
                'units': conf["units"],
                'activation': conf["activation"],
                'dense_func': torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**args)
            mlp_init = self.init_factory.create(**conf["initializer"])
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            num_mlp_out = conf["units"][-1]
            self._disc_logistics = torch.nn.Linear(num_mlp_out, 1)
            torch.nn.init.uniform_(self._disc_logistics.weight, -1, 1)
            torch.nn.init.zeros_(self._disc_logistics.bias)

            self._disc = nn.Sequential(self._disc_mlp, self._disc_logistics)

        def disc(self, normalized_obs):
            return self._disc(normalized_obs)

        def disc_load_state_dict(self, state_dict):
            for name, param in self._disc_mlp.named_parameters():
                key = 'a2c_network._disc_mlp.' + name
                if key in state_dict:
                    param.data = state_dict[key].data
                else:
                    raise KeyError(f'{key} is not found in the disc checkpoint state_dict')

            for name, param in self._disc_logistics.named_parameters():
                key = 'a2c_network._disc_logistics.' + name
                if key in state_dict:
                    param.data = state_dict[key].data
                else:
                    raise KeyError(f'{key} is not found in the disc checkpoint state_dict')

        @property
        def disc_logistics_weights(self):
            return torch.flatten(self._disc_logistics.weight)

        @property
        def disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logistics.weight))
            return weights
