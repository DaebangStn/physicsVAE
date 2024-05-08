from typing import Optional

import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.core.model import CoreModel


class StyleModel(CoreModel):
    def __init__(self, network):
        super().__init__(network)

    class NetworkWrapper(CoreModel.NetworkWrapper):
        """torch.nn.Module which post-processes the network(variable 'net') output
        to compatible with rl-games.
        """
        def __init__(self, net, **kwargs):
            super().__init__(net, **kwargs)
            if self.normalize_input:
                self.disc_running_mean_std = RunningMeanStd((kwargs['disc']['num_inputs'],))

        def norm_disc_obs(self, obs):
            with torch.no_grad():
                return self.disc_running_mean_std(obs) if self.normalize_input else obs

        def forward(self, input_dict):
            output_dict = super().forward(input_dict)
            if input_dict.get('is_train', False):
                output_dict['rollout_disc_logit'] = self.disc(input_dict['normalized_rollout_disc_obs'])
                output_dict['replay_disc_logit'] = self.disc(input_dict['normalized_replay_disc_obs'])
                output_dict['demo_disc_logit'] = self.disc(input_dict['normalized_demo_disc_obs'])
            return output_dict

        def disc(self, normalized_obs):
            return self.a2c_network.disc(normalized_obs)

        def disc_load_state_dict(self, state_dict):
            self.a2c_network.disc_load_state_dict(state_dict)

        def disc_running_mean_load_state_dict(self, state_dict):
            for name, param in self.disc_running_mean_std.named_parameters():
                key = 'disc_running_mean_std.' + name
                if key in state_dict:
                    param.data = state_dict[key].data
                else:
                    raise KeyError(f'{key} is not found in the disc checkpoint state_dict')

        @property
        def disc_logistics_weights(self):
            return self.a2c_network.disc_logistics_weights

        @property
        def disc_weights(self):
            return self.a2c_network.disc_weights
