import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from learning.base.model import BaseModel


class StyleModel(BaseModel):
    def __init__(self, network):
        super().__init__(network)

    class NetworkWrapper(BaseModel.NetworkWrapper):
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
                output_dict['rollout_disc_logit'] = self.disc(input_dict['rollout_obs'])
                output_dict['replay_disc_logit'] = self.disc(input_dict['replay_obs'])
                output_dict['demo_disc_logit'] = self.disc(input_dict['demo_obs'])
            return output_dict

        def disc(self, obs):
            return self.a2c_network.disc(obs)

        @property
        def disc_logistics_weights(self):
            return self.a2c_network.disc_logistics_weights

        @property
        def disc_weights(self):
            return self.a2c_network.disc_weights
