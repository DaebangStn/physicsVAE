import torch

from learning.style.model import StyleModel


class SkillModel(StyleModel):
    def __init__(self, network):
        super().__init__(network)

    class NetworkWrapper(StyleModel.NetworkWrapper):
        """torch.nn.Module which post-processes the network(variable 'net') output
        to compatible with rl-games.
        """
        def __init__(self, net, **kwargs):
            super().__init__(net, **kwargs)

        def forward(self, input_dict):
            output_dict = super().forward(input_dict)
            if input_dict.get('is_train', False):
                output_dict['enc'] = self.enc(input_dict['rollout_obs'])
            return output_dict

        def actor(self, obs):
            obs = self.norm_obs(obs)
            # TODO, since network is separated, action and value could evaluate separately
            with torch.no_grad():
                mu, logstd, _, _ = self.a2c_network({
                    'obs': obs,
                    'is_train': False
                })
            return mu, torch.exp(logstd)

        def critic(self, obs):
            obs = self.norm_obs(obs)
            # TODO, since network is separated, action and value could evaluate separately
            with torch.no_grad():
                _, _, value, _ = self.a2c_network({
                    'obs': obs,
                    'is_train': False
                })
            return value

        def enc(self, obs):
            return self.a2c_network.enc(obs)

        def enc_load_state_dict(self, state_dict):
            self.a2c_network.enc_load_state_dict(state_dict)

        @property
        def enc_weights(self):
            return self.a2c_network.enc_weights
