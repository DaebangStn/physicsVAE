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

        def attach_latent_and_norm_obs(self, obs: torch.Tensor, latent: torch.Tensor):
            latent_feature = self.a2c_network.latent_feature(latent)
            normalized_obs = self.norm_obs(obs)
            normalized_obs = torch.cat([normalized_obs, latent_feature], dim=-1)
            return normalized_obs

        def actor_latent(self, obs, latent):
            return self.actor_module(self.attach_latent_and_norm_obs(obs, latent))

        def critic_latent(self, obs, latent):
            normalized_value = self.critic_module(self.attach_latent_and_norm_obs(obs, latent))
            return normalized_value

        def enc(self, obs):
            normalized_obs = self._norm_disc_obs(obs)
            return self.a2c_network.enc(normalized_obs)

        def enc_load_state_dict(self, state_dict):
            self.a2c_network.enc_load_state_dict(state_dict)

        def _compute_obs(self, input_dict: dict) -> torch.Tensor:
            return self.attach_latent_and_norm_obs(input_dict['obs'], input_dict['latent'])

        @property
        def enc_weights(self):
            return self.a2c_network.enc_weights

        @property
        def input_size(self):
            return self.a2c_network.input_size
