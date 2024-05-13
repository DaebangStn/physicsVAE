from typing import Tuple

import torch

from learning.style.model import StyleModel


class SkillModelASE(StyleModel):
    def __init__(self, network):
        super().__init__(network)

    class NetworkWrapper(StyleModel.NetworkWrapper):
        """torch.nn.Module which post-processes the network(variable 'net') output
        to compatible with rl-games.
        """

        def __init__(self, net, **kwargs):
            super().__init__(net, **kwargs)

        def forward(self, input_dict):
            normed_obs = input_dict['obs']
            ase_latents = input_dict['latent']
            use_hidden_latents = input_dict.get('use_hidden_latents', False)

            mu, logstd = self.a2c_network.eval_actor(normed_obs, ase_latents, use_hidden_latents)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            normalized_value = self.a2c_network.eval_critic(normed_obs, ase_latents, use_hidden_latents)

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
                    'rollout_disc_logit': self.a2c_network.disc(input_dict['normalized_rollout_disc_obs']),
                    'replay_disc_logit': self.a2c_network.disc(input_dict['normalized_replay_disc_obs']),
                    'demo_disc_logit': self.a2c_network.disc(input_dict['normalized_demo_disc_obs']),
                    'enc': self.a2c_network.enc(input_dict['normalized_rollout_disc_obs']),
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

        def actor_latent(self, normed_obs, latent):
            mu, logstd = self.a2c_network.eval_actor(normed_obs, latent)
            sigma = torch.exp(logstd)
            return mu, sigma

        def critic_latent(self, normed_obs, latent):
            normalized_value = self.a2c_network.eval_critic(normed_obs, latent)
            value = self.denorm_value(normalized_value)
            return value

        def enc(self, normalized_obs):
            return self.a2c_network.enc(normalized_obs)

        def enc_load_state_dict(self, state_dict):
            self.a2c_network.enc_load_state_dict(state_dict)

        @property
        def enc_weights(self):
            return self.a2c_network.enc_weights

        @property
        def input_size(self):
            return self.a2c_network.input_size