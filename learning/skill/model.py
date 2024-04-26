from typing import Tuple

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

        # def forward(self, input_dict):
        #     output_dict = super().forward(input_dict)
        #     if input_dict.get('is_train', False):
        #         output_dict['enc'] = self.enc(input_dict['normalized_rollout_disc_obs'])
        #     return output_dict

        def forward(self, input_dict):
            """
            norm_obs_actor, norm_obs_critic = self.attach_latent_and_norm_obs(input_dict['obs'], input_dict['latent'])
            mu, logstd = self.actor_module(norm_obs_actor)
            # mu, logstd = self.actor_module(
            #     self.attach_latent_and_norm_obs(input_dict['obs'], input_dict['latent'], True))
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            normalized_value = self.critic_module(norm_obs_critic)
            # normalized_value = self.critic_module(
            #     self.attach_latent_and_norm_obs(input_dict['obs'], input_dict['latent']))
            """

            obs, _ = self.attach_latent_and_norm_obs(input_dict['obs'], input_dict['latent'])
            input_dict2 = {
                'obs': obs,
                'is_train': input_dict['is_train'],
                'prev_actions': input_dict.get('prev_actions'),
                'normalized_rollout_disc_obs': input_dict.get('normalized_rollout_disc_obs'),
                'normalized_replay_disc_obs': input_dict.get('normalized_replay_disc_obs'),
                'normalized_demo_disc_obs': input_dict.get('normalized_demo_disc_obs'),
            }
            mu, logstd, normalized_value, states = self.a2c_network(input_dict2)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

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
                    'rollout_disc_logit': self.disc(input_dict['normalized_rollout_disc_obs']),
                    'replay_disc_logit': self.disc(input_dict['normalized_replay_disc_obs']),
                    'demo_disc_logit': self.disc(input_dict['normalized_demo_disc_obs']),
                    'enc': self.enc(input_dict['normalized_rollout_disc_obs']),
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

        def attach_latent_and_norm_obs(self, obs: torch.Tensor, latent: torch.Tensor, latent_network: bool = False
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
            if latent_network:
                latent_feature = self.a2c_network.latent_feature(latent)
            else:
                latent_feature = latent
            normalized_obs = self.norm_obs(obs)
            normalized_obs_actor = torch.cat([normalized_obs, latent_feature], dim=-1)
            normalized_obs_critic = torch.cat([normalized_obs, latent], dim=-1)

            # obs = torch.cat([obs, latent], dim=-1)
            # normalized_obs = self.norm_obs(obs)
            return normalized_obs_actor, normalized_obs_critic

        def actor_latent(self, obs, latent):
            normed_obs, _ = self.attach_latent_and_norm_obs(obs, latent, True)
            # return self.actor_module(normed_obs)
            mu, logstd, _, _ = self.a2c_network({
                'obs': normed_obs,
                'is_train': False
            })
            return mu, torch.exp(logstd)

        def critic_latent(self, obs, latent):
            _, normed_obs = self.attach_latent_and_norm_obs(obs, latent, False)
            # normalized_value = self.critic_module(normed_obs)
            # return normalized_value
            _, _, normalized_value, _ = self.a2c_network({
                'obs': normed_obs,
                'is_train': False
            })
            return normalized_value

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
