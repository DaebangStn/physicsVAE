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
        #         output_dict['enc'] = self.enc(input_dict['rollout_obs'])
        #     return output_dict

        def _norm_obs(self, obs, latent_dim):
            # obs_naive = obs[:, :-latent_dim]
            # obs_latent = obs[:, -latent_dim:]
            # obs_naive = self.norm_obs(obs_naive)
            # return torch.cat([obs_naive, obs_latent], dim=-1)
            return self.norm_obs(obs)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self._norm_obs(input_dict['obs'], 0)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma,
                    'rollout_disc_logit': self.disc(input_dict['rollout_obs']),
                    'replay_disc_logit': self.disc(input_dict['replay_obs']),
                    'demo_disc_logit': self.disc(input_dict['demo_obs']),
                    'enc': self.enc(input_dict['rollout_obs']),
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result

        def actor(self, obs):
            obs = self._norm_obs(obs, 0)
            # TODO, since network is separated, action and value could evaluate separately
            with torch.no_grad():
                mu, logstd, _, _ = self.a2c_network({
                    'obs': obs,
                    'is_train': False
                })
            return mu, torch.exp(logstd)

        def critic(self, obs):
            obs = self._norm_obs(obs, 0)
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
