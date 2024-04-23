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
            input_dict['obs'] = self._attach_latent_to_obs(input_dict['obs'], input_dict['latent'])

            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            is_train = input_dict.get('is_train', True)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(input_dict['prev_actions'], mu, sigma, logstd)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma,
                    'enc': self.enc(input_dict['rollout_obs']),
                    'rollout_disc_logit': self.disc(input_dict['rollout_obs']),
                    'replay_disc_logit': self.disc(input_dict['replay_obs']),
                    'demo_disc_logit': self.disc(input_dict['demo_obs']),
                }
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.denorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma,
                }
            return result

        def actor(self, obs, **kwargs):
            assert self.a2c_network.separate, 'actor is not supported for non-separate network'
            assert 'latent' in kwargs, 'latent is not provided'
            with torch.no_grad():
                return self._actor_module(self._attach_latent_to_obs(obs, kwargs['latent']))

        def critic(self, obs, **kwargs):
            assert self.a2c_network.separate, 'critic is not supported for non-separate network'
            assert 'latent' in kwargs, 'latent is not provided'
            with torch.no_grad():
                return self._critic_module(self._attach_latent_to_obs(obs, kwargs['latent']))

        def enc(self, obs):
            return self.a2c_network.enc(obs)

        def enc_load_state_dict(self, state_dict):
            self.a2c_network.enc_load_state_dict(state_dict)

        def _attach_latent_to_obs(self, obs, latent):
            latent_feature = self.a2c_network.latent_feature(latent)
            obs = self.norm_obs(obs)
            obs = torch.cat([obs, latent_feature], dim=-1)
            return obs

        @property
        def enc_weights(self):
            return self.a2c_network.enc_weights
