import torch
from rl_games.algos_torch.model_builder import ModelBuilder

from learning.skill.model import SkillModel
from learning.skill.networkBuilder import SkillNetworkBuilder
from learning.ase_model_mine import SkillModelASE
from learning.ase_network_builder import ASEBuilder
from utils import load_checkpoint_to_network
from utils.rl_games import register_net_n_model
from utils.config import load_config, build_args, set_seed


CKPT_PATH = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900_converted.pth'
CKPT_PATH_REF = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900.pth'


def cvt_cfg_ref(train_conf):
    train_conf['model']['name'] = 'skill_ref'
    train_conf['network']['name'] = 'skill_ref'
    train_conf['checkpoint'] = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900.pth'
    return train_conf


def check_model(name: str, net1, net2, *net_input):
    net1_out = net1(*net_input)
    net2_out = net2(*net_input)

    if isinstance(net1_out, dict):
        for k, v in net1_out.items():
            if k in ["neglogpacs", "actions"]:
                continue
            if isinstance(v, torch.Tensor):
                assert torch.allclose(v, net2_out[k], atol=1e-5), \
                    f"{name}: {k} mismatch!. [Mean] mine: {v.mean()}, ref: {net2_out[k].mean()}"
                print(f"{name}: {k} match!")
            else:
                print(f"{name}: {k} is not tensor!")
    elif isinstance(net1_out, tuple):
        for i in range(len(net1_out)):
            assert torch.allclose(net1_out[i], net2_out[i], atol=1e-5), f"{name}: {i} mismatch!"
            print(f"{name}: {i} match!")
    elif isinstance(net1_out, torch.Tensor):
        assert torch.allclose(net1_out, net2_out, atol=1e-5), f"{name} mismatch!"
    print(f"{name} match!")


def main():
    latent_dim = 64
    obs_dim = 253
    disc_obs_dim = 140
    num_env = 32
    disc_traj_len = 10
    total_disc_obs_dim = disc_obs_dim * disc_traj_len

    args = build_args()
    cfg_run, cfg_train = load_config(args)
    set_seed(cfg_train['seed'])

    builder = ModelBuilder()
    register_net_n_model('skill', SkillNetworkBuilder, SkillModel)
    register_net_n_model('skill_ref', ASEBuilder, SkillModelASE)

    rlg_param = {
        'actions_num': 31,
        'input_shape': (obs_dim,),
        'num_seqs': num_env,
        'value_size': 1,
        'normalize_value': True,
        'normalize_input': True,
        'obs_shape': (obs_dim,),
    }

    net = builder.load(cfg_train).build(**cfg_train['network'], **rlg_param)

    cvt_cfg_ref(cfg_train)
    ref_net = builder.load(cfg_train).build(**cfg_train['network'], **rlg_param)

    load_checkpoint_to_network(net, CKPT_PATH)
    load_checkpoint_to_network(ref_net, CKPT_PATH_REF)

    latent = torch.rand(num_env, latent_dim)
    obs = torch.rand(num_env, obs_dim)
    disc_obs = torch.rand(num_env, total_disc_obs_dim)
    in_317 = torch.rand(num_env, 317)
    in_512 = torch.rand(num_env, 512)
    in_1024 = torch.rand(num_env, 1024)

    print("==> Testing model evaluation pass...")

    net.eval()
    ref_net.eval()

    # check_model('Latent Feature', net.a2c_network.latent_feature, ref_net.a2c_network.actor_mlp.eval_style, latent)
    # check_model('Mu', net.a2c_network.mu, ref_net.a2c_network.mu, in_512)
    # check_model('Actor1', net.a2c_network.actor_mlp[0], ref_net.a2c_network.actor_mlp._dense_layers[0], in_317)
    # check_model('Actor2', net.a2c_network.actor_mlp[2], ref_net.a2c_network.actor_mlp._dense_layers[1], in_1024)
    # check_model('Actor3', net.a2c_network.actor_mlp[4], ref_net.a2c_network.actor_mlp._dense_layers[2], in_1024)

    # obs_feature, obs_naive = net.attach_latent(obs, latent)
    # actor1 = net.a2c_network.actor_mlp(obs_naive)
    # actor2 = ref_net.a2c_network.actor_mlp(obs, latent, True)
    # assert torch.allclose(actor1, actor2, atol=1e-5), "Naive actor mismatch!"
    #
    # actor1 = net.a2c_network.actor_mlp(obs_feature)
    # actor2 = ref_net.a2c_network.actor_mlp(obs, latent, False)
    # assert torch.allclose(actor1, actor2, atol=1e-5), "Feature actor mismatch!"
    #
    # actor1, _ = net.actor_module(obs_naive)
    # actor2, _ = ref_net.a2c_network.eval_actor(obs, latent, True)
    # assert torch.allclose(actor1, actor2, atol=1e-5), "Naive actor mismatch!"
    #
    # actor1, _ = net.actor_module(obs_feature)
    # actor2, _ = ref_net.a2c_network.eval_actor(obs, latent, False)
    # assert torch.allclose(actor1, actor2, atol=1e-5), "Feature actor mismatch!"

    # g1 = make_dot(actor1, params=dict(net.a2c_network.actor_mlp.named_parameters()))
    # g2 = make_dot(actor2, params=dict(ref_net.a2c_network.actor_mlp.named_parameters()))
    # g1.render('actor1', format='png', cleanup=True)
    # g2.render('actor2', format='png', cleanup=True)

    check_model('Actor', net.actor_latent, ref_net.actor_latent, obs, latent)
    check_model('Critic', net.critic_latent, ref_net.critic_latent, obs, latent)
    check_model('Disc', net.disc, ref_net.disc, disc_obs)
    check_model('Enc', net.enc, ref_net.enc, disc_obs)

    input_dict = {
        'obs': obs,
        'latent': latent,
        'is_train': False,
    }

    check_model('Forward[Test]', net, ref_net, input_dict)

    input_dict['is_train'] = True
    input_dict['prev_actions'] = torch.rand(num_env, 31)
    input_dict['normalized_rollout_disc_obs'] = torch.rand(num_env, total_disc_obs_dim)
    input_dict['normalized_replay_disc_obs'] = torch.rand(num_env, total_disc_obs_dim)
    input_dict['normalized_demo_disc_obs'] = torch.rand(num_env, total_disc_obs_dim)
    net.train()
    ref_net.train()
    check_model('Forward[Train]', net, ref_net, input_dict)


if __name__ == "__main__":
    main()
    print("Done!")
