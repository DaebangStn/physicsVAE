from isaacgym import gymapi, gymtorch
import torch
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch import torch_ext

from learning.skill.model import SkillModel
from learning.skill.networkBuilder import SkillNetworkBuilder
from learning.ase_model_mine import SkillModelASE
from learning.ase_network_builder import ASEBuilder
from utils.rl_games import register_net_n_model
from utils.config import load_config, build_args, set_seed


CKPT_PATH = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900_converted.pth'
CKPT_PATH_REF = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900.pth'


def cvt_cfg_ref(train_conf):
    train_conf['model']['name'] = 'skill_ref'
    train_conf['network']['name'] = 'skill_ref'
    train_conf['checkpoint'] = 'runs/keypointMaxObsTask_ep_3900_rew_6840.244_3900.pth'
    return train_conf


def main():
    latent_dim = 64
    obs_dim = 253
    disc_obs_dim = 140
    num_env = 32

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

    ckpt = torch_ext.load_checkpoint(CKPT_PATH)
    ref_ckpt = torch_ext.load_checkpoint(CKPT_PATH_REF)
    try:
        net.load_state_dict(ckpt['model'])
        ref_net.load_state_dict(ref_ckpt['model'])
    except RuntimeError as e:
        print(f"Error: {e}")

    latent = torch.rand(num_env, latent_dim)
    obs = torch.rand(num_env, obs_dim)
    disc_obs = torch.rand(num_env, disc_obs_dim)

    print("==> Testing model evaluation pass...")

    input_dict = {
        'obs': obs,
        'latent': latent,
        'is_train': False,
    }

    net.eval()
    ref_net.eval()

    net_out = net(input_dict)
    ref_net_out = ref_net(input_dict)

    for k, v in net_out.items():
        assert torch.allclose(v, ref_net_out[k], atol=1e-5), \
            f"{k} mismatch!. [Mean] mine: {v.mean()}, ref: {ref_net_out[k].mean()}"
        print(f"{k} match!")


if __name__ == "__main__":
    main()
    print("Done!")
