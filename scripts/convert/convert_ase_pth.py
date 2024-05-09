from copy import deepcopy as dc
import torch


ase_pth_path = 'runs/ASE_Humanoid_08-22-07-17_1024_512_loco2/nn/Humanoid.pth'

ase_pth = torch.load(ase_pth_path)
model = ase_pth['model']

model["a2c_network.actor_mlp.0.weight"] = dc(model["a2c_network.actor_mlp._dense_layers.0.weight"])
model["a2c_network.actor_mlp.0.bias"] = dc(model["a2c_network.actor_mlp._dense_layers.0.bias"])
model["a2c_network.actor_mlp.2.weight"] = dc(model["a2c_network.actor_mlp._dense_layers.1.weight"])
model["a2c_network.actor_mlp.2.bias"] = dc(model["a2c_network.actor_mlp._dense_layers.1.bias"])
model["a2c_network.actor_mlp.4.weight"] = dc(model["a2c_network.actor_mlp._dense_layers.2.weight"])
model["a2c_network.actor_mlp.4.bias"] = dc(model["a2c_network.actor_mlp._dense_layers.2.bias"])

model["a2c_network.critic_mlp.0.weight"] = dc(model["a2c_network.critic_mlp._mlp.0.weight"])
model["a2c_network.critic_mlp.0.bias"] = dc(model["a2c_network.critic_mlp._mlp.0.bias"])
model["a2c_network.critic_mlp.2.weight"] = dc(model["a2c_network.critic_mlp._mlp.2.weight"])
model["a2c_network.critic_mlp.2.bias"] = dc(model["a2c_network.critic_mlp._mlp.2.bias"])
model["a2c_network.critic_mlp.4.weight"] = dc(model["a2c_network.critic_mlp._mlp.4.weight"])
model["a2c_network.critic_mlp.4.bias"] = dc(model["a2c_network.critic_mlp._mlp.4.bias"])

model["a2c_network._latent_enc_mlp.0.weight"] = dc(model["a2c_network.actor_mlp._style_mlp.0.weight"])
model["a2c_network._latent_enc_mlp.0.bias"] = dc(model["a2c_network.actor_mlp._style_mlp.0.bias"])
model["a2c_network._latent_enc_mlp.2.weight"] = dc(model["a2c_network.actor_mlp._style_mlp.2.weight"])
model["a2c_network._latent_enc_mlp.2.bias"] = dc(model["a2c_network.actor_mlp._style_mlp.2.bias"])
model["a2c_network._latent_enc_mlp.4.weight"] = dc(model["a2c_network.actor_mlp._style_dense.weight"])
model["a2c_network._latent_enc_mlp.4.bias"] = dc(model["a2c_network.actor_mlp._style_dense.bias"])

model["a2c_network._disc.0.0.weight"] = dc(model["a2c_network._enc_mlp.0.weight"])
model["a2c_network._disc.0.0.bias"] = dc(model["a2c_network._enc_mlp.0.bias"])
model["a2c_network._enc.0.0.weight"] = dc(model["a2c_network._enc_mlp.0.weight"])
model["a2c_network._enc.0.0.bias"] = dc(model["a2c_network._enc_mlp.0.bias"])
model["a2c_network._disc.0.2.weight"] = dc(model["a2c_network._enc_mlp.2.weight"])
model["a2c_network._disc.0.2.bias"] = dc(model["a2c_network._enc_mlp.2.bias"])
model["a2c_network._enc.0.2.weight"] = dc(model["a2c_network._enc_mlp.2.weight"])
model["a2c_network._enc.0.2.bias"] = dc(model["a2c_network._enc_mlp.2.bias"])
model["a2c_network._disc.0.4.weight"] = dc(model["a2c_network._enc_mlp.4.weight"])
model["a2c_network._disc.0.4.bias"] = dc(model["a2c_network._enc_mlp.4.bias"])
model["a2c_network._enc.0.4.weight"] = dc(model["a2c_network._enc_mlp.4.weight"])
model["a2c_network._enc.0.4.bias"] = dc(model["a2c_network._enc_mlp.4.bias"])

model["a2c_network._disc.1.weight"] = dc(model["a2c_network._disc_logits.weight"])
model["a2c_network._disc.1.bias"] = dc(model["a2c_network._disc_logits.bias"])
model["a2c_network._disc_logistics.weight"] = dc(model["a2c_network._disc_logits.weight"])
model["a2c_network._disc_logistics.bias"] = dc(model["a2c_network._disc_logits.bias"])

model["a2c_network._enc.1.weight"] = dc(model["a2c_network._enc.weight"])
model["a2c_network._enc.1.bias"] = dc(model["a2c_network._enc.bias"])
model["a2c_network._enc_linear.weight"] = dc(model["a2c_network._enc.weight"])
model["a2c_network._enc_linear.bias"] = dc(model["a2c_network._enc.bias"])

model['running_mean_std.running_mean'] = dc(ase_pth['running_mean_std']['running_mean'])
model['running_mean_std.running_var'] = dc(ase_pth['running_mean_std']['running_var'])
model['running_mean_std.count'] = dc(ase_pth['running_mean_std']['count'])

model['value_mean_std.running_mean'] = dc(ase_pth['reward_mean_std']['running_mean'])
model['value_mean_std.running_var'] = dc(ase_pth['reward_mean_std']['running_var'])
model['value_mean_std.count'] = dc(ase_pth['reward_mean_std']['count'])

model['disc_running_mean_std.running_mean'] = dc(ase_pth['amp_input_mean_std']['running_mean'])
model['disc_running_mean_std.running_var'] = dc(ase_pth['amp_input_mean_std']['running_var'])
model['disc_running_mean_std.count'] = dc(ase_pth['amp_input_mean_std']['count'])

torch.save(ase_pth, ase_pth_path.replace('.pth', '_converted.pth'))
