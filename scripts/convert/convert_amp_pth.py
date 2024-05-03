from copy import deepcopy as dc
import torch


amp_pth_path = 'runs/Humanoid_00000100.pth'

amp_pth = torch.load(amp_pth_path)
model = amp_pth['model']

model['a2c_network.disc.0.0.weight'] = dc(model['a2c_network._disc_mlp.0.weight'])
model['a2c_network.disc.0.0.bias'] = dc(model['a2c_network._disc_mlp.0.bias'])

model['a2c_network.disc.1.weight'] = dc(model['a2c_network._disc_logits.weight'])
model['a2c_network.disc.1.bias'] = dc(model['a2c_network._disc_logits.bias'])

model['a2c_network.disc.0.2.weight'] = dc(model['a2c_network._disc_mlp.2.weight'])
model['a2c_network.disc.0.2.bias'] = dc(model['a2c_network._disc_mlp.2.bias'])

model['a2c_network._disc_logistics.weight'] = dc(model['a2c_network._disc_logits.weight'])
model['a2c_network._disc_logistics.bias'] = dc(model['a2c_network._disc_logits.bias'])

model.pop('a2c_network._disc_logits.weight')
model.pop('a2c_network._disc_logits.bias')

model['running_mean_std.running_mean'] = dc(amp_pth['running_mean_std']['running_mean'])
model['running_mean_std.running_var'] = dc(amp_pth['running_mean_std']['running_var'])
model['running_mean_std.count'] = dc(amp_pth['running_mean_std']['count'])

model['value_mean_std.running_mean'] = dc(amp_pth['reward_mean_std']['running_mean'])
model['value_mean_std.running_var'] = dc(amp_pth['reward_mean_std']['running_var'])
model['value_mean_std.count'] = dc(amp_pth['reward_mean_std']['count'])

model['disc_running_mean_std.running_mean'] = dc(amp_pth['amp_input_mean_std']['running_mean'])
model['disc_running_mean_std.running_var'] = dc(amp_pth['amp_input_mean_std']['running_var'])
model['disc_running_mean_std.count'] = dc(amp_pth['amp_input_mean_std']['count'])

torch.save(amp_pth, amp_pth_path.replace('.pth', '_converted.pth'))
