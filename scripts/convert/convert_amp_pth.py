from copy import deepcopy as dc
import torch


amp_pth_path = 'runs/Humanoid_00003300.pth'

amp_pth = torch.load(amp_pth_path)
model = amp_pth['model']

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
