from copy import deepcopy as dc
import torch

# This conversion is for brought ASE network
ase_pth_path = 'runs/ASE_Humanoid_08-22-07-17_1024_512_loco2/nn/Humanoid.pth'

ase_pth = torch.load(ase_pth_path)
model = ase_pth['model']

model['running_mean_std.running_mean'] = dc(ase_pth['running_mean_std']['running_mean'])
model['running_mean_std.running_var'] = dc(ase_pth['running_mean_std']['running_var'])
model['running_mean_std.count'] = dc(ase_pth['running_mean_std']['count'])

model['value_mean_std.running_mean'] = dc(ase_pth['reward_mean_std']['running_mean'])
model['value_mean_std.running_var'] = dc(ase_pth['reward_mean_std']['running_var'])
model['value_mean_std.count'] = dc(ase_pth['reward_mean_std']['count'])

model['disc_running_mean_std.running_mean'] = dc(ase_pth['amp_input_mean_std']['running_mean'])
model['disc_running_mean_std.running_var'] = dc(ase_pth['amp_input_mean_std']['running_var'])
model['disc_running_mean_std.count'] = dc(ase_pth['amp_input_mean_std']['count'])

torch.save(ase_pth, ase_pth_path.replace('.pth', '_converted2.pth'))
