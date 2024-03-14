import os
import torch
import numpy as np
from isaacgym import gymapi


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def get_tensor_like_r_body_state(gym, sim, num_envs=None):
    value = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)
    tensor_like = np.zeros((len(value['pose']['p']['x']), 13), dtype=np.float32)
    for i, item in enumerate(value):
        tensor_like[i] = [
            item['pose']['p']['x'], item['pose']['p']['y'], item['pose']['p']['z'],
            item['pose']['r']['x'], item['pose']['r']['y'], item['pose']['r']['z'], item['pose']['r']['w'],
            item['vel']['linear']['x'], item['vel']['linear']['y'], item['vel']['linear']['z'],
            item['vel']['angular']['x'], item['vel']['angular']['y'], item['vel']['angular']['z']
        ]
    if num_envs is not None:
        row_per_env = tensor_like.shape[0] // num_envs
        tensor_like = tensor_like.reshape(num_envs, row_per_env, 13)
    return tensor_like


def set_tensor_like_r_body_state(gym, sim, value, num_envs=None):
    if num_envs is not None:
        rows = num_envs * value.shape[1]
        value = value.reshape(rows, 13)
    dtype = [
        ('pose', [
            ('p', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]),
            ('r', [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('w', '<f4')])
        ]),
        ('vel', [
            ('linear', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]),
            ('angular', [('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        ])
    ]
    numpy_like = np.empty((value.shape[0]), dtype=dtype)
    for i, item in enumerate(value):
        numpy_like[i]['pose']['p']['x'] = item[0]
        numpy_like[i]['pose']['p']['y'] = item[1]
        numpy_like[i]['pose']['p']['z'] = item[2]
        numpy_like[i]['pose']['r']['x'] = item[3]
        numpy_like[i]['pose']['r']['y'] = item[4]
        numpy_like[i]['pose']['r']['z'] = item[5]
        numpy_like[i]['pose']['r']['w'] = item[6]
        numpy_like[i]['vel']['linear']['x'] = item[7]
        numpy_like[i]['vel']['linear']['y'] = item[8]
        numpy_like[i]['vel']['linear']['z'] = item[9]
        numpy_like[i]['vel']['angular']['x'] = item[10]
        numpy_like[i]['vel']['angular']['y'] = item[11]
        numpy_like[i]['vel']['angular']['z'] = item[12]
    gym.set_sim_rigid_body_states(sim, numpy_like, gymapi.STATE_ALL)
