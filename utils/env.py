import math
from typing import List
import numpy as np
from isaacgym import gymapi


def default_asset_option():
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True
    asset_options.use_mesh_materials = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    #  Simulator stability
    asset_options.armature = 0.01
    asset_options.angular_damping = 0.01
    asset_options.max_angular_velocity = 100.0
    return asset_options


def humanoid_asset_option():
    asset_options = default_asset_option()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    return asset_options


def soft_asset_option():
    asset_options = default_asset_option()
    asset_options.thickness = 0.1    # important to add some thickness to the soft body to avoid penetrations
    return asset_options


def drop_transform(height: float):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    return pose


def env_create_parameters(num_envs: int, spacing: float):
    num_per_row = math.ceil(np.sqrt(num_envs))
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    return lower, upper, num_per_row


def create_sensors(gym, asset, names: List[str], pose=None):
    if pose is None:
        pose = gymapi.Transform()
    for name in names:
        idx = gym.find_asset_rigid_body_index(asset, name)
        gym.create_asset_force_sensor(asset, idx, pose)


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


def set_tensor_like_dof_pose_state(gym, envs, handles, values: np.ndarray):
    assert values.shape[0] == len(envs), f"Received {values.shape[0]} data, but environments have different {len(envs)}"

    for i, env in enumerate(envs):
        value = values[i]
        gym.set_actor_dof_position_targets(env, handles[i], value)
        pass
