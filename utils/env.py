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
