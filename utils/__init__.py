import os
from typing import Tuple
import torch

from poselib.motion_lib import MotionLib
from utils import angle

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


class MotionLibFetcher:
    def __init__(self, **kwargs):
        self._num_envs = kwargs.pop('num_envs')
        self._traj_len = kwargs.pop('traj_len')
        self._dt = kwargs.pop('dt')
        self._device = kwargs.get('device')
        self._motion_lib = MotionLib(**kwargs)

    def fetch(self, n: int = 1):
        motion_ids = self._motion_lib.sample_motions(n * self._num_envs)

        motion_length = self._dt * (self._traj_len + 1)
        start_time = self._motion_lib.sample_time(motion_ids, truncate_time=motion_length)
        end_time = (start_time + motion_length).unsqueeze(-1)
        time_steps = - self._dt * torch.arange(0, self._traj_len, device=self._device)
        capture_time = (time_steps + end_time).view(-1)

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._traj_len]).view(-1)
        state = retarget_obs(self._motion_lib.get_motion_state(motion_ids, capture_time))
        # TODO: check the motion index is in order.
        # TODO: whether [traj1, traj1, ..., traj2, traj2, ...] or [traj1, traj2, ..., traj1, traj2, ...]
        return state.view(n * self._num_envs, -1)


class TensorHistoryFIFO:
    def __init__(self, max_size: int):
        self._q = self.TensorFIFO(max_size)

    def push(self, x: torch.Tensor, resets: torch.Tensor):
        assert x.shape[0] == resets.shape[0], f"[{self.__class__}]shape mismatch: {x.shape[0]} != {resets.shape[0]}"
        if len(self._q) == 0:
            for i in range(self._q.max_len):
                self._q.push(x)
        elif torch.any(resets):
            for i in range(self._q.max_len):
                self._q.set_row(i, x, resets)
        self._q.push(x)

    @property
    def history(self):
        return torch.cat(self._q.list, dim=1)

    def __len__(self):
        return len(self._q)

    class TensorFIFO:
        def __init__(self, max_size: int):
            self._q = []
            self._max_size = max_size

        def push(self, item: torch.Tensor):
            self._q.insert(0, item)
            if len(self._q) > self._max_size:
                self._q.pop()

        def set_row(self, idx: int, item: torch.Tensor, set_flag: torch.Tensor):
            assert idx < len(self._q), f"[{self.__class__}] index out of range: {idx} >= {len(self._q)}"
            assert item.shape[0] == set_flag.shape[0], \
                f"[{self.__class__}] set_flag shape mismatch: {item.shape[0]} != {set_flag.shape[0]}"
            assert item.shape == self._q[0].shape, \
                f"[{self.__class__}] item shape mismatch: {item.shape} != {self._q[0].shape}"

            self._q[idx] = torch.where(set_flag, item, self._q[idx])

        @property
        def max_len(self):
            return self._max_size

        @property
        def list(self):
            return self._q

        def __getitem__(self, idx):
            return self._q[idx]

        def __repr__(self):
            return repr(self._q)

        def __len__(self):
            return len(self._q)


@torch.jit.script
def retarget_obs(motion_lib_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos) = motion_lib_state
    return torch.cat((dof_pos, dof_vel, root_pos, root_rot, root_vel, root_ang_vel), dim=-1)
