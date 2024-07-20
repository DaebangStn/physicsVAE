import torch

from poselib.motion_lib import MotionLib
from utils import *


class InterMotionLib(MotionLib):
    def __init__(self, motion_file_path: str, dof_body_ids: List[int], dof_offsets: List[int], key_body_ids: List[int],
                 device: torch.device):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device

        self._motions_cpu = None
        self._load_motions(motion_file_path)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

    def _load_motions(self, motion_file):
        self._motions = []
        self._motions_cpu = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            self._load_intergen_dataset(curr_file)

    def _load_intergen_dataset(self, file_path: str):
        person = "person1"

        bdata = np.load(file_path, allow_pickle=True)
        fps = float(bdata["mocap_framerate"])
        dt = 1.0 / fps
        self._motion_fps.append(fps)
        self._motion_dt.append(dt)

        bdata = bdata[person]
        betas = torch.tensor(bdata["betas"], device=self._device)

        num_frames = bdata['trans'].shape[0]
        _len = dt * (num_frames - 1)
        self._motion_num_frames.append(num_frames)
        self._motion_lengths.append(_len)

        root_pos = bdata["trans"]
        root_rot = bdata["root_orient"]
        r_body_pos = bdata["pose_body"]
