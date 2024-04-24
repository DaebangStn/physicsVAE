import torch

from learning.logger.hdfBase import HdfBaseLogger


class MotionLogger(HdfBaseLogger):
    """
    See hierarchy in HdfBaseLogger

    logger_specific_group:
        root(None):
            motion_id: float (N, D) - N: number of environment, D: number of logged motion id (resizeable)
    """

    def __init__(self, filename: str, exp_name: str, num_envs: int, cfg: dict):
        super().__init__(filename, exp_name, cfg)
        if "motion_id" not in self._base_group:
            self._base_group.create_dataset("motion_id", shape=(num_envs, 0), maxshape=(num_envs, None),
                                            dtype='i4', chunks=True)

    def log(self, motion_indices: torch.Tensor):
        assert motion_indices.shape[0] == self._base_group["motion_id"].shape[0], \
            f"num_env({self._base_group['motion_id'].shape[0]}) != data({motion_indices.shape[0]})"

        self._append_ds(self._base_group["motion_id"], motion_indices.unsqueeze(0).to("cpu").numpy(), axis=1)
