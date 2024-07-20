from utils import *
from poselib.inter_motion_lib import InterMotionLib


DB_PATH = "assets/motions/intergen/ballroom.pkl"
JOINT_INFO = "assets/urdf/joint_information.yaml"
MODEL_NAME = "smpl_humanoid"


def test_inter_motion_lib():
    j_info = load_yaml(JOINT_INFO)
    j_info = j_info[MODEL_NAME]

    imlib = InterMotionLib(DB_PATH, j_info["dof_body_ids"], j_info["dof_offsets"], j_info["key_body_ids"],
                           torch.device("cpu"))


if __name__ == "__main__":
    test_inter_motion_lib()
    print("test_inter_motion_lib.py: PASSED")
