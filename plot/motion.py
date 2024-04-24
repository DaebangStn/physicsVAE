import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import *


PLOT_LAST = True


def plot_motion(f, exp_name):
    base_group = f[exp_name]["MotionLogger"]
    num_env = base_group['motion_id'].shape[0]
    rollout_len = base_group['motion_id'].shape[1]
    print(f"Dataset: {exp_name}")
    print(f"Env#: {num_env}, Rollout Length: {rollout_len}")

    data = np.array(base_group['motion_id'])
    categories, counts = np.unique(data, return_counts=True)
    plt.bar(categories, counts, alpha=0.5, label=exp_name)

    exp_name = shorten_middle(exp_name, 25)
    plt.title(f"{exp_name}\n(motions)")

    motion_id_max = np.max(data)
    motion_id_min = np.min(data)
    plt.xticks(np.arange(motion_id_min, motion_id_max + 1, 1))

    plt.xlabel("Motion Id")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)
    plt.show()


if __name__ == '__main__':
    args = build_args()
    # plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend

    if not args.hdf:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    with h5py.File(args.hdf, 'r+') as f:
        if PLOT_LAST:
            plot_motion(f, list(f.keys())[-1])
        else:
            for exp_name in f:
                try:
                    plot_motion(f, exp_name)
                except Exception as e:
                    print(f"Error occurred while plotting {exp_name}: {e}")
                    continue
