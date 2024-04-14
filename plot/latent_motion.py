import h5py
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from utils.plot import *


def sort_list_by_key(_sorted: list, _key: list):
    paired_sorted = sorted(zip(_key, _sorted), key=lambda x: x[0])
    _, sorted_tuple = zip(*paired_sorted)
    return list(sorted_tuple)


def plot_latent_and_motion_id(f, exp_name):
    base_group = f[exp_name]["LatentMotionLogger"]
    latent_num = base_group['latent'].shape[0]
    print(f"Dataset: {exp_name}")
    print(f"Latent#: {latent_num}")

    motion_ids_latent_idx = []
    motion_ids = []
    total_var = 0
    total_items = 0
    for i in range(latent_num):
        motion_id = np.array(base_group['motion_id'][str(i)])
        if len(motion_id) == 0:
            continue
        motion_ids_latent_idx.append((i, motion_id))
        motion_ids.append(motion_id)
        total_var += np.var(motion_id)
        total_items += len(motion_id)

    std = np.sqrt(total_var / total_items)
    print(f"Mean Std: {std:.4f}")

    full_mid = np.concatenate(motion_ids)
    motion_id_max = np.max(full_mid)
    motion_id_min = np.min(full_mid)

    motion_id_freq = np.empty((len(motion_ids), motion_id_max - motion_id_min + 1), dtype=float)
    motion_id_means = np.zeros(len(motion_ids), dtype=float)
    for latent_idx, motion_id in motion_ids_latent_idx:
        size = motion_id.shape[0]
        unique, counts = np.unique(motion_id, return_counts=True)
        motion_id_freq[latent_idx, unique - motion_id_min] = counts / size
        motion_id_means[latent_idx] = np.mean(motion_id)

    latent_mean_indices = np.argsort(motion_id_means)
    motion_id_freq = motion_id_freq[latent_mean_indices]

    cmap = plt.get_cmap('viridis')
    norm = colors.Normalize(vmin=0, vmax=1)

    heatmap = plt.imshow(motion_id_freq.T, cmap=cmap, norm=norm, aspect='auto')
    plt.colorbar(heatmap)

    exp_name = shorten_middle(exp_name, 35)
    plt.title(f"{exp_name}\n(latent and motion_id)")
    plt.xlabel("Latent index")
    plt.ylabel("Motion Id")
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(motion_id_min, motion_id_max + 1, 1))
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)
    plt.show()


if __name__ == '__main__':
    args = build_args()
    # plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend

    if not args.hdf:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    with h5py.File(args.hdf, 'r+') as f:
        for exp_name in f:
            try:
                plot_latent_and_motion_id(f, exp_name)
            except Exception as e:
                print(f"Error occurred while plotting {exp_name}: {e}")
                continue
