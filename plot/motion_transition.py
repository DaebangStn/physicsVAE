import h5py
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gs

from utils.plot import *


THRESHOLD = 0.8  # for some latent, if the motion_id frequency is lower than this, it is regarded as noise motion
PLOT_LAST = True
PLOT_INIT_TERM = True


def get_motion_id(motion_id: np.ndarray, noise_id: int) -> int:
    num_id = motion_id.shape[0]
    if num_id == 0:
        return noise_id
    unique, counts = np.unique(motion_id, return_counts=True)
    max_count = np.max(counts)
    if max_count / num_id < THRESHOLD:
        return noise_id
    return unique[np.argmax(counts)]


def plot_motion_transition(f, exp_name):
    base_group = f[exp_name]["MotionTransitionLogger"]
    num_env = base_group['motion_id'].shape[0]
    rollout_len = base_group['motion_id'].shape[1]
    print(f"Dataset: {exp_name}")
    print(f"Env#: {num_env}, Rollout Length: {rollout_len}")

    envs_trans_indices = base_group['transition']
    envs_motion_id = base_group['motion_id']
    motion_id_max = np.max(envs_motion_id) + 1  # +1 for the noise motion
    motion_id_min = np.min(envs_motion_id)

    motion_transitions = []  # (motion_id_from, motion_id_to)
    initial_motion_ids = []
    termination_motion_ids = []
    for i in range(num_env):
        trans_indices = envs_trans_indices[str(i)]
        motion_ids = envs_motion_id[i]
        if len(trans_indices) == 0:
            continue
        last_transition = 0
        init_added = False
        after_mid = None
        for j in range(trans_indices.shape[0]):
            trans_idx_now = trans_indices[j]
            trans_idx_next = trans_indices[j + 1] if j < (trans_indices.shape[0] - 1) else rollout_len
            if trans_idx_now == last_transition or trans_idx_now == trans_idx_next:
                continue
            before_trans_motion_ids = motion_ids[last_transition:trans_idx_now]
            after_trans_motion_ids = motion_ids[trans_idx_now:trans_idx_next]
            before_mid = get_motion_id(before_trans_motion_ids, motion_id_max)
            after_mid = get_motion_id(after_trans_motion_ids, motion_id_max)
            motion_transitions.append((before_mid, after_mid))

            last_transition = trans_idx_now
            if not init_added:
                initial_motion_ids.append(before_mid)
                init_added = True
        if after_mid is not None:
            termination_motion_ids.append(after_mid)

    print("Transitions#: ", len(motion_transitions))

    init_mid, init_cnt = np.unique(initial_motion_ids, return_counts=True)
    term_mid, term_cnt = np.unique(termination_motion_ids, return_counts=True)
    init_freq = np.zeros(motion_id_max + 1, dtype=np.int64)
    term_freq = np.zeros(motion_id_max + 1, dtype=np.int64)
    init_freq[init_mid] = init_cnt
    term_freq[term_mid] = term_cnt

    motion_transitions = np.array(motion_transitions)
    trans, frequencies = np.unique(
        motion_transitions.view([('', motion_transitions.dtype)] * motion_transitions.shape[1]),
        return_counts=True)

    grid_size = motion_id_max - motion_id_min + 1
    heatmap_data = np.zeros((grid_size, grid_size), dtype=np.int64)
    for ((id_from, id_to), freq) in zip(trans, frequencies):
        heatmap_data[id_from - motion_id_min, id_to - motion_id_min] = freq

    sum_y_axis = np.sum(heatmap_data, axis=1)
    total_freq = sum_y_axis + term_freq
    heatmap_data_norm = heatmap_data / sum_y_axis[:, None]
    norm = Normalize(vmin=0, vmax=1)

    # fig = plt.figure(figsize=(8, 6))
    # gs = Gs.GridSpec(2, 1, height_ratios=[10, 1])
    # fig.add_subplot(gs[0])

    heatmap = plt.imshow(heatmap_data_norm.T, aspect='auto', cmap='seismic', interpolation='nearest', norm=norm)
    plt.colorbar(heatmap)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(i, j, f"{heatmap_data_norm[i, j]:.2f}\n({heatmap_data[i, j]})", ha='center', va='center',
                     color='red' if heatmap_data_norm[i, j] < 0.5 else 'blue')

    exp_name = shorten_middle(exp_name, 35)
    plt.title(f"{exp_name}\n(motion transition probability)-Thresh:{THRESHOLD}")
    tick_positions = np.arange(motion_id_min, motion_id_max + 1, 1)
    if PLOT_INIT_TERM:
        plt.xlabel("Motion From\n(Terminated Motion count)\n[Total Motion count]")
        plt.ylabel("Motion To\n(Initiated Motion count)")
        xtick_labels = ([str(i) + f"\n({term_freq[i]})" + f"\n[{total_freq[i]}]" for i in range(motion_id_min, motion_id_max)] +
                        ["noise" + f"\n({term_freq[-1]})" + f"\n[{total_freq[-1]}]"])
        ytick_labels = ([str(i) + f"\n({init_freq[i]})" for i in range(motion_id_min, motion_id_max)] +
                        ["noise" + f"\n({init_freq[-1]})"])
    else:
        plt.xlabel("Motion From")
        plt.ylabel("Motion To")
        xtick_labels = [str(i) for i in range(motion_id_min, motion_id_max)] + ["noise"]
        ytick_labels = xtick_labels
    plt.xticks(tick_positions, xtick_labels)
    plt.yticks(tick_positions, ytick_labels)
    plt.gca().invert_yaxis()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)

    # ax = fig.add_subplot(gs[1])
    # ax.set_axis_off()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.text(0, 1, f"Initial Motion IDs: {init_mid} Frequency: {init_freq}", ha='left', va='top')
    # ax.text(0, 0, f"Termination Motion IDs: {term_mid} Frequency: {term_freq}", ha='left', va='top')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = build_args()
    # plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend

    if not args.hdf:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    with h5py.File(args.hdf, 'r+') as f:
        if PLOT_LAST:
            plot_motion_transition(f, list(f.keys())[-1])
        else:
            for exp_name in f:
                try:
                    plot_motion_transition(f, exp_name)
                except Exception as e:
                    print(f"Error occurred while plotting {exp_name}: {e}")
                    continue
