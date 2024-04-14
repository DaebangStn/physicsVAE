import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import *


def plot_action(f, exp_name):
    data = np.array(f[exp_name]["ActionLogger"]['action'])
    print(f"Dataset: {exp_name}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Standard Deviation: {np.std(data):.4f}")
    print(f"Minimum: {np.min(data):.4f}")
    print(f"Maximum: {np.max(data):.4f}")

    flattened_data = data.flatten()

    exp_name = shorten_middle(exp_name, 25)
    plt.hist(flattened_data, bins=100, alpha=0.5, label=exp_name, density=True)
    plt.title(f"{exp_name}\n(actions)")
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.xlim(-1, 1)
    plt.ylim(0, 5)
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)
    plt.show()


if __name__ == '__main__':
    args = build_args()
    # plt.switch_backend('TkAgg')  # Since pycharm IDE embeds matplotlib, it is necessary to switch backend

    if not args.hdf:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    with h5py.File(args.hdf, 'r') as f:
        for exp_name in f:
            try:
                plot_action(f, exp_name)
            except Exception as e:
                print(f"Error occurred while plotting {exp_name}: {e}")
                continue
