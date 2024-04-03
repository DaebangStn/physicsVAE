import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_args():
    parameters = [
        {"name": "--hdf", "type": str, "default": "",
         "help": "Path to the HDF file containing the latent vectors."},
    ]
    parser = argparse.ArgumentParser(description="Visualize Latent Space")
    for param in parameters:
        kwargs = {k: v for k, v in param.items() if k != "name"}
        parser.add_argument(param["name"], **kwargs)
    return parser.parse_args()


def plot_distribution(label: str, data: np.ndarray):
    print(f"Dataset: {label}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"Standard Deviation: {np.std(data):.4f}")
    print(f"Minimum: {np.min(data):.4f}")
    print(f"Maximum: {np.max(data):.4f}")

    flattened_data = data.flatten()
    plt.hist(flattened_data, bins=100, alpha=0.5, label=label, density=True)
    plt.title(f"{label}")
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.xlim(-1, 1)
    plt.ylim(0, 5)
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'q' else None)
    plt.show()


if __name__ == '__main__':
    args = build_args()

    if not args.hdf:
        raise ValueError("Please provide the path to the HDF5 file containing the latent vectors.")

    with h5py.File(args.hdf, 'r') as f:
        for dataset in f:
            plot_distribution(dataset, np.array(f[dataset]))
