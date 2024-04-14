from time import sleep

import h5py
import pytest
from matplotlib import pyplot as plt

from plot.action import plot_action
from plot.latent_motion import plot_latent_and_motion_id


@pytest.mark.plot
def test_plot_action():
    with h5py.File("test.hdf5", 'r') as f:
        for exp_name in f:
            if "ActionLogger" in f[exp_name]:
                plot_action(f, exp_name)
                sleep(1)
                plt.close()


@pytest.mark.plot
def test_plot_latent_and_motion_id():
    with h5py.File("test.hdf5", 'r') as f:
        for exp_name in f:
            if "LatentMotionLogger" in f[exp_name]:
                plot_latent_and_motion_id(f, exp_name)
                sleep(1)
                plt.close()
