"""
Tutorial for UWB Data Augmentation
==============================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

######################################################################
# Plot function
# -----------------------------------

def plot_uwb_heatmap(uwb_data):
    """
    Plot UWB heatmap.

    """
    data_shape = uwb_data.shape

    if len(data_shape) == 2:
        uwb_to_plot = uwb_data
    elif len(data_shape) == 3:
        uwb_to_plot = np.mean(uwb_data, axis=0)
    else:
        raise ValueError("The input data should have at least 2 dimensions.")
    
    plt.figure(figsize=(15, 8))
    plt.imshow(uwb_to_plot, aspect='auto', cmap='viridis')
    plt.colorbar(label='UWB Amplitude')
    plt.title('UWB Heatmap')
    plt.xlabel('Time Index')
    plt.ylabel('Channel Index')
    plt.show()

######################################################################
# Load UWB Data 
# -----------------------------------
root_dir = './data/sleep_pose_net_data' 
data_dir = os.path.join(root_dir, 'Dataset I')
x = np.load(os.path.join(data_dir, 'X.npy'))
x = x[:,:,30:130]
x_amp_sample = np.abs(x)[82,:,:]
plot_uwb_heatmap(x_amp_sample)

######################################################################
# Magnitude Warping
# -----------------------------------

from pysensing.uwb.augmentation.magnitude_warping import *

x_amp_sample_ = x_amp_sample.transpose()
Mag_warping_x = MagWarp(x_amp_sample_, sigma = 0.4, knot= 4)
plot_uwb_heatmap(Mag_warping_x.transpose())

######################################################################
# Time Warping
# -----------------------------------

from pysensing.uwb.augmentation.time_warping import *

x_amp_sample_ = x_amp_sample.transpose()
time_warping_x = TimeWarp(x_amp_sample_, sigma = 0.4, knot= 4)
plot_uwb_heatmap(time_warping_x.transpose())

######################################################################
# Time Shifting
# -----------------------------------

from pysensing.uwb.augmentation.time_shifting import *

x_amp_sample_ = x_amp_sample.transpose()
time_shifted_x = signal_shift_time(x_amp_sample_, shft_arr = [10, -10])
plot_uwb_heatmap(time_shifted_x[0].transpose())
plot_uwb_heatmap(time_shifted_x[1].transpose())

######################################################################
# Range Shifting
# -----------------------------------

from pysensing.uwb.augmentation.range_shifting import *

x_amp_sample_ = x_amp_sample.transpose()
range_shifted_x = signal_shift(x_amp_sample_, shft_arr = [30, -30])
plot_uwb_heatmap(range_shifted_x[0].transpose())
plot_uwb_heatmap(range_shifted_x[1].transpose())

######################################################################
# Scaling
# -----------------------------------

from pysensing.uwb.augmentation.scaling import *

x_amp_sample_ = x_amp_sample.transpose()
scaled_x = DA_Scaling(x_amp_sample_, sigma= 0.1)
plot_uwb_heatmap(scaled_x.transpose())


