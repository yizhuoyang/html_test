"""
Tutorial for UWB Data Preprocessing
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
x_amp_sample = np.abs(x)[1,:,:]
plot_uwb_heatmap(x_amp_sample)

######################################################################
# Clutter Suppression
# -----------------------------------
# Load corresponding Clutter Suppression functions

from pysensing.uwb.preprocessing.clutter_suppression import *

######################################################################
# DC Suppression
# -----------------------------------

x_amp_sample_ = np.expand_dims(x_amp_sample, axis=0)
dc_suppressed_x = dc_suppression(x_amp_sample_)
plot_uwb_heatmap(dc_suppressed_x)

######################################################################
# Static Background Suppression
# -----------------------------------

x_amp_sample_copy = np.expand_dims(x_amp_sample, axis=0)
static_background_suppressed_x = static_background_suppression(x_amp_sample_copy)
plot_uwb_heatmap(static_background_suppressed_x)

######################################################################
# Running Background Suppression
# -----------------------------------

x_amp_sample_copy = np.expand_dims(x_amp_sample, axis=0)
running_background_suppressed_x = running_background_suppression(x_amp_sample_copy, alpha=0.1)
plot_uwb_heatmap(running_background_suppressed_x)

######################################################################
# Cropping
# -----------------------------------
# Load corresponding Cropping functions

from pysensing.uwb.preprocessing.cropping import *

######################################################################
# Range Selection
# -----------------------------------

x_amp_sample_ = np.expand_dims(x_amp_sample, axis=0)
cropped_x, spatial_highest_position = range_selection(x_amp_sample_, spatial_size = 50)
plot_uwb_heatmap(cropped_x)

######################################################################
# Filtering
# -----------------------------------
# Load corresponding Filtering functions

from pysensing.uwb.preprocessing.filtering import *

######################################################################
# Band Pass Butterworth Filtering
# -----------------------------------

x_amp_sample_ = np.expand_dims(x_amp_sample, axis=0)
butterworth_filter = bandpass_butterworth_filter(low_cut=2, high_cut=4, sample_rate=10, order=4)
filtered_uwb_data = butterworth_filter(x_amp_sample_)
plot_uwb_heatmap(filtered_uwb_data)

######################################################################
# NaN Removal
# -----------------------------------

from pysensing.uwb.preprocessing.nan_removal import *

x_amp_sample_ = x_amp_sample
nan_removed_data = np.zeros(x_amp_sample_.shape)
for i in range(len(x_amp_sample_)):
    nan_removed_data[i,:] = remove_nan(x_amp_sample_[i,:])
plot_uwb_heatmap(nan_removed_data)

######################################################################
# Normalization
# -----------------------------------

from pysensing.uwb.preprocessing.normalization import *

x_amp_sample_ = x_amp_sample
normalized_uwb_data = normalize_data(x_amp_sample_)
plot_uwb_heatmap(normalized_uwb_data)

######################################################################
# Transformation
# -----------------------------------
# Load corresponding Transformation functions

from pysensing.uwb.preprocessing.transformation import *

######################################################################
# Time Difference Transform
# -----------------------------------

x_input = np.abs(x)
time_difference_uwb_data = time_difference_transform(x_input, norm = True)
plot_uwb_heatmap(time_difference_uwb_data[0])

######################################################################
# Weighted RTF Transform
# -----------------------------------

x_input = np.abs(x)
wrtft_uwb_data = weighted_rtf_transform(x_input, NFFT = 25, stride = 2, norm = True)
plot_uwb_heatmap(wrtft_uwb_data[0])


