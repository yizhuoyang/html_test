"""
Acoustic Hand Gesture Recognition Tutorial
==============================================================
"""

######################################################################
# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for acoustic Hand Gesture Recognition
# 
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn as nn
import tqdm
import sys
from pysensing.acoustic.datasets.hgr import AMG
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
######################################################################
# Hand Gesture Recognition with Acoustic Myography and Wavelet Scattering Transform
# -----------------------------------
# Reimplementation of "Hand Gesture Recognition with Acoustic Myography and Wavelet Scattering Transform"
#
# This dataset contains acoustic myography of different hand gestures. The aucoustic data is in 8 channel. In this library, subjects
# AA01, CU14, DH18, NL20, NM08 and SR11 are selected as testing data, while the remaining used for training.
#
# The classes contains in the dataset are [Pronation, Supination, Wrist Flexion, Wrist Extension, Radial Deviation, Ulnar Deviation,
# Hand close, Hand open, Hook grip, Fine pinch, Tripod grip, Index finger flexion, Thumb finger flexion, and No movement (Rest)]

######################################################################
# Load the data
# ------------------------
# Method 1: Use get_dataloader
from pysensing.acoustic.datasets.get_dataloader import *
train_loader,test_loader = load_hgr_dataset(
    root='./data',
    download=True)

# Method 2: Manually setup the dataloader
root = './data' # The path contains the samosa dataset
amg_traindataset = AMG(root,'train')
amg_testdataset = AMG(root,'test')
# Define the Dataloader
amg_trainloader = DataLoader(amg_traindataset,batch_size=32,shuffle=False,drop_last=True)
amg_testloader = DataLoader(amg_testdataset,batch_size=32,shuffle=False,drop_last=True)
#List the activity classes in the dataset
dataclass = amg_traindataset.class_dict
# Example of the samples in the dataset
index = 128
# Randomly 3elect an index
spectrogram,activity= amg_traindataset.__getitem__(index)
plt.figure(figsize=(10,6))
plt.imshow(spectrogram.numpy()[0])
plt.title("Spectrogram for activity: {}".format(activity))
plt.show()
######################################################################
# And that's it. We're done with our acoustic hand gesture recognition tutorials. Thanks for reading.
