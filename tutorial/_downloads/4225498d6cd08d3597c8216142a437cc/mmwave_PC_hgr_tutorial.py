
"""
Tutorial for Human Gesture Recognition
==============================================================
"""

######################################################################
#!/usr/bin/env python
# coding: utf-8

######################################################################
# In[1]:


import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os

######################################################################
# Dataset with M-Gesture: 
# ------------------------

######################################################################
# Point cloud gesture dataset collected using FMCW mmWave Radar, TI-IWR1443 single-chip 76-GHz to 81-GHz mmWave sensor evaluation module. 2 scenarios are included: 
# short range (i.e. Human-Radar Distance(HRD) < 0.5 m) and long range (i.e. 2m < HRD < 5m); Only long-range gesture recognition 
# is supported as only long-range dataset contain point cloud data.

######################################################################
# Load the data
# ------------------------

######################################################################
# In[2]:


from pysensing.mmwave.PC.dataset.hgr import load_hgr_dataset
# The path contains the radHAR dataset
root =  '/mnt/data_nas/data/junqiao/data/mGesture'
train_dataset, test_dataset = load_hgr_dataset("M-Gesture", root)

######################################################################
# Visualize the point cloud
# ------------------------

######################################################################
# In[3]:


from matplotlib import pyplot as plt
from pysensing.mmwave.PC.tutorial.plot import plot_3d_graph
# Example of the samples in the dataset
index = 9  # Randomly select an index
pc,gesture = train_dataset.__getitem__(index)
print(pc.shape, type(gesture))
plot_3d_graph(None, pc[8])

######################################################################
# Create model 
# ------------------------

######################################################################
# M-Gesture utilizes CNN-based model, EVL_NN with feature engineering module called RPM as the baseline hgr method. From model.hgr, we can import desired hgr model designed for mmWave PC. The model parameter for EVL_NN reimplemented for M-Gesture is as follows:

######################################################################
# In[4]:


from pysensing.mmwave.PC.model.hgr import EVL_NN
model = EVL_NN(dataset="M-Gesture", num_classes=4)
print(model)

######################################################################
# Model Train
# ------------------------

######################################################################
# pysensing library support quick training of model with the following steps. The training interface incorporates pytorch loss functions, optimizers and dataloaders to facilate training. An example is provided for how to define the aforemetioned terms.

######################################################################
# In[5]:


# Create pytorch dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)


# Define pytorch loss function as criterion 
criterion = nn.CrossEntropyLoss()


# Define pytorch optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# GPU acceleration with cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# A quick training using har_train. The resulted model parameters will be saved into "train_{num_epochs}.pth".

######################################################################
# In[6]:


# Pysensing training interface
from pysensing.mmwave.PC.inference.hgr import hgr_train
# hgr_train(model, train_loader, num_epochs=1, optimizer=optimizer, criterion=criterion, device=device)


######################################################################
# Model inference
# ------------------------

######################################################################
# Load the pretrained model, e.g. from  https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/mmwave_pc/HGR/M-Gesture_EVL_NN.pth
#, and perform human gesture recognition!

######################################################################
# In[7]:

# load pretrained model
from pysensing.mmwave.PC.inference import load_pretrain
model = load_pretrain(model, "M-Gesture", "EVL_NN").to(device)
model.eval()

######################################################################
# Test the model on testing dataset.

######################################################################
# In[8]:
from pysensing.mmwave.PC.inference.hgr import hgr_test
# hgr_test(model, test_loader, criterion=criterion, device=device)

######################################################################
# Model inference on sample and deep feature embedding of input modality in HGR task.

######################################################################
# In[9]:


idx = 5
pc, label= test_dataset.__getitem__(idx)
print(pc.shape)
pc  = torch.tensor(pc).unsqueeze(0).float().to(device)
predicted_result = model(pc)
print("The predicted gesture is {}, while the ground truth is {}".format(label,torch.argmax(predicted_result).cpu()))

# Deep feature embedding
from pysensing.mmwave.PC.inference.embedding import embedding
emb = embedding(input = pc, model=model, dataset_name = "M-Gesture", model_name = "EVL_NN", device=device)
print("The shape of feature embedding is: ", emb.shape)