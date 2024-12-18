
"""
Tutorial for Human Activity Recognition
==============================================================
"""

######################################################################
#!/usr/bin/env python
# coding: utf-8




# ------------------------
# In[1]:


import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os

######################################################################
# Dataset with radHAR: 
# ------------------------

######################################################################
# radHAR dataset is designed to use mmWave PC data collected by IWR1443Ti to predict the actions of the users.
# There are totally 5 actions in the dataset: ['boxing','jack','jump','squats','walk']
# In the library, we provide a dataloader to use mmWave PC data , converted into voxel image, and predict these actions. 


######################################################################
# Load the data
# ------------------------

######################################################################
# In[2]:


from pysensing.mmwave.PC.dataset.har import load_har_dataset
# The path contains the radHAR dataset
train_dataset, test_dataset = load_har_dataset("radHAR")

######################################################################
# Visualize the voxel image

######################################################################
# In[3]:


from matplotlib import pyplot as plt
# Example of the samples in the dataset
index = 9  # Randomly select an index
voxels,activity = train_dataset.__getitem__(index)


print(voxels.shape, type(voxels))

plt.figure(figsize=(10,6))
plt.imshow(voxels[0].transpose(1,2,0).mean(-1))
plt.title("Voxel image for activity: {}".format(activity))
plt.show()

######################################################################
# Create model 
# ------------------------

######################################################################
# raHAR utilizes MLP-based model as a baseline har method. From model.har, we can import desired har model designed for mmWave PC. The model parameter for har_MLP reimplemented for radHAR is as follows:

######################################################################
# In[4]:


from pysensing.mmwave.PC.model.har import har_MLP
model = har_MLP(dataset="radHAR", num_classes=5)
print(model)

######################################################################
# Model Train
# ------------------------

######################################################################
# pysensing library support quick training of model with the following steps. The training interface incorporates pytorch loss functions, optimizers and dataloaders to facilate training. An example is provided for how to define the aforemetioned terms.

######################################################################
# # In[5]:


# Create pytorch dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)


# Define pytorch loss function as criterion 
criterion = nn.CrossEntropyLoss()


# Define pytorch optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# GPU acceleration with cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# A quick training using har_train. The resulted model parameters will be saved into "train_{num_epochs}.pth".

######################################################################
# In[6]:


# Pysensing training interface
from pysensing.mmwave.PC.inference.har import har_train
# har_train(model, train_loader, num_epochs=1, optimizer=optimizer, criterion=criterion, device=device)

######################################################################
# Model inference
# ------------------------


######################################################################
# Load the pretrained model, e.g. from  https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/mmwave_pc/HAR/radHAR_MLP.pth
#, and perform human action recognition!

######################################################################
# In[7]:

# load pretrained model
from pysensing.mmwave.PC.inference import load_pretrain
model = load_pretrain(model, "radHAR", "har_MLP").to(device)
model.eval()

######################################################################
# Test the model on testing dataset.

######################################################################
# In[8]:
from pysensing.mmwave.PC.inference.har import har_test
# har_test(model, test_loader, criterion=criterion, device=device)

######################################################################
# Model inference on sample and deep feature embedding of input modality in HAR task.

######################################################################
# In[9]:


idx = 5
pc, label= test_dataset.__getitem__(idx)
print(pc.shape)
pc = torch.tensor(pc).unsqueeze(0).float().to(device)
predicted_result = model(pc)
print("The predicted gesture is {}, while the ground truth is {}".format(label,torch.argmax(predicted_result).cpu()))

# Deep feature embedding
from pysensing.mmwave.PC.inference.embedding import embedding
emb = embedding(input = pc, model=model, dataset_name = "radHAR", model_name = "har_MLP", device=device)
print("The shape of feature embedding is: ", emb.shape)




