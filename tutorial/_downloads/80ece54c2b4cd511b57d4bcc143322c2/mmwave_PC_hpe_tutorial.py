
"""
Tutorial for Human Pose Estimation
==============================================================
"""

######################################################################
# In[1]:

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os

######################################################################
# Dataset with MetaFi: 
# ------------------------

######################################################################
# Point cloud Pose reconstruction dataset collected by Ti 6843 mmWave radar. 40 subjects are included and the human poses are obtained by 2 RGB camera.
# We provide cross-subject experiment settings with all daily activities. 
# In the library, we provide a dataloader to use mmWave PC data, and predict these human poses. 


######################################################################
# Load the data
# ------------------------

######################################################################
# In[3]:

from pysensing.mmwave.PC.dataset.hpe import load_hpe_dataset
# The path contains the radHAR dataset

train_dataset, test_dataset = load_hpe_dataset("MetaFi")


######################################################################
# Visualize the PC data
# ------------------------

######################################################################
# In[6]:

from matplotlib import pyplot as plt
from pysensing.mmwave.PC.tutorial.plot import plot_3d_graph
# Example of the samples in the dataset
index = 10  # Randomly select an index
pc,pose = train_dataset.__getitem__(index)
print(pc.shape, type(pose))
plot_3d_graph(pose, pc[0])

######################################################################
# Create model 
# ------------------------

######################################################################
# mmFi utilizes PointTransformer model as a baseline hpe method. From model.hpe, we can import 
# desired hpe model designed for mmWave PC. The model parameter for PointTransformer reimplemented 
# for mmFi is as follows:

######################################################################
# In[7]:

from pysensing.mmwave.PC.model.hpe import PointTransformerReg
model = PointTransformerReg(
                    input_dim = 5,
                    nblocks = 5,
                    n_p = 17
                )
print(model)


######################################################################
# A shortcut for loading the hpe model to avoid the tedious hyper-parameter setting.


######################################################################
# In[8]:


from pysensing.mmwave.PC.model.hpe import load_hpe_model
model = load_hpe_model("MetaFi", "PointTransformer")
print(model)


######################################################################
# Model Train
# ------------------------

######################################################################
# pysensing library support quick training of model with the following steps. The training interface 
# incorporates pytorch loss functions, optimizers and dataloaders to facilate training. 
# An example is provided for how to define the aforemetioned terms.


######################################################################
# In[11]:


# Create pytorch dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)

# Define pytorch loss function as criterion 
criterion = nn.CrossEntropyLoss()

# Define pytorch optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# GPU acceleration with cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# A quick training using hpe_train. The resulted model parameters will be saved into "train_{num_epochs}.pth".

######################################################################
# In[12]:


# Pysensing training interface
from pysensing.mmwave.PC.inference.hpe import hpe_train
# hpe_train(model, train_loader, num_epochs=1, optimizer=optimizer, criterion=criterion, device=device)


######################################################################
# Model inference
# ------------------------

######################################################################
# Load the pretrained model, e.g. from  https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/mmwave_pc/hpe/MetaFi_PointTransformer.pth 
#, and perform human pose estimation!

######################################################################
# In[13]:

# load pretrained model
from pysensing.mmwave.PC.inference import load_pretrain
model = load_pretrain(model, "MetaFi", "PointTransformer").to(device)
model.eval()


######################################################################
# Test the model on testing dataset.

######################################################################
# In[14]:
from pysensing.mmwave.PC.inference.hpe import hpe_test
# hpe_test(model, test_loader, criterion=criterion, device=device)

######################################################################
# Model inference on sample and deep feature embedding of input modality in HPE task.

######################################################################
# In[15]:

# Model inference
idx = 5
points, pose= test_dataset.__getitem__(idx)
points = torch.tensor(points).unsqueeze(0).float().to(device)
predicted_result = model(points)
print("The predicted pose is {}, while the ground truth is {}".format(predicted_result.cpu(),pose))

# Deep feature embedding
from pysensing.mmwave.PC.inference.embedding import embedding
emb = embedding(input = points, model=model, dataset_name = "MetaFi", model_name = "PointTransformer", device=device)
print("The shape of feature embedding is: ", emb.shape)



######################################################################
# mmDiff: diffusion model for mmWave radar HPE
# ------------------------

######################################################################
# Load Diffusion Runner with model initialized. This process will define the setting for model and dataset. Currently two settings are implemented: 
# 1. "mmBody + P4Transformer": 
#     Phase 1: Input [b, 4, 5000, 6]; Output: [b, 17, 3] and [b, 17, 64]. 
#     Phase 2: GRC, LRC, TMC, SLC
# 2. "MetaFi + PointTransformer": 
#     Phase 1: Input [b, 5, 150, 5]; Output: [b, 17, 3] and [b, 17, 32]. 
#     Phase 2: GRC, TMC, SLC

######################################################################
# In[16]:
from pysensing.mmwave.PC.model.hpe.mmDiff.load_mmDiff import load_mmDiff
mmDiffRunner = load_mmDiff("MetaFi")


######################################################################
# Phase 1 Training: Can train phase 1 from scratch (is_train = True) or load pretrained phase 1 model (is_train = False).
#  Set is_save = True to facilitate phase 2 training acceleration.
# If phase 1 features are saved, set is_save = False.

######################################################################
# In[17]:

mmDiffRunner.phase1_train(train_dataset, test_dataset, is_train=False, is_save=False)

######################################################################
# Phase 1 can also receive self defined model and the model should follow the setting defined above. The Self-defined model should output coarse joints and coarse joint features.

######################################################################
# In[18]:

# Self defined model should output coarse joints and coarse joint features
from pysensing.mmwave.PC.model.hpe.pointTrans import PointTransformerReg_feat
model = PointTransformerReg_feat(
                    input_dim = 5,
                    nblocks = 5,
                    n_p = 17
                )
print(model)
mmDiffRunner.phase1_train(train_dataset, test_dataset, model_self=model, is_train=False, is_save=False)

######################################################################
# Phase 2 Training: Can train from scratch (is_train = True) or load pretrained phase 2 model (is_train = False).

######################################################################
# In[19]:

mmDiffRunner.phase2_train(train_loader = None, is_train = False)

######################################################################
# Testing mmDiff

######################################################################
# In[20]:

#mmDiffRunner.test()
