"""
Tutorial for UWB Human Activity Recognition
==============================================================
"""
import torch
import torch.nn as nn
import os

from pysensing.uwb.datasets.get_dataloader import *
from pysensing.uwb.models.get_model import *
from pysensing.uwb.training.har import *
from pysensing.uwb.inference.predict import *
from pysensing.uwb.inference.embedding import *

######################################################################
# Download Data from Cloud Storage
# -----------------------------------
# 
# Open the following link in your browser to download HAR datasets:
# 
# [Download Sleep_Pose_Net Dataset](https://pysensing.oss-ap-southeast-1.aliyuncs.com/data/uwb/sleep_pose_net_data.zip) \
# [...]()
# 
# Unzip the downloaded file and move to your data folder. For HAR, the data folder should look like this:
# ```
# |---data 
# |------|---HAR 
# |------|------|---sleep_pose_net_data 
# |------|------|------|---Dataset I 
# |------|------|------|---Dataset II 
# ```

######################################################################
# Load the data
# -----------------------------------
# 
# Human action recognition dataset: 
# 
# Sleep Pose Net Dataset
# UWB size : n x 160 x 100
# x_diff and x_wrtft size is depended on preprocessing parameters
# 
# Dataset 1
# - number of classes : 6
# - train number : 623
# - test number : 307
# 
# Dataset 2
# - number of classes : 7
# - train number : 739
# - test number : 365
# 
# Dataset name choices are: 
# - 'Sleepposenet_dataset1'
# - 'Sleepposenet_dataset2_session1_ceiling'
# - 'Sleepposenet_dataset2_session1_wall'
# - 'Sleepposenet_dataset2_session1_all'
# - 'Sleepposenet_dataset2_session2_ceiling'
# - 'Sleepposenet_dataset2_session2_wall'
# - 'Sleepposenet_dataset2_session2_all'
# - 'Sleepposenet_dataset2_sessionALL_ceiling'
# - 'Sleepposenet_dataset2_sessionALL_wall'
# - 'Sleepposenet_dataset2_sessionALL_all'

root = './data' 
train_loader, test_loader = load_har_dataset(root, 'Sleepposenet_dataset2_session1_all')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data in train_loader:
    x_diff, x_wrtft, labels = data
    print(x_diff.size())
    print(x_wrtft.size())
    print(labels.size())
    break

######################################################################
# Load the model
# -----------------------------------
# Model zoo:
# Sleep Pose Net model

model = load_har_model(dataset_name = 'sleep_pose_net_dataset2', model_name = 'sleepposenet')
print(model)

######################################################################
# Model train
# -----------------------------------

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sleepposenet_training(
    root= root,
    dataset_name='Sleepposenet_dataset2_session1_all',
    datasetname_model='sleep_pose_net_dataset2',
    model_name='sleepposenet',
    num_epochs=5,
    learning_rate=0.001,
    save_weights=True
)

######################################################################
# Model inference
# -----------------------------------
# You need to define the pre-trained weight path in the `predictor` object's `pt_weight_path` variable. Otherwise, the varibale will set to None and no weight will be loaded.

har_predictor = predictor(
    task='har', 
    dataset_name='sleep_pose_net_dataset2', 
    model_name='sleepposenet',
    pt_weights = './sleepposenet_weights.pth'
)
for data in test_loader:
    x_diff, x_wrtft, labels = data
    break
outputs = har_predictor.predict([x_diff, x_wrtft])
print("output:", outputs)

######################################################################
# Generate embedding
# -----------------------------------
# - noted that the `model_name` variable defined in `load_model` function represents the model structure name, and in `load_pretrain_weights` function represents the model structure and pretrain dataset name.

model = load_har_model(dataset_name = 'sleep_pose_net_dataset2', model_name = 'sleepposenet')
model = load_pretrain_weights(model, dataset_name = 'sleep_pose_net_dataset2', model_name = 'sleepposenet', device=device)
uwb_embedding = har_uwb_embedding(x_diff, x_wrtft, model, device)
print('uwb_embedding shape: ', uwb_embedding.shape)


