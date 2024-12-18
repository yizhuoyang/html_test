"""
Tutorial for UWB Localization
=======
"""
import torch
import torch.nn as nn
import os

from pysensing.uwb.datasets.get_dataloader import *
from pysensing.uwb.models.get_model import *
from pysensing.uwb.training.localization import *
from pysensing.uwb.inference.predict import *
from pysensing.uwb.inference.embedding import *

######################################################################
# Download Data from Cloud Storage
# -----------------------------------
# 
# Open the following linke in your browser to download Localization datasets:
# 
# [Download Pedestrian_Tracking Dataset](https://pysensing.oss-ap-southeast-1.aliyuncs.com/data/uwb/Pedestrian_Tracking.zip) \
# [...]()
# 
# Unzip the downloaded file and move to your data folder. For HAR, the data folder should look like this:
# ```
# |---data 
# |------|---localization 
# |------|------|---Pedestrian_Tracking 
# |------|------|------|---processed_data
# |------|------|------|------|---AnchorPos.mat
# |------|------|------|------|---Bg_CIR_VAR.mat
# |------|------|------|------|---Dyn_CIR_VAR.mat
# |------|------|------|---raw_data
# ......
# ```

######################################################################
# Load the data
# -----------------------------------
# Human action recognition dataset: 
# 
# Human Tracking Dataset
# - UWB size : n x 1 x 500 x 2
# 
# Dataset name choices are:
# - 'pedestrian_tracking_mod1_CIR'
# - 'pedestrian_tracking_mod2_CIR'
# - 'pedestrian_tracking_mod3_CIR'
# - 'pedestrian_tracking_mod1_Var'
# - 'pedestrian_tracking_mod2_Var'
# - 'pedestrian_tracking_mod3_Var'

root = './data' 
train_loader, val_loader, test_loader = load_localization_dataset(root, 'pedestrian_tracking_mod1_CIR')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data in train_loader:
    x, y = data
    print(x.size())
    print(y.size())
    break

######################################################################
# Load the model
# -----------------------------------
# Model zoo:
# ResNet

model = load_localization_model(dataset_name = 'human_tracking', model_name = 'resnet')
print(model)

######################################################################
# Model train
# -----------------------------------

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

localization_training(
    root = root,
    dataset_name='pedestrian_tracking_mod1_CIR',
    model_name='resnet',
    num_epochs=1,
    learning_rate=0.001,
    save_weights=True,
)

######################################################################
# Model inference
# -----------------------------------

localization_predictor = predictor(
    task='localization', 
    dataset_name='human_tracking', 
    model_name='resnet',
    pt_weights = './human_tracking_weights.pth'
)
for data in test_loader:
    x, y = data
    break
outputs = localization_predictor.predict(x)
print("output shape:", outputs.shape)

######################################################################
# Generate embedding
# -----------------------------------
# - noted that the `model_name` variable defined in `load_model` function represents the model structure name, and in `load_pretrain_weights` function represents the model structure and pretrain dataset name.

model = load_localization_model(dataset_name = 'human_tracking', model_name = 'resnet')
model = load_pretrain_weights(model, dataset_name = 'human_tracking', model_name = 'CIR_model', device=device)
uwb_embedding = localization_uwb_embedding(x, model, device)
print('uwb_embedding shape: ', uwb_embedding.shape)


