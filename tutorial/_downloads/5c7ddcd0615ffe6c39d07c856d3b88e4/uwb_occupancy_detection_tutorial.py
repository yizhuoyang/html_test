"""
Tutorial for UWB Occupancy Detection
=======
"""

import torch
import torch.nn as nn
import os

from pysensing.uwb.datasets.get_dataloader import *
from pysensing.uwb.models.get_model import *
from pysensing.uwb.training.occupancy_detection import *
from pysensing.uwb.inference.predict import *
from pysensing.uwb.inference.embedding import *

######################################################################
# Download Data from Cloud Storage
# -----------------------------------
# 
# Open the following link in your browser to download HAR datasets:
# 
# [Download nlos_human_detection Dataset](https://pysensing.oss-ap-southeast-1.aliyuncs.com/data/uwb/nlos_human_detection.zip) \
# [...]()
# 
# Unzip the downloaded file and move to your data folder. For HAR, the data folder should look like this:
# ```
# |---data 
# |------|---occupancy_detection 
# |------|------|---nlos_human_detection 
# |------|------|------|---dynamic
# |------|------|------|---static
# ```

######################################################################
# Load the data
# -----------------------------------
# Occupancy Detection dataset: 
# 
# NLOS Huaman Detection Dataset
# - UWB size : n x 256
# 
# Dataset name choices are:
# - 'nlos_human_detection_raw_dynamic'
# - 'nlos_human_detection_raw_static'

root = './data' 
train_loader, val_loader = load_occupancy_detection_dataset(root, datasetname='nlos_human_detection_raw_dynamic')
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
# - LSTM
# - AutoEncoder

model = load_occupancy_detection_model(dataset_name = 'nlos_human_detection', model_name = 'autoencoder')
print(model)

######################################################################
# Model train
# -----------------------------------

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlos_human_detection_training(
    root = root,
    dataset_name='nlos_human_detection_raw_dynamic',
    model_name='autoencoder',
    num_epochs=1,
    learning_rate=0.001,
    save_weights=True,
)

######################################################################
# Model inference
# -----------------------------------

occupancy_detector = predictor(
    task='occupancy_detection',
    dataset_name='nlos_human_detection',
    model_name='autoencoder',
    pt_weights='./nlos_human_detection_weights.pth'
    )
for data in val_loader:
    x, y = data
    break
outputs = occupancy_detector.predict(x)
print("output:", outputs)

######################################################################
# Generate embedding
# -----------------------------------
# - noted that the `model_name` variable defined in `load_model` function represents the model structure name, and in `load_pretrain_weights` function represents the model structure and pretrain dataset name.

model = load_occupancy_detection_model(dataset_name = 'nlos_human_detection', model_name = 'autoencoder')
model = load_pretrain_weights(model, dataset_name = 'nlos_human_detection_static', model_name = 'autoencoder',device=device)
uwb_embedding = occupancy_detection_uwb_embedding(x, model, device)
print('uwb_embedding shape: ', uwb_embedding.shape)


