"""
CSI human pose estimation Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for CSI human pose estimation task
# 
import sys
sys.path.append('../..')
import torch
import pysensing.csi.dataset.get_dataloader as get_dataloader
import pysensing.csi.model.get_model as get_model
import pysensing.csi.inference.predict as predict
import pysensing.csi.inference.embedding as embedding
import pysensing.csi.inference.train as train
import itertools

######################################################################
# Load the data
# -----------------------------------

# MMFi, the first multi-modal non-intrusive 4D human dataset with 27 daily or rehabilitation action categories, leveraging LiDAR, mmWave radar, and WiFi signals for device-free human sensing.. MM-Fi consists of over 320k synchronized frames of five modalities from 40 human subjects.

# WiPose consists of 166,600 packets of .mat format. These packets contain pose annotations and WiFi channel state information (CSI) of 12 different actions performed by 12 volunteers, including wave, walk, throw, run, push, pull, jump, crouch, circle, sit down, stand up, and bend.


train_loader, val_loader = get_dataloader.load_hpe_dataset(dataset_name='MMFi', protocol='protocol1', split_to_use='random_split', random_seed=0, random_ratio=0.8, batch_size=1, data_unit='frame')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i, data in enumerate(train_loader):
    csi = data['input_wifi-csi'].type(torch.FloatTensor).to(device)
    label = data['output'].to(device)
    break
print('csi: ', csi)
print('label: ', label)

######################################################################
# Load the model
# -----------------------------------
# For MMFi dataset, model zoo contains WPNet and WPFormer

model = get_model.load_hpe_model('MMFi', 'WPNet')
print(model)

######################################################################
# Model train
# ------------------------
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_num = 1
train_loader_small = list(itertools.islice(train_loader, 10))

train.hpe_train(train_loader_small, model, epoch_num, optimizer, criterion, device)

######################################################################
# Model inference
# ------------------------

model = get_model.load_pretrain(model, 'MMFi', 'WPNet', device=device)
output = predict.hpe_predict(csi, 'MMFi', model, device).to(device)
print('output: ', output)


######################################################################
# Evaluate the loss
# ------------------------

criterion = torch.nn.MSELoss().to(device)
loss = criterion(output, label)
print(loss)

######################################################################
# Generate embedding
# ------------------------

csi_embedding = embedding.hpe_csi_embedding(csi, 'MMFi', model, device)
print('csi_embedding: ', csi_embedding)


######################################################################
# And that's it. We're done with our CSI human pose estimation tutorials. Thanks for reading.
