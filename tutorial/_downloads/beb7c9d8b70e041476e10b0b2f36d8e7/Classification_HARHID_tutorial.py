"""
CSI classification tasks Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for CSI classification tasks, including Human activity recognition and Human identity detection
# 
import sys
sys.path.append('../..')
import pysensing.csi.dataset.get_dataloader as get_dataloader
import pysensing.csi.model.get_model as get_model
import pysensing.csi.inference.predict as predict
import pysensing.csi.inference.train as train
import pysensing.csi.inference.embedding as embedding
import torch

######################################################################
# Load the data
# -----------------------------------

# Human action recognition dataset: 

# UT-HAR
# CSI size : 1 x 250 x 90
# number of classes : 7
# classes : lie down, fall, walk, pickup, run, sit down, stand up
# train number : 3977
# test number : 996

# NTU-HAR
# CSI size : 3 x 114 x 500
# number of classes : 6
# classes : box, circle, clean, fall, run, walk
# train number : 936
# test number : 264

# Widar
# BVP size : 22 x 20 x 20
# number of classes : 22
# classes :
# Push&Pull, Sweep, Clap, Slide, Draw-N(H), Draw-O(H),Draw-Rectangle(H),
# Draw-Triangle(H), Draw-Zigzag(H), Draw-Zigzag(V), Draw-N(V), Draw-O(V), Draw-1,
# Draw-2, Draw-3, Draw-4, Draw-5, Draw-6, Draw-7, Draw-8, Draw-9, Draw-10
# train number : 34926
# test number : 8726

# Human identity detection dataset:

# NTU-HumanID
# CSI size : 3 x 114 x 500
# number of classes : 14
# classes : gaits of 14 subjects
# train number : 546
# test number : 294
# Examples of NTU-Fi data


train_loader, test_loader = get_dataloader.load_classification_dataset('UT_HAR', batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data in train_loader:
   
    csi, label = data
    csi = csi.to(device)
    label = label.type(torch.LongTensor).to(device)
    print('data:', csi)
    print('label:', label)
    break

######################################################################
# Load the model
# -----------------------------------
# Model zoo:
# MLP
# LeNet
# ResNet
# RNN
# GRU
# LSTM
# BiLSTM
# CNN+GRU
# ViT

model = get_model.load_har_model('UT_HAR', 'MLP')
print(model)

######################################################################
# Model train
# ------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_num = 1

train.har_train(train_loader, model, epoch_num, optimizer, criterion, device)

######################################################################
# Model inference
# ------------------------

model = get_model.load_pretrain(model, 'UT_HAR', 'MLP', device=device)
output = predict.har_predict(csi, 'UT_HAR', model, device).type(torch.FloatTensor).to(device)
print("output:", output)

######################################################################
# Evaluate the loss
# ------------------------

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, label)
print(loss)

######################################################################
# Generate embedding
# ------------------------

csi_embedding = embedding.har_csi_embedding(csi, 'UT_HAR', model, device)
print('csi_embedding: ', csi_embedding)



######################################################################
# And that's it. We're done with our CSI humna activity recognition and human identity detection tutorials. Thanks for reading.
