"""
Acoustic Pedestrian Detection Tutorial
==============================================================
"""

######################################################################
# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for acoustic Human pose estimation
# 

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import pysensing.acoustic.preprocessing.transform as transform
from pysensing.acoustic.inference.utils import *
from pysensing.acoustic.datasets.ped_det import AVPed,AFPILD
from pysensing.acoustic.models.ped_det import PED_CNN,PED_CRNN
from pysensing.acoustic.models.get_model import load_ped_det_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
######################################################################
# AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness
# ----------------------------------------------------------------------------------
# Reimplementation of "AV-PedAware: Self-Supervised Audio-Visual Fusion for Dynamic Pedestrian Awareness".
#
# This dataset contains the footstep sound of the pedestains which used for pedestrian localization..
#
# Note: Different from original paper which utilizes both audio and visual data to train the network. This library only focuses on using only audio data for pedestrian localization.
#
# The dataset can be downloaded from https://github.com/yizhuoyang/AV-PedAware

######################################################################
# Load the data
# ------------------------
# The dataset can be downloaded from this github repo: https://github.com/yizhuoyang/AV-PedAware

root = './data' # The path contains the AVPed dataset
avped_traindataset = AVPed(root,'train')
avped_testdataset = AVPed(root,'test')
index = 20
# Randomly select an index
spectrogram,position,lidar= avped_traindataset.__getitem__(index)
plt.figure(figsize=(5,3))
plt.imshow(spectrogram.numpy()[0])
plt.title("Spectrogram")
plt.show()
######################################################################
# Load model
# ------------------------

# Method 1:
avped_model = PED_CNN(0.2).to(device)
# Method 2:
avped_model = load_ped_det_model('ped_cnn',pretrained=True).to(device)

######################################################################
# Modle Training and Testing
# ------------------------

# Model training
from pysensing.acoustic.inference.training.ped_det_train import *
avped_trainloader = DataLoader(avped_traindataset,batch_size=64,shuffle=True,drop_last=True)
avped_testloader  = DataLoader(avped_traindataset,batch_size=64,shuffle=True,drop_last=True)
epoch = 1
optimizer = torch.optim.Adam(avped_model.parameters(), 0.001)
loss = ped_det_train_val(avped_model,avped_trainloader,avped_testloader, epoch, optimizer, device, save_dir='/data',save = False)

# Model testing
loss = ped_det_test(avped_model,avped_testloader,  device)
######################################################################
# Modle Inference
# ------------------------

# Method 1
spectrogram,position,lidar= avped_testdataset.__getitem__(1)
avped_model.eval()
#Direct prediction use the model
predicted_result = avped_model(spectrogram.unsqueeze(0).float().to(device))
position = position.unsqueeze(0).numpy()
predicted_result = predicted_result.cpu().detach().numpy()
draw_scenes(lidar,position,predicted_result)

# Method 2
#Use inference.predict
from pysensing.acoustic.inference.predict import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted_result  = ped_det_predict(spectrogram,'AVPed',avped_model, device=device)
predicted_result = predicted_result.cpu().detach().numpy()
draw_scenes(lidar,position,predicted_result)

######################################################################
# Modle Embedding
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = ped_det_embedding(spectrogram,'AVPed',avped_model, device=device)


######################################################################
# AFPILD: Acoustic footstep dataset collected using one microphone array and LiDAR sensor for person identification and localization
# ----------------------------------------------------------------------------------
# Reimplementation of "AFPILD: Acoustic footstep dataset collected using one microphone array and LiDAR sensor for person identification and localization".

# This dataset contains footstep sound of the pedestains which used for pedestrian localization and classification
######################################################################
# Load the data
# ------------------------

# Method 1: Use get_dataloader
from pysensing.acoustic.datasets.get_dataloader import *
train_loader,test_loader = load_ped_det_dataset(
    root='./data',
    dataset='AFPILD',
    download=True)

# Method 2
root = './data' # The path contains the AFPILD dataset
afpild_traindataset = AFPILD(root,'ideloc_ori_cloth','train')
afpild_testdataset = AFPILD(root,'ideloc_ori_cloth','test')
# Define the Dataloader
afpild_trainloader = DataLoader(afpild_traindataset,batch_size=64,shuffle=True,drop_last=True)
afpild_testloader = DataLoader(afpild_testdataset,batch_size=64,shuffle=True,drop_last=True)
#List the activity classes in the dataset
index = 330
# Randomly select an index
data_dict,label = afpild_testdataset.__getitem__(index)

fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].imshow(data_dict['spec'][:,:,0], aspect='auto', origin='lower')
axs[0].set_title('Spectrogram')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Frequency')

axs[1].imshow(data_dict['gcc'][:,:,0], aspect='auto', origin='lower')
axs[1].set_title('GCC')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Lag')

person_id, angle = label
print(f"Person ID: {person_id}, Angle: {angle:.2f} radians")



######################################################################
# Model training
# ------------------------
from pysensing.acoustic.inference.training.AFPILD_utils.training import afpild_train

afpild_train(
      config_file="./data/AFPILD/afpild_spec_gcc_fusion.json",
      root_dir='./data/AFPILD',
      task='accil_ana_shoe',
      epochs=1,
      num_workers=4,
      dataset_dir='./data/AFPILD/')

######################################################################
# Model testing
# ------------------------
from pysensing.acoustic.inference.training.AFPILD_utils.testing import afpild_testing
afpild_testing(
    config_file="./data/AFPILD/afpild_spec_gcc_fusion.json",
    root_dir= "./data/AFPILD",
    dataset_dir="./data/AFPILD/",
    resume_path="./data/AFPILD/saved/AFPILD-CRNN/20241030055348/model/model_best.pth", # Path to the trained model
    task='accil_ori_rd',
)

######################################################################
# Model inference
# ------------------------

# Load the model 1
avped_model = PED_CRNN(task='ideloc_ori_cloth.pth').to(device)
# avped_model.load_state_dict(torch.load('path to weights',weights_only=True)['models']['model'])

# Load the model 2
avped_model = load_ped_det_model('ped_crnn',pretrained=True,task='ideloc_ori_cloth').to(device)

# Model prediction 1
data_dict_tensor = {k: torch.Tensor(v).to(device).unsqueeze(0).float() for k, v in data_dict.items()}
output = avped_model(data_dict_tensor).squeeze(0).detach().cpu().numpy()
#print("The predicted person id is: {}, the ground truth is: {}".format(np.argmax(output[:40]),int(label[0])))
#print("The predicted angle is: {}, the ground truth is: {}".format(output[-1],label[1]))

# Model prediction 2
from pysensing.acoustic.inference.predict import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
predicted_result  = ped_det_predict(data_dict,'AFPILD',avped_model, device=device)
predicted_result = predicted_result.cpu().detach().numpy()
print("The predicted person id is: {}, the ground truth is: {}".format(np.argmax(output[:40]),int(label[0])))
print("The predicted angle is: {}, the ground truth is: {}".format(output[-1],label[1]))

######################################################################
# Model embedding
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = ped_det_embedding(data_dict,'AFPILD',avped_model, device=device)

######################################################################
# And that's it. We're done with our acoustic humna pose estimation tutorials. Thanks for reading.
