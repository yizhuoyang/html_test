"""
Acoustic Human Activity Recognition Tutorial
==============================================================
"""

######################################################################
# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for acoustic Human activity recognition
# 
import torch
torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt
import numpy as np
import pysensing.acoustic.datasets.har as har_datasets
import pysensing.acoustic.models.har as har_models
import pysensing.acoustic.models.get_model as acoustic_models
import pysensing.acoustic.inference.embedding as embedding
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

######################################################################
# SAMoSA: Sensoring Activities with Motion abd Subsampled Audio
# -----------------------------------
# SAMSoSA dataset is designed to use audio and IMU data collected by a watch to predict the actions of the users.
#
# There are totally 27 actions in the dataset. 
#
# In the library, we provide a dataloader to use only audio data to predict these actions. 

######################################################################
# Load the data
# ------------------------
# Method 1: Use get_dataloader
from pysensing.acoustic.datasets.get_dataloader import *
train_loader,test_loader = load_har_dataset(
    root='./data',
    dataset='samosa',
    download=True)

# Method 2: Manually setup the dataloader
root = './data' # The path contains the samosa dataset
samosa_traindataset = har_datasets.SAMoSA(root,'train')
samosa_testdataset = har_datasets.SAMoSA(root,'test')
# Define the dataloader
samosa_trainloader = DataLoader(samosa_traindataset,batch_size=64,shuffle=True,drop_last=True)
samosa_testloader = DataLoader(samosa_testdataset,batch_size=64,shuffle=True,drop_last=True)
dataclass = samosa_traindataset.class_dict
datalist  = samosa_traindataset.audio_data
# Example of the samples in the dataset
index = 50  # Randomly select an index
spectrogram,activity= samosa_traindataset.__getitem__(index)
plt.figure(figsize=(10,5))
plt.imshow(spectrogram.numpy()[0])
plt.title("Spectrogram for activity: {}".format(activity))
plt.show()
######################################################################
# Load the model
# ------------------------
# Method 1:
samosa_model = har_models.HAR_SAMCNN(dropout=0.6).to(device)
# Method 2:
samosa_model = acoustic_models.load_har_model('samcnn',pretrained=True).to(device)


######################################################################
# Model training and testing
# ------------------------
from pysensing.acoustic.inference.training.har_train import *
# Model training
epoch = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(samosa_model.parameters(), 0.0001)
har_train_val(samosa_model,samosa_trainloader,samosa_testloader, epoch, optimizer, criterion, device, save_dir = './data',save = True)

# Model testing
test_loss = har_test(samosa_model,samosa_testloader,criterion,device)

######################################################################
# Modle inference for single sample
# ------------------------
# Method 1
# You may aslo load your own trained model by setting the path
# samosa_model.load_state_dict(torch.load('path_to_model')) # the path for the model
spectrogram,activity= samosa_testdataset.__getitem__(3)
samosa_model.eval()
#Direct use the model for sample inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted_result = samosa_model(spectrogram.unsqueeze(0).float().to(device))
print("The ground truth is {}, while the predicted activity is {}".format(activity,torch.argmax(predicted_result).cpu()))

# Method 2
# Use inference.predict
from pysensing.acoustic.inference.predict import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted_result  = har_predict(spectrogram,'SAMoSA',samosa_model, device)
print("The ground truth is {}, while the predicted activity is {}".format(activity,torch.argmax(predicted_result).cpu()))

######################################################################
# Modle inference for single sample
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = har_embedding(spectrogram,'SAMoSA',samosa_model, device=device)

######################################################################
#  Implementation of "AudioIMU: Enhancing Inertial Sensing-Based Activity Recognition with Acoustic Models"
# -----------------------------------
# This dataset is designed to use audio and IMU data collected by a watch to predict the actions of the users, 23 different activities are collected in the dataset.
#
# But different from the orginal paper, the reimplemeted paper only takes the audio data for human activity recognition. Subjects 01, 02, 03, 04 are used for testing while the other are used for training.

######################################################################
# Load the data
# ------------------------
# Method 1: Use get_dataloader
from pysensing.acoustic.datasets.get_dataloader import *
train_loader,test_loader = load_har_dataset(
    root='./data',
    dataset='audioimu',
    download=True)

# Method2
root = './data' # The path contains the audioimu dataset
audioimu_traindataset = har_datasets.AudioIMU(root,'train')
audioimu_testdataset = har_datasets.AudioIMU(root,'test')
# Define the Dataloader
audioimu_trainloader = DataLoader(audioimu_traindataset,batch_size=64,shuffle=False,drop_last=True)
audioimu_testloader = DataLoader(audioimu_testdataset,batch_size=64,shuffle=False,drop_last=True)
#List the activity classes in the dataset
dataclass = audioimu_traindataset.classlist
# Example of the samples in the dataset
index = 0  # Randomly select an index
spectrogram,activity= audioimu_testdataset.__getitem__(index)
print(spectrogram.shape)
plt.figure(figsize=(18,6))
plt.imshow(spectrogram.numpy()[0])
plt.title("Spectrogram for activity: {}".format(activity))
plt.show()

######################################################################
# Load the model
# ------------------------
# Method 1
audio_model = har_models.HAR_AUDIOCNN().to(device)
# Method2
audio_model = acoustic_models.load_har_model('audiocnn',pretrained=True).to(device)


######################################################################
# Model training and testing
# ------------------------
from pysensing.acoustic.inference.training.har_train import *
epoch = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(audio_model.parameters(), 0.0001)
loss = har_train_val(audio_model,audioimu_trainloader,audioimu_testloader, epoch, optimizer, criterion, device, save_dir='./data',save = False)

# Model testing
test_loss = har_test(audio_model,audioimu_testloader,criterion,device)


######################################################################
# Model inference
# ------------------------
# You may aslo load your own trained model by setting the path
# audio_model.load_state_dict(torch.load('path_to_model')) # the path for the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
spectrogram,activity= audioimu_testdataset.__getitem__(6)
audio_model.eval()
predicted_result = audio_model(spectrogram.unsqueeze(0).float().to(device))
print("The ground truth is {}, while the predicted activity is {}".format(activity,torch.argmax(predicted_result).cpu()))

#Method 2
#Use inference.predict
from pysensing.acoustic.inference.predict import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
predicted_result  = har_predict(spectrogram,'AudioIMU',audio_model, device)
print("The ground truth is {}, while the predicted activity is {}".format(activity,torch.argmax(predicted_result).cpu()))

######################################################################
# Model embedding
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = har_embedding(spectrogram,'AudioIMU',audio_model, device=device)

######################################################################
# And that's it. We're done with our acoustic humna activity recognition tutorials. Thanks for reading.
