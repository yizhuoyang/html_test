"""
This notebook is a tutroial for using the mmwave raw data before the FFT
( Time, chirps, virtual antennas, virtual antenna per chirp) specified
in cubelearn

https://github.com/zhaoymn/cubelearn

"""


######################################################################
# Data loading and prepocessing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

import numpy as np

# loading example data sample from the cubelearn HGR data
# {user}_{gesture(label)}_{idx}.npy}
# data have shape size (2, T, 128, 12, 256), where 2 is the real and complex part of the raw data, 
# T is the timestamps (10 for HGR and AGR, 20 for HAR), 128 is the number of chirps in a frame, 12 is the virtual antennas 
# https://github.com/zhaoymn/cubelearn?tab=readme-ov-file
user = 7
label = 2
sample = 1

#replace with your data path, please download and unzip data from https://github.com/zhaoymn/cubelearn?tab=readme-ov-file
#HAR data path should be .../HAR_data/activity_organized/{user}_{label}_{sample}.npy
raw_data = np.load(f'./{user}_{label}_{sample}.npy')

#combine the real and complex part
data = raw_data[0, :, :, :, :] + raw_data[1,:,:,:,:] * 1j

#DAT and RDAT models takes partial input for efficiency, skip this in other model
data = data[:,:64,:,:128]

#Data type is complex64
data = np.array(data, dtype=np.complex64)


######################################################################
# model loading and inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

import torch
import requests
from pysensing.mmwave_raw.models.network import DAT_2DCNNLSTM

# URL of the pretrained model
pretrained_model_url = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/mmwave_raw/HAR/DAT_2DCNNLSTM_HAR.pth"
# */pretrain/modality/task/model_name.pth
# modelname = {DAT_2DCNNLSTM_HAR,DAT_2DCNNLSTM_AGR,DAT_2DCNNLSTM_HGR,RDAT_3DCNNLSTM_HAR,RDAT_3DCNNLSTM_AGR,RDAT_3DCNNLSTM_HGR}
local_model_path = "./DAT_2DCNNLSTM_HAR.pth"

# Download the pretrained weights
response = requests.get(pretrained_model_url)
with open(local_model_path, "wb") as f:
    f.write(response.content)

#loading the model and pretrained weight
model = DAT_2DCNNLSTM(HAR=True)
model.load_state_dict(torch.load(local_model_path, weights_only=True)['model_state_dict'])
model.eval()

#convert data to torch tensor
data = torch.tensor(data)

#unsqueeze for the batch dimension
x = data.unsqueeze(0) 
one_hot = model(x)

#class prediction
class_idx = torch.argmax(one_hot)

print(f"The prediction is {class_idx==label}")


######################################################################
# Embedding extraction
# --------------------
# 


######################################################################
# For lstm models the embedding is extracted after the lstm (recommened)
# 

from pysensing.mmwave_raw.inference.embedding import embedding

emb = embedding(x,model,'cpu',True)

print(emb.shape)


######################################################################
# For non-lstm model the embedding is extracted after the final max
# pooling layer berfore the FCs, might have different shape for different
# models
# 

from pysensing.mmwave_raw.models.network import DAT_3DCNN

model_ = DAT_3DCNN()

emb = embedding(x,model_,'cpu',False)

print(emb.shape)


######################################################################
# for non DAT and RDAT models don’t forget to use the whole data
# 

from pysensing.mmwave_raw.models.network import RAT_3DCNN
model_ = RAT_3DCNN()
data_ = raw_data[0, :, :, :, :] + raw_data[1,:,:,:,:] * 1j
data_ = data_[:,:128,:,:256] #whole data cube
data_ = np.array(data_, dtype=np.complex64)
data_ = torch.tensor(data_)
x_ = data_.unsqueeze(0) 
emb = embedding(x_,model_,'cpu',False)
print(emb.shape)
