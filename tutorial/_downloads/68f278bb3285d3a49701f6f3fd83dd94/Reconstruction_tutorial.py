"""
CSI reconstruction Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for CSI human pose estimation task
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
# CSI reconstruction dataset: 

# HandFi
# CSI size : 6, 20, 114
# image : 144, 144
# joints2d :  2, 42
# joints3d : 2, 21
# train number : 3600
# test number : 400


train_loader, test_loader = get_dataloader.load_recon_dataset('HandFi', batch_size=32, return_train=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for data in train_loader:
    ((joints,image), csi) = data
    joint = joints[:,:,0:21].to(device,dtype=torch.float)
    img=image.to(device,dtype=torch.float)
    csi=csi.to(device,dtype=torch.float)
    joint2d = joint[:,0:2,:] 
    joint2d = joint2d.view(-1,42)
    joint3d = joint[:,2,:] 
    
    print('data:', csi)
    print('img:', img)
    print('joint:', joint)
    break

######################################################################
# Load the model
# -----------------------------------
# For HandFi dataset, model zoo contains AutoEncoder.

model = get_model.load_recon_model('HandFi', 'AutoEncoder')
print(model)

######################################################################
# Model train
# ------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_num = 1

train.recon_train(train_loader, model, epoch_num, optimizer, device)

######################################################################
# Model inference
# ------------------------

model = get_model.load_pretrain(model, 'HandFi', 'AutoEncoder', device=device)
output = predict.recon_predict(csi, 'HandFi', model, device)
_, mask, twod, threed = output 
print("mask:", mask.shape)
print("twod:", twod.shape)
print("threed:", threed.shape)


######################################################################
# Evaluate the loss
# ------------------------

IoUerr = train.IoU(img,mask) 
mPAerr = train.mPA(img,mask)
mpjpe, pck = train.mpjpe_pck(joint2d,joint3d, twod, threed)

print(  f'mPA: {mPAerr:.3f} | => IoU: {IoUerr:.3f} | => mpjpe: {mpjpe:.3f} | =>pck: {pck:.3f}\n')
######################################################################
# Generate embedding
# ------------------------

csi_embedding = embedding.recon_csi_embedding(csi, 'HandFi', model, device)
print('csi_embedding: ', csi_embedding)


######################################################################
# And that's it. We're done with our CSI reconstruction tutorials. Thanks for reading.
