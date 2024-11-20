"""
Acoustic Human Pose estimation Tutorial
==============================================================
"""

######################################################################
# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for acoustic Human pose estimation
# 

from pysensing.acoustic.datasets.utils.hpe_vis import *
from pysensing.acoustic.models.hpe import Speech2pose,Wipose_LSTM
from pysensing.acoustic.models.get_model import load_hpe_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
######################################################################
# Listening Human Behavior: 3D Human Pose Estimation with Acoustic Signals
# ----------------------------------------------------------------------------------
#Implementation of "Listening Human Behavior: 3D Human Pose Estimation with Acoustic Signals".
#
#This dataset contains the audio reflected by human to estimate the 3D human pose with the acoustic signals.
#
#Reference: https://github.com/YutoShibata07/AcousticPose_Public

######################################################################
# Load the data
# ------------------------

# Method 1: Use get_dataloader
from pysensing.acoustic.datasets.get_dataloader import *
train_loader,val_loader,test_loader = load_hpe_dataset(
    root='./data',
    dataset_name='pose_regression_timeseries_subject_1',
    download=True)

# Method 2
csv = './data/hpe_dataset/csv/pose_regression_timeseries_subject_1/test.csv' # The path contains the samosa dataset
data_dir = './data'
hpe_testdataset = SoundPose2DDataset(csv,sound_length=2400,input_feature='logmel',
                                     mean=np.array(get_mean()).astype("float32")[:4],
                                     std=np.array(get_std()).astype("float32")[:4],
                                     )
index = 10 # Randomly select an index
sample= hpe_testdataset.__getitem__(index)
print(sample['targets'].shape)
print(sample['sound'].shape)
######################################################################
# Load Speech2pose model
# ------------------------

# Method 1
hpe_model = Speech2pose(out_cha=63).to(device)
# model_path = 'path to pretrian weights'
# state_dict = torch.load(model_path,weights_only=True)
# hpe_model.load_state_dict(state_dict)

# Method 2
hpe_model = load_hpe_model('speech2pose',pretrained=True,task='subject8').to(device)
######################################################################
# Modle Inference
# ------------------------

#Method 1
sample= hpe_testdataset.__getitem__(index)
hpe_model.eval()
predicted_result = hpe_model(sample['sound'].unsqueeze(0).float().to(device))
vis_images = make_images(sample['targets'].numpy(),predicted_result.cpu().detach().numpy().squeeze(0))

#Method 2
from pysensing.acoustic.inference.predict import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted_result  = hpe_predict(sample['sound'],'SoundPose2DDataset',hpe_model, device=device)
vis_images = make_images(sample['targets'].numpy(),predicted_result.cpu().detach().numpy().squeeze(0))

seq_num = 0
fig = plt.figure(figsize=(12, 12))
plt.imshow(vis_images[seq_num]['img'])
plt.axis('off')
plt.show()

######################################################################
# Modle Embedding
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = hpe_embedding(sample['sound'],'SoundPose2DDataset',hpe_model, device=device)

######################################################################
# Modle Training
# ------------------------
from pysensing.acoustic.inference.training.AcousticPose_utils.hpe_train import train_model,generate_configs

args = {
    "root_dir": "./data/hpe_dataset/testing_result",
    "save_name": "seq1",
    "input_feature": "logmel",
    "batchsize": 64,
    "max_epoch": 50,
    "csv_path": "./data/hpe_dataset/csv",
    "dataset_name": "pose_regression_timeseries_subject_1",
    "model": "speech2pose",
    "sound_length": 2400,
    "learning_rate": 0.01,
}
config_path = args["root_dir"]+'/'+args["save_name"]+"/config.yaml"
generate_configs(args)
resume_training = False
random_seed = 0

train_model(
    config_path=config_path,
    resume=resume_training,
    seed=random_seed,
)

# Modle Training
# ------------------------
from pysensing.acoustic.inference.training.AcousticPose_utils.hpe_test import evaluate_model
args = {
    "root_dir": "./data/hpe_dataset/testing_result",
    "save_name": "seq1",
    "batchsize": 64,
    "max_epoch": 20,
    "csv_path": "./data/hpe_dataset/csv",
    "dataset_name": "pose_regression_timeseries_subject_1",
    "model": "speech2pose",
    "sound_length": 2400,
    "learning_rate": 0.01,
}
config_path = args["root_dir"]+'/'+args["save_name"]+"/config.yaml"
evaluation_mode = "test"
model_path = None

evaluate_model(
    config_path=config_path,
    mode=evaluation_mode,
    model_path=model_path)


######################################################################
# Load the Wipose_LSTM model
# ------------------------

# Method 1
hpe_model = Wipose_LSTM(in_cha=4,out_cha=63).to(device)
# model_path = 'path to trained model'
# state_dict = torch.load(model_path,weights_only=True)

# Method 2
hpe_model = load_hpe_model('wipose',pretrained=True,task='subject8').to(device)

######################################################################
# Load the data
# ------------------------
csv = './data/hpe_dataset/csv/pose_regression_timeseries_subject_8/test.csv' # The path contains the samosa dataset
hpe_testdataset = SoundPoseLSTMDataset(csv,sound_length=2400,input_feature='raw',mean=np.array(get_raw_mean()).astype("float32"),std=np.array(get_raw_std()).astype("float32"))
index = 0 # Randomly select an index
sample= hpe_testdataset.__getitem__(index)

######################################################################
# Model inference
# ------------------------

# Method 1
hpe_model.eval()
predicted_result = hpe_model(sample['sound'].unsqueeze(0).float().to(device))
vis_images = make_images(sample['targets'],predicted_result.cpu().detach().squeeze(0))

#Method 2
from pysensing.acoustic.inference.predict import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predicted_result  = hpe_predict(sample['sound'],'SoundPoseLSTMDataset',hpe_model, device=device)
vis_images = make_images(sample['targets'].numpy(),predicted_result.cpu().detach().numpy().squeeze(0))

seq_num = 2
fig = plt.figure(figsize=(12, 12))
plt.imshow(vis_images[seq_num]['img'])
plt.axis('off')
plt.show()

######################################################################
# Model embedding
# ------------------------
from pysensing.acoustic.inference.embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_embedding = hpe_embedding(sample['sound'],'SoundPoseLSTMDataset',hpe_model, device=device)

######################################################################
# And that's it. We're done with our acoustic humna pose estimation tutorials. Thanks for reading.
