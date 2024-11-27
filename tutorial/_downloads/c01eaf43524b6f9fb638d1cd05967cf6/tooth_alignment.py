"""
Tutorial for tooth alignment
============================

"""

import os
import sys
import torch
import requests

from pysensing.intraoral_scan.preprocessing.ta_utils import *
from pysensing.intraoral_scan.inference.utils import ta_dataloader
from pysensing.intraoral_scan.inference.ta_predict import predict

def download_weights(remote_url, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading weights from {remote_url}...")
        response = requests.get(remote_url, stream=True)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print("Weights already exist. Skipping download.")


######################################################################
# Load Model (picking one from following three models)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# Load CurveNet
from pysensing.intraoral_scan.models.tooth_alignment.curvenet import CurveNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = CurveNet()
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_alignment/CurveNet.pth"
LOCAL_WEIGHT_PATH = "models/CurveNet"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True, map_location="cuda"))

# Load DGCNN
from pysensing.intraoral_scan.models.tooth_alignment.dgcnn import DGCNN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = DGCNN()
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_alignment/DGCNN.pth"
LOCAL_WEIGHT_PATH = "models/DGCNN_TA"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True, map_location="cuda"))

# Load TANet
from pysensing.intraoral_scan.models.tooth_alignment.tanet import TANet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = TANet()
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_alignment/TANet.pth"
LOCAL_WEIGHT_PATH = "models/TANet"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True, map_location="cuda"))


######################################################################
# Load Dataset
# ~~~~~~~~~~~~
# 

dataset_path = "../datasets/tooth_alignment/example/data"
batch_size = 3

data_loader = ta_dataloader.DataLoader(dataset_path, batch_size)
print(len(data_loader))


######################################################################
# Model Inference
# ~~~~~~~~~~~~~~~
# 

# Predict the results
num_point_tooth = 400   # the number of points for each tooth pointcloud
preds = predict(data_loader, num_point_tooth, model, "cuda")

# transform the prediction to pose
root = "../datasets/tooth_alignment/example"
ans_pose = trans_pred(data_loader, preds, root)


######################################################################
# Visualization of Tooth Alignment Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

data_idx = 0    # Set the data_idx for visualization
(_, dirs) = get_idx_dirs(root)
objs = getTooth(data_idx, root, dirs)

# Show original dentition
pose = getAxis(f'{root}/{dirs[data_idx]}/TeethAxis_Ori.txt', keep_fdi=True)
showTooth(objs, pose).show()

# Show aligned dentition
showTooth(objs, ans_pose[data_idx]).show()

# Show GT dentition
pose = getAxis(f'{root}/{dirs[data_idx]}/TeethAxis_T2.txt', keep_fdi=True)
showTooth(objs, pose).show()