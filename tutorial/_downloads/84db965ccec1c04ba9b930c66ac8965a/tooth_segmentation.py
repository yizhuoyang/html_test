"""
Tutorial for tooth segmentation
===============================

"""

import os
import sys
import torch
import trimesh
import requests
from torch import nn

from pysensing.intraoral_scan.inference.utils.segmenter import Segmenter
from pysensing.intraoral_scan.inference.ts_predict import predict, visualization
from pysensing.intraoral_scan.inference.utils.ts_dataloader import extract_data_from_root

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

# Load PCT
from pysensing.intraoral_scan.models.tooth_segmentation.pct import PCT
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model = PCT().to(torch.device("cuda"))
model = nn.DataParallel(model)
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_segmentation/PCT.pth"
LOCAL_WEIGHT_PATH = "models/PCT"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True))
segmenter = Segmenter(model)

# Load DGCNN
from pysensing.intraoral_scan.models.tooth_segmentation.dgcnn import DGCNN_partseg
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model = DGCNN_partseg().to(torch.device("cuda"))
model = nn.DataParallel(model)
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_segmentation/DGCNN.pth"
LOCAL_WEIGHT_PATH = "models/DGCNN_TS"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True))
segmenter = Segmenter(model)

# Load DBGANet
from pysensing.intraoral_scan.models.tooth_segmentation.dbganet import DBGANet
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

model = DBGANet().to(torch.device("cuda"))
model = nn.DataParallel(model)
REMOTE_WEIGHT_URL = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/intraoral_scan/tooth_segmentation/DBGANet.pth"
LOCAL_WEIGHT_PATH = "models/DBGANet"
download_weights(REMOTE_WEIGHT_URL, LOCAL_WEIGHT_PATH)
model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH, weights_only=True))
segmenter = Segmenter(model)


######################################################################
# Load Dataset and Inference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

root_path = "../datasets/tooth_segmentation/data"
data = extract_data_from_root(root_path)
prediction = predict(data, segmenter=segmenter)


######################################################################
# Visualization of Tooth Segmentation Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

data_idx = 0    # Set the data_idx for visualization
raw_path = '../datasets/tooth_segmentation/example' 
jaw_dir, cases_dir = data[data_idx]['jaw_dir'], data[data_idx]['case_dir']
with open(f"{raw_path}/{jaw_dir}/{cases_dir}/{cases_dir}_{jaw_dir}.obj") as F:
    mesh = trimesh.exchange.obj.load_obj(F)

# Show original IOS
orig_label = [0 for _ in range(len(data[data_idx]['label']))]
result = visualization(mesh, orig_label)
result.show()

# Show segmented IOS
predicted_label = prediction[data_idx]
result = visualization(mesh, predicted_label)
result.show()

# Show GT IOS
gt_label = segmenter.convert_clzz_to_label(data[data_idx]['label'])
result = visualization(mesh, gt_label)
result.show()