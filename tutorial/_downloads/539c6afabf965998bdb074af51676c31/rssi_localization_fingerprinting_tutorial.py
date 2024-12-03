"""
RSSI Localization Fingerprinting Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing codes for rssi localization fingerprinting
#
import os
import torch
from torch.optim import Adam, SGD
from pysensing.rssi.datasets.ntu_iot_rssi import get_dataloader_ntuiotrssi
from pysensing.rssi.datasets.uji_indoor_loc import get_dataloader_ujiindoorloc
from pysensing.rssi.datasets.download import download_dataset
from pysensing.rssi.inference.embedding import rssi_embedding
from pysensing.rssi.models.localization.load_model import create_model
from pysensing.rssi.models.localization.fingerprinting import MLP, CNN, WKNN, LSTM
from pysensing.rssi.inference.train import train_model
import warnings

warnings.filterwarnings('ignore')

######################################################################
# Download the NTU IoT Lab RSSI Data
# ------------------------

root_data = "./"
download_dataset("NTUIoTRSSI", root_data)
dir_data = os.path.join(root_data, "NTUIoTRSSI")

path_train = os.path.join(dir_data, "data_train.txt")
loader_train = get_dataloader_ntuiotrssi(path_train, batch_size=32, is_train=True)
n_samples_train = len(loader_train.dataset)
path_test = os.path.join(dir_data, "data_test.txt")
loader_test = get_dataloader_ntuiotrssi(path_test, batch_size=32, is_train=False)
n_samples_test = len(loader_test.dataset)

######################################################################
# Non-Trainable Method: Weighted K-Nearest Neighbors
# ------------------------

dataset_train = loader_train.dataset
dataset_train.get_compact()
dataset_test = loader_test.dataset
sample_test_rss, sample_test_pos = dataset_test.__getitem__(0)

pos_train, rss_train = dataset_train.compact_set[:, :2], dataset_train.compact_set[:, 2:]
wknn_estimator = WKNN(pos_train, rss_train)
wknn_est_pos = wknn_estimator(sample_test_rss, K=5)
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), wknn_est_pos))

######################################################################
# Trainable Method: MLP
# ------------------------

mlp_estimator = train_model(model_name="MLP", dataset_name="NTUIoTRSSI", regression=True, optimizer="SGD", epochs=50, batch_size=32, lr=1e-3, dir_save=None)

mlp_estimator = mlp_estimator.cpu()
mlp_estimator.eval()
mlp_est_pos = mlp_estimator(sample_test_rss)
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), mlp_est_pos.cpu().detach().numpy()))

######################################################################
# Trainable Method: CNN (1D)
# ------------------------

cnn_estimator = train_model(model_name="CNN", dataset_name="NTUIoTRSSI", regression=True, optimizer="SGD", epochs=50, batch_size=32, lr=1e-3, dir_save=None)

cnn_estimator = cnn_estimator.cpu()
cnn_estimator.eval()
cnn_est_pos = cnn_estimator(sample_test_rss.unsqueeze(0))
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), cnn_est_pos.cpu().detach().numpy()))

######################################################################
# Trainable Method: LSTM
# ------------------------

lstm_estimator = train_model(model_name="LSTM", dataset_name="NTUIoTRSSI", regression=True, optimizer="SGD", epochs=50, batch_size=32, lr=1e-3, dir_save=None)

lstm_estimator = lstm_estimator.cpu()
lstm_estimator.eval()
lstm_est_pos = lstm_estimator(sample_test_rss.unsqueeze(0))
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), lstm_est_pos.cpu().detach().numpy()))


######################################################################
# Train CNN models for the UJI Indoor Loc Dataset
# ------------------------

######################################################################
# Load the UJI Indoor Loc Dataset
# ------------------------

root_data = "./"
download_dataset("UJIIndoorLoc", root_data)
dir_data = os.path.join(root_data, "UJIIndoorLoc")

uji_train_loader = get_dataloader_ujiindoorloc(os.path.join(dir_data, "trainingData.csv"), ['longitude', 'latitude', 'floor', 'buildingid'], 32, True)
n_samples_train = len(uji_train_loader.dataset)

uji_test_loader = get_dataloader_ujiindoorloc(os.path.join(dir_data, "validationData.csv"), ['longitude', 'latitude', 'floor', 'buildingid'], 1111, False)
n_samples_test = len(uji_test_loader.dataset)
dataset_test = uji_test_loader.dataset
test_rssi, test_coord, test_area = next(iter(uji_test_loader))

######################################################################
# Train the Area (Building ID + Floor ID) classification model
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_area = torch.nn.CrossEntropyLoss().to(device)

cnn_area = CNN(dim_in=520, dim_out=15, dim_embed=64, channels=[16, 64]).to(device)
optimizer_area = Adam(cnn_area.parameters(), lr=1e-3)

train_epochs = 10
for epoch in range(train_epochs):
    total = 0
    correct = 0
    epoch_loss_area = 0

    for batch_id, batch_data in enumerate(uji_train_loader):
        rssi, coord, area = batch_data
        # print(rssi)
        total += rssi.size(0)
        if not torch.isfinite(rssi).all():
            raise ValueError("Invalid value encountered")
        rssi = rssi.to(device)
        coord = coord.to(device)
        area = area.to(device)

        area_logits_pred = cnn_area(rssi)
        loss_area = criterion_area(area_logits_pred, area)

        optimizer_area.zero_grad()
        loss_area.backward()
        optimizer_area.step()

        area_label_pred = torch.argmax(area_logits_pred, dim=1)
        correct += area_label_pred.eq(area).sum().item()
        epoch_loss_area += loss_area.item()

    epoch_acc = 100. * (correct / total)
    epoch_loss_area /= (batch_id + 1)
    info = 'Epoch {}/{}: Train Accuracy = {}, Train Loss = {:.5f}'.format(epoch + 1,
                                                                          train_epochs,
                                                                          epoch_acc,
                                                                          epoch_loss_area)
    # print(info)

cnn_area = cnn_area.cpu()
cnn_area.eval()
test_area_logits_pred = cnn_area(test_rssi)
test_area_label_pred = torch.argmax(test_area_logits_pred, dim=1)
test_correct = test_area_label_pred.eq(test_area).sum().item()
test_total = test_rssi.size(0)
test_accuracy = test_correct / test_total

print("The testing accuracy of Area identification is: {}".format(test_accuracy))

######################################################################
# Train the Location (Coordinates) regression model
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_coord = torch.nn.MSELoss().to(device)

cnn_coord = CNN(dim_in=520, dim_out=2, dim_embed=64, channels=[16, 64]).to(device)
optimizer_coord = Adam(cnn_coord.parameters(), lr=5e-4)

train_epochs = 10
for epoch in range(train_epochs):
    epoch_loss_coord = 0

    for batch_id, batch_data in enumerate(uji_train_loader):
        rssi, coord, area = batch_data
        # print(rssi)
        if not torch.isfinite(rssi).all():
            raise ValueError("Invalid value encountered")
        rssi = rssi.to(device)
        coord = coord.to(device)
        area = area.to(device)

        coord_pred = cnn_coord(rssi)
        loss_coord = criterion_coord(coord_pred, coord)

        optimizer_coord.zero_grad()
        loss_coord.backward()
        optimizer_coord.step()

        epoch_loss_coord += loss_coord.item()

    epoch_loss_coord /= (batch_id + 1)

    info = 'Epoch {}/{}: Train Loss (Localization Error) = {:.5f}'.format(epoch + 1,
                                                                          train_epochs,
                                                                          epoch_loss_coord)
    # print(info)

cnn_coord = cnn_coord.cpu()
cnn_coord.eval()
test_coord_pred = cnn_coord(test_rssi)
test_accuracy = criterion_coord(test_coord_pred, test_coord)

print("The testing accuracy of coordinate localization is: {}".format(test_accuracy))

######################################################################
# Extract embeddings from pre-trained localization model (Instance: CNN for UJIIndoorLoc)
# ------------------------

uji_test_loader = get_dataloader_ujiindoorloc(os.path.join(dir_data, "validationData.csv"), ['longitude', 'latitude', 'floor', 'buildingid'], 1111, False)
n_samples_test = len(uji_test_loader.dataset)
dataset_test = uji_test_loader.dataset
test_rssi, test_coord, test_area = next(iter(uji_test_loader))

from pysensing.rssi.models.localization.load_model import load_pretrain

device = torch.device("cpu")

model = load_pretrain(model_name="CNN", dataset_name="UJIIndoorLoc", regression=False, path_model=None, device="cpu")
emb_area = rssi_embedding(rssi=test_rssi, dataset="UJIIndoorLoc", model=model, device=device)


