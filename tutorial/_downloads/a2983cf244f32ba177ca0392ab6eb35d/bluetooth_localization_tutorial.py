"""
Bluetooth Localization Tutorial
===============================
"""

######################################################################

# !pip install pysensing

######################################################################
# This is the tutoral for Bluetooth RSSI Based Localization using Fingerprinting Methods
# 
import torch
from torch.optim import Adam, SGD
from pysensing.bluetooth.datasets.wmu_ble_loc import get_dataloader_wmubleloc
from pysensing.bluetooth.datasets.amazonas_indoor_env import get_dataloader_amazonasindoorenv
from pysensing.bluetooth.models.localization.fingerprinting import MLP, CNN, WKNN, LSTM
from pysensing.acoustic.datasets.get_dataloader import download_and_extract
import warnings

warnings.filterwarnings('ignore')
######################################################################
# Data download links
# -----------------------------------
dataset_url = {
'WMUBLELoc': 'https://pysensing.oss-ap-southeast-1.aliyuncs.com/data/ble/WMUBLELoc.zip',
'Amazonas':'https://pysensing.oss-ap-southeast-1.aliyuncs.com/data/ble/AmazonasIndoorEnv.zip'
}

######################################################################
# Load the WMU BLE Localization Data
# ------------------------

data_dir = './WMUBLELoc'
# download the dataset if the dataset have not been download
download_and_extract(dataset_url['WMUBLELoc'],data_dir)
wmu_path = "./WMUBLELoc/iBeacon_RSSI_Labeled.csv"
loader_train = get_dataloader_wmubleloc(wmu_path, batch_size=32, is_train=True, train_seed=0)
n_samples_train = len(loader_train.dataset)
loader_test = get_dataloader_wmubleloc(wmu_path, batch_size=32, is_train=False, train_seed=0)
n_samples_test = len(loader_test.dataset)

######################################################################
# Non-Trainable Method: Weighted K-Nearest Neighbors
# ------------------------

dataset_train = loader_train.dataset
dataset_test = loader_test.dataset

train_pos = dataset_train.ble_sets.to_numpy()[:, :2]
train_ble = dataset_train.ble_sets.to_numpy()[:, -14:]

sample_test_ble, sample_test_pos = dataset_test.__getitem__(0)
sample_test_ble = sample_test_ble.detach().cpu().numpy()

wknn_estimator = WKNN(train_pos, train_ble)
wknn_est_pos = wknn_estimator(sample_test_ble, K=5)
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), wknn_est_pos))
######################################################################
# Trainable Method: MLP
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10
criterion = torch.nn.MSELoss()

mlp_estimator = MLP(dim_in=14, dim_hidden=[64, 32, 16], dim_out=2).to(device)
optimizer = Adam(mlp_estimator.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    for batch_id, (x, y) in enumerate(loader_train):
        x, y = x.to(device), y.to(device)
        y_pred = mlp_estimator(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

mlp_estimator.eval()
mlp_est_pos = mlp_estimator(torch.tensor(sample_test_ble).to(device))
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), mlp_est_pos.cpu().detach().numpy()))

######################################################################
# Trainable Method: CNN (1D)
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10
criterion = torch.nn.MSELoss()

cnn_estimator = CNN(dim_in=14, dim_out=2, dim_embed=64, channels=[16, 32]).to(device)
optimizer = Adam(cnn_estimator.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    for batch_id, (x, y) in enumerate(loader_train):
        if not torch.isfinite(x).all():
            raise ValueError("Invalid value encountered")
        x, y = x.to(device), y.to(device)
        y_pred = cnn_estimator(x)
        if not torch.isfinite(y_pred).all():
            print("Invalid value computed")
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

cnn_estimator.eval()
cnn_est_pos = cnn_estimator(torch.tensor(sample_test_ble).unsqueeze(0).to(device))
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), cnn_est_pos.cpu().detach().numpy()))

######################################################################
# Trainable Method: LSTM
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 10
criterion = torch.nn.MSELoss()

lstm_estimator = LSTM(dim_in=14, dim_embed=64, dim_lstm=32, dim_out=2).to(device)
optimizer = SGD(lstm_estimator.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    for batch_id, (x, y) in enumerate(loader_train):
        if not torch.isfinite(x).all():
            raise ValueError("Invalid value encountered")
        x, y = x.to(device), y.to(device)
        y_pred = lstm_estimator(x)
        if not torch.isfinite(y_pred).all():
            print("Invalid value computed")
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

lstm_estimator.eval()
lstm_est_pos = lstm_estimator(torch.tensor(sample_test_ble).unsqueeze(0).to(device))
print("Ground Truth Position: {}. Estimated Position: {}".format(sample_test_pos.cpu().numpy(), lstm_est_pos.cpu().detach().numpy()))

######################################################################
# Train CNN models for the Amazonas Indoor Environment Dataset
# ------------------------

######################################################################
# Load the Amazonas Indoor Environment Dataset
# ------------------------
data_dir = './Amazonas'
# download the dataset if the dataset have not been download
download_and_extract(dataset_url['Amazonas'],data_dir)
amazonas_path = './Amazonas/AmazonasIndoorEnv'
amazonas_train_loader = get_dataloader_amazonasindoorenv(amazonas_path, batch_size=32, is_train=True, receiver='ALL', train_seed=0)
n_samples_train = len(amazonas_train_loader.dataset)

amazonas_test_loader = get_dataloader_amazonasindoorenv(amazonas_path, batch_size=32, is_train=False, receiver='ALL', train_seed=0)
n_samples_test = len(amazonas_test_loader.dataset)
amazonas_dataset_test = amazonas_test_loader.dataset
test_ble, test_pos = next(iter(amazonas_test_loader))

######################################################################
# Train the Location (Coordinates) regression model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_pos = torch.nn.MSELoss().to(device)

cnn_pos = CNN(dim_in=15, dim_out=2, dim_embed=64, channels=[16, 64]).to(device)
optimizer_pos = Adam(cnn_pos.parameters(), lr=5e-4)

train_epochs = 10
for epoch in range(train_epochs):
    epoch_loss_pos = 0

    for batch_id, batch_data in enumerate(amazonas_train_loader):
        ble, pos = batch_data
        if not torch.isfinite(ble).all():
            raise ValueError("Invalid value encountered")
        ble = ble.to(device)
        pos = pos.to(device)

        pos_pred = cnn_pos(ble)
        loss_pos = criterion_pos(pos_pred, pos)

        optimizer_pos.zero_grad()
        loss_pos.backward()
        optimizer_pos.step()

        epoch_loss_pos += loss_pos.item()

    epoch_loss_pos /= (batch_id + 1)

    info = 'Epoch {}/{}: Train Loss (Localization Error) = {:.5f}'.format(epoch + 1,
                                                                          train_epochs,
                                                                          epoch_loss_pos)
    # print(info)

cnn_pos.eval()
test_pos_pred = cnn_pos(test_ble.to(device))
test_accuracy = criterion_pos(test_pos_pred, test_pos.to(device))

print("The testing accuracy of coordinate localization is: {}".format(test_accuracy))


######################################################################
# Extract embeddings from pre-trained localization model (Instance: CNN for UJIIndoorLoc)
# ------------------------
uji_test_loader = get_dataloader_amazonasindoorenv(amazonas_path, batch_size=4765, is_train=False, receiver='ALL', train_seed=0)
n_samples_test = len(uji_test_loader.dataset)
dataset_test = uji_test_loader.dataset
test_ble, test_pos = next(iter(uji_test_loader))

from pysensing.bluetooth.models.localization.load_model import load_pretrain

device = torch.device("cpu")
model = CNN(dim_in=15, dim_out=2, dim_embed=64, channels=[16, 64]).to(device)

url_pretrain = "https://pysensing.oss-ap-southeast-1.aliyuncs.com/pretrain/ble/localization/amazonas_coord_cnn.pth"
model = load_pretrain(model, url_pretrain, device)

emb_area = model.generate_embeddings(test_ble)


######################################################################
# And that's it. We're done with our bluetooth localization tutorial tutorials. Thanks for reading.
