# encoding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager
import scipy.io as sio
from os.path import dirname, join as pjoin
from sklearn import svm
from pathlib import Path
import glob
import matlab
import matlab.engine
from mpl_toolkits.mplot3d import Axes3D  # 导入3D样式库
from sklearn import metrics
from sklearn.metrics import f1_score
import math
import cmath
import seaborn as sns

path_read = []
xx, yy = np.meshgrid(np.linspace(-10, 10, 7000), np.linspace(-10, 10, 7000))
dataPath = 'venv/data/indoorA305'
X_train = np.empty(shape=[0, 183])
X_test = np.empty(shape=[0, 183])
y_true_test = np.empty(shape=[26945, 1])
csi_len = np.empty(shape=[1, 90])
csi_ang = np.empty(shape=[1, 90])


# 读取数据的算法：读取.mat数据里的三个rss值，再调用matlab里的函数计算出最终的数据值
def check_if_dir(data_path):
    temp_list = os.listdir(data_path)
    for temp_list_each in temp_list:
        if os.path.isfile(data_path + '/' + temp_list_each):
            temp_path = data_path + '/' + temp_list_each
            if os.path.splitext(temp_path)[-1] == '.mat':
                path_read.append(temp_path)
            else:
                continue
        else:
            check_if_dir(data_path + '/' + temp_list_each)


torch.set_default_tensor_type(torch.DoubleTensor)
check_if_dir(dataPath)
kk = 0
# eng = matlab.engine.start_matlab()
# path_read = list(dataPath.glob('**/*.mat'))
for item in path_read:
    mat_data = sio.loadmat(item)
    key_name = list(mat_data.keys())[-1]
    # key_name = 'csi_trace'
    # 根据key获取数据
    data = mat_data[key_name]
    total_num = data.shape[0]
    if total_num != 0:
        for i in range(total_num):
            rssia = data[i][0]['rssi_a'][0][0][0][0].item()
            rssib = data[i][0]['rssi_b'][0][0][0][0].item()
            rssic = data[i][0]['rssi_c'][0][0][0][0].item()

            csi = data[i][0]['csi'][0][0][0].flatten().reshape(1, 90)
            for j in range(90):
                csi_real = math.fabs(csi[0][j].real)
                csi_imag = math.fabs(csi[0][j].imag)
                csi_len[0][j] = math.sqrt(pow(csi_real, 2) + pow(csi_imag, 2))
                csi_ang[0][j] = cmath.phase(csi[0][j])
            agc = data[i][0]['agc'][0][0][0][0].item()
            rss = np.array([[rssia, rssib, rssic]])
            # all_data1 = np.c_[rss, csi]
            all_data2 = np.c_[rss, csi_len]
            all_data = np.c_[all_data2, csi_ang]
            length = len(item)
            if (item[length - 10] == '1' or item[length - 10] == '2') and item[length - 9] == '/' and item[length - 11] == '_':
                X_test = np.append(X_test, all_data, axis=0)
                y_true_test[kk] = 0
                kk += 1
            else:
                X_train = np.append(X_train, all_data, axis=0)
                X_test = np.append(X_test, all_data, axis=0)
                y_true_test[kk] = 1
                kk += 1
####
# 超参数
# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 16
LR = 0.01

X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
train_set = Data.TensorDataset(X_train)
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 压缩
        input_size = X_train.size(1)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )

        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_size),
            nn.Sigmoid(),  # compress to a range (0, 1) # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# 损失函数
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    total_loss = 0
    for i, x in enumerate(train_loader):
       # ans = x.shape
        ans=x[0]
        x_recon = autoencoder(ans)
        loss = loss_func(x_recon, ans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
    total_loss /= len(X_train)
    print('Epoch {}/{} : loss: {:.4f}'.format(
        epoch + 1, EPOCH, loss.item()))


def get_recon_err(X):
    return torch.mean((autoencoder(X) - X) ** 2, dim=1).detach().numpy()

ans=X_train.size()
recon_err_train = get_recon_err(X_train).reshape(23549,1)
recon_err_test = get_recon_err(X_test).reshape(26945,1)
recon_err = np.concatenate([recon_err_train, recon_err_test])
labels = np.concatenate([np.zeros((recon_err_train.shape[0],1)), y_true_test])
index = np.arange(0, len(labels))

sns.kdeplot(recon_err[labels == 0], shade=True)
#sns.kdeplot(recon_err[labels == 1], shade=True)
plt.show()
