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
dataPath = 'venv/data/indoorA305'
# ae csi train data
X_ae_train = np.empty(shape=[0, 180])
X_ae_test = np.empty(shape=[0, 180])

# osvm rss+csi train data
X_all_train = np.empty(shape=[0,5])
X_all_train=np.empty(shape=[0,5])

X_rss_test= np.empty(shape=[0,3])
X_rss_train= np.empty(shape=[0,3])

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
            rss = np.array([[rssia, rssib, rssic]])
            csi = data[i][0]['csi'][0][0][0].flatten().reshape(1, 90)
            for j in range(90):
                csi_real = math.fabs(csi[0][j].real)
                csi_imag = math.fabs(csi[0][j].imag)
                csi_len[0][j] = math.sqrt(pow(csi_real, 2) + pow(csi_imag, 2))
                csi_ang[0][j] = cmath.phase(csi[0][j])
            # csi_data1 = np.c_[rss, csi]
            csi_data = np.c_[csi_len, csi_ang]
            length = len(item)
            if (item[length - 10] == '1' or item[length - 10] == '2') and item[length - 9] == '/' and item[length-11] == '_':
                X_ae_test = np.append(X_ae_test, csi_data, axis=0)
                X_rss_test = np.append(X_rss_test,rss,axis=0)
                y_true_test[kk] = -1
                kk += 1
            else:
                X_ae_train = np.append(X_ae_train, csi_data, axis=0)
                X_ae_test = np.append(X_ae_test, csi_data, axis=0)
                X_rss_test = np.append(X_rss_test, rss, axis=0)
                X_rss_train =np.append(X_rss_train, rss, axis=0)
                y_true_test[kk] = 1
                kk += 1
####
# 超参数
# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 16
LR = 0.01

X_ae_train, X_ae_test = torch.from_numpy(X_ae_train), torch.from_numpy(X_ae_test)
train_set = Data.TensorDataset(X_ae_train)
train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 压缩
        input_size = X_ae_train.size(1)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_size),
            nn.Sigmoid(),  # compress to a range (0, 1) # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
encoder_csi,_ = autoencoder(X_ae_train)
encoder_csi_test,_ = autoencoder(X_ae_test)
all_train=encoder_csi.detach().numpy()
all_test = encoder_csi_test.detach().numpy()
X_all_train=np.c_[X_rss_train,all_train]
X_all_test = np.c_[X_rss_test,all_test]


# osvm 算法
def oneClassSVM(Xtrain, Xtest):
    svm_detector = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma='auto')
    svm_detector.fit(Xtrain)

    # 判断数据是在超平面内还是超平面外，返回+1或-1，正号是超平面内，负号是在超平面外
    y_pred_train = svm_detector.predict(Xtrain)
    y_pred_test = svm_detector.predict(Xtest)
    toal_test_data = np.hstack((Xtest, y_pred_test.reshape(Xtest.shape[0], 1))) # 将测试集和检测结果合并
    y_pred=y_pred_test.reshape(Xtest.shape[0], 1)
    y_true=y_true_test
    ans = y_pred.shape
    ans2=y_true_test.shape
    normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # 获得异常检测结果中集
    outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # 获得异常检测结果异常数据
    n_test_outliers = outlier_test_data.shape[0]  # 获得异常的结果数量
    total_count_test = toal_test_data.shape[0]  # 获得测试集样本量

    print('outliers: {0}/{1}'.format(n_test_outliers, total_count_test))  # 输出异常的结果数量
    print('{:*^60}'.format(' all result data (limit 5) '))  # 打印标题
    print(toal_test_data[:5])  # 打印输出前5条合并后的数据集
    print("binary:")
    print(f1_score(y_true_test, y_pred, average='binary'))
    print("None:")
    print(f1_score(y_true_test, y_pred, average=None))
    print("micro:")
    print(f1_score(y_true_test, y_pred, average='micro'))
    print("macro:")
    print(f1_score(y_true_test, y_pred, average='macro'))

    # ======================= metrics ============================
    precision, recall, threshold = metrics.precision_recall_curve(y_true_test, y_pred)
    print(recall)
    print(precision)
    print(threshold)

    pr_auc = metrics.auc(recall, precision)  # 梯形块分割，建议使用
    pr_auc0 = metrics.average_precision_score(y_true_test, y_pred)  # 小矩形块分割

    print(pr_auc)
    print(pr_auc0)

    # ======================= PLoting =============================
    plt.figure(1)
    plt.plot(recall, precision, label=f"PR_AUC = {pr_auc:.2f}\nAP = {pr_auc0:.2f}",
             linewidth=2, linestyle='-', color='r', marker='o')
    plt.fill_between(recall, y1=precision, y2=0, step=None, alpha=0.2, color='b')
    plt.title("PR-Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.show()
    # 统计预测错误的个数
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size

    print(n_error_train, ",", n_error_test)

print(np.size(X_all_train, 0), ",", np.size(X_all_test, 0))
oneClassSVM(X_all_train, X_all_test)