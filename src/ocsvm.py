# encoding:utf-8
import numpy as np
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

path_read = []
xx, yy = np.meshgrid(np.linspace(-10, 10, 7000), np.linspace(-10, 10, 7000))
dataPath = 'venv/data/indoorA305'
X_train = np.empty(shape=[0, 3])
X_test = np.empty(shape=[0, 3])
y_true_test = np.empty(shape=[26945,1])

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


check_if_dir(dataPath)
kk=0
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

            csi = data[i][0]['csi'][0][0][0].flatten()
            agc = data[i][0]['agc'][0][0][0][0].item()
            rss = np.array([[rssia, rssib, rssic]])
            length = len(item)
            if (item[length-10] == '1' or item[length-10] == '2') and item[length-9] == '/' and item[length-11] == '_':
                X_test = np.append(X_test, rss, axis=0)
                y_true_test[kk]= -1
                kk+=1
           # elif '6' <= item[length - 10] <= '9' and ('4' <= item[length - 12] <= '8'): #or item[length - 12] == 0):
            else:
                X_train = np.append(X_train, rss, axis=0)
                X_test = np.append(X_test, rss, axis=0)
                y_true_test[kk] = 1
                kk += 1
            # else:
            #     X_train = np.append(X_train, rss, axis=0)
            #     X_test = np.append(X_test, rss, axis=0)
            #     y_true_test[kk] = 1
            #     kk+=1

print(X_train)
print(X_test.shape[0])


# 测试集用1-14的数据，训练集用3-14的数据


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

    # plt.style.use('ggplot')  # 使用ggplot样式库
    # fig = plt.figure()  # 创建画布对象
    # ax = Axes3D(fig)  # 将画布转换为3D类型
    # s1 = ax.scatter(normal_test_data[:, 0], normal_test_data[:, 1], normal_test_data[:, 2], s=100, edgecolors='k',
    #                 c='g',
    #                 marker='o')  # 画出正常样本点
    # s2 = ax.scatter(outlier_test_data[:, 0], outlier_test_data[:, 1], outlier_test_data[:, 2], s=100, edgecolors='k',
    #                 c='r',
    #                 marker='o')  # 画出异常样本点
    # s3 = ax.scatter(Xtrain[:, 0], Xtrain[:, 1], Xtrain[:, 2], s=100, edgecolors='k', c='w', marker='o')
    # ax.w_xaxis.set_ticklabels([])  # 隐藏x轴标签，只保留刻度线
    # ax.w_yaxis.set_ticklabels([])  # 隐藏y轴标签，只保留刻度线
    # ax.w_zaxis.set_ticklabels([])  # 隐藏z轴标签，只保留刻度线
    # ax.legend([s1, s2, s3], ['normal points', 'outliers', 'train points'], loc=0)  # 设置两类样本点的图例
    # plt.title('novelty detection')  # 设置图像标题
    # plt.show()


print(np.size(X_train, 0), ",", np.size(X_test, 0))
oneClassSVM(X_train, X_test)