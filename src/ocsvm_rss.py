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
from sklearn import metrics
from sklearn.metrics import f1_score

train_path = 'venv/data/APrssdata/AP_testrssdata.mat'
test_path = 'venv/data/APrssdata/rssdata.mat'
mat_data = sio.loadmat(train_path)
test_mat_data = sio.loadmat(test_path)

test_data = test_mat_data['rssdata']['rss'][0][0]
X_test = np.transpose(test_data)

train_data = mat_data['rsstraindata']
X_train = np.transpose(train_data)

y_true = test_mat_data['rssdata']['la'][0][0][0]
y_true_test = np.empty(shape=[X_test.shape[0],1])

ans = X_test.shape[0]
for i in range(ans):
    y_true_test[i]=y_true[i]


# osvm 算法
def oneClassSVM(Xtrain, Xtest):
    xx, yy = np.meshgrid(np.linspace(-50, 50, 10000), np.linspace(-50, 50, 10000))
    svm_detector = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
    svm_detector.fit(Xtrain)

    # 判断数据是在超平面内还是超平面外，返回+1或-1，正号是超平面内，负号是在超平面外
    y_pred_train = svm_detector.predict(Xtrain)
    y_pred_test = svm_detector.predict(Xtest)

    toal_test_data = np.hstack((Xtest, y_pred_test.reshape(Xtest.shape[0], 1)))  # 将测试集和检测结果合并
    y_pred= y_pred_test.reshape(Xtest.shape[0], 1)
    normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # 获得异常检测结果中集
    outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # 获得异常检测结果异常数据
    n_test_outliers = outlier_test_data.shape[0]  # 获得异常的结果数量
    total_count_test = toal_test_data.shape[0]  # 获得测试集样本量

    ans1=y_true_test.dtype
    ans2 = y_pred.dtype
    print(y_pred_test)
    print(y_true_test)
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

print(np.size(X_train, 0), ",", np.size(X_test, 0))
oneClassSVM(X_train, X_test)

   #  print(n_error_train, ",", n_error_test)
   #  # 计算网格数据到超平面的距离，含正负号
   #  Z = svm_detector.decision_function(np.c_[xx.ravel(), yy.ravel()])  # ravel表示数组拉直
   #  Z = Z.reshape(xx.shape)
   #  """
   #  绘图
   #  """
   #  plt.title("Novelty Detection")
   #  plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)  # 绘制异常区域的轮廓， 把异常区域划分为7个层次
   #  a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')  # 绘制轮廓，SVM的边界点（到边界距离为0的点
   #  plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')  # 绘制正常样本的区域，使用带有填充的轮廓
   #
   #  s = 40  # 样本点的尺寸大小
   #  b1 = plt.scatter(X_train[:, 0], c='white', s=s, edgecolors='k')  # 绘制训练样本，填充白色，边缘”k“色
   #  b2 = plt.scatter(X_test[:, 0], c='blueviolet', s=s, edgecolors='k')  # 绘制测试样本--正常样本，填充蓝色，边缘”k“色
   # # c = plt.scatter(X_outliers[:, 0], , c='gold', s=s, edgecolors='k')  # 绘制测试样本--异常样本，填充金色，边缘”k“色
   #
   #  plt.axis('tight')
   #  plt.xlim((-5, 5))
   #  plt.ylim((-5, 5))
   #
   #  # 集中添加图注
   #  plt.legend([a.collections[0], b1, b2],
   #             ["learned frontier", "training data",
   #              "test regular data", "test abnormal data"],
   #             loc="upper left",
   #             prop=matplotlib.font_manager.FontProperties(size=11))
   #  plt.xlabel("error train: %d/200 ;   errors novel regular: %d/40 ;   errors novel abnormal: %d/40"
   #             % (n_error_train, n_error_test, n_error_outliers))
   #  plt.show()



