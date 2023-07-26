'''This is adapted from GitHub repo https://github.com/manigalati/usad.,
original paper was published at KDD 2020 at https://dl.acm.org/doi/10.1145/3394486.3403392, 
titled "USAD: UnSupervised Anomaly Detection on Multivariate Time Series".
Please also check the authors' original paper and implementation for reference.'''

# Set current working directory to the main branch of RLMSAD
import sys
sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
import sklearn
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score ,train_test_split

def data_process_SWaT():
    # 1.读取训练数据集
    data = pd.read_csv(r"C:\Users\GJH\Desktop\RLMSAD-master\svm_test\fix_data\sumfix2.csv")  # 目标车辆
    # data_row = pd.read_csv(r"C:\Users\gjh\svm_密度821\id_821\821_添加密度fix - 未添加异常.csv")     # 目标车辆
    # data = pd.read_csv(r"C:\Users\GJH\Desktop\RLMSAD-master\svm_test\fix_data\huizong.csv")  # 目标车辆
    examples_num = data.shape[0]  # 样本数

    x1 = 'xVelocity'
    x2 = 'xAcceleration'  #
    x3 = 'dhw'  #
    x4 = 'density_80_pinghua'  #
    x5 = 'pre_finaly'  #
    x6 = 'E'

    X = data[[x1, x2, x3, x4, x5, x6]].values.reshape(examples_num, 6)  # (examples_num,4)整理数据
    # X = data[[x1,x2,x3]].values.reshape(examples_num,3)  # (examples_num,4)整理数据
    y = data[['label']].values.reshape(examples_num, )  # (examples_num,1)整理数据

    # 加噪声 # 数值随便指定，指定了之后对应的数值唯一
    np.random.seed(1)
    noise1 = np.random.normal(loc=0, scale=0.2, size=len(X))  #     实验1  0.2和0.2   实验2   0.3和0.6
    noise2 = np.random.normal(loc=0, scale=0.5, size=len(X))  #
    for i in range(len(X)):
        X[i][0] += noise1[i]                   # x方向 速度加噪
        X[i][2] += noise2[i]                   # x方向 车距加噪

    # 2.标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    random.seed(3)
    train_size = 0.7  # 训练集的比例
    train_indices = random.sample(range(len(X)), int(train_size * len(X)))
    train_X = [X[i] for i in train_indices]
    train_label = [y[i] for i in train_indices]

    # 将剩余数据作为测试集
    test_X = [X[i] for i in range(len(X)) if i not in train_indices]
    test_label = [y[i] for i in range(len(y)) if i not in train_indices]

    train_X, test_X, train_label, test_label = sklearn.model_selection.train_test_split(X, y, random_state=1,test_size=0.3)
    #计算异常比例
    all = y.sum()
    T = all/len(y)

    return train_X, test_X, train_label, test_label, T