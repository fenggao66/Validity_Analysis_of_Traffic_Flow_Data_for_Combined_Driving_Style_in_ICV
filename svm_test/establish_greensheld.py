from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

list_k_v = []

for i in range(1, 500, 20):  # 速度30-40
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_57\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=7, contamination=0.25, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 420
    list_k_v.append([mes, k])

for i in range(6400, 6800, 20):  # 速度30-40
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_57\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=7, contamination=0.25, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 420
    list_k_v.append([mes, k])

for i in range(4025, 4400, 20):  # 4025 - 5290           速度 25-27
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_11\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=15, contamination=0.2, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 450
    list_k_v.append([mes, k])

for i in range(4900, 5290, 20):  # 4025 - 5290          速度23-25
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_11\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=15, contamination=0.2, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 450
    list_k_v.append([mes, k])

for i in range(8458, 8901, 40):  # 8458-8901   速度21-23
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_11\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=13, contamination=0.25, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 450
    list_k_v.append([mes, k])

for i in range(8190, 8959, 30):  # 8190-8959      速度11-15
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=13, contamination=0.25, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 420
    list_k_v.append([mes, k])

for i in range(5, 4444, 50):  # 速度5-20
    data = pd.read_csv("C:\\Users\\gjh\\svm_密度821\\data_46\\" + str(i) + ".csv")
    car_num = len(data)
    X = data[['xVelocity', 'yVelocity']].values.reshape(car_num, 2)  # (examples_num,1)整理数据
    model1 = LOF(n_neighbors=13, contamination=0.4, novelty=True)  # 定义一个LOF模型，异常比例是10%
    model1.fit(X)
    y = model1._predict(X)
    X_train = []

    # 去除异常点
    for j in range(len(y)):
        if y[j] == 1:
            X_train.append(X[j].tolist())  # 转化为list格式    np----->list
    X_train = np.array(X_train)  # 格式转换          list----->np

    # 计算均值
    sum = 0
    mes = 0
    for j in range(len(X_train)):
        sum += X_train[j][0]
    mes = abs(sum / len(X_train))

    # 计算密度
    k = car_num / 400
    list_k_v.append([mes, k])


# coding=utf-8
import pylab
import numpy as np
x = np.array(list_k_v)[:,1]
y = np.array(list_k_v)[:,0]
z1 = np.polyfit(x, y, 2)              # 曲线拟合，返回值为多项式的各项系数
p1 = np.poly1d(z1)                    # 返回值为多项式的表达式，也就是函数式子
print(p1)
y_pred = p1(x)                        # 根据函数的多项式表达式，求解 y
# print(np.polyval(p1, 29))            #  根据多项式求解特定 x 对应的 y 值
# print(np.polyval(z1, 29))             # 根据多项式求解特定 x 对应的 y 值

plot1 = pylab.plot(x, y, '*', label='original values',color='0.2')
plot2 = pylab.plot(x, y_pred, 'r', label='fit values',color='0.2')
pylab.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
pylab.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# pylab.title('密度-速度关系图')
pylab.xlabel('密度')
pylab.ylabel('速度')
pylab.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0.65, 0.8))
pylab.axis([0,0.2,0,50])
pylab.savefig('k-v.png', dpi=600, bbox_inches='tight')
# plt.savefig('density.svg')
pylab.show()


# 寻找最大流量对应的速度和密度
q = 0
t = 0
list_q = []               #流量列表
for i in range(len(list_k_v)):
    q_real = list_k_v[i][0]*list_k_v[i][1]
    list_q.append(q_real)
    if q_real>q:
        q = q_real
        t = list_k_v[i]