import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
import random
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score ,train_test_split
from data_process import *
import joblib

# 1.读取训练数据集
data = pd.read_csv(r"C:\Users\GJH\Desktop\RLMSAD-master\svm_test\fix_data\sumfix2.csv")     # 目标车辆
# data = pd.read_csv(r"C:\Users\GJH\Desktop\RLMSAD-master\svm_test\fix_data\huizong.csv")     # 目标车辆
examples_num = data.shape[0]  # 样本数

x1 = 'xVelocity'
x2 = 'xAcceleration'        #
x3 = 'dhw'                  #
x4 = 'density_80_pinghua'   #
x5 = 'pre_finaly'           #
x6 = 'E'

# X = data[[x1,x2,x3,x4,x5,x6]].values.reshape(examples_num,6)  # (examples_num,4)整理数据
X = data[[x1,x2,x3]].values.reshape(examples_num,3)  # (examples_num,4)整理数据
y = data[['label']].values.reshape(examples_num,)  # (examples_num,1)整理数据

# 加噪声 # 数值随便指定，指定了之后对应的数值唯一
np.random.seed(1)
noise1 = np.random.normal(loc=0, scale=0.5, size=len(X))     # 实验二（0，0.3）
noise2 = np.random.normal(loc=0, scale=0.03, size=len(X))
noise3 = np.random.normal(loc=0, scale=0.9, size=len(X))     # 验二（0，0.6）

for i in range(len(X)):
    X[i][0] += noise1[i]                   # x方向 速度加噪
    # X[i][1] += noise2[i]                 # x方向 加速度速度加噪
    X[i][2] += noise3[i]                   # x方向 车距速度加噪

# plt.figure(figsize=(16,8))
# plt.subplot(421)
# plt.plot(data['xVelocity'])
# plt.subplot(423)
# plt.plot(data['xAcceleration'])
# plt.subplot(425)
# plt.plot(data['dhw'])
# plt.subplot(427)
# plt.plot(noise1)
# plt.show()
#
# plt.figure(figsize=(16,8))
# plt.subplot(211)
# plt.axis([0,len(data),3,18])
# plt.plot(data['xVelocity']+noise1)
# plt.subplot(212)
# plt.axis([0,len(data),0,1.1])
# plt.plot(data['label'],color='red')
# plt.show()


# 2.标准化
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# train_X, test_X, train_label, test_label = sklearn.model_selection.train_test_split(X,y, random_state=1,test_size=0.3)# ,shuffle=False


train_X, test_X, train_label, test_label,T = data_process_SWaT()

# random.seed(2)
# train_size = 0.6 # 训练集的比例
# train_indices = random.sample(range(len(X)), int(train_size * len(X)))
# train_X = [X[i] for i in train_indices]
# train_label = [y[i] for i in train_indices]
#
# # 将剩余数据作为测试集
# test_X = [X[i] for i in range(len(X)) if i not in train_indices]
# test_label = [y[i] for i in range(len(y)) if i not in train_indices]


# plt.figure(figsize=(16,8))
# plt.subplot(421)
# plt.plot(test_X)
# plt.show()

# 3.初始化参数
W = 0.5  # 惯性因子  0.5
c1 = 0.6  # 学习因子  0.8
c2 = 0.6  # 学习因子  0.5
n_iterations = 3  # 迭代次数
n_particles = 30  # 种群规模


# 4.设置适应度值 输出分类精度得分，返回比较分类结果和实际测得值，可以把分类结果的精度显示在一个混淆矩阵里面
def fitness_function(position):  # 输出
    # 全局极值   svm分类器  核函数gamma  惩罚参数c
    svclassifier = SVC(kernel='rbf', gamma=position[0], C=position[1])
    # 参数gamma和惩罚参数c以实数向量的形式进行编码作为PSO的粒子的位置
    svclassifier.fit(train_X, train_label)
    #     score = cross_val_score(svclassifier, X, Y, cv=10).mean()                # 交叉验证
    #     print('分类精度',score)                                                    # 分类精度
    #     Y_pred = cross_val_predict(svclassifier, X, Y, cv=10) s                   # 获取预测值
    train_y_pred = svclassifier.predict(train_X)
    test_y_pred = svclassifier.predict(test_X)

    score_train = cross_val_score(svclassifier, train_X, train_label, cv=5).mean()
    #   score_test = cross_val_score(svclassifier, test_X, test_y, cv=3).mean()
    print("训练集分数：", score_train)
    #   print("测试集分数：",score_test)
    return confusion_matrix(train_label, train_y_pred)[0][1] + confusion_matrix(train_label, train_y_pred)[1][0], \
           confusion_matrix(test_label, test_y_pred)[0][1] + confusion_matrix(test_label, test_y_pred)[1][0]


# 5.粒子图
def plot(position):
    x = []
    y = []
    for i in range(0, len(particle_position_vector)):
        x.append(particle_position_vector[i][0])
        y.append(particle_position_vector[i][1])
    colors = (0, 0, 0)
    plt.scatter(x, y, c=colors, alpha=0.1)
    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
    plt.xlabel('gamma')  # 核函数
    plt.ylabel('C')  # 惩罚函数
    plt.axis([0, 9, 0, 9], )
    plt.gca().set_aspect('equal', adjustable='box')  # #设置横纵坐标缩放比例相同，默认的是y轴被压缩了。
    return plt.show()


# # 6.初始化粒子位置，进行迭代
# # 粒子位置向量
particle_position_vector = np.array([np.array([random.random() * 15, random.random() *15]) for _ in range(n_particles)])
pbest_position = particle_position_vector  # 个体极值等于最初位置
pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])  # 个体极值的适应度值
gbest_fitness_value = np.array([float('inf'), float('inf')])  # 全局极值的适应度值
gbest_position = np.array([float('inf'), float('inf')])
velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])  # 粒子速度

# 迭代更新
iteration = 0
while iteration < n_iterations:
    # plot(particle_position_vector)  # 粒子具体位置
    for i in range(n_particles):  # 对每个粒子进行循环
        fitness_cadidate = fitness_function(particle_position_vector[i])  # 每个粒子的适应度值=适应度函数（每个粒子的具体位置）
        # print("粒子误差", i, "is (training, test)", fitness_cadidate, " At (gamma, c): ",
        # particle_position_vector[i])

        if (pbest_fitness_value[i] > fitness_cadidate[
            1]):  # 每个粒子的适应度值与其个体极值的适应度值(pbest_fitness_value)作比较，如果更优的话，则更新个体极值，
            pbest_fitness_value[i] = fitness_cadidate[1]
            pbest_position[i] = particle_position_vector[i]

        if (gbest_fitness_value[1] > fitness_cadidate[1]):  # 更新后的每个粒子的个体极值与全局极值(gbest_fitness_value)比较，如果更优的话，则更新全局极值
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

        elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    for i in range(n_particles):  # 更新速度和位置，更新新的粒子的具体位置
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                               gbest_position - particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1

#7.输出最终结果
print("全局最优点的位置是 ", gbest_position, "在第", iteration, "步迭代中（训练集，测试集）错误个数:",
      fitness_function(gbest_position))

pso_svm_model = SVC(C= gbest_position[0],gamma = gbest_position[1],kernel='rbf')     #c=7,g=7

# pso_svm_model = SVC(C= 4,gamma = 5,kernel='rbf')     #c=7,g=7

pso_svm_model.fit(train_X,train_label)
y_preds = pso_svm_model.predict(test_X)

# save model
# joblib.dump(pso_svm_model, r'C:\Users\GJH\Desktop\论文实验及其数据\软著\pso_svm_model.pkl')
# load model
# pso_svm_model = joblib.load(r'C:\Users\gjh\highD按时间切分\POS_SVM模型训练\save_model/pso_svm_model.pkl')
# print(pso_svm_model.predict(test_X))

train_score = pso_svm_model.score(train_X, train_label)
test_score = pso_svm_model.score(test_X, test_label)

# print('训练集：', train_score)
# print('测试集：', test_score)
report = metrics.classification_report(test_label,y_preds,digits=4)   # 0是正确 1是错误
print(report)
