import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pylab import mpl
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
data = pd.read_csv(r'C:\Users\GJH\data_train_test\LSTM_pre_v\train_lstm\244_v_e.csv')
print(len(data))

TRAIN_SPLIT = 170
tf.random.set_seed(13)

import pickle
density_80_pinghua = data['density_80_pinghua']
model_path=r'C:\Users\GJH\svm_密度821\p1.sav'
p1 = pickle.load(open(model_path, 'rb'))
pre_v = p1(density_80_pinghua)     # density_80_pinghua   avg_density_pinghua

# 添加列 预测和最后都用
data['pre_v'] = pre_v
len(pre_v)

features_considered = ['xAcceleration','density_80_pinghua','pre_v','E','dhw','xVelocity']  #
features = data[features_considered]
# features.index = data['Unnamed: 0.2']
# true_v.index = data['Unnamed: 0.2']

features.head()

from sklearn import preprocessing
ss = preprocessing.StandardScaler()
dataset = ss.fit_transform(features.iloc[:,:-1])
true_v = features['xVelocity']

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        # indices = range(i-history_size, i, step) #索引为range(0, 720, 6)，range(1, 721, 6) range(2, 722, 6)
        indices = range(i-history_size, i)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size]) #（720+72）（721+72）
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

past_history = 5  # 记忆时间   5最好
future_target = 0
STEP = 0
  #    dataset[:,:5]前5个特征, dataset[:, -1]
x_train_single, y_train_single = multivariate_data(dataset, true_v, 0,     # 从0开始
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, true_v,TRAIN_SPLIT,     # 从切分开始
                                                None, past_history,
                                               future_target, STEP,
                                               single_step=True)

BATCH_SIZE = 128    # 256
BUFFER_SIZE = 1000  # 1000
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(16,
                                           input_shape=x_train_single.shape[-2:]))
model.add(tf.keras.layers.Dropout(0.4))
# single_step_model.add(tf.keras.layers.LSTM(8))# return_sequences=True,,activation='relu',

model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.003), loss='mse') #   mse均方误差    mean_squared_error

import os

EVALUATION_INTERVAL = 20   # 多少步验证
EPOCHS = 70    # 训练次数

checkpoint_path = f"best_model/244model_save_and_load.ckpt"   # 模型保存路径

# checkpoint_dir = f"best_model/1197model_save_and_load.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
# last_model_path = tf.train.latest_checkpoint(checkpoint_dir)
# print(last_model_path)
# if last_model_path is not None and os.path.exists(last_model_path):
#   model.load_weights(last_model_path)                       # 只加载权重，对应model.save_weights保存的文件
#   model = tf.keras.models.load_model(last_model_path)       # 加载完整模型，对应model.save保存的文件
#   print('加载模型:{}'.format(last_model_path))


# 创建callback，用于保存训练时的模型权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,    # 只保存权重
                                                 save_best_only=True,       # 只保存最佳模型
                                                 monitor='val_loss',
                                                 mode='min',
                                                 verbose=1)

single_step_history = model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50,
                                            callbacks=[cp_callback]) #

# 数据预测
checkpoint_path = f"best_model/244model_save_and_load.ckpt"
model = tf.keras.models.load_model(checkpoint_path)

pre_finaly = model.predict(x_val_single)

true_v = y_val_single

time = []
n = len(pre_finaly)
for i in range(0,n):
    time.append(i)
# 预测值
fig = plt.figure(figsize=(16,8))
plt.plot(time,pre_finaly, 'b-', label = 'pre_finaly')

# 真实值
plt.plot(time, true_v, 'r', label = 'actual')
plt.xticks(rotation = '60');
plt.legend()

# plt.plot(time, pre_v[TRAIN_SPLIT+past_history:], 'g', label = 'pre_v')
# plt.xticks(rotation = '60');
# plt.legend()

from sklearn.metrics import mean_squared_error,r2_score
from sklearn import metrics
# 两个参数分别是实际值、预测值  features['pre_v']  labels真实值

MSE1 = metrics.mean_squared_error(true_v,pre_finaly)
RMSE1 = metrics.mean_squared_error(true_v,pre_finaly)**0.5
MAE1 = metrics.mean_absolute_error(true_v,pre_finaly)
MAPE1 = metrics.mean_absolute_percentage_error(true_v,pre_finaly)

MSE2 = metrics.mean_squared_error(true_v,pre_v[TRAIN_SPLIT+past_history:])
RMSE2 = metrics.mean_squared_error(true_v,pre_v[TRAIN_SPLIT+past_history:])**0.5
MAE2 = metrics.mean_absolute_error(true_v,pre_v[TRAIN_SPLIT+past_history:])
MAPE2 = metrics.mean_absolute_percentage_error(true_v,pre_v[TRAIN_SPLIT+past_history:])

print('均方误差：',MSE1,MSE2)
print('均方根误差：',RMSE1,RMSE2)
print('平均绝对误差：',MAE1,MAE2)
print('平均绝对百分比误差：',MAPE1,MAPE2)

xVelocity = data['xVelocity'][TRAIN_SPLIT+past_history:]
density_80_pinghua = data['density_80_pinghua'][TRAIN_SPLIT+past_history:]
E = data['E'][TRAIN_SPLIT+past_history:]
pre_v_ = pre_v[TRAIN_SPLIT+past_history:]   # 多一个杠 防止数据重复（导致减少）
dhw = data['dhw'][TRAIN_SPLIT+past_history:]
xAcceleration = data['xAcceleration'][TRAIN_SPLIT+past_history:]
pre_finaly
len(xVelocity),len(density_80_pinghua),len(E),len(pre_v_),len(dhw),len(xAcceleration),len(pre_finaly)


dataset = pd.DataFrame()

dataset['xAcceleration'] = xAcceleration
dataset['density_80_pinghua'] = density_80_pinghua
dataset['E'] = E
dataset['pre_v'] = pre_v_
dataset['dhw'] = dhw
dataset['pre_v_fina'] = pre_finaly
dataset['xVelocity'] = xVelocity

dataset.to_csv(r"C:\Users\gjh\Desktop\RLMSAD-master\svm_test\row_data\244.csv")
