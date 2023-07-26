#  定义 DataFrame 求差分  目的求导数 time已经定义
import pandas as pd
import numpy as np
import scipy.signal
import csv
import matplotlib.pyplot as plt

# 数据导入
data = pd.read_csv(r"C:\Users\gjh\Desktop\RLMSAD-master\base_detectors\raw_data\SWaT\Normal_sum_2.csv")
xAcceleration = data['xAcceleration']

# time为data总长度
time = []
n = len(data)
for i in range(0,n):
    time.append(i)

df=pd.DataFrame(data={'time':time,'xAcceleration':xAcceleration})
df["xAcceleration导数"]=df["xAcceleration"].diff()/0.04    #计算速度的差分  0.04为采样频率

# 将第一个默认为0.00
df['xAcceleration导数'][0] = 0.00

# 去除分界线加速度（过高）
for i in range(len(df['xAcceleration导数'])):
    if abs(df['xAcceleration导数'][i])>3:
        df['xAcceleration导数'][i] = 0.2

plt.figure(figsize=(20,8))  #设定图画大小
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(time,df['xAcceleration导数'],label="'加速度导数'")  #通过将1阶差分除以步长step得出模拟导数值
plt.show()

# 平滑
z_acc = scipy.signal.savgol_filter(df['xAcceleration导数'][1:], window_length = 15, polyorder = 4)
len(z_acc)
plt.figure(figsize=(20,8))  #设定图画大小
# plt.subplot(223)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title('车辆抽搐度')
plt.xlabel('时刻')
plt.ylabel('导数')
# plt.axis([0,800,-1,1])

plt.plot(time[1:],z_acc,label='加速度导数（平滑）')
plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0.88, 0.95))
plt.show()

# 量化窗口内
J = 0.2
list_E = []
window = 50       # 窗口大小
density_80 = data['density_80']
for i in range(len(z_acc)-window+1):
    list_window = z_acc[i:i+window]
    R = np.std(list_window)        # 第25*0.04秒时候的冲击标准差
    e = R/J
    E = e*(density_80[i+window]/0.08)         # 使用密度量化
    list_E.append(E)
# len(time),len(list_E)
# 驾驶风格绘图
plt.figure(figsize=(20,8))  #设定图画大小
plt.plot(time[:len(list_E)],list_E,label='驾驶风格')
plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0.88, 0.95))
plt.show()

# 驾驶风格离散化
list_scarrer = []

# 将窗口大小驾驶风格补全为2（一般）
for i in range(window):
    list_scarrer.append(2)

for i in range(len(list_E)):
    if 0 <= list_E[i] < 0.75:
        list_scarrer.append(1)
    if 0.75 <= list_E[i] < 1.5:
        list_scarrer.append(2)
    if 1.5 <= list_E[i]:
        list_scarrer.append(3)

len(time), len(list_scarrer), len(list_E)

#驾驶风格离散化展示
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(111)
plt.title('驾驶风格量化')
plt.xlabel('时间',fontsize=10)
plt.ylabel('风格',fontsize=10)

ax1.scatter(time,list_scarrer,s = 20)
# plt.legend('sample',borderaxespad=0., bbox_to_anchor=(0.8, 0.8))
plt.show()