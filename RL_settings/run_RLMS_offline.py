import sys

sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') 
from RL_settings.env import *
from base_detectors.get_preds_thres import *
from stable_baselines3 import DQN

# Make predictions based on the pretrained models, obtain the raw scores and thresholds
# 根据预训练的模型进行预测，获得原始分数和阈值
list_pred_sc, list_gtruth, T = raw_scores_gtruth(model_path='C:/Users/gjh/Desktop/RLMSAD-master/base_detectors')

# thresholds
# list_thresholds=[]
# for i in range(len(list_pred_sc)):    # 5个分类器阈值
#     list_thresholds.append(raw_thresholds(list_pred_sc[i],contamination=0.16))  #0.12

# 将整数数组转换为float64类型
list_thresholds=[]
arr_int = np.array([0.5,0.5,0.5,0.5])
list_thresholds = arr_int.astype('float64')
# for i in range(len(list_pred_sc)):    # 5个分类器阈值
#     list_thresholds.append((list_pred_sc[i].max()+list_pred_sc[i].min())/2)



EXP_TIMES = 10 # 10 How many runs to average the results

# Store the precision, recall, F1-score
store_prec = np.zeros(EXP_TIMES)
store_rec = np.zeros(EXP_TIMES)
store_f1 = np.zeros(EXP_TIMES)
store_acc = np.zeros(EXP_TIMES)

for times in range(EXP_TIMES):
    # Set up the training environment on all the dataset 在所有数据集上设置训练环境
    env_off=TrainEnvOffline(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
    # env_off=TrainEnvOffline_consensus_conf(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
    # env_off=TrainEnvOffline_dist_conf(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)
    # env_off=TrainEnvOffline_None(list_pred_sc=list_pred_sc, list_thresholds=list_thresholds, list_gtruth=list_gtruth)

    # Train the model on all the dataset   在所有数据集上训练模型
    model = DQN(
                'MlpPolicy',
                env_off,
                verbose=0,
                learning_rate=5e-4,   # 0.0025
                policy_kwargs={"net_arch" : [128, 64]}
                )
    # model=A2C('MlpPolicy', env_off, verbose=0)
    model.learn(total_timesteps=len(list_pred_sc[0])) # 此处的total_timesteps代表总共的时间步的数量，也就是你的模拟能够得到的state,action,reward,next state的采样数量。
    model.save("DQN_offline_model")
    # model.save("A2C_offline_model")

    # Evaluate the model on all the dataset  在所有数据集上评估模型
    model = DQN.load("DQN_offline_model")
    # model=A2C.load("A2C_offline_model")

    prec, rec, f1, _, list_preds, acc, total_reward= eval_model(model, env_off)

    store_acc[times]=acc
    store_prec[times]=prec
    store_rec[times]=rec
    store_f1[times]=f1
    if acc>0.98:
        print("times: %2d, acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f, total_reward: %.2f" % (times, acc, prec, rec, f1, total_reward))

# Compute the mean and standard deviation of the results
average_acc=np.mean(store_acc)
average_prec=np.mean(store_prec)
average_rec=np.mean(store_rec)
average_f1=np.mean(store_f1)

std_acc=np.std(store_acc)
std_prec=np.std(store_prec)
std_rec=np.std(store_rec)
std_f1=np.std(store_f1)

print("Total number of reported anomalies: ",sum(list_preds))
print("Total number of true anomalies: ",sum(list_gtruth))

print("Average acc: %.4f, std: %.4f" % (average_acc, std_acc))
print("Average precision: %.4f, std: %.4f" % (average_prec, std_prec))
print("Average recall: %.4f, std: %.4f" % (average_rec, std_rec))
print("Average F1-score: %.4f, std: %.4f" % (average_f1, std_f1))
