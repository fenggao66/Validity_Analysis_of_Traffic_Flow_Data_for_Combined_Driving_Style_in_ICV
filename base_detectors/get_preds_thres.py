# Set current working directory to the main branch of RLMSAD
import os
import sys
sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
from data_process import *
from base_detectors.USAD.usad_model import *
import numpy as np
import torch
import torch.utils.data as data_utils
import pickle

def raw_scores_gtruth(model_path='C:/Users/gjh/Desktop/RLMSAD-master/base_detectors'):

    '''Prepare for computing raw scores and get ground truth labels for the WHOLE dataset
    准备计算原始分数并获得整个数据集的基本事实标签（平均值采样后 = 5帧）'''

    train_X, test_X, train_label, test_label, T = data_process_SWaT()
    
    BATCH_SIZE =  20   #7919
    N_EPOCHS = 70
    hidden_size = 40

    # w_size=windows_normal.shape[1]*windows_normal.shape[2]
    #
    # test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    #                     torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))),
    #                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load the pretrained model
    model_list = [f for f in sorted(os.listdir(model_path)) if f.endswith('.sav')] # model named sorted in alphabetical order
    # # Extract the model names
    # self.model_names = [f.split('_')[0] for f in self.model_list]

    # 每个模型在attack上给出分数
    list_pred_sc = []
    for f in model_list:
        filepath=os.path.join(model_path,f)
        loaded_model = pickle.load(open(filepath, 'rb'))
        # if f.split('_')[0] == 'usad':
        #     sc_pred=training_scores(loaded_model,test_X)
        #     sc_pred=np.concatenate([torch.stack(sc_pred[:-1]).flatten().detach().cpu().numpy(),
        #                             sc_pred[-1].flatten().detach().cpu().numpy()])
        if f.split('_')[0] == 'SVM' or 'Log' or 'Dtree' or 'Mlp':
            # windows_attack=windows_attack.reshape((windows_attack.shape[0],-1)) # Flatten the windows
            # sklearn defined this as the opposite as the original paper, so here it is the lower the more abnormal
            # sklearn将其定义为与原始论文相反，因此在这里它越低越不正常
            # sc_pred=-loaded_model.decision_function(test_X)
            prob = loaded_model.predict_proba(test_X)
            sc_pred = prob[:,1]
        # else:
        #     windows_attack=windows_attack.reshape((windows_attack.shape[0],-1)) # Flatten the windows
        #     sc_pred=loaded_model.decision_function(windows_attack)
        list_pred_sc.append(sc_pred)

    # Ground truth labels  窗口所有标签真实
    # windows_labels=[]
    # for i in range(len(labels_down)-windows_normal.shape[1]):
    #     windows_labels.append(list(np.int_(labels_down[i:i+windows_normal.shape[1]])))
    # list_gtruth=[1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    return list_pred_sc, test_label, T

def train_test(list_pred_sc,list_gtruth, train_portion=0.7):
    '''Train-test split for the predicted scores and ground truth labels
    预测分数和地面真相标签的训练测试分割'''
    full_size=len(list_pred_sc[0]) # Full size of the dataset
    list_pred_sc_train=[list_pred_sc[i][:int(full_size*train_portion)] for i in range(len(list_pred_sc))]
    list_pred_sc_test=[list_pred_sc[i][int(full_size*train_portion):] for i in range(len(list_pred_sc))]
    list_gtruth_train=list_gtruth[:int(full_size*train_portion)]
    list_gtruth_test=list_gtruth[:int(full_size*train_portion)]
    return list_pred_sc_train, list_pred_sc_test, list_gtruth_train, list_gtruth_test


def raw_thresholds(raw_scores, contamination=0.1):
    '''raw_scores: each 1D numpy array, the raw anomaly scores'''
    return np.sort(raw_scores)[int(len(raw_scores)*(1-contamination))]

