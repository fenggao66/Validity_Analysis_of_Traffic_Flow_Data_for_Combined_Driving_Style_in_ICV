# Set current working directory to the main branch of RLMSAD
import sys
# sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
from sklearn.neural_network import MLPClassifier
from data_process import *
import pickle
from sklearn import metrics

# path_normal= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/train_sum_2_.csv"
# path_attack= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/1111_v_E.csv"

#数据获取
train_X, test_X, train_label, test_label, T = data_process_SWaT()

# mlp
mlp_clf = MLPClassifier(random_state=42,max_iter = 400)
mlp_clf.fit(train_X,train_label)

# 验证test
y_preds = mlp_clf.predict(test_X)
report = metrics.classification_report(test_label,y_preds,digits=4)   # 0是正确 1是错误
print(report)

# save the model
# filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/Mlp_SWaT.sav'

# filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/Mlp_SWaT_2.sav'
# pickle.dump(mlp_clf, open(filename, 'wb'))