# Set current working directory to the main branch of RLMSAD
import sys
# sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import SVC
from data_process import *
import pickle
from sklearn import metrics

# path_normal= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/train_sum_2_.csv"
# path_attack= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/1111_v_E.csv"
train_X, test_X, train_label, test_label, T = data_process_SWaT()

# SVM
# for i in range(1,10,1):
#     for j in range(1, 10,1):
#         svm_model = SVC(C=i, gamma=j, kernel='rbf', probability=True)     #C= , gamma =  [11.56584394 10.19786332]
#         svm_model.fit(train_X,train_label)
#         # 验证test
#         y_preds = svm_model.predict(test_X)
#         report = metrics.classification_report(test_label,y_preds,digits=4)   # 0是正确 1是错误
#         print(report,i,j)

svm_model = SVC(C=1, gamma=10, kernel='rbf', probability=True)     #C= , gamma =  [11.56584394 10.19786332]
svm_model.fit(train_X,train_label)
# 验证testz
y_preds = svm_model.predict(test_X)
report = metrics.classification_report(test_label,y_preds,digits=4)   # 0是正确 1是错误
print(report)
# save the model
#filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/SVM_SWaT.sav'

# filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/SVM_SWaT_2.sav'
# pickle.dump(svm_model, open(filename, 'wb'))