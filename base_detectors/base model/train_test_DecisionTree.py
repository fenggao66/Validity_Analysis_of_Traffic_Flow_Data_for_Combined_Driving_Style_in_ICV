# Set current working directory to the main branch of RLMSAD
import sys
sys.path.append('C:/Users/gjh/Desktop/RLMSAD-master') # This is the path setting on my computer, modify this according to your need
from sklearn import tree
from data_process import *
import pickle
from sklearn import metrics

# path_normal= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/train_sum_2_.csv"
# path_attack= "C:/Users/gjh/Desktop/RLMSAD-master/base_detectors/raw_data/SWaT/1111_v_E.csv"
train_X, test_X, train_label, test_label,T = data_process_SWaT()

# DecisionTreeClassifier
rnd_clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='best', min_samples_leaf=5)# random_state=42，  gini或者 entropy,
rnd_clf.fit(train_X,train_label)

# 验证test
y_preds = rnd_clf.predict(test_X)
report = metrics.classification_report(test_label,y_preds,digits=4)   # 0是正确 1是错误
print(report)

# save the model
# filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/Dtree_SWaT.sav'

# filename = 'C:/Users/GJH/Desktop/RLMSAD-master/base_detectors/Dtree_SWaT_2.sav'
# pickle.dump(rnd_clf, open(filename, 'wb'))