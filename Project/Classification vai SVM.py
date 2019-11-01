# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:09:51 2019

@author: mhrahman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.multiclass import OneVsRestClassifier

Train_data = pd.read_csv(r'C:\Users\mhrahman\Desktop\Machine learning_5063\Project\Digit\train.csv', index_col = False)
Test_data = pd.read_csv(r'C:\Users\mhrahman\Desktop\Machine learning_5063\Project\Digit\test.csv',index_col = False)
Train_Lbael = Train_data.iloc[:,0].values
Train_feature = Train_data.iloc[:,1:].values
Test = Test_data.iloc[:,:].values

# Plot of Label
plot = sns.countplot(Train_Lbael)

#Normalizing the data
Feature = (Train_feature - np.mean(Train_feature))/np.std(Train_feature)
#sc_x = StandardScaler()
#Feature_sk = sc_x.fit_transform(Train_feature)

# Train_Test data
X_train,Y_train,X_test,Y_test = train_test_split(Feature,Train_Lbael,test_size = 0.25, random_state = None)
V_train,W_train,V_test,W_test = train_test_split(X_train,X_test,test_size = 0.2,random_state = None)

#Hyperparameter tuning
parameter = [
        {'C' : [1,10,100], 'degree' : [1,2,3],'kernel': ['poly']},
        {'C' : [1,10,100], 'gamma' : [0.1,0.01,0.001],'kernel': ['rbf']}
        ]
V_clf = GridSearchCV(estimator=SVC(),param_grid=parameter, cv = 5)
V_clf.fit(W_train,W_test)

print ("Best Parameter found :")
print()
print(V_clf.best_params_)
print()
mean = V_clf.cv_results_['mean_test_score']
stds = V_clf.cv_results_['std_test_score']
for mean, std, params in zip(mean, stds, V_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


#SVM implementation
clf = SVC(C = 10.0,gamma= 'auto',kernel='poly',degree = 2)
clf.fit(X_train,X_test)
Y_pred = clf.predict(Y_train)

#Evaluation
Model_acc = clf.score(X_train,X_test)
Accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy:{}".format(Accuracy))
conf = confusion_matrix(Y_test,Y_pred)
df_conf = pd.DataFrame(conf,range(10),range(10))
ax = plt.axes()
sns.set(font_scale=1.2)
sns.heatmap(df_conf,annot= True,cmap="YlGnBu_r",fmt='g')
ax.set(xlabel = 'Predicted',ylabel = 'True')
ax.set_title("Confusion Matrix")
ax.axis.label.set_size(15)
#ROC_analysis
Pred_f = []
True_f = []
Y_one = label_binarize(Train_Lbael, classes=[0,1,2,3,4,5,6,7,8,9])
n_class = Y_one.shape[1]
x_train,x_test, y_train,y_test = train_test_split(Feature,Y_one,test_size = 0.25, random_state = None)
classifier = OneVsRestClassifier(SVC(C = 10.0,degree = 2,gamma= 'auto',kernel='poly'))
Y_score = classifier.fit(x_train,y_train).decision_function(x_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i],_ = roc_curve(y_test[:,i],Y_score[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()