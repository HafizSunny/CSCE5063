# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 00:08:18 2019

@author: mhrahman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('data.csv',index_col=False)
data=data.drop(['Index'],axis=1)

#Data normalization
norm_data=(data-data.min())/(data.max()-data.min())

#Linear Regression
X=norm_data.iloc[:,0:15]
Y=norm_data.iloc[:,15:16]
bias_vector=np.ones([X.shape[0],1])
X=np.concatenate((X,bias_vector),axis=1)
#Y=b+x1.T1+x2.T2+.....+x15.T15
theta=np.zeros([1,16])

#Train-Test split
indicies=X.shape[0]
num_trainng=int(indicies*0.8)
num_testing=indicies-num_trainng

X_train,X_test=X[:num_trainng],X[num_trainng:]
Y_train,Y_test=Y.iloc[:num_trainng],Y[num_trainng:]

#Cost with regularization
def cost(X,Y,theta,lamb):
    m=len(Y)
    ini_cost=np.square((Y-X.dot(theta.T)))
    J=(1/(2*m))*np.sum(ini_cost) + (lamb/(2*m))*np.sum(abs(theta))
    return J
#cost with regularization
def cost_without(X,Y,theta):
    m=len(Y)
    ini_cost=np.square((Y-X.dot(theta.T)))
    J=(1/(2*m))*np.sum(ini_cost)
    return J
#c=cost(X,Y,theta)
def gradient_desecent(X,Y,theta,lr,epsilon,regu):
    converged=False
    m=len(X)
    c=cost(X,Y,theta,regu)
    itr=0
    err=[]
    while not converged:
        for i in range(m):
            Y_0=np.dot(X,theta.T)
            theta=theta - (1/m)*lr*(X.T.dot(Y_0-Y)).T -lr*(regu/(2*m))*np.sign(theta)
#            for i in range(len(theta[0])):
#                if theta[0][i]<0.005:
#                    theta[0][i]=0
            error=cost(X,Y,theta,regu)
            cond=(abs(c-error)*100)/c
            if cond[0]<epsilon:
                print('Converged')
                converged=True
                break;
            err.append(error[0])
            c=error
        itr+=1
    return theta,itr,err
#plot of the loss function(J_K)
d=gradient_desecent(X_train,Y_train,theta,0.01,0.1,1)
plt.plot(d[2])
plt.title("Loss function(J) vs the number of iteration (k)")
plt.xlabel('Total number of iteration')
plt.ylabel('Loss')
#computation of the error for test data
test_cost=cost_without(X_test,Y_test,theta)
print(test_cost)
           
        
            
    
        