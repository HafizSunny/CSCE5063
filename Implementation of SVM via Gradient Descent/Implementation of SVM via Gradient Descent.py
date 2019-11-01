# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 01:11:32 2019

@author: mhrahman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta 
data=pd.read_csv('data.csv', index_col=False)
Feature=data.iloc[:,0:8]
Label=data.iloc[:,8:9]
Y=Label.values
w=np.zeros([1,8])
bias=0
X=Feature.values

def cost(C,Input,y,bias,w):
    x=Input
    mul=np.dot(x,w.T)+bias
    J=C*np.sum(np.maximum(0,1-(y*mul)))+ 0.5*np.sum(w**2) 
    return J
def J_w(x,y,b,w,C):
        zs=np.dot(x,w.T)+b
        fu=np.where(y*zs>=1,0,-y)
        d_w=np.zeros(x.T.shape[0])
        if len(x)==8:
            for i in range(len(x)):
                L=x*fu.T
                d_w[i] = C * np.sum(L)
        else:
            for i in range (x.T.shape[0]):
                L=x[:,i]*fu.T
                d_w[i] = C * np.sum(L)
        J_w=w+d_w
        return J_w
    
def b_w(x,y,b,w,C):
    zs=np.dot(x,w.T) + b
    L=np.where(y*zs>=1,0,-y)
    b_w=C*np.sum(L)
    return b_w

def bgd(x,y,b,w,lr,epsilon,C):
    m=len(x)
    converged=False
    c=cost(C,x,y,b,w)
    k=0
    err=[]
    err.append(c)
    while not converged:
#        for i in range(m):
        w=w-lr*(J_w(x,y,b,w,C))
        error=cost(C,x,y,b,w)
        err.append(error)
        cond=(abs(c-error)*100)/c
        if cond<epsilon:
            print('Converged')
            converged=True
            break;
        c=error
        b=b-lr*b_w(x,y,b,w,C)
        k=k+1
    return w,b,k,err
bgd_start_time= time.time()   
b_gd=bgd(X,Y,bias,w,1e-9,0.04,10)
bgd_end_time=time.time()
print('bgd convergence : {}'.format(bgd_end_time - bgd_start_time))

def batch_create(x,y,batch_size,itr):
#    batch_n=len(x_1)/batch_size
    #batch_id = np.int(np.mod(itr,(len(x)+batch_size-1)/batch_size))
    batch_id =np.mod(itr,len(x)//batch_size)
    mini_x = x[(batch_id*batch_size):((batch_size*(batch_id + 1)))]
    mini_y=y[(batch_id*batch_size):(batch_size*(batch_id + 1))]
    return mini_x, mini_y
    
def mini_bgd(x,y,b,w,lr,epsilon,C,batch_size,shuffle=False):
    k=0
    data_1=np.hstack((x,y))
    if shuffle==True:
        np.random.shuffle(data_1)
    x_1=data_1[:,0:8]
    y_1=data_1[:,8:9]
#    m=len(x)
    x1,y1=batch_create(x_1,y_1,batch_size,k)
    c=cost(C,x,y,b,w)
    convererged=False
    del_K_1=0
    err=[]
    err.append(c)
    while not convererged:
        x1,y1=batch_create(x_1,y_1,batch_size,k)
        w = w - lr*(J_w(x1,y1,b,w,C))
        error = cost(C,x,y,b,w)
        err.append(error)
        cond=(abs(c-error)*100)/c
        del_K=0.5*del_K_1 + 0.5 * cond
        if del_K < epsilon:
            print ('Converged')
            convererged = True
            break; 
        c=error
        b=b-lr*b_w(x1,y1,b,w,C)
        del_K_1 = del_K
        k = k + 1
    return w,b,k,err
# Stochastic gradient descent
sgd_start_time = time.time()
sgd=mini_bgd(X,Y,bias,w,1e-8,0.0003,10,1,shuffle=True)
sgd_end_time = time.time()
print('sgd convergence : {}'.format(sgd_end_time - sgd_start_time))

# Mini-batch gadient descent
mbgd_start_time = time.time()
mbgd=mini_bgd(X,Y,bias,w,1e-8,.004,10,4,shuffle=True)
mbgd_end_time = time.time()
print('mbgd convergence : {}'.format(mbgd_end_time - mbgd_start_time))


#plt.plot(b_gd[3])
#plt.plot(sgd[3])
plt.plot(mbgd[3])
plt.title('Cost vs the number of iterations for Mini batch gradient Descent')
            
            
            
            
            
    