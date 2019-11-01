# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:39:38 2019

@author: mhrahman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Input = pd.read_csv('./data/X.csv',index_col=False,header= None)
X = Input.values
Output = pd.read_csv('./data/Y.csv',index_col = False,header = None)
Y = Output.values
# bias
b = np.ones([X.shape[0],1])
## Weight matrix_________
init_w1 = pd.read_csv('./data/initial_W1.csv',index_col = False, header = None)
init_w2 = pd.read_csv('./data/initial_W2.csv',index_col = False, header = None)
initial_W1 = init_w1.values
initial_W2 = init_w2.values
def one_hot_vector(Y):
    row_num = Y.shape[0]
    col_num = len(np.unique(Y))
    one_hot = np.zeros([row_num,col_num])
    for i,j in enumerate(Y):
        one_hot[i,j[0]-1] = 1
    return one_hot
Y= one_hot_vector(Y)

#activation function
def sigmoid(X):
    return  1/(1 + np.exp(-X))
# Gradient of sigma 
def Grad_sigm(X):
    return (sigmoid(X)*(1 - sigmoid(X)))


# Feed Forward part
def feed_forward(X,bias,W_1,W_2):
    X = np.concatenate((bias,X),axis=1)
    Z_1 = np.dot(X,W_1.T)
    H = sigmoid(Z_1)
    H = np.concatenate((bias,H), axis = 1)
    Z_2 = np.dot(H,W_2.T)
    Y_pred = sigmoid(Z_2)
    return Y_pred, H, Z_1
#### Debugging Feed-Forward part
W1 = pd.read_csv('./data/W1.csv', index_col= False, header = None)
W1 = W1.values
W2 = pd.read_csv('./data/W2.csv', index_col = False, header = None)   
W2 = W2.values 
Prim_Y,f,u = feed_forward(X,b,W1,W2)
#### _________________________________________________________________
# Accuracy checking
def calc(Y_real, Y_pred):
    pred = np.argmax(Y_pred,axis = 1)
    actual = np.argmax(Y_real,axis =1)
    loss = sum([x != y for x,y in zip (pred,actual)])
    count = len(pred)
    return 1- loss/count
acc = calc(Y, Prim_Y)

# cost (categorical cross-entropy)
def cost(Y_real,Y_pred,lamb,W_1,W_2):
#    J=0
    cat_cross = 1/len(Y_pred)*(np.sum(-Y_real * np.log(Y_pred) - (np.ones([len(Y_real),1]) - Y_real)* np.log(np.ones([len(Y_real),1]) - Y_pred)))
#    for i in range(len(Y_pred)):
#        for k in range(Y_real.shape[1]):
#            J = J + (-Y_real[i,k]*np.log(Y_pred[i,k]))
    Regu = lamb/(2 * len(Y_pred))*(np.sum((W_1[:,1:W_1.shape[1]])**2) + np.sum((W_2[:,1:W_2.shape[1]])**2))
    return cat_cross + Regu 

# Backpropagation
def back_prop(Y_pred, Y_real,X,bias,W_1,W_2,H,Z_1, lamb):
    X = np.concatenate((bias,X),axis=1)
    regu_W_1 = W_1
    regu_W_1[:,0] = 0
    regu_W_2 = W_2
    regu_W_2[:,0] = 0
    b_2 = Y_pred - Y_real
    b_1 = np.dot(b_2,W_2[:,1:])*Grad_sigm(Z_1)
    Del_W2 = 1/len(Y_pred)*(np.dot(b_2.T,H) + lamb*(regu_W_2))
    Del_W1 = 1/len(Y_pred)*(np.dot(b_1.T,X) + lamb*(regu_W_1))
    return Del_W1, Del_W2
## Back prop check
# =============================================================================
# W1_grad = pd.read_csv('./data/W1_grad.csv',index_col = False, header = None)
# W2_grad = pd.read_csv('./data/W2_grad.csv',index_col = False, header = None)
# W1_grad = W1_grad.values
# W2_grad = W2_grad.values
# =============================================================================

def batch_grad(X,bias,Y,W_1,W_2,lamb, lr, epoch):
    error = []
    Y_pred_1,h,z = feed_forward(X,bias,W_1,W_2)
    c = cost(Y, Y_pred_1,lamb,W_1,W_2)
    error.append(c)
    for i in range(epoch):
        Y_pred, H, Z1 = feed_forward(X,bias,W_1,W_2)
        Del_W1, Del_W2 = back_prop(Y_pred,Y,X,bias,W_1,W_2,H,Z1,lamb)
        W_1 = W_1 - lr*Del_W1
        W_2 = W_2 - lr*Del_W2
        err = cost(Y,Y_pred,lamb,W_1,W_2)
        error.append(err)
        accuracy = calc(Y, Y_pred)
        print('Accuracy {} for epoch {}'.format(accuracy,i))
    return error
    
    
final = batch_grad(X,b,Y,initial_W1,initial_W2,3,0.2,500) 
plt.plot(final)
plt.title("Loss vs number of iterations")
plt.xlabel("Iteraions")
plt.ylabel("Loss")