    # -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:29:37 2020

@author: wangtianyu6162
"""

import math
import numpy as np
import pandas as pd
from gurobipy import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.constraints import non_neg 
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from scipy.optimize import minimize 
from e2e_network import *
from param import *
class ReLuNet(e2e_network):
    def __init__(self, df_price, df_share, base_price, base_share, cost):
        e2e_network.__init__(self, df_price, df_share, base_price, base_share, cost)
    #-----------------------------------------------#
    #abs linear optimization problem linear + relu + linear#  
    def train_network_revenue(self, df_fit_price, df_fit_share):
        best_price = np.zeros(self.product_number)
        #standarized for large num of products
        if choice == 1:
            x_train = (self.df_price - lower_price)/10 
            y_train = (60 - self.revenue)/10
        
        #standarized for small number of products
        elif choice == 0:
            x_train = (self.df_price - lower_price)/10
            y_train = (6 - self.revenue)/5
        
        #create the model

        model = Sequential()
        
        model.add(Dense(units = 10, input_dim = self.product_number))
        model.add(Activation("relu"))
        
        model.add(Dense(units = 1))
        model.add(Activation("linear"))
        #kernel_constraint is to ensure its convexity
        
        model.summary()
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
        reduce_lr = LearningRateScheduler(self.scheduler)
        #auto_reduce the learning rate
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10*product_number, mode = 'auto')
        model.fit(x_train, y_train, epochs = 200*product_number, validation_split = 0.1, batch_size = 32, callbacks = [reduce_lr])        
        
        
        
        # find the values of weights
        self.weights = np.array(model.get_weights())
        self.weights[0] = self.weights[0].T
        self.weights[2] = np.array([self.weights[2][i][0] for i in range(len(self.weights[2]))])

    def opt_network_revenue(self):
        #optimize the function fitted by neural network
        cost = lambda price: np.sum([self.weights[2][i]*max(np.dot(self.weights[0][i], price) + self.weights[1][i], 0) for i in range(len(self.weights[2]))])
        initial_price = np.zeros(self.product_number) 
        bounds = tuple([(x,y) for x,y in zip(np.zeros(self.product_number), np.ones(self.product_number)*(upper_price - lower_price)/10)])
        res = minimize(cost, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
        print(res)
        return res.x
        
    
    #fit the real data in the model
    def opt_network_revenue_fit(self, df_fit_price, df_fit_share):
        fit_price = np.array(df_fit_price)
        fit_share = np.array(df_fit_share)
        mse = 0
        fit_number = len(fit_price)
        predict_R = np.zeros(fit_number)
        #evaluation the created error
        for i in range(fit_number):
            true_R = np.dot(fit_price[i] - self.cost, fit_share[i])
            if choice == 1:
                predict_R[i] = 60 - 10*(np.sum([self.weights[2][j]*max(np.dot(self.weights[0][j], (fit_price[i] - lower_price)/10) + self.weights[1][j], 0) for j in range(len(self.weights[2]))]) + self.weights[3][0])
            else:
                predict_R[i] = 6 - 5*(np.sum([self.weights[2][j]*max(np.dot(self.weights[0][j], (fit_price[i] - lower_price)/10) + self.weights[1][j], 0) for j in range(len(self.weights[2]))]) + self.weights[3][0])
            #print(true_R, predict_R)
            mse = mse + (true_R - predict_R[i])**2
        #the predict_R is for further use to readjust the error in the LE model.
        return [math.sqrt(mse/fit_number), predict_R]        
