# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:38:23 2020

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
class SigmoidNet(e2e_network):
    def __init__(self, df_price, df_share, base_price, base_share, cost):
        e2e_network.__init__(self, df_price, df_share, base_price, base_share, cost)
    
    #-------------------------directly train demand-------------------------
    #--------------------- and then use info to retrive optimized revenue-------------
    #train the demand function of each product
    def train_network_demand_level1(self, i = 0, epochs = 1000, discount_factor = 0.9):
        if choice == 0:
            x_train = (self.df_price - lower_price)/50
        elif choice == 1:
            x_train = (self.df_price - lower_price)/10
        y_train = self.df_share.T[i]
        
        #parameters of transformation
        coefficient = discount_factor/y_train.max()
        y_train = coefficient*y_train
        
        #create the model
        model = Sequential()
        
        model.add(Dense(units = 1, input_dim = self.product_number))
        model.add(Activation("sigmoid"))
        
        
        
        model.summary()
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
        #auto reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10*product_number, mode = 'auto')
        model.fit(x_train, y_train, epochs = 100*product_number, validation_split = 0.1, batch_size = 32, callbacks = [reduce_lr])  
        
        
        [weight, intercept] = model.get_weights()
        return [[weight[i][0] for i in range(len(weight))], intercept[0], coefficient]
    
    #incorporate the number of product functions into below:
    def train_network_demand_whole(self):
        weight_list = []
        inter_list = []
        coeff_list = []
        for i in range(self.product_number):
            [weight, intercept, coefficient] = self.train_network_demand_level1(i)
            weight_list.append(weight)
            inter_list.append(intercept)
            coeff_list.append(coefficient)
        print("finish the distributed training process for fitting demand curve!")
        weight_list = np.array(weight_list)
        print("the intercept list is below",inter_list)
        print("=====================\n")
        print("the coefficient list is below", coeff_list)
        self.weights = [weight_list, inter_list, coeff_list]
    
    def opt_network_demand(self):
        
        if choice == 0:
            total_cost = lambda price: np.sum([(self.cost[i] - 10*price[i] - lower_price)/(self.weights[2][i]*
                                           (1 + math.exp(-np.dot(price, self.weights[0][i]) - self.weights[1][i]))) for i in range(self.product_number)])
        else:
            total_cost = lambda price: np.sum([(self.cost[i] - 50*price[i] - lower_price)/(self.weights[2][i]*
                                           (1 + math.exp(-np.dot(price, self.weights[0][i]) - self.weights[1][i]))) for i in range(self.product_number)])

        initial_price = np.ones(self.product_number)*(upper_price - lower_price)/50
        if choice == 0:
            bounds = tuple([(x,y) for x,y in zip(np.zeros(self.product_number), np.ones(self.product_number)*(upper_price - lower_price)/10)])
        elif choice == 1:
            bounds = tuple([(x,y) for x,y in zip(np.zeros(self.product_number), np.ones(self.product_number)*(upper_price - lower_price)/50)])
        res = minimize(total_cost, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
        print(res)
        return res.x
    
    #fit the real data in the model
    def train_network_demand_fit(self, df_fit_price, df_fit_share):
        fit_price = np.array(df_fit_price)
        fit_share = np.array(df_fit_share)
        mse = 0
        fit_number = len(fit_price)
        #evaluation the created error
        for k in range(fit_number):
            true_R = np.dot(fit_price[k] - self.cost, fit_share[k])
            #fit_price[k] = (fit_price[k] - 50)/50
            if choice == 0:
                predict_R = np.sum([(fit_price[k][i] - self.cost[i])/(self.weights[2][i]*
                                               (1 + math.exp(-np.dot((fit_price[k] - lower_price)/10, self.weights[0][i]) - self.weights[1][i]))) for i in range(self.product_number)])

            elif choice == 1:
                predict_R = np.sum([(fit_price[k][i] - self.cost[i])/(self.weights[2][i]*
                                               (1 + math.exp(-np.dot((fit_price[k] - lower_price)/50, self.weights[0][i]) - self.weights[1][i]))) for i in range(self.product_number)])
            #print(true_R, predict_R)
            mse = mse + (true_R - predict_R)**2
        return math.sqrt(mse/fit_number)      
    
    def demand_estimate(self):
        
    #estimate each product's demand in the original dataset.
    
        demand_estimate = []
        for t in range(len(self.df_price)):
            demand_list = np.zeros(self.product_number)
            for i in range(self.product_number):
                if choice == 1:
                    demand_list[i] = 1/(self.weights[2][i]*(1 + math.exp(-np.dot((self.df_price[t] - lower_price)/50, self.weights[0][i]) - self.weights[1][i]))) 
                elif choice == 0:
                    demand_list[i] = 1/(self.weights[2][i]*(1 + math.exp(-np.dot((self.df_price[t] - lower_price)/10, self.weights[0][i]) - self.weights[1][i]))) 
            demand_estimate.append(demand_list)
        return np.array(demand_estimate)    
            