# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:12:34 2020

@author: wangt
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


class e2e_network:
    
    def __init__(self, df_price, df_share, base_price, base_share, cost):
        self.df_price = np.array(df_price)
        self.df_share = np.array(df_share)
        self.base_price = base_price
        self.base_share = base_share
        self.product_number = len(base_price[0])
        self.elasticity = np.zeros((self.product_number*self.product_number))
        self.cost = cost
        self.weights = 0
        self.revenue = np.zeros(len(self.df_price))
        self.transform()
    def transform(self):

        for i in range(len(self.df_price)):
            self.revenue[i] = np.dot(self.df_price[i] - np.array(self.cost), self.df_share[i])
    
#-----------------------------------------------#
#convex optimization problem#
    def train_network_convex(self):
        x_train = (self.df_price - 50)/100
        y_train = 90 - self.revenue
        
        model = Sequential()
        
        model.add(Dense(units = 100, input_dim = self.product_number))
        model.add(Activation("sigmoid"))
        
        #add the constraint: kernel_constraint =non_neg() to ensure its convexity
        model.add(Dense(units = 1, kernel_constraint = non_neg()))
        model.add(Activation("linear"))
        model.summary()
        model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
        model.fit(x_train, y_train, validation_split = 0.2, epochs = 300, batch_size = 32)        
        
        price_new = (np.random.randn(10,10)*10 + 25)/100
        
#        s = model.predict(price_new)
#        print(s)
        self.weights = np.array(model.get_weights())
        self.weights[0] = self.weights[0].T
        price = np.ones(self.product_number)
        s = np.array([self.weights[2][i][0] for i in range(len(self.weights[2]))])
        
#        print('aij',self.weights[0])
#        print('bi',self.weights[1])
#        print('ci', self.weights[2])
#        print('d', self.weights[3])
#        print(len(self.weights[2]))
#        
#        true_value = np.dot(np.array([1/(1 + math.exp(-np.dot(self.weights[0][i], price) - self.weights[1][i])) 
#                                for i in range(len(self.weights[2]))]), s) + self.weights[3][0]         
#        print(true_value, model.predict(np.array([price])))
    #directly train the error of revenues
    #compute the gradient of the approximate function
    def nn_gradient(self):
        
        gradient = lambda price: np.array([np.sum([self.weights[2][j][0]*(self.weights[0][j][i]*np.sum(math.exp(-np.dot(self.weights[0][j], price) - self.weights[1][j]))/
                                            (1 + np.sum(math.exp(-np.dot(self.weights[0][j], price) - self.weights[1][j])))**2) for j in range(len(self.weights[2]))]) for i in range(self.product_number)])
        return gradient
    
  
        
    # solve the optimization 
    def opt_network_convex(self):

        weight_2 = np.array([self.weights[2][i][0] for i in range(len(self.weights[2]))])
        
        
        cost = lambda price: np.dot(np.array([1/(1 + math.exp(-np.dot(self.weights[0][i], price) - self.weights[1][i])) 
                                for i in range(len(self.weights[2]))]), weight_2) + self.weights[3][0]         

        
        initial_price = np.zeros(self.product_number) +50
        bounds = tuple([(x,y) for x,y in zip(np.zeros(self.product_number), np.ones(self.product_number)*50)])
        
        res = minimize(cost, initial_price, method = 'SLSQP', jac = self.nn_gradient(), constraints= (), bounds=bounds) 
        #print(res)
        #print(gradient)
    
    def scheduler(self, epoch, reduce_level = 0.8):
        # every finite number of epaches, we reduce the learning rate to the reduce_level as before.
        if epoch % 500 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * reduce_level)
            print("lr changed to {}".format(lr*reduce_level))
        return K.get_value(model.optimizer.lr)
    


    #---------------directly train demand use softmax--------------------
            
    