# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:24:14 2020

@author: wangtianyu6162
"""

import numpy as np
import pandas as pd
import math
from gurobipy import *
from scipy.optimize import minimize 
from param import *
##estimate the demand using the MNL model
class est_opt2:
    best_adjustment = 0
    def __init__(self, df_price, df_share, base_price, base_share, cost):
        self.df_price = np.array(df_price)
        self.df_share = np.array(df_share)
        self.base_price = base_price[0]
        self.base_share = base_share[0]
        self.product_number = len(base_price[0])
        self.base_p = np.zeros(self.product_number)
        self.var_p = np.zeros(self.product_number)
        
        self.cost = cost
        
    def mle_opt(self):
    #get the optimal parameters to fit the MNL models
#        print(self.df_price)
    #MLE target function
        mle = lambda arg: -np.sum([np.sum([self.df_share[i][j]*(arg[j] - arg[j + self.product_number]*self.df_price[i][j] - 
                                          math.log(1 + np.sum([math.exp(arg[k] - arg[k + self.product_number]* self.df_price[i][k]) 
                                          for k in range(self.product_number)]))) for j in range(self.product_number)])
                                          for i in range(len(self.df_price))])
        max_bound = np.append(np.ones(self.product_number)*5, np.ones(self.product_number)/5)   
    #fit boundary for parameter of a and b
        bounds = tuple([(x,y) for x,y in zip(max_bound / 100, max_bound)])
        initial_arg = max_bound/10
        
        res = minimize(mle, initial_arg, method = 'SLSQP', constraints = (), bounds = bounds)
        self.base_p = res.x[0: self.product_number]
        self.var_p = res.x[self.product_number: 2*self.product_number]
        return [self.base_p, self.var_p]
    
    def mle_opt_real(self, base_p, var_p):
    #send the real parameters if it is really a MNL model
        self.base_p = base_p
        self.var_p = var_p
        
    def fixed_point_function(self, theta):
    #used for unconstraint MNL pricing model
        els = self.var_p
        print(self.base_p, els)
        return np.sum([math.exp(self.base_p[i] - els[i]*self.cost[i] - math.log(els[i]) - 1 - els[i]*theta)
                        for i in range(self.product_number)])
    def revenue_opt(self):
    #to maximize the revenue with the given model
    ## reference from Wang (2012)
    ## brutal binary search to find the fixed point
#        start_point = 0
#        error = 0.01
#        end_point = self.df_price.max() - self.cost.min() # the possible largest estimation
#        if self.fixed_point_function(end_point) > end_point:
#            print("failure")
#        else:
#            while end_point - start_point > error:
#                mid = (start_point + end_point)/2
#                if self.fixed_point_function(mid) < mid:
#                    end_point = mid
#                elif self.fixed_point_function(mid) > mid:
#                    start_point = mid
#                else:
#                    break
#        #steps of binary search
#        #find the optimal maximum revenue
#        best_price = mid + [self.cost[i] + 1/self.var_p[i] for i in range(self.product_number)]
        
        #just optimize brutely
        revenue = lambda price: -10*np.sum([(price[i] - self.cost[i])*math.exp(self.base_p[i] - self.var_p[i]*price[i])/(1 + np.sum([math.exp(self.base_p[j] - self.var_p[j]*price[j]) for j in range(self.product_number)])) for i in range(self.product_number)])
        initial_price = np.zeros(self.product_number) + lower_price
        bounds = tuple([(x,y) for x,y in zip(np.ones(self.product_number)*(upper_price - lower_price), np.ones(self.product_number)*upper_price)])
        res = minimize(revenue, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
        print(res)
        return res.x
    
    #fit the real data in the model
    def mnl_fit(self, df_fit_price, df_fit_share):
        fit_price = np.array(df_fit_price)
        fit_share = np.array(df_fit_share)
        mse = 0
        fit_number = len(fit_price)
        #evaluation the created error
        for k in range(fit_number):
            true_R = np.dot(fit_price[k] - self.cost, fit_share[k])
            predict_R = np.sum([(fit_price[k][i] - self.cost[i])*math.exp(self.base_p[i] - self.var_p[i]*fit_price[k][i])/(1 + np.sum([math.exp(self.base_p[j] - self.var_p[j]*fit_price[k][j]) 
                                for j in range(self.product_number)])) for i in range(self.product_number)])
            #print(true_R, predict_R)
            mse = mse + (true_R - predict_R)**2
        return mse/fit_number        

    
        