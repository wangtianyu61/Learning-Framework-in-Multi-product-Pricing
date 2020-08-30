# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:46:52 2020

@author: wangtianyu6162
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize 
from gurobipy import *
from param import *
##estimate the demand using the linear_elatsicity model
##and then alternate the directions for beta and x.
class task_est_opt:
    best_adjustment = 0
    def __init__(self, df_price, df_share, base_price, base_share, cost):
        #get the training data (price and share) from csv
        self.df_price = np.array(df_price)
        self.df_share = np.array(df_share)
        self.base_price = base_price[0]
        self.base_share = base_share[0]
        self.product_number = len(base_price[0])
        
        #the estimate parameters in this model
        self.elasticity = np.zeros((self.product_number*self.product_number))
        self.robust_target = 0
        self.cost = cost
    
    def initial_opt(self):
        m = Model("mse_estimation_model")
        beta = m.addVars(self.product_number, self.product_number, lb = -2, ub = 2)
        obj = 0
        test_number = len(self.df_price)
        
    #mse error loss function
        for t in range(test_number):
            for j in range(self.product_number):
                obj_inner = 0
                for i in range(self.product_number):
                    
                    obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                obj = obj + (obj_inner + self.base_share[j] - self.df_share[t][j])* (obj_inner + self.base_share[j] - self.df_share[t][j])

        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.robust_target = m.objVal/test_number
        #get the optimal values of beta
        self.elasticity = m.getAttr('x', beta).values()
        self.elasticity = np.array(self.elasticity)
    
    
    #task-based learning approach
    def ad_opt(self, robust_level, opt_sign = True):
        #opt_sign = True means max P max beta
        #opt sign = False means max P min beta
        self.robust_level = robust_level
        self.opt_sign = opt_sign
        self.initial_opt()
        for i in range(10):
            [adj_price, revenue_max1] = self.opt_price()
            revenue_max2 = self.opt_elasticity(adj_price)
            #print(revenue_max1, revenue_max2)
            if abs(revenue_max2 - revenue_max1) < 0.05:
                break
        self.elasticity = self.elasticity.reshape((self.product_number, self.product_number))
        return [adj_price + self.base_price, revenue_max2]
    def opt_price(self):
        self.elasticity = self.elasticity.reshape(self.product_number*self.product_number)
    #optimization target
        revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*opt_price[i] for i in range(self.product_number)]) + self.base_share[j])*
                            (self.base_price[j] + opt_price[j] - self.cost[j]) for j in range(self.product_number)])
        initial_price = self.base_price
    #bound as the price range from [50] to [100]
        bounds = tuple([(x,y) for x,y in zip(lower_price - self.base_price, upper_price - self.base_price)])
        
        
        res = minimize(revenue, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
    

        #get the optimal values of delta price
        #print(res.x + self.base_price)
        return [res.x, -res.fun]
    
    def opt_elasticity(self, adj_price):
        m = Model("mse_opt_elasticity")
        beta = m.addVars(self.product_number, self.product_number, lb = -2, ub = 2)
        obj = 0
        for j in range(self.product_number):
            obj_inner = self.base_share[j]
            for i in range(self.product_number):
                obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*adj_price[i]
            obj = obj + obj_inner*(self.base_price[j] + adj_price[j] - self.cost[j])
        if self.opt_sign == True:
            m.setObjective(obj, GRB.MAXIMIZE)
        else:
            m.setObjective(obj, GRB.MINIMIZE)
#        for t in range(len(self.df_price)):
#            constr_t = 0
#            for j in range(self.product_number):
#                constr_inner = 0
#                for i in range(self.product_number):                        
#                    constr_inner = constr_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
#                constr_t = constr_t + (constr_inner + self.base_share[j] - self.df_share[t][j])* (constr_inner + self.base_share[j] - self.df_share[t][j])
#            m.addConstr(constr_t <= self.robust_target*self.robust_level, " beta constraints")
#        
        obj = 0
        test_number = len(self.df_price)
        
    #mse error loss function
        for t in range(test_number):
            for j in range(self.product_number):
                obj_inner = 0
                for i in range(self.product_number):
                    
                    obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                obj = obj + (obj_inner + self.base_share[j] - self.df_share[t][j])* (obj_inner + self.base_share[j] - self.df_share[t][j])
        m.addConstr(obj <= self.robust_target*self.robust_level*test_number, 'beta constr')                    
    
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.elasticity = np.array(m.getAttr('x', beta).values())
        return m.objVal
        
    #robustness task-based learning approach
    def robust_ad_opt(self, loss_percent):
        min_robust_level = 1
        max_robust_level = 4
        min_require = (1 - loss_percent)*self.ad_opt(1, False)[1]
        print(min_require)
        for i in range(100):
            temp = (min_robust_level + max_robust_level)/2
            [price, revenue] = self.ad_opt(temp, False)
            if revenue >= min_require:
                min_robust_level = (min_robust_level + max_robust_level)/2
            else:
                max_robust_level = (min_robust_level + max_robust_level)/2
            if max_robust_level - min_robust_level <=0.1:
                print(max_robust_level)
                return price
    
    #fit the real data in the model
    def le_fit(self, df_fit_price, df_fit_share):
        fit_price = np.array(df_fit_price)
        fit_share = np.array(df_fit_share)
        mse = 0
        fit_number = len(fit_price)
        #evaluation the created error
        for k in range(fit_number):
            true_R = np.dot(fit_price[k] - self.cost, fit_share[k])
            predict_R = np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*(fit_price[k][i] - self.base_price[i]) for i in range(self.product_number)]) + self.base_share[j])*
                            (fit_price[k][j]- self.cost[j]) for j in range(self.product_number)])
            print(true_R, predict_R)
            mse = mse + (true_R - predict_R)**2
        return mse/fit_number      