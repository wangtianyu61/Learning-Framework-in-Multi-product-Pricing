# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:57:31 2020

@author: wangt
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize 
from gurobipy import *
from param import *
##estimate the demand using the linear_elatsicity model
class est_opt:
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
        
        self.cost = cost
        
    def mse_opt(self):
    #get the optimal elasticity estimated to fit the models
    #solved by gurobi
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
#        
#        #get the optimal values of beta
        self.elasticity = m.getAttr('x', beta).values()
        return np.array(self.elasticity)
    
    def mse_opt1(self):
    #solved by scipy
        lse = lambda beta: np.sum([np.sum([( np.sum([self.base_share[j]/self.base_price[i]*beta[j*self.product_number + i]*(self.df_price[t][i]-self.base_price[i]) for i in range(self.product_number)])
                                            + self.base_share[j] - self.df_share[t][j])* 
                                  (np.sum([self.base_share[j]/self.base_price[i]*beta[j*self.product_number + i]*(self.df_price[t][i]-self.base_price[i]) for i in range(self.product_number)]) 
                                            + self.base_share[j] - self.df_share[t][j]) for j in range(self.product_number)]) 
                                   for t in range(len(self.df_price))])
        initial_beta = np.zeros(self.product_number*self.product_number)
        res = minimize(lse, initial_beta, method = 'SLSQP', constraints = ())
        print(res)
    
    def mse_opt1_addopt(self, risk, revenue_predict, demand_predict, mix_choice):
        #we consider the result in end-to-end learning
        #mix_choice ==2 means using both end-to-end models
        #risk is a 2-d array
        #revenue_predict is for the relu fucntion and demand_predict is for the sigmoid function
        
        m = Model("mse_estimation_model")
        beta = m.addVars(self.product_number, self.product_number, lb = -2, ub = 2)
        obj = 0
        test_number = len(self.df_price)
        #original MSE error loss function
        for t in range(test_number):
            for j in range(self.product_number):
                obj_inner = 0
                for i in range(self.product_number):
                    
                    obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                obj = obj + (obj_inner + self.base_share[j] - self.df_share[t][j])* (obj_inner + self.base_share[j] - self.df_share[t][j])

        if mix_choice == 0 or mix_choice == 2:
        #we consider the results in ReLu function
            for t in range(test_number):
                revenue_LE = 0
                for j in range(self.product_number):
                    demand_inner = self.base_share[j]
                    for i in range(self.product_number):
                        demand_inner = demand_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i] - self.base_price[i])
                    revenue_LE = revenue_LE + demand_inner*(self.df_price[t][i] - self.cost[j])
                obj = obj + risk[0]*(revenue_LE - revenue_predict[t])*(revenue_LE - revenue_predict[t])
            
        
        #we consider in the sigmoid function
        elif mix_choice == 1 or mix_choice == 2:
            for t in range(test_number):
                for j in range(self.product_number):
                    obj_inner = 0
                    for i in range(self.product_number):
                    
                        obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                    obj = obj + risk[1]*(obj_inner + self.base_share[j] - demand_predict[t][j])* (obj_inner + self.base_share[j] - demand_predict[t][j])
            
            
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
#        
#        #get the optimal values of beta
        self.elasticity = m.getAttr('x', beta).values()
        return np.array(self.elasticity)
    
    
    def mse_opt1_justopt(self, revenue_predict, demand_predict, mix_choice):
    #an extreme case of the function above that is we just use the revenue predicted in the revenue predict.
        m = Model("mse_estimation_model")
        beta = m.addVars(self.product_number, self.product_number, lb = -2, ub = 2)
        obj = 0
        test_number = len(self.df_price)
    #stands for the function above when risk goes to infty.
        if mix_choice == 0:
            for t in range(test_number):
                revenue_LE = 0
                for j in range(self.product_number):
                    demand_inner = self.base_share[j]
                    for i in range(self.product_number):
                        demand_inner = demand_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i] - self.base_price[i])
                    revenue_LE = revenue_LE + demand_inner*(self.df_price[t][j] - self.cost[j])
                obj = obj + (revenue_LE - revenue_predict[t])*(revenue_LE - revenue_predict[t])
        #we consider in the sigmoid function
        elif mix_choice == 1:
            for t in range(test_number):
                for j in range(self.product_number):
                    obj_inner = 0
                    for i in range(self.product_number):
                    
                        obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                    obj = obj + (obj_inner + self.base_share[j] - demand_predict[t][j])* (obj_inner + self.base_share[j] - demand_predict[t][j])

            
            
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        print(m.ObjVal)
#        
#        #get the optimal values of beta
        self.elasticity = m.getAttr('x', beta).values()
        return np.array(self.elasticity)
    
    def revenue_opt(self, els): 
    #get the optimal revenue estimated by the linear elatsicity model
    #sometimes unconvex and sovled by scipy
        self.elasticity = els.reshape(self.product_number*self.product_number)
    #optimization target
        revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*opt_price[i] for i in range(self.product_number)]) + self.base_share[j])*
                            (self.base_price[j] + opt_price[j] - self.cost[j]) for j in range(self.product_number)])
        initial_price = self.base_price
    #bound as the price range from [50] to [100]
        bounds = tuple([(x,y) for x,y in zip(lower_price - self.base_price, upper_price - self.base_price)])
        
        
        res = minimize(revenue, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
    
        
        #get the optimal values of delta price
        print(res.x + self.base_price)
        return res.x + self.base_price
    #to maximize the revenue with the given strategy
    def revenue_opt_mix(self, els, robust_level, weights, mix_choice):
    #we consider when the optimizing this function, we add the constraint of other optimization targets.
    #if mix_choice == 2 we will use both functions for adaptions
    
    #here robust_level is 2d array. robust_level[0] for the ReLu fucntion and robust_level[1] for the Sigmoid function.
    
    #weights are parameters from the end-to-end models and weights[0] from the ReLu model; weights[1] from the Sigmoid model.
        self.elasticity = els.reshape(self.product_number*self.product_number)
        #optimization target
        if mix_choice == 0:
            revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*(opt_price[i] - self.base_price[i]) for i in range(self.product_number)]) + self.base_share[j])*
                                                     (opt_price[j] - self.cost[j]) for j in range(self.product_number)]) + robust_level[0]*np.sum([weights[0][2][i]*max(np.dot(weights[0][0][i], (opt_price - lower_price)/10) + weights[0][1][i], 0) 
                                                     for i in range(len(weights[0][2]))])
            
        elif mix_choice == 1:
            if choice == 1:
                revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*(opt_price[i] - self.base_price[i]) for i in range(self.product_number)]) + self.base_share[j])*
                            (opt_price[j] - self.cost[j]) for j in range(self.product_number)]) + robust_level[1]*np.sum([(self.cost[i] - 50*opt_price[i])/(weights[1][2][i]*
                                           (1 + math.exp(-np.dot((opt_price - lower_price*np.ones(self.product_number))/50, weights[1][0][i]) - weights[1][1][i]))) for i in range(self.product_number)])
            else:
                revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*(opt_price[i] - self.base_price[i]) for i in range(self.product_number)]) + self.base_share[j])*
                            (opt_price[j] - self.cost[j]) for j in range(self.product_number)]) + robust_level[1]*np.sum([(self.cost[i] - 50*opt_price[i])/(weights[1][2][i]*
                                           (1 + math.exp(-np.dot((opt_price - lower_price*np.ones(self.product_number))/10, weights[1][0][i]) - weights[1][1][i]))) for i in range(self.product_number)])
         
        elif mix_choice == 2:
            revenue = lambda opt_price: -np.sum([(np.sum([self.base_share[j]/self.base_price[i]*self.elasticity[j*self.product_number + i]*(opt_price[i] - self.base_price[i]) for i in range(self.product_number)]) + self.base_share[j])*
                            (opt_price[j] - self.cost[j]) for j in range(self.product_number)]) + robust_level[0]*np.sum([weights[0][2][i]*max(np.dot(weights[0][0][i], (opt_price - lower_price)/10) + weights[0][1][i], 0) 
                            for i in range(len(weights[0][2]))]) + robust_level[1]*np.sum([(self.cost[i] - opt_price[i])/(weights[1][2][i]*(1 + math.exp(-np.dot(opt_price, weights[1][0][i]) - weights[1][1][i]))) 
                            for i in range(self.product_number)])

        initial_price = self.base_price
        
        #bound as the price range from [50] to [100]
        bounds = tuple([(x,y) for x,y in zip(lower_price*np.ones(self.product_number), upper_price*np.ones(self.product_number))])
        
        
        res = minimize(revenue, initial_price, method = 'SLSQP', constraints= (), bounds=bounds) 
    
        
        #get the optimal values of delta price
        return res.x 
        return 0
        
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
            #print(true_R, predict_R)
            mse = mse + (true_R - predict_R)**2
        return math.sqrt(mse/fit_number)        
    
    def mse_readjust_alpha(self, els):
        beta = els.reshape(self.product_number, self.product_number)
        m = Model("mse_estimation_model")
        alpha = m.addVars(self.product_number, lb = 0, ub = 1)
        obj = 0
        test_number = len(self.df_price)
        
        #mse error loss function
        for t in range(test_number):
            for j in range(self.product_number):
                obj_inner = 0
                for i in range(self.product_number):
                    
                    obj_inner = obj_inner + self.base_share[j]/self.base_price[i]*beta[j, i]*(self.df_price[t][i]-self.base_price[i]) 
                obj = obj + alpha[j]*(obj_inner + self.base_share[j] - self.df_share[t][j])* (obj_inner + self.base_share[j] - self.df_share[t][j])

        m.addConstr(alpha.sum() == 1, 'budget')
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
#        
#        #get the optimal values of beta
        alpha_value = m.getAttr('x', alpha).values()
        return alpha_value