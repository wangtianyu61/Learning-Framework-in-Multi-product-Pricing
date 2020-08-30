# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:21:42 2020

@author: wangt
"""
import math
import numpy as np
import pandas as pd
import csv

#linear elatiscity model
class simu_linear_demand:
    
    def __init__(self, product_number, train_number, lower_price, upper_price, lower_cost, upper_cost, noise_var):
        self.csv_name_share = "test_share.csv"
        self.csv_name_price = "test_price.csv"
        self.product_number = product_number
        self.train_number = train_number
        self.elasticity = 0
        self.lower_price = lower_price
        self.lower_cost = lower_cost
        self.upper_price = upper_price
        self.upper_cost = upper_cost
        self.noise_var = noise_var
        if self.lower_cost == self.upper_cost:
            self.cost = np.zeros(self.product_number)
        else:
            self.cost = np.random.randint(low = self.lower_cost, high = self.upper_cost, size = self.product_number)
        
        #for the use of linear elasticity model
        self.base_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)

        self.base_share = np.random.rand(product_number)
        self.base_share = self.base_share / np.sum(self.base_share) - 1/(self.product_number*self.product_number)
        self.base_share = np.array([max(a, 0) for a in list(self.base_share)])
        
    def data_generate(self):
        csvFile_share = open(self.csv_name_share,'a',newline = "")
        writer_share = csv.writer(csvFile_share)
        csvFile_price = open(self.csv_name_price,'a',newline = "")
        writer_price = csv.writer(csvFile_price)
        price_head = ["price " + str(i + 1) for i in range(self.product_number)]
        share_head = ["share " + str(i + 1) for i in range(self.product_number)]
        writer_price.writerow(price_head)
        writer_share.writerow(share_head)
        if self.product_number != 2:

            writer_share.writerow(self.base_share)
            writer_price.writerow(self.base_price)

    
        
    ## elasticity follows from uniform distribution from -0.5 to 0.5
    ## with diagonal follows uniform distribution from -2 to -1
            self.price_elasticity = np.random.rand(self.product_number, self.product_number)-0.5
            for i in range(self.product_number):
                self.price_elasticity[i][i] = np.random.rand(1) - 2
            
            for i in range(self.train_number - 1): 
                new_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)
                writer_price.writerow(new_price)
                delta_price = new_price - self.base_price
                new_demand = self.base_share.copy()
                #otherwise, it would send the address of the variable
                for j in range(self.product_number):
                    for k in range(self.product_number):
                        new_demand[j] = new_demand[j] + (self.base_share[j]/self.base_price[k])*self.price_elasticity[j][k]*delta_price[k]
                    #add the noise to the demand function
                    new_demand[j] = max(new_demand[j] + np.random.normal(0,self.noise_var), 0)
                    #print(new_demand)
                writer_share.writerow(new_demand)
        else:
            self.price_elasticity = np.array([[0.05, 0.01], [0.01, 0.04]])
            self.base_share = np.array([0.8, 0.92])
            for i in range(self.train_number - 1):
                new_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)
                writer_price.writerow(new_price)
                new_demand = np.zeros(self.product_number)
                for j in range(self.product_number):
                    new_demand[j] = self.base_share[j] - np.dot(self.price_elasticity[j], new_price) + np.random.normal(0, self.noise_var*500)
                writer_share.writerow(new_demand)
                
        csvFile_share.close()
        csvFile_price.close()
 


##linear elatsicity model test
    def test(self,best_price, test_number):

        revenue = np.zeros(test_number)
        
        delta_price = best_price - self.base_price
        new_demand = np.zeros((test_number, self.product_number))
        for i in range(test_number):
            if self.product_number != 2:
                new_demand[i] = self.base_share.copy()    
                #otherwise, it would send the address of the variable
                for j in range(self.product_number):
                    for k in range(self.product_number):
                        new_demand[i][j] = max(new_demand[i][j] + (self.base_share[j]/self.base_price[k])*self.price_elasticity[j][k]*delta_price[k] + np.random.normal(0, self.noise_var), 0)
                    #add the noise to the demand function
            else:
                for j in range(self.product_number):
                    new_demand[i][j] = self.base_share[j] - np.dot(self.price_elasticity[j], best_price) + np.random.normal(0, self.noise_var*500)
            #print('new demand is',new_demand)
            revenue[i] = np.dot(new_demand[i], best_price - self.cost) 
        #print(revenue)
        return np.mean(revenue), new_demand

##MNL Model
class simu_mnl:
    def __init__(self, product_number, train_number, lower_price, upper_price, lower_cost, upper_cost, noise_var):
        self.csv_name_share = "test_share.csv"
        self.csv_name_price = "test_price.csv"
        self.product_number = product_number
        self.train_number = train_number
        if upper_cost == lower_cost:
            self.cost = np.zeros(product_number)
        else:
            self.cost = np.random.randint(low = lower_cost, high = upper_cost, size = product_number)
        
        # a in MNL model
        self.base_share = np.random.rand(product_number)*1 + 2
        
        # b in MNL model
        self.price_elasticity = np.random.rand(product_number)/20 + 0.04
        ##fix param for 2 product
        if product_number == 2:
            self.base_share = np.array([1, 2.5])
            self.price_elasticity = np.array([0.6, 0.3])
        self.lower_price = lower_price
        self.lower_cost = lower_cost
        self.upper_price = upper_price
        self.upper_cost = upper_cost
        self.noise_var = noise_var
        
        
    def data_generate(self):
        price_head = ["price " + str(i + 1) for i in range(self.product_number)]
        share_head = ["share " + str(i + 1) for i in range(self.product_number)]
        csvFile_share = open(self.csv_name_share,'a',newline = "")
        writer_share = csv.writer(csvFile_share)
        csvFile_price = open(self.csv_name_price,'a',newline = "")
        writer_price = csv.writer(csvFile_price)
        writer_price.writerow(price_head)
        writer_share.writerow(share_head)
        
        for i in range(self.train_number):
            base_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)
            noise = np.random.normal(0, self.noise_var, self.product_number)
            market_value = [max(math.exp(self.base_share[i] - self.price_elasticity[i]*base_price[i]) + noise[i],0) for i in range(self.product_number)]
            new_demand = market_value / (1 + np.sum(market_value))
        
            writer_price.writerow(base_price)
            writer_share.writerow(new_demand)
        csvFile_share.close()
        csvFile_price.close()
    
    def test(self, best_price, test_number):
        revenue = np.zeros(test_number)
        new_demand = np.zeros((test_number, self.product_number))
        for i in range(test_number):
            noise = np.random.normal(0, self.noise_var, self.product_number)
            market_value = [max(math.exp(self.base_share[i] - self.price_elasticity[i]*best_price[i]) + noise[i],0) for i in range(self.product_number)]
            new_demand[i] = np.array(market_value / (1 + np.sum(market_value)))
            revenue[i] = np.dot(new_demand[i], best_price - self.cost) 
        #print(revenue)
        return np.mean(revenue), new_demand          


#mixed model in linear_elatsicity 
class simu_mix:
    def __init__(self, product_number, train_number, lower_price, upper_price, lower_cost, upper_cost, noise_var, mix_prop = 0.5):
        self.csv_name_share = "test_share.csv"
        self.csv_name_price = "test_price.csv"
        self.product_number = product_number
        self.train_number = train_number
        self.cost = np.random.randint(low = lower_cost, high = upper_cost, size = product_number)
    
        
        #elatisicity in linear_elatsicity model
        self.elasticity_1 = np.random.rand(self.product_number, self.product_number)*0.5 - 0.5
        self.elasticity_2 = np.random.rand(self.product_number, self.product_number)*0.5 
        for i in range(self.product_number):
            self.elasticity_1[i][i] = np.random.rand(1)*0.5 - 1.5
            self.elasticity_2[i][i] = np.random.rand(1)*0.5 - 2
        self.lower_price = lower_price
        self.lower_cost = lower_cost
        self.upper_price = upper_price
        self.upper_cost = upper_cost
        self.noise_var = noise_var
        #the noise var show the noise added to the linear elatsicity model
        self.mix_prop = mix_prop
        #show how these two models are mixed together, the value indicated the percentage of the model in linear elasticity
        #for the use of linear elasticity model
        self.base_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)

        self.base_share = np.random.rand(product_number)
        self.base_share = self.base_share / np.sum(self.base_share) - 1/(self.product_number*self.product_number)
        self.base_share = np.array([max(a, 0) for a in list(self.base_share)])

    def data_generate(self):
        csvFile_share = open(self.csv_name_share,'a',newline = "")
        writer_share = csv.writer(csvFile_share)
        csvFile_price = open(self.csv_name_price,'a',newline = "")
        writer_price = csv.writer(csvFile_price)
        price_head = ["price " + str(i + 1) for i in range(self.product_number)]
        share_head = ["share " + str(i + 1) for i in range(self.product_number)]
        writer_price.writerow(price_head)
        writer_share.writerow(share_head)
        writer_share.writerow(self.base_share)
        writer_price.writerow(self.base_price)

        for i in range(self.train_number - 1):
            new_price = np.random.randint(low = self.lower_price, high = self.upper_price, size = self.product_number)
            writer_price.writerow(new_price)
            delta_price = new_price - self.base_price
            new_demand = self.base_share.copy()
        #otherwise, it would send the address of the variable
            
            for j in range(self.product_number):
                for k in range(self.product_number):
                    if new_price[j] < (self.lower_price + self.upper_price)/2:
                        new_demand[j] = new_demand[j] + (self.base_share[j]/self.base_price[k])*self.elasticity_1[j][k]*delta_price[k]
                    #add the noise to the demand function
                    else:
                        new_demand[j] = new_demand[j] + (self.base_share[j]/self.base_price[k])*self.elasticity_2[j][k]*delta_price[k]
                    new_demand[j] = max(new_demand[j] + np.random.normal(0,self.noise_var), 0)
                #print(new_demand)
            writer_share.writerow(new_demand)
        csvFile_share.close()
        csvFile_price.close()
    #test the data with mnl
    ##linear elatsicity model test
    def test(self,best_price, test_number):

        revenue = np.zeros(test_number)
        delta_price = best_price - self.base_price
        new_demand = np.zeros((test_number, self.product_number))
        for i in range(test_number):
            new_demand[i] = self.base_share.copy()
        #otherwise, it would send the address of the variable
           
            for j in range(self.product_number):
                for k in range(self.product_number):
                    if best_price[j] < (self.lower_price + self.upper_price)/2:    
                        new_demand[i][j] = new_demand[i][j] + (self.base_share[j]/self.base_price[k])*self.elasticity_1[j][k]*delta_price[k]
                    else:
                        new_demand[i][j] = new_demand[i][j] + (self.base_share[j]/self.base_price[k])*self.elasticity_2[j][k]*delta_price[k]
                    #add the noise to the demand function
                    
                    new_demand[i][j] = max(new_demand[i][j] + np.random.normal(0,self.noise_var), 0)
            #print('new demand is',new_demand)
            
            revenue[i] = np.dot(new_demand[i], best_price - self.cost) 
        #print(revenue)
        return np.mean(revenue), new_demand  