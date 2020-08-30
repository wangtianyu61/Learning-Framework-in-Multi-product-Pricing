# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:50:18 2020

@author: wangt
"""

import pandas as pd
import numpy as np
import os
import math
from est_opt import *
from est_opt2 import *
from task_est_opt import *
from e2e_network import *
from ReLuNet import *
from SigmoidNet import *
from simulation import *
from param import *




#parameter readjustment for reverse engineering
risk = [0.00001, 1] #for correcting the loss function
robust_level = [10, 0.1] #for correcting the optimal price & generate a more robust constraint.

#simulation and generate data
choice = 0
if choice == 0:
    simu1 = simu_linear_demand(product_number, train_fit_number, lower_price, upper_price, 
                               lower_cost, upper_cost, noise_var)
elif choice == 1:
    simu1 = simu_mnl(product_number, train_fit_number, lower_price, upper_price, 
                       lower_cost, upper_cost, noise_var*10)
else:
    simu1 = simu_mix(product_number, train_fit_number, lower_price, upper_price,
                       lower_cost, upper_cost, noise_var, 0.9)
    

if __name__ == '__main__':
    policy_rev = np.zeros((4, 10))
    policy_mse = np.zeros((4, 5))
    policy_rev_add = np.zeros((4, 4))
    choice = 1
    for train_number in [50, 100, 150]:
        train_fit_number = train_number + 50
        
        for i in range(10):
            simu1 = simu_linear_demand(product_number, train_fit_number, lower_price, upper_price,
                       lower_cost, upper_cost, noise_var)
            simu1.data_generate()
            #read the generated data and split the train and test number dataset
            df_share = pd.read_csv("test_share.csv")
            df_fit_share = df_share.iloc[train_number:train_fit_number]
            df_share = df_share.iloc[0:train_number]
        
            df_price = pd.read_csv("test_price.csv")
            df_fit_price = df_price.iloc[train_number:train_fit_number]
            df_price = df_price.iloc[0:train_number]
        
            os.remove(simu1.csv_name_share)
            os.remove(simu1.csv_name_price)
            base_share = np.array(df_share.iloc[0:1])
            base_price = np.array(df_price.iloc[0:1])
            #print(df_share)
            model1 = est_opt(df_price, df_share, base_price, base_share, simu1.cost)
            est_elasticity = model1.mse_opt()
            best_price = model1.revenue_opt(est_elasticity)
            revenue1 = simu1.test(best_price, test_number)[0]
            
            task_model = task_est_opt(df_price, df_share, base_price, base_share, simu1.cost)
            best_price = task_model.ad_opt(1.2, False)[0]
            
            revenue_task1 = simu1.test(best_price, test_number)[0]
            
            best_price = task_model.ad_opt(1.2, True)[0]
            
            revenue_task2 = simu1.test(best_price, test_number)[0]
            ##
            
            best_price_t3 = task_model.robust_ad_opt(0.1)
            revenue_task3 = simu1.test(best_price_t3, test_number)[0]
            
            #policy_rev_add[int(train_number/50) - 1] += np.array([revenue1, revenue_task1, revenue_task2, revenue_task3])/10
            
                           ### our model: whole framework of end-to-end learning
            model_ReLu = ReLuNet(df_price, df_share, base_price, base_share, simu1.cost)
            ##model2_version1: linear + relu + linear
            model_ReLu.train_network_revenue(df_fit_price, df_fit_share)
            price_nn = model_ReLu.opt_network_revenue()
            revenue4 = simu1.test(price_nn*10 + 50, test_number)[0]
            [mse4, revenue_predict_ReLu] = model_ReLu.opt_network_revenue_fit(df_fit_price, df_fit_share)
            [a, revenue_predict_ReLu] = model_ReLu.opt_network_revenue_fit(df_price, df_share)
        
            #model2_version2: linear + sigmoid
            model_Sigmoid = SigmoidNet(df_price, df_share, base_price, base_share, simu1.cost)
            model_Sigmoid.train_network_demand_whole()
            price_nn_demand = model_Sigmoid.opt_network_demand()
            revenue3 = simu1.test(price_nn_demand*50 + 50, test_number)[0]
            mse3 = model_Sigmoid.train_network_demand_fit(df_fit_price, df_fit_share)
            demand_predict_Sigmoid = model_Sigmoid.demand_estimate()
            #
            #
            #
            weights = [model_ReLu.weights, model_Sigmoid.weights]
            #
            #
            #### model1: use linear_elatsicity model to fit and then optimize (benchmark)
            model1 = est_opt(df_price, df_share, base_price, base_share, simu1.cost)
            est_elasticity = model1.mse_opt()
            best_price = model1.revenue_opt(est_elasticity)
            revenue1 = simu1.test(best_price, test_number)[0]
            
            mse1 = model1.le_fit(df_fit_price, df_fit_share)
            #
            #
            ###Type 1 ---ADD THE RESULT OF E2E LEARNING (ReLuNet)--------------------##
            #
            #
            #######################Q: the mse error is too large.########################
            est_elasticity_1 = model1.mse_opt1_addopt(risk, revenue_predict_ReLu, np.nan, 0)
            best_price_adj = model1.revenue_opt(est_elasticity_1)
            revenue1_adj = simu1.test(best_price_adj, test_number)[0]
            mse1_adj = model1.le_fit(df_fit_price, df_fit_share)
            
            best_price_adjj = model1.revenue_opt_mix(est_elasticity, robust_level, weights, 0)
            revenue1_adjj = simu1.test(best_price_adjj, test_number)[0]
            ##
            ##
            ###just use the result of E2E LEARNING
#            est_elasticity_2 = model1.mse_opt1_justopt(revenue_predict_ReLu,  demand_predict_Sigmoid, 0)
#            best_price_adj2 = model1.revenue_opt(est_elasticity_2)
#            revenue1_adj2 = simu1.test(best_price_adj2, test_number)[0]
#            mse1_adj2 = model1.le_fit(df_fit_price, df_fit_share)
            #
            
            ##Type 2 ---ADD THE RESULT OF E2E LEARNING (SigmoidNet)--------------------##
            est_elasticity_1 = model1.mse_opt1_addopt(risk, np.nan, demand_predict_Sigmoid, 1)
            best_price_adj3 = model1.revenue_opt(est_elasticity_1)
            revenue1_adj3 = simu1.test(best_price_adj3, test_number)[0]
            mse1_adj3 = model1.le_fit(df_fit_price, df_fit_share)
            
            best_price_adjj2 = model1.revenue_opt_mix(est_elasticity, robust_level, weights, 1)
            revenue1_adjj2 = simu1.test(best_price_adjj2, test_number)[0]
            #
            #
            ##just use the result of E2E LEARNING
#            est_elasticity_3 = model1.mse_opt1_justopt(revenue_predict_ReLu, demand_predict_Sigmoid, 1)
#            best_price_adj4 = model1.revenue_opt(est_elasticity_3)
#            revenue1_adj4 = simu1.test(best_price_adj4, test_number)[0]
#            mse1_adj4 = model1.le_fit(df_fit_price, df_fit_share)
            #
            #
            #
            ###Type 3 ---ADD THE RESULT OF E2E LEARNING (ReLuNet & SigmoidNet)--------------------##
        #    est_elasticity_1 = model1.mse_opt1_addopt(risk, revenue_predict_ReLu, demand_predict_Sigmoid, 2)
        #    best_price_adj5 = model1.revenue_opt(est_elasticity_1)
        #    revenue1_adj5 = simu1.test(best_price_adj5, test_number)[0]
        #    mse1_adj5 = model1.le_fit(df_fit_price, df_fit_share)
        #    ##
        #    best_price_adjj3 = model1.revenue_opt_mix(est_elasticity, robust_level, weights, 2)
        #    revenue1_adjj3 = simu1.test(best_price_adjj3, test_number)
            #
            #
            #
            #####its best version
            ######best_best = model1.revenue_opt(simu1.elasticity)
            #####revenue0 = simu1.test(best_best, test_number)[0]
            #####print(best_price)
            ##
            ###### model2: use mnl model to fit and then optimize (benchmark)
            ##model_mnl = est_opt2(df_price, df_share, base_price, base_share, simu1.cost)
            ##model_mnl.mle_opt()
            ##best_price = model_mnl.revenue_opt()
            ##revenue2 = simu1.test(best_price, test_number)[0]
            ##mse2 = model_mnl.mnl_fit(df_fit_price, df_fit_share)
            ##
            ##
            ######its best version
            ###model_mnl.mle_opt_real(simu1.base_share, simu1.price_elasticity)
            ###best_best = model_mnl.revenue_opt()
            ###revenue0 = simu1.test(best_best, test_number)[0]
            ###mse0 = model_mnl.mnl_fit(df_fit_price, df_fit_share)
            ##
            #
            #
            ##Result Table
#            print("For revenue part\n================\n")
#            print("best price", np.max(model_ReLu.revenue))
#            print("Linear Elasticity Model:", np.mean(revenue1))
#            print("ReLuNet: " + str(np.mean(revenue4)) + "  SigmoidNet: " + str(np.mean(revenue3)))
#            print("Demand Parameter adj (linear&relu) : " + str(np.mean(revenue1_adj)))
#            #print("Demand Parameter adj (relu) : " + str(np.mean(revenue1_adj2)))
#            print("Demand Parameter adj (linear&sigmoid) : " + str(np.mean(revenue1_adj3)))
#            #print("Demand Parameter adj (sigmoid) : " + str(np.mean(revenue1_adj4)))
#            print("Revenue func adj (linear&relu) : " + str(np.mean(revenue1_adjj)))
#            print("Revenue func adj (linear&sigmoid) : " + str(np.mean(revenue1_adjj2)))
            print("=============================\n")
            
            policy_rev[int(train_number/50) - 1] += np.array([revenue1, revenue_task1, revenue_task2, revenue_task3, revenue4, revenue3, revenue1_adj, revenue1_adj3, revenue1_adjj, revenue1_adjj2])
            
            print("For MSE part\n==================\n")
            print("Linear Elasticity Model:", math.sqrt(mse1))
            print("ReLuNet: " + str(math.sqrt(mse4)) + "  SigmoidNet: " + str(math.sqrt(mse3)))
            print("Demand Parameter adj (linear&relu) : " + str(math.sqrt(mse1_adj)))
            #print("Demand Parameter adj (relu) : " + str(math.sqrt(mse1_adj2)))
            print("Demand Parameter adj (linear&sigmoid) : " + str(math.sqrt(mse1_adj3)))
            #print("Demand Parameter adj (sigmoid) : " + str(math.sqrt(mse1_adj4)))
            policy_mse[int(train_number/50) - 1] += np.array([mse1, mse4, mse3, mse1_adj, mse1_adj3])

    df = pd.DataFrame(policy_rev)
    df.to_csv(str(product_number) + '_1.csv')

    df2 = pd.DataFrame(policy_mse)
    df2.to_csv(str(product_number) +'_rmse_' + '_1.csv')
#            