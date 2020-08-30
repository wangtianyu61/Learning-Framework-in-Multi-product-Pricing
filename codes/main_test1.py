import pandas as pd
import numpy as np
import os
import math
import csv
from est_opt import *
from est_opt2 import *
from task_est_opt import *
from e2e_network import *
from ReLuNet import *
from SigmoidNet import *
from simulation import *
from param import *
from sklearn.kernel_ridge import KernelRidge
product_number = 2
train_fit_number = 100*product_number#test for fitting the revenue function
train_number = 30*product_number

test_number = 50#test for best price


#parameter readjustment for reverse engineering
risk = [0.1, 0.1] #for correcting the loss function
robust_level = [1, 0.5] #for correcting the optimal price & generate a more robust constraint.

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
    
    

def revenue_fit(df_price, df_share):
    revenue = np.zeros(len(df_price))
    for i in range(len(df_price)):
        revenue[i] = np.dot(list(df_share.loc[i]), list(df_price.loc[i]))
    return revenue

def grid_search(model):
    re_max = 0
    x_max = 0
    y_max = 0
    for i in range(0, 101, 1):
        for j in range(0, 101, 1):
            re_temp = model.predict(np.array([np.array([i/10, j/10])]))[0]
            if re_temp > re_max:
                re_max = re_temp
                x_max = i/10
                y_max = j/10
            #print(re_temp)
    return re_max, x_max, y_max
if __name__ == '__main__':
    csvFile = open('result_task2.csv','w',newline = "")
    writer_price = csv.writer(csvFile)
    writer_price.writerow(['train_num', 'ordinary', 'robustness', 'sigmoid', 'relu', 'loss_fun_mix1','loss_fun_mix2', 'revenue_mix1', 'revenue_mix2'])
    csvFile2 = open('result_task2_mse.csv', 'w', newline = '')
    writer_mse = csv.writer(csvFile2)
    writer_mse.writerow(['train_num', 'ordinary', 'sigmoid', 'relu', 'loss_fun_mix1', 'loss_fun_mix2'])
    for train_number in [10, 20, 30, 50]:
        for count in range(5):
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
    #    
    #    y_train = revenue_fit(df_price, df_share
    #    X_train = np.array(df_price)
    #    clf = KernelRidge(alpha = 1, kernel = 'rbf', gamma = 10)
    #    clf.fit(X_train, y_train)
    #    max_revenue, x, y = grid_search(clf)
    #
            base_share = np.array(df_share.iloc[0:1])
            base_price = np.array(df_price.iloc[0:1])
    #    
            model1 = est_opt(df_price, df_share, base_price, base_share, simu1.cost)
            est_elasticity = model1.mse_opt()
            best_price = model1.revenue_opt(est_elasticity)
            revenue1 = simu1.test(best_price, test_number)[0]
            print(np.mean(revenue1))
            mse1 = model1.le_fit(df_fit_price, df_fit_share)
    #    
    #    
    ##    model_mnl = est_opt2(df_price, df_share, base_price, base_share, simu1.cost)
    ##    model_mnl.mle_opt_real(simu1.base_share, simu1.price_elasticity)
    ##    best_best = model_mnl.revenue_opt()
    ##    revenue0 = simu1.test(best_best, test_number)[0]
    #    
            task_model = task_est_opt(df_price, df_share, base_price, base_share, simu1.cost)
            
#            best_price_t1 = task_model.ad_opt(2, False)[0]
#            revenue_task1 = simu1.test(best_price_t1, test_number)[0]
#    ##            print(best_price_t1, np.mean(revenue_task1))
#            best_price_t2 = task_model.ad_opt(2, True)[0]
#            revenue_task3 = simu1.test(best_price_t2, test_number)[0]
            best_price_t3 = task_model.robust_ad_opt(0.1)
            revenue_task2 = simu1.test(best_price_t3, test_number)[0]
    
            #print(np.mean(revenue1), np.mean(revenue_task1), np.mean(revenue_task2), np.mean(revenue_task3))
            model_ReLu = ReLuNet(df_price, df_share, base_price, base_share, simu1.cost)
            ##model2_version1: linear + relu + linear
            model_ReLu.train_network_revenue(df_fit_price, df_fit_share)
            price_nn = model_ReLu.opt_network_revenue()
            revenue4 = simu1.test(price_nn*10 + 2, test_number)[0]
            [mse4, revenue_predict_ReLu] = model_ReLu.opt_network_revenue_fit(df_fit_price, df_fit_share)
            [a, revenue_predict_ReLu] = model_ReLu.opt_network_revenue_fit(df_price, df_share)
            
            #model2_version2: linear + sigmoid
            model_Sigmoid = SigmoidNet(df_price, df_share, base_price, base_share, simu1.cost)
            model_Sigmoid.train_network_demand_whole()
            price_nn_demand = model_Sigmoid.opt_network_demand()
            revenue3 = simu1.test(price_nn_demand*10 + 2, test_number)[0]
            mse3 = model_Sigmoid.train_network_demand_fit(df_fit_price, df_fit_share)
            demand_predict_Sigmoid = model_Sigmoid.demand_estimate()
            
            
            weights = [model_ReLu.weights, model_Sigmoid.weights]
    
            est_elasticity_1 = model1.mse_opt1_addopt(risk, revenue_predict_ReLu, np.nan, 0)
            best_price_adj = model1.revenue_opt(est_elasticity_1)
            revenue1_adj = simu1.test(best_price_adj, test_number)[0]
            mse1_adj = model1.le_fit(df_fit_price, df_fit_share)
                
            best_price_adjj = model1.revenue_opt_mix(est_elasticity, robust_level, weights, 0)
            revenue1_adjj = simu1.test(best_price_adjj, test_number)[0]
    
            est_elasticity_1 = model1.mse_opt1_addopt(risk, np.nan, demand_predict_Sigmoid, 1)
            best_price_adj3 = model1.revenue_opt(est_elasticity_1)
            revenue1_adj3 = simu1.test(best_price_adj3, test_number)[0]
            mse1_adj3 = model1.le_fit(df_fit_price, df_fit_share)
                
            best_price_adjj2 = model1.revenue_opt_mix(est_elasticity, robust_level, weights, 1)
            revenue1_adjj2 = simu1.test(best_price_adjj2, test_number)[0]
    
        
            
            
            revenue_list = [train_number, np.mean(revenue1), np.mean(revenue_task2), np.mean(revenue3), np.mean(revenue4), np.mean(revenue1_adj), np.mean(revenue1_adj3), np.mean(revenue1_adjj), np.mean(revenue1_adjj2)]
            writer_price.writerow(revenue_list)
            mse_list = [train_number, mse1, mse3, mse4, mse1_adj, mse1_adj3]
            writer_mse.writerow(mse_list)
    csvFile.close()
    csvFile2.close()