import numpy as np
import math
choice = 1

#a simple case
if choice == 0:
    #basic params setting
    product_number = 2
    train_fit_number = 20*product_number#test for fitting the revenue function
    train_number = int(5*product_number)
    rolling_number = 5*product_number
    test_number = 20 #test for best price
    lower_price = 2
    upper_price = 10
    lower_cost = 0
    upper_cost = 0
    noise_var = 0.00001

#a more complex case
## initial parameters for multiproducts
elif choice == 1:
    lower_price = 50
    upper_price = 100
    lower_cost = 10
    upper_cost = 20
    noise_var = 0.004
    
    product_number = 10
    train_number = 10*product_number
    train_fit_number = train_number + 50#test for fitting the revenue function

    test_number = 20#test for best price

def revenue(price):
    base_p = np.array([1, 2.5])
    var_p = np.array([0.6, 0.3])
    return np.sum([price[i]*math.exp(base_p[i] - var_p[i]*price[i])/(1 + np.sum([math.exp(base_p[j] - var_p[j]*price[j]) 
            for j in range(len(price))])) for i in range(len(price))])
    
def best_revenue():
    best_revenue = 0
    max_i = 0
    max_j = 0
    for i in range(0, 100):
        print(i)
        for j in range(0, 100):
            current_revenue = revenue([0.1*i, 0.1*j])
            if current_revenue > best_revenue:
                best_revenue = current_revenue
                max_i = i
                max_j = j
    return best_revenue, max_i, max_j