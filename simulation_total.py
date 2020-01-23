import pandas as pd
import autograd.numpy as np

from envelope_class import *

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


si_10 = np.array([[(i+0.5)/10,(j+0.5)/10] for i in range(10) for j in range(10)])
si_15 = np.array([[(i+0.5)/15,(j+0.5)/15] for i in range(15) for j in range(15)])
si_20 = np.array([[(i+0.5)/20,(j+0.5)/20] for i in range(20) for j in range(20)])

si_rand10 = np.array([(randint(0,10)/10,randint(0,10)/10) for i in range(10**2)])
si_rand15 = np.array([(randint(0,10)/10,randint(0,10)/10) for i in range(15**2)])
si_rand20 = np.array([(randint(0,10)/10,randint(0,10)/10) for i in range(20**2)])

si_dict = {"si_10":si_10,"si_15":si_15,"si_20":si_20,"si_10_rand":si_rand10,"si_15_rand":si_rand15,"si_20_rand":si_rand20}

theta_1st = np.array([0,0])
theta_2nd = np.array([1,0.5])
theta_3rd = np.array([1,0.5])

theta_dict = {"case1":theta_1st,"case2":theta_2nd,"case3":theta_3rd}

def run_simulation(si_name,theta_name):
    si,theta = si_dict[si_name],theta_dict[theta_name]
    
    X_list,Y_list = generate(si,theta)
    thershold = 1e-5
    u = 2
    case_length = len(si)
    
    err_raw_lr = list()
    err_raw_env = list()
    err_raw_spc_env = list()
    # for index in range(6,7):
    for index in range(case_length):
#         print("Start for {}th project".format(index))
        X_train,Y_train = np.delete(X_list,index,0),np.delete(Y_list,index,0)
        si_train = np.delete(si,index,0)
        X_pred,Y_pred = X_list[index].reshape(1,6),Y_list[index].reshape(1,5)
        err_lr = LinearRegression().fit(X_train,Y_train).predict(X_pred) - Y_pred
        err_raw_lr.append(err_lr)

        alpha_env,beta_env = envelope(X_train,Y_train,u)
        err_env = alpha_env + np.matmul(X_pred,beta_env) - Y_pred
        err_raw_env.append(err_env)

        theta_use = theta + np.random.uniform(0,1,2)
        alpha_spc_env,beta_spc_env = spatial_envelope(X_train,Y_train,si_train,theta_use,u,thershold)
        err_spc_env = alpha_spc_env + np.matmul(X_pred,beta_spc_env) - Y_pred 
        err_raw_spc_env.append(err_spc_env)
    
    err_lr_final = sum([np.sum(err_i**2) for err_i in err_raw_lr])/case_length
    err_env_final = sum([np.sum(err_i**2) for err_i in err_raw_env])/case_length
    err_spc_env_final = sum([np.sum(err_i**2) for err_i in err_raw_spc_env])/case_length
    
    return(si_name,theta_name,err_lr_final, err_env_final, err_spc_env_final)


si_theta_name_list = [(si,theta) for si in si_dict.keys() for theta in theta_dict.keys()]

res = list()
for si,theta in si_theta_name_list:
    res.append(run_simulation(si,theta))

out_df = pd.DataFrame(res).transpose()
out_df.columns = ["si","case","LR","ENV","SPEC_ENV"]
out_df.to_csv("res_total.csv")