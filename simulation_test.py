import pandas as pd
import autograd.numpy as np

from envelope_class import *

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


si_4 = np.array([[(i+0.5)/4,(j+0.5)/4] for i in range(4) for j in range(4)])
si = si_4
theta = np.array([0.5,1])
X_list, Y_list = generate(si,theta,seed = 2020)
thershold = 1e-5
u = 2

err_raw_lr = list()
err_raw_env = list()
err_raw_spc_env = list()
# for index in range(6,7):
for index in range(16):
    print("Start for {}th project".format(index))
    X_train,Y_train = np.delete(X_list,index,0),np.delete(Y_list,index,0)
    si_train = np.delete(si,index,0)
    X_pred,Y_pred = X_list[index].reshape(1,6),Y_list[index].reshape(1,5)
    err_lr = LinearRegression().fit(X_train,Y_train).predict(X_pred) - Y_pred
    err_raw_lr.append(err_lr)

    alpha_env,beta_env = envelope(X_train,Y_train,u)
    err_env = alpha_env + np.matmul(X_pred,beta_env) - Y_pred
    err_raw_env.append(err_env)

    alpha_spc_env,beta_spc_env = spatial_envelope(X_train,Y_train,si_train,theta,u,thershold)
    err_spc_env = alpha_spc_env + np.matmul(X_pred,beta_spc_env) - Y_pred 
    err_raw_spc_env.append(err_spc_env)

n = len(X_list)
err_lr_final = sum([np.sum(err_i**2) for err_i in err_raw_lr])/n
err_env_final = sum([np.sum(err_i**2) for err_i in err_raw_env])/n
err_spc_env_final = sum([np.sum(err_i**2) for err_i in err_raw_spc_env])/n

pd.DataFrame(list(err_lr_final,err_env_final,err_spc_env_final)).to_csv('test_result.csv')