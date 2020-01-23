# import numpy as np
import autograd.numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern as skMatern
from sklearn.decomposition import PCA

from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from scipy.optimize import minimize
from scipy.linalg import sqrtm
from random import randint

import warnings
warnings.filterwarnings("ignore")


def generate(si,theta,seed = 0):
    np.random.seed(seed)
    n,p,r = len(si),6,5
    X = np.random.normal(0,1,n*p).reshape(n,p)
    X_kar = np.kron(np.eye(r),X)
    beta_star = np.random.uniform(0,1,p*r).reshape(p*r,1)

    A = np.random.uniform(0,1,25).reshape(5,5)
    Gamma = PCA().fit(A).components_
    Gamma1 = Gamma[:,:2]
    Gamma0 = Gamma[:,2:]
    out1,out0 = list(),list()
    for i in range(2):
        out1.append([(-.9)**abs(i - j) for j in range(2)])
    for i in range(3):
        out0.append([(-.5)**abs(i - j) for j in range(3)])
    Omega1 = np.array(out1)
    Omega0 = np.array(out0)
    Sigma = np.matmul(np.matmul(Gamma1,Omega1),Gamma1.T) + np.matmul(np.matmul(Gamma0,Omega0),Gamma0.T)
        
    if (theta[0] == 0) & (theta[1] == 0): 
        h = np.eye(len(si))
    else:
        h= np.array(rho(si,theta))

    Sigma_kr = np.kron(Sigma,h)
    Err_kr = np.random.multivariate_normal(np.repeat(0,n*r),Sigma_kr).reshape(n*r,1)
    Y_kr = np.matmul(X_kar,beta_star) + np.array(Err_kr)
    Y = Y_kr.reshape(n,r)
    return(X,Y)

def envelope(X_env,Y_env,u):

    p,r = X_env.shape[1],Y_env.shape[1]
    linear_model = LinearRegression().fit(X_env,Y_env)
    err = Y_env - linear_model.predict(X_env)
    Sigma_res = np.cov(err.transpose())
    Sigma_Y = np.cov(Y_env.transpose())

    def cost(Gamma):
        X = np.matmul(Gamma,Gamma.T)
        out = -np.log(np.linalg.det(np.matmul(np.matmul(X,Sigma_res),X) + np.matmul(np.matmul(np.eye(r) - X,Sigma_Y),np.eye(r) - X)))
        return(np.array(out))

    manifold = Grassmann(r,u)
    # manifold = Stiefel(r, u)
    problem = Problem(manifold=manifold, cost=cost,verbosity=0)
    solver = SteepestDescent()
    Gamma = solver.solve(problem)
    PSigma1_hat = np.matmul(Gamma,Gamma.T)
    PSigma2_hat = np.eye(r) - PSigma1_hat

    beta_hat = np.matmul(PSigma1_hat,linear_model.coef_)
    Sigma1_hat = np.matmul(np.matmul(PSigma1_hat,Sigma_res),PSigma1_hat)
    Sigma2_hat = np.matmul(np.matmul(np.eye(r) - PSigma1_hat,Sigma_res),np.eye(r) - PSigma1_hat)
    alpha_hat = np.mean(Y_env - np.matmul(X_env,beta_hat.T),axis = 0)
     
    return(alpha_hat.reshape(1,r),beta_hat.reshape(p,r))

def rho(loc,theta):
    K = skMatern(nu=abs(theta[1]), length_scale=abs(theta[0]))
    covMat = K(loc)
    return(covMat)

def spatial_envelope(X_env,Y_env,si,theta,u,thershold):
    #const:
    #n,p,r
    #H,G
    #beta_MLE
    #si: matrix of loacation
    n,p,r = X_env.shape[0],X_env.shape[1],Y_env.shape[1]
    H = Y_env - np.kron(np.mean(Y_env,axis = 0).reshape(1,r),np.repeat(1,n).reshape(n,1))
    G = X_env - np.kron(np.mean(X_env,axis = 0).reshape(1,p),np.repeat(1,n).reshape(n,1))
    
    linear_model = LinearRegression().fit(X_env,Y_env)
    err = Y_env - linear_model.predict(X_env)
    beta_MLE = linear_model.coef_

    # changable
    # judge, iter_count, err_count, theta
    # V0_hat, V1_hat, PV0_hat, PV1_hat
    # Sigma_Y, Sigma_res
    # rho_h_theta
    Sigma_res = np.cov(err.T)
    Sigma_Y = np.cov(Y_env.T)

    judge = True
    iter_count = 0
    cal_thershold,min_thershold = thershold,thershold
    prot = 1
    err_count = 0
    while judge:
        try:
            if err_count > 10:
                print("Start new")
                theta = np.random.uniform(0,0.5,2)
                err_count = 0
            # Step1: Optimize on Gamma to get V0,V1,PV0,PV1
            def cost(Gamma):
                X = np.matmul(Gamma,Gamma.T)
                out = -np.log(np.linalg.det(np.matmul(np.matmul(X,Sigma_res),X) + np.matmul(np.matmul(np.eye(r) - X,Sigma_Y),np.eye(r) - X)))
                return(np.array(out))

            manifold = Grassmann(r,u)
            #manifold = Stiefel(r,u)
            problem = Problem(manifold=manifold, cost=cost,verbosity=0)
            solver = SteepestDescent()
            Gamma = solver.solve(problem)
            PV1_hat = np.matmul(Gamma,Gamma.T)
            PV0_hat = np.eye(r) - PV1_hat

            V1_hat = np.matmul(np.matmul(PV1_hat,Sigma_Y),PV1_hat)
            V0_hat = np.matmul(np.matmul(PV1_hat,Sigma_res),PV1_hat)    
            # Step2: Optimize on theta
            def theta_fun(theta):
                rho_h_theta= np.array(rho(si,theta))
                item1 = np.matmul(sqrtm(np.linalg.inv(rho_h_theta).real).real,G)
                project = lambda x: np.eye(n) - np.matmul(np.matmul(x,np.linalg.inv(np.matmul(x.T,x)).real),x.T)
                item2 = np.matmul(np.matmul(project(item1),sqrtm(np.linalg.inv(rho_h_theta).real).real),H)
                item3 = np.matmul(np.matmul(item2,np.linalg.pinv(V1_hat).real),item2.T)
                item4 = np.matmul(sqrtm(np.linalg.inv(rho_h_theta).real).real,H)
                item5 = np.matmul(np.matmul(item4,np.linalg.pinv(V0_hat).real),item4.T)
                loss = r*np.linalg.det(rho_h_theta) + 0.5 * np.trace(item3 + item5)
                return(loss)
#             print("Theta: {}".format(theta))
            opt_res = minimize(theta_fun,theta,method="BFGS")
#             print("Pass")
            weight = max(min(1,1/cal_thershold),(thershold/prot)**(1 - 1/(iter_count+1)))
            theta_opt = np.abs(np.array(opt_res.x))
            theta_new = (1 - weight) * theta + weight * theta_opt
            theta = theta_new
#             theta = np.array(opt_res.x)
            # Step3 update Sigma_Y, Sigma_Res based on theta
            rho_h_theta= np.array(rho(si,theta))
            term1 = np.matmul(np.matmul(H.T,np.linalg.inv(rho_h_theta).real),H)
            term2 = np.matmul(np.matmul(G.T,np.linalg.inv(rho_h_theta).real),H)
            term3 = np.matmul(np.matmul(G.T,np.linalg.inv(rho_h_theta).real),G)

            Sigma_Y = term1
            Sigma_res = term1 - np.matmul(np.matmul(term2.T,np.linalg.inv(term3).real),term2)

            if iter_count == 0:
                iter_count += 1
                oldV0_hat,oldV1_hat,old_theta = V0_hat,V1_hat,theta
                continue
#             print("Before thershold")
            cal_thershold = np.sum((oldV1_hat - V1_hat)**2) + np.sum((oldV0_hat - V0_hat)**2) + np.sum((old_theta - theta)**2)
            print("Gap: {}, Theta: {}, weight: {}".format(cal_thershold,theta,weight))
            if cal_thershold < thershold:
                judge = False
                min_thershold = min(min_thershold,cal_thershold)
                prot = cal_thershold/min_thershold

            oldV0_hat,oldV1_hat,old_theta = V0_hat,V1_hat,theta
            iter_count += 1
        except:
            err_count += 1
            theta = theta + np.array([randint(-10,10)*thershold*prot,randint(-10,10)*thershold*prot])
            X_env = X_env + np.random.normal(0,1e-6,n*p).reshape(n,p)
            Y_env = Y_env + np.random.normal(0,1e-6,n*r).reshape(n,r)
            continue
            
    beta_final = np.matmul(PV1_hat,beta_MLE)
    Y_bar = np.mean(Y_env,axis = 0)
    X_bar = np.mean(X_env,axis = 0)
    alpha_final = Y_bar - np.matmul(X_bar,beta_final.T)
    output = (alpha_final.reshape(1,r),beta_final.reshape(p,r))
#     print("stop, iter = {}".format(iter_count+err_count))
        
    return(output)