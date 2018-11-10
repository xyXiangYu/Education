#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:17:45 2018

@author: xiang
"""



# Number of iterations
n_iter = 300


d = 50
n = 10000
idx = np.arange(d)

# lbda = 1. / n 

# run all the model with lower ridges
lbda_low = 1. / n 

# Ground truth coefficients of the model
x_model_truth = (-1)**idx * np.exp(-idx / 10.)


A, b = simu_linreg(x_model_truth, n, std=1., corr=0.9)
loss = loss_linreg
grad = grad_linreg
grad_i = grad_i_linreg
lipschitz_constant = lipschitz_linreg
#lbda = 1. / n ** (0.5)



# _A, _b = simu_linreg(x_model_truth, n, std=1., corr=0.1)
# #_A, _b = simu_logreg(x_model_truth, n, std=1., corr=0.7)



x_init = np.zeros(d)
x_min, f_min, _ = fmin_l_bfgs_b(loss, x_init, grad, args=(A, b, lbda_low), pgtol=1e-30, factr=1e-30)
max_squared_sum = np.max(np.sum(A ** 2, axis=1))

sol_ini_alg = [gd, agd, scipy_runner(fmin_cg), scipy_runner(fmin_l_bfgs_b), sgd, sag, svrg]
monitors_lower = run_all(sol_ini_alg, A, b, lbda_low )





# In[1]:



# run all the model with higher ridges
lbda_higher = 1. / (n ** 0.25) 
x_init = np.zeros(d)
x_min, f_min, _ = fmin_l_bfgs_b(loss, x_init, grad, args=(A, b, lbda_higher), pgtol=1e-30, factr=1e-30)
max_squared_sum = np.max(np.sum(A ** 2, axis=1))

sol_ini_alg = [gd, agd, scipy_runner(fmin_cg), scipy_runner(fmin_l_bfgs_b), sgd, sag, svrg]
monitors_higher = run_all(sol_ini_alg, A, b, lbda_higher )


# In[ ]:


nr_m = len(sol_ini_alg)

monitor_time_high_ridge = []
# monitor_time_high_ridge = [monitor_gd.tohttp://localhost:8888/notebooks/lab2_xiang_yu_and_babin_jean.ipynb#tal_time, monitor_agd.total_time, monitor_cg.total_time, 
#          monitor_bfgs.total_time, monitor_sgd.total_time, monitor_sag.total_time, 
#          monitor_svrg.total_time]

print (nr_m)
monitor_time_low_ridge = []
for i in np.arange(nr_m):  
    monitor_time_low_ridge.append(monitors_lower[i].total_time)
    monitor_time_high_ridge.append(monitors_higher[i].total_time)

pd.DataFrame({'Running time - low ridge' : monitor_time_low_ridge, 
              'Running time - high ridge' : monitor_time_high_ridge},
            solvers)



# compare SAG and CG with two different lbda values 
solvers_lambda = [ "ACG", "ACG_lower", "CG", "CG_lower", "SAG", "SAG_lower","SVRG", "SVRG_lower",]
monitors_lambda = [monitors_higher[1], monitors_lower[1], monitors_higher[2], monitors_lower[2], 
                   monitors_higher[5], monitors_lower[5], 
                   monitors_higher[6], monitors_lower[6]]
plot_epochs(monitors_lambda, solvers_lambda)
plot_time(monitors_lambda, solvers_lambda)


# Both  Broyden-Fletcher-Goldfarb-Shanno (BFGS) and ConjugateGradients(CG) methods are Quasi-Newton Methods, which approximate the Hessian using recent function and gradient evaluations. Since the gradient of our objective function is a linear function of $\lambda: \nabla f_i(x) = (a_i^\top x - b_i) a_i + \lambda x$. Therefore changing the magnitude of $\lambda$ does not influence the calculation time of $\nabla f_i(x)$ very much in each step. These two methods use the approximate gradients to come up with a new
# search direction in which they do a combination of fixed-step, analytic-step and line-search minimizations. In the steps of finding a new search direction, $\lambda$ also does not affect the calculation time so much, therefore we see that the compuational time for CG and BFGS stays quite stable for two differet lambdas. 
# 