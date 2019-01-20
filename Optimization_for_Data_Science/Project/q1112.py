#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 22:44:59 2019

@author: xiang
"""


import pandas as pd

df = pd.read_csv('winequality-red.csv', delimiter=';')
df.tail()


# In[572]:


df['quality'].unique()


# Now let's extract `X` and `y`

# In[573]:


data = df.values
X = data[:, :-1]
y = data[:, -1] - 3
X.shape, y.shape, np.unique(y)


# Let's do a basic scaling of the features:

# In[574]:


from sklearn.preprocessing import scale
X = scale(X)


# Now test the functions above with this dataset:

# In[575]:


x_init = np.zeros(X.shape[1] + np.unique(y).size - 1)
Y = binarize(y)
negloglik(x_init, X=X, Y=Y)
grad_negloglik(x_init, X=X, Y=Y)



from sklearn.base import BaseEstimator, ClassifierMixin


class ProportionalOdds(BaseEstimator, ClassifierMixin):
    """scikit-learn estimator for the proportional odds model
    
    Parameters
    ----------
    lbda : float
        The regularization parameter
    penalty : 'l1' | 'l2'
        The type of regularization to use.
    max_iter : int
        The number of iterations / epochs to do on the data.
    solver : 'pgd' | 'apgd' | 'lbfgs'
        The type of regularization to use.
        'lbfgs' is only supported with penalty='l2'.
        
    Attributes
    ----------
    alpha_ : ndarray, (n_classes - 1,)
        The alphas.
    beta_ : ndarray, (n_features,)
        The regression coefficients.
    """
    def __init__(self, lbda=1., penalty='l2', max_iter=2000,
                 solver='lbfgs'):
        self.lbda = lbda
        self.penalty = penalty
        self.max_iter = max_iter
        self.solver = solver
        assert self.penalty in ['l1', 'l2']
        assert self.solver in ['pgd', 'apgd', 'lbfgs'] 

    def fit(self, X, y):
        """Fit method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.
        y : ndarray, shape (n_samples,)
            The target. Must be integers between 0 and n_classes - 1.
        """
        n_classes = int(np.max(y)) + 1
        assert np.all(np.unique(y) == np.arange(n_classes))
        Y = binarize(y)
        n_samples, n_features = X.shape
        # TODO
        
        # find the min value with lbfgs
        p = n_features
        k = Y.shape[1]
        
        lips_const = np.linalg.norm(X) ** 2 / 4
        step = 1 /  lips_const 
        

        x_init = np.zeros(p + k - 1)
        x_init[:k - 1] = np.arange(k - 1)
        
        print (self.solver, self.penalty)
        
        if self.penalty == 'l1': 
            objf = pobj_l1
            proxf = prox_l1
        elif  self.penalty == 'l2':   
            objf = pobj_l2
            proxf = prox_l2
            
        if  self.solver == "lbfgs": 
            if self.penalty == 'l2':
                bounds = [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p
                result = fmin_l_bfgs_b(negloglik, x_init, fprime=None, args=(X, Y), approx_grad=1,bounds=bounds)
                
                x = result[0]
#                print (x)
            
        elif self.solver == "pgd" or  self.solver == "apgd":
            if self.solver == "pgd": 
                solver = pgd
            else: 
                solver = apgd
            
            monitor_solver =  monitor(solver, objf, x_min=None, args=(X, Y, lbda))
            x = x_init 
            n_iter = self.max_iter
            step = step
            
            monitor_solver.run(x_init, grad_negloglik, proxf, self.max_iter, step, 1, (X, Y), (step, lbda, k))
            x = monitor_solver.x_list[-1]
            
            
#              monitor_si.run(loss, x_init, grad,  maxiter=n_iter, args=(A, b, lbda), pgtol=1e-30)
            
#            def pgd(x_init, grad, prox, n_iter=100, step=1., store_every=1,
#                    grad_args=(), prox_args=()):
#                """Proximal gradient descent algorithm."""
#                x = x_init.copy()
#                x_list = []
#                for i in range(n_iter):
        
        ### TODO 
        # grad should be the gradient of negloglik
#        x = prox(x - step * grad(x, *grad_args), *prox_args)       
#
#        
#        ### END TODO
#        if i % store_every == 0:
#            x_list.append(x.copy())
#    return x, x_list
            
#            monitor_apgd_l1 =  monitor(apgd, pobj_l1, x_min, args=(X, Y, lbda))
#            monitor_apgd_l1.run(x_init, grad_negloglik, prox_l1, n_iter, step,
#                   grad_args=(X, Y), prox_args=(step, lbda, k))
        print (x)
        beta = x[-n_features:]
        n_thresh = k - 1
        eta = x[:n_thresh]

        # END TODO
        self.params_ = x
        self.alpha_ = eta.cumsum()
        self.beta_ = beta
        return self

    def predict(self, X):
        """Predict method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            The predicted target.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Predict proba method
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The features.

        Returns
        -------
        y_proba : ndarray, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return predict_proba(self.params_, X)


for solver in ['pgd', 'apgd', 'lbfgs']:
    clf = ProportionalOdds(lbda=1., penalty='l2', max_iter=1000, solver=solver)
    clf.fit(X, y)
    print('Solver with L2: %s   -   Score : %s' % (solver, clf.score(X, y)))

for solver in ['pgd', 'apgd']:
    clf = ProportionalOdds(lbda=1., penalty='l1', max_iter=1000, solver=solver)
    clf.fit(X, y)
    print('Solver with L1: %s   -   Score : %s' % (solver, clf.score(X, y)))


# <div class="alert alert-success">
#     <b>QUESTION 11:</b>
#     <ul>
#     <li>
#         Compare the cross-validation performance of your model with a multinomial
#         logistic regression model that ignores the order between the classes. You will comment your results.
#     </li>
#     </ul>
# </div>

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# TODO

# END TODO