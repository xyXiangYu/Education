
# coding: utf-8

# # PROJECT: Course Optimization for Data Science
# ## Optimization strategies for the proportional odds model
# 
# 
# Author: Alexandre Gramfort
# 
# If you have questions or if something is not clear in the text below please contact us
# by email.
# 
# ## Aim:
# 
# - Derive mathematically and implement the loss and gradients of the proportional odds model
# - Implement your own solvers for L1 or L2 regularization with: (Accelerated) Proximal gradient descent and L-BFGS (only for L2)
# - Implement your own scikit-learn estimator for the proportional odds model and test it on the `wine quality` dataset.
# 
# ### Remarks:
# 
# - This project involves some numerical difficulty due to the presence of many `log` and `exp` functions.
# - The correct and stable computation of the gradient is quite difficult. For this reason you have the possibility to use the `autograd` package to compute the gradient by automatic differentiation. `autograd` inspired the design of `pytorch`. It is a pure python package which makes it easy to install, and it is sufficient for our usecase.
# 
# ## VERY IMPORTANT
# 
# This work must be done by pairs of students.
# Each student must send their work before the 20th of January at 23:59, using the moodle platform.
# This means that **each student in the pair sends the same file**
# 
# On the moodle, in the "Optimization for Data Science" course, you have a "devoir" section called "Project".
# This is where you submit your jupyter notebook file.
# 
# The name of the file must be constructed as in the next cell
# 
# ### Gentle reminder: no evaluation if you don't respect this EXACTLY
# 
# #### How to construct the name of your file

# In[527]:


# Change here using YOUR first and last names
fn1 = "Yu"
ln1 = "Xiang"
fn2 = "Jean"
ln2 = "Babin"

filename = "_".join(map(lambda s: s.strip().lower(), 
                        ["project", ln1, fn1, "and", ln2, fn2])) + ".ipynb"
print(filename)


# Some imports

# In[528]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt


# ## Part 0: Proportional odds model
# 
# This model is an ordinal regression model. It is a supervised learning model, in the case where the target space $Y$ is discrete: $Y=\{1, \dots, k\}$; this is the case in multiclass classification for example. Its specificity is that we assume there is an order in the output space. 
# 
# Intuitively, it means that if the true label is 2, predicting 5 is worse that predicting 3 (as 3 is closer to 2 than 5 is). For a usual classification loss, each bad prediction costs the same. In the case of the proportional odds model, **it costs more to predict values that are farther from the true target**.
# 
# The proportional odds model can be seen as an extension to the logistic regression model as we will see now.

# Working with observations in $\mathbb{R}^p$, the proportional odds model has the following structure for $1 \leq j \leq k-1$:
# 
# $$
# \log \left ( \frac{P(Y \leq j \mid x)}{P(Y > j \mid x)} \right ) = \alpha_j + \beta^T x ,
# $$
# 
# where $\beta \in \mathbb{R}^p$ and $\alpha = \{ \alpha_j \}_{j=1}^{k-1}$ is an increasing sequence of constants ($\alpha_1 \leq \alpha_2 \leq \dots \leq \alpha_{k-1}$). We omit here the last term since $P(Y \leq k) = 1$.
# Since $P(Y > j | x) = 1 - P(Y \leq j | x)$, we can rewrite the previous equation as:
# $$
# P(Y \leq j \mid x) = \frac{e^{\alpha_j + \beta^T x}}{e^{\alpha_j + \beta^T x} + 1} = \phi(\alpha_j + \beta^T x)
# $$
# 
# and 
# 
# $$
# P(Y = j \mid x) = \frac{e^{\alpha_j + \beta^T x}}{e^{\alpha_j + \beta^T x} + 1} - \frac{e^{\alpha_{j-1} + \beta^T x}}{e^{\alpha_{j-1} + \beta^T x} + 1} = \phi(\alpha_j + \beta^T x) - \phi(\alpha_{j-1} + \beta^T x)
# $$
# 
# for $2 \leq j \leq k-1$, where $\phi$ denotes the sigmoid function $\phi(t) = 1 / (1 + \exp(-t))$.
# 
# After one-hot encoding of the target variable ([`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [`LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html)), denoting $\{ y_{ij} \}_{j=1}^{k}$ the indicator sequence for the class of the $i^{\text{th}}$ observation $x_i$ (i.e., exactly one of the $y_{ij}$ equals one and the rest are zero) the negative log likelihood becomes:
# 
# $$
# f(\alpha, \beta) =
# - \sum_{i=1}^{n} \left [ y_{i1} \log(\phi(\alpha_1 + \beta^T x_i)) 
# + \sum_{j=2}^{k-1} \Big( y_{ij} \log( 
# \phi(\alpha_j + \beta^T x_i) - \phi(\alpha_{j-1} + \beta^T x_i)) \Big)
# + y_{ik} \log(1 - \phi(\alpha_{k-1} + \beta^T x_i)) \right ] .
# $$

# Introducing some $\ell_1$ or $\ell_2$ regularization on the parameter $\beta$ with regularization parameter $\lambda \ge 0$, the penalized likelihood estimation problem reads:
# $$
#     (\mathcal{P}_\alpha): \left\{
# 	\begin{aligned}
# 	\min_{\alpha, \beta} \quad f(\alpha, \beta) + \lambda \mathcal{R}(\beta) \\
#     \alpha_1 \leq \dots \leq \alpha_{k-1}
# 	\end{aligned}
#     \right.
# $$
# where $\mathcal{R}(\beta) = \|\beta\|_1$ or $\tfrac{1}{2} \|\beta\|^2_2$

# <div class="alert alert-success">
#     <b>QUESTION 1:</b>
#      <ul>
#       <li>Justify that $(\mathcal{P}_\alpha)$ is a convex problem.</li>
#     </ul>
# </div>

# In the first exercise of this course(exe_convexity_smoothness.pdf), we have proved the following statements: 
# 
# 1\.  $\|\beta\|_1$ and $\tfrac{1}{2} \|\beta\|^2_2$ are both convex functions, and actually every norm is convex.
# 
# 2\.  For every convex function $f : y ∈ \mathbb{R}^m → f (y)$, we have that $g : x ∈ \mathbb{R}^d → f (Ax − b)$ is a
# convex function, where $A ∈ \mathbb{R}^{m*d}$ and $b ∈ \mathbb{R}^m$. 
# 
# 3\. $f_i : \mathbb{R}^d → \mathbb{R}$ be convex for $i = 1, ..., m$, then we have  that $\sum_{i=1}^{m}f_i$ is convex
# 
# Further, we could prove that:
# 
# 4\. $h(x) = \log (1+ e^x)$ is convex, (here $e$ is the exponential constant)
# 
# 5\. $g(x) = \log (\frac{1}{e^{-x}-1})$ is convex
# 
# 

# Statement 4 and Statement 5 could be easily proved by calculating the second order derivatives and check that they are positive for any real number x. 
# 
# $$h^{''}(x) = \frac{e^x}{(1 + e^x)^2} > 0, \forall x$$
# $$g^{''}(x) = \frac{e^x}{(e^x - 1)^2} > 0, \forall x $$
# 
# 

# With above statements, we could prove that $(\mathcal{P}_\alpha)$ is a convex problem. 
# First we show $f(\alpha, \beta)$ is a convex function. 
# 
# Denote $z_j := \alpha_j + \beta^T x_i$, then 
#  $$f(\alpha, \beta) = \sum_{i=1}^{n} [-y_{i1} \log(\phi(z_1)) - \sum_{j=2}^{k-1} \Big( y_{ij} \log(\phi(z_j) - \phi(z_{j-1})\big) -  y_{ik} \log(1 - \phi(z_{k-1})) ]$$
#  
# By statment 3, we only need to prove the addend of function $f(\alpha, \beta)$ is convex, i.e. we need to show the following three parts are convex functions to prove the convexity of function $f(\alpha, \beta)$. 
# -  Addend 1: $-\log(\phi(z_1))$
# -  Addend 2: $-\log(\phi(z_j) - \phi(z_{j-1})$
# -  Addend 3$-\log(1 -\phi(z_{k-1}))$
# 
# We have 
# $$ \phi(x) = \frac{e^x}{1+ e^x} = (1+ e^{-x})^{-1}$$
# and Addend 1: 
# $$-\log(\phi(z_1)) = \log (\phi(z_1)^{-1}) = \log (1+ e^{-z_1}): = h(-z_1)$$
# and from statement 4, we have $h(z_j(x_i))$ is convex, combining statement 4 and statement 2, we know that $h(-z_1)$ is also convex ($z_1$ is linear transformation of $x$), i.e. we have shown that $-\log(\phi(z_1))$ is convex
# 
# Now we prove the convexity of Addend 2: 
# 
# $$-\log(\phi(z_j) - \phi(z_{j-1}) = -\log((1+ e^{-z_j})^{-1} - (1+ e^{-z_{j-1}})^{-1}) = \log(1+ e^z_j) + \log(1+ e^z_{j-1}) - \log(e^z_{j-1} - e^z_j)$$
# 
# We know from statement 4 and 2 that $\log(1+ e^z_j)$ and $\log(1+ e^{j-1})$ are convex, moreover, we could prove that 
# $-\log(e^z_{j-1} - e^z_j) $ is also convex. 
# Denote: $\delta = z_j - z_{j-1} = \alpha_j - \alpha_{j-1}$
# 
# Then 
# $$-\log(e^z_{j-1} - e^z_j) = -\log(e^{z_j - \delta}- e^z_j) = -\log (e^z_j(\frac{1}{e^\delta} -1)) = -z_j + \log((e^{-\delta} - 1)^{-1})$$
# $z_j$ is convex, and the second order derivative of $\log((e^{-\delta} - 1)^{-1})$ is  $\frac{e^{-\delta}}{(e^{-\delta}-1)^2}$, which is positive $\forall x$
# 
# Thus we have proved that Addend 2: $-\log(\phi(z_j) - \phi(z_{j-1})$ is also convex. 
# 
# Similarly we could prove that Addend 3: is also convex. 

# Since function $f(\alpha, \beta)$ is just sum of Addend 1, a number of Addend 2 and Addend 3. Addend 1, 2 and 3 are all convex, therefore by statement 3, $f(\alpha, \beta)$ is convex. 
# 
# Moreover, by statement 1, $\mathcal{R}(\beta)$ is convex. 
# 
# And the domain of $\alpha, \beta$ is convex, since it is real number space. 
# Then  $(\mathcal{P}_\alpha)$ is a convex function defined on a convex domain, thus it is a convex problem. 
# 

# ## Simulation
# 
# Generate data under the above model and then estimate $\alpha$ and $\beta$ using maximum likelihood

# In[529]:


import numpy as np

n = 1000  # number of samples
p = 2  # number of features
k = 3  # number of classes


# #### Generate parameters and compute probability distributions for each sample

# In[530]:


rng = np.random.RandomState(42)
X = 15 * rng.normal(size=(n, p))
alpha = np.sort(np.linspace(-10, 10, k - 1) + rng.randn(k - 1))
beta = rng.randn(p)


# We want to compute the quantity $P(Y = j \mid x_i)$ for $j= 1, \dots , k$, and $i= 1, \dots, n$.
# 
# First, let us compute an array containing the values $P(Y < j \mid x_i)$ for $j= 1, \dots , k+1$ and $i=1, \dots, n$. (we denote this array `F`):

# In[531]:


def phi(t):
    return 1. / (1. + np.exp(-t))

F = phi(np.dot(X, beta)[:, np.newaxis] + alpha)
F = np.concatenate([np.zeros((n , 1)), F, np.ones((n , 1))], axis=1)
F


# In[532]:


# compute P(Y = j | x)
proba = np.diff(F, axis=1)
assert proba.shape == (n, k)
proba


# The sum of all probas for each sample should be 1:

# In[533]:


np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(n))
# numpy.testing.assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.


# #### Simulate $Y$ according to $P(Y = j \mid x)$

# In[534]:


y = np.array([rng.choice(np.arange(k), size=1, p=pi)[0] for pi in proba])
y[:10]


# In[535]:


for j in range(k):
    Xj = X[y == j]
    plt.plot(Xj[:, 0], Xj[:, 1], 'o', label='y = %d' % j, alpha=0.5)

plt.legend();


# # Log-Likelihood function
# 
# We adopt the parametrization from $(\mathcal{P}_\alpha)$. The vector of parameters `params` has `k-1 + p` entries. The first `k-1` are the alphas $\alpha$ and the last `p` entries correspond to $\beta$. The function that predicts the probabilities of each sample reads:

# In[536]:


def predict_proba_alphas(params, X=X):
    """Compute the probability of each sample in X.
    
    Parameters:
    -----------
    params: array, shape (k - 1 + p,)
        Parameters of the model. The first k - 1 entries are the alpha_j,
        the remaining p ones are the entries of beta.
        
    X: array, shape (n, p)
        Design matrix.
        
    Returns
    -------
    proba : ndarray, shape (n, k)
        The proba of belonging to each class for each sample.
    """
    n_samples, n_features = X.shape
    n_thresh = params.size - n_features
    alpha = params[:n_thresh]
    beta = params[n_thresh:]
    F = phi(np.dot(X, beta)[:, np.newaxis] + alpha)
    F = np.concatenate(
        [np.zeros((n_samples , 1)), F, np.ones((n_samples , 1))], axis=1)
    proba = np.diff(F, axis=1)
    return proba


# One-hot encoding of `y` can be done with scikit-learn `LabelBinarizer`. As it's a matrix, we call it `Y`:

# In[537]:


from sklearn import preprocessing

def binarize(y):
    le = preprocessing.LabelBinarizer()
    Y = le.fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.concatenate([1 - Y, Y], axis=1)
    return Y

Y = binarize(y)
Y[:10]


# The negative log-likelihood then reads:

# In[538]:


def negloglik_alphas(params, X=X, Y=Y):
    proba = predict_proba_alphas(params, X)
    assert Y.shape == proba.shape
    return -np.sum(np.log(np.sum(proba * Y, axis=1) + np.finfo('float').eps))

params = np.concatenate([alpha, beta])
negloglik_alphas(params)


# <div class="alert alert-success">
#     <b>QUESTION 2:</b>
#      <ul>
#       <li>Justify why applying coordinate descent or proximal gradient descent to $(\mathcal{P}_\alpha)$ is not easy (or even possible?).</li>
#     </ul>
# </div>

# When choosing one coordinate to minimize, e.g. minimizing $\mathcal{P}_\alpha$ with respect to $\alpha_i$, it may happen that $\alpha_i$ becomes bigger than $\alpha_{i+1}$,  leading to a negative probablity, i.e. the constraints can not be fulfilled. Even if fortunately, the final result satisfies the constriants, we may not get the optimal solution becasue in some iterations, we get a fake optimal value of $\alpha_i$ (a value does not meet the constraints), leading to a doubtful(mostly likely not optimal) final result. 
# 
# 
# For proximal gradient descent, we will have difficulity to find a very simple explicit formula for the alpha term. 
# 
# 

# ## Reparametrization
# 
# To fix the problem, we propose to reparametrize the problem with a new vector $\eta \in \mathbb{R}^{k-1}$ such that $\alpha_j = \sum_{l=1}^{j} \eta_l$ with $\eta_j \geq 0$ for $j \geq 2$.
# 
# We denote by $\mathcal{L}(\eta, \beta)$ the corresponding negative log-likelihood:
# 
# $$
# \mathcal{L}(\eta, \beta) =
# - \sum_{i=1}^{n} \left [ y_{i1} \log \left ( \phi(\eta_1 + \beta^T x_i) \right )
# + \sum_{j=2}^{k-1} y_{ij} \log \left ( \phi(\sum_{l=1}^j \eta_l + \beta^T x_i) - \phi(\sum_{l=1}^{j-1} \eta_l + \beta^T x_i) \right ) + y_{ik} \log \left ( 1 - \phi(\sum_{l=1}^{k-1} \eta_l + \beta^T x_i) \right ) \right ] .
# $$

# <div class="alert alert-success">
#     <b>QUESTION 3:</b>
#      <ul>
#       <li>Show that $(\mathcal{P}_\alpha)$ can be rewritten as an unconstrained convex problem $(\mathcal{P}_\eta)$.
# $$
#     (\mathcal{P}_\eta): \left\{
# 	\begin{aligned}
# 	\min_{\eta \in \mathbb{R}^{k-1}, \beta \in \mathbb{R}^{p}} \quad \mathcal{L}(\eta, \beta) + \lambda \mathcal{R}(\beta) + \sum_{j=2}^{k-1} g_j(\eta_j)\\
# 	\end{aligned}
#     \right.
# $$
#           You will detail what are the functions $g_j$.
#     </li>
#     <li>
#         Justify that the problem can be solved with Proximal Gradient Descent, Proximal Coordinate Descent and the L-BFGS-B algorithm (implemented in scipy.optimize).
#     </li>
#     </ul>
# </div>

# Define: $g(x) = L * (|x| - x)$, where $L$ is a number very large(approaching positive infinity) i.e. 
# $$g(x) = \{
# \begin{aligned} 
# &  0,  \  if \  x >= 0\\
# & L \  ( where \  L \to + \infty), \  else
# \end{aligned}$$
# 
# 
# Then  $(\mathcal{P}_\eta)$ is equivalent to $(\mathcal{P}_\alpha)$ since the contraints in $(\mathcal{P}_\alpha)$ is translated into function $g_j$ for $j = 2, 3, ..., k-1$
# 
# Notice that the constraints in $(\mathcal{P}_\alpha)$ are:  $\alpha_1 \leq \dots \leq \alpha_{k-1}$, 
# 
# and now we have $\alpha_1 = \eta_1$, $\alpha_2 = \eta_1 + \eta_2$, ..., $\alpha_{k-1} = \sum_{l=1}^{k-1} \eta_l$, which are equivalently to that all $\eta_2, \eta_3, ..., \eta_{k-1}$ are positive. 
# 
# $ g_j(\eta_j), j = 2, 3, ..., k-1$ equals to zero as long as $\eta_j > 0 $ and will become infinity if not. Which means the objective value of $(\mathcal{P}_\alpha)$ and  $(\mathcal{P}_\eta)$ will be the same when condition $\alpha_1 \leq \dots \leq \alpha_{k-1} $ in $(\mathcal{P}_\alpha)$ (the equivalently condition in $\mathcal{L}(\eta, \beta)$: $\eta_j >=0, j = 2, 3, ..., k-1$) is satisfied.  
# 
# Therefore,  $(\mathcal{P}_\eta)$ is equivalent to $(\mathcal{P}_\alpha)$. 

# In Question 2, we have proved that $(\mathcal{P}_\alpha)$ is convex, and function $ f(\alpha, \beta)$ in $(\mathcal{P}_\alpha)$ are combinations of linear, exponential, and log functions, which are all continous and differentiable. Therefore,  $ f(\alpha, \beta)$ is convex and differentiable. Similarly, we could prove that   $  \mathcal{L}(\eta, \beta) $ is convex and differentiable. Moreover, both $\lambda \mathcal{R}(\beta)$ and $\sum_{j=2}^{k-1} g_j(\eta_j)$ are convex and separable. 
# 
# Then we could use Proximal Gradient Descent, Proximal Coordinate Descent and and the L-BFGS-B algorithm  as stated in the lecture slides, since they all only require to calculate the first order gradient. 
# 
# The convexity of $\sum_{j=2}^{k-1} g_j(\eta_j)$  could be proved by showing that its addend is convex, as follows: 
# \begin{aligned}
# g(\theta x + (1-\theta) y) &= L (|\theta x + (1-\theta) y| - \theta x + (1-\theta) y) \\
# & <= L (|\theta x| + |(1-\theta) y| - \theta x + (1-\theta) y) \\
# & = \theta L(|x| - x) + (1-\theta) L (|y| - y) \\
# & = \theta g(x) + (1-\theta) g(y) 
# \end{aligned}
# where $\theta \in [0,1]$

# Without losing generosity, we let $L = 1e+8$

# In[539]:


L = 1e+8


# <div class="alert alert-success">
#     <b>QUESTION 4:</b>
#      <ul>
#       <li>Introducing the functions $f_2(\eta, \beta) = \tfrac{\lambda}{2}\|\beta\|_2^2 + \sum_{j=2}^{k-1} g_j(\eta_j)$ (corresponding to the case where $\mathcal{R}=\tfrac{1}{2}\|\beta\|_2^2$) and $f_1(\eta, \beta) = \lambda \|\beta\|_1 + \sum_{j=2}^{k-1} g_j(\eta_j)$ (corresponding to the case where $\mathcal{R}=\|\beta\|_1$), compute and implement the proximal operators of $f_1$ and $f_2$.
#     </li>
#     </ul>
# </div>
# 
# In the code below, `lambda` being a reserved keyword in Python, we denote $\lambda$ by `reg`.

# In[540]:


def prox_f2(params, reg=1., n_classes=k):
    
    n_samples, n_features = X.shape
    n_thresh = params.size - n_features
    
    eta = params[:n_thresh]  # eta:  eta1 could be any number
    # eta2, eta3, ..., eta_{k-1} must be non-negative
    beta = params[n_thresh:]  # the first n_thresh beta
    beta_new = beta / (1 + reg)
    
    eta_new = eta.copy()  
    
    eta_new[eta < -2 * L] = eta[eta< -2 * L] + 2 * L
    eta_new[np.logical_and(-2 * L <= eta, eta <= 0)] = 0 
    eta_new[0] = eta[0] # For the first term of eta, there is no constraint
    
    params[:n_thresh] = eta_new
    params[n_thresh:] = beta_new

    return params


def prox_f1(params, reg=1., n_classes=k):
    # TODO
    
    n_samples, n_features = X.shape
    n_thresh = params.size - n_features
    
    eta = params[:n_thresh]
    beta = params[n_thresh:]  # the first n_thresh beta
    
    beta_new = np.sign(beta) * np.maximum(np.abs(beta) - reg, 0)   
    eta_new = eta.copy()
    
    eta_new[eta < -2 * L] = eta[eta< -2 * L] + 2 * L
    eta_new[np.logical_and(-2 * L <= eta, eta <= 0)] = 0 
    eta_new[0] = eta[0] # For the first term of eta, there is no constraint
    
    
    params[:n_thresh] = eta_new
    params[n_thresh:] = beta_new


    return params

rng = np.random.RandomState(5)
x = rng.randn(p + k - 1)
l_l1 = 1.
l_l2 = 2.
ylim = [-1, 3]

plt.figure(figsize=(15.0, 4.0))
plt.subplot(1, 3, 1)
plt.stem(x)
plt.title("Original parameter", fontsize=16)
plt.ylim(ylim)
plt.subplot(1, 3, 2)
plt.stem(prox_f1(x, l_l1))
plt.title("Proximal Lasso", fontsize=16)
plt.ylim(ylim)
plt.subplot(1, 3, 3)
plt.stem(prox_f2(x, l_l2))
plt.title("Proximal Ridge", fontsize=16)
plt.ylim(ylim)


# ## Part 1: Implementation of the solvers
# 
# ### L-BFGS-B Solver
# 
# We will start by using the L-BFGS solver provided by `scipy`, without specifying the gradient function. In this case, the [`fmin_l_bfgs_b`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html) function will approximate the gradient using a finite difference method.

# <div class="alert alert-success">
#     <b>QUESTION 5:</b>
#     <ul>
#     <li>
#         Implement the new predict_proba function using the new parametrization with $\eta$
#     </li>
#     </ul>
# </div>

# In[541]:


def predict_proba(params, X=X):
    """Compute the probability of every sample in X.
    
    Parameters
    ----------
    params : ndarray, shape (k - 1 + p,)
        The parameters. The first k-1 values are the etas
        and the last p ones are beta.
        
    X: array, shape (n, p)
        Design matrix.
    
    Returns
    -------
    proba : ndarray, shape (n, k)
        The proba of belonging to each class for each sample.
    """
    n_samples, n_features = X.shape
    n_thresh = params.size - n_features
    eta = params[:n_thresh]
    beta = params[n_thresh:]
    alpha = eta.cumsum()

    
    F = phi(np.dot(X, beta)[:, np.newaxis] + alpha)
    F = np.concatenate(
        [np.zeros((n_samples , 1)), F, np.ones((n_samples , 1))], axis=1)
    proba = np.diff(F, axis=1)
    
    return proba


def negloglik(params, X=X, Y=Y):
    """Compute the negative log-likelihood.
    
    Parameters
    ----------
    params : ndarray, shape (p + k - 1,)
        The parameters. The first k-1 values are the etas
        and the remaining ones are the entries of beta.
    
    Returns
    -------
    nlk : float
        The negative log-likelihood to be minimized.
    """
    proba = predict_proba(params, X=X)
    # print (Y.shape, proba.shape)
    assert Y.shape == proba.shape
    return -np.sum(np.log(np.sum(proba * Y, axis=1) + np.finfo('float').eps))


# The next cell is to check your implementation:

# In[542]:


# Check your implementation
def alpha_to_eta(alpha):
    eta = alpha.copy()
    eta[1:] = np.diff(alpha)
    return eta

# Compute with P_alpha parametrization:
negloglik_alphas(np.concatenate([alpha, beta]))

# Compute with P_eta parametrization:
eta = alpha_to_eta(alpha)
params = np.concatenate([eta, beta])

# Check that log-likelihoods match
assert abs(negloglik(params) - negloglik_alphas(np.concatenate([alpha, beta]))) < 1e-10


# <div class="alert alert-success">
#     <b>QUESTION 6:</b>
#     <ul>
#     <li>
#         Solve the optimization using the `fmin_l_bfgs_b` function.
#     </li>
#     </ul>
# </div>
# 
# HINT: You can specify positivity contraints for certain variables using the `bounds` parameter of `fmin_l_bfgs_b`. Infinity for numpy is `np.inf`.
# 
# The estimate of $\beta$ (resp. $\eta$ and $\alpha$) should be called `beta_hat` (resp. `eta_hat` and `alpha_hat`)

# In[543]:


from scipy.optimize import fmin_l_bfgs_b

x0 = np.zeros(p + k - 1)
x0[:k - 1] = np.arange(k - 1)  # initiatlizing with etas all equal to zero is a bad idea!

bounds = [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p

result = fmin_l_bfgs_b(negloglik, x0, fprime=None, args=(), 
                              approx_grad=1,bounds=bounds)

params = result[0]
n_samples, n_features = X.shape
n_thresh = params.size - n_features
eta_hat = params[:n_thresh]
beta_hat = params[n_thresh:]


# In[544]:


Y_proba = predict_proba(np.concatenate([eta_hat, beta_hat]))
y_pred = np.argmax(Y_proba, axis=1)

for j in range(k):
    Xj = X[y_pred == j]
    plt.plot(Xj[:, 0], Xj[:, 1], 'o', label='y = %d' % j, alpha=0.5)

plt.legend();


# ### Computation of the gradients
# 
# We have so far been lazy by asking `fmin_l_bfgs_b` to approximate the gradient.
# You are going to fix this using either one of the next 2 options:
# 
# <div class="alert alert-success">
#     <b>QUESTION 7 (option 1):</b>
#     <ul>
#     <li>
#         Implement the function grad_negloglik that computes the gradient of negloglik.
#     </li>
# </ul>
# </div>
# 
# <div class="alert alert-success">
#     <b>QUESTION 7 (option 2):</b>
#     <ul>
#     <li>
#         Implement the function grad_negloglik that computes the gradient of negloglik
#         using the <a href="https://github.com/HIPS/autograd">autograd</a> package.
#     </li>
#     </ul>
# </div>
# 
# **HINT** : QUESTION 7 (option 1) you can use the fact that: $\log(\phi(t))' = 1 - \phi(t)$ and $\phi(t)' = \phi(t) (1 - \phi(t))$
# 
# You can check your implementation of the function `grad_negloglik` with the check_grad function. However **WARNING** your code is likely to be numerically quite unstable due to the numerous `log` and `exp` with tiny values that are probabilities. You may want to work with log of probabilities but **warning** this is not easy...

# $$
# \mathcal{L}(\eta, \beta) =
# - \sum_{i=1}^{n} \left [ y_{i1} \log \left ( \phi(\eta_1 + \beta^T x_i) \right )
# + \sum_{j=2}^{k-1} y_{ij} \log \left ( \phi(\sum_{l=1}^j \eta_l + \beta^T x_i) - \phi(\sum_{l=1}^{j-1} \eta_l + \beta^T x_i) \right ) + y_{ik} \log \left ( 1 - \phi(\sum_{l=1}^{k-1} \eta_l + \beta^T x_i) \right ) \right ] .
# $$

# Denote $z_j := \alpha_j + \beta^T x_i = \sum_{l=1}^{j} \eta_l +  \beta^T x_i$, then 
# 
# $$\frac{\partial z_j}{\partial \eta_m} = \{
# \begin{aligned} 
# & 1,  if \ j >= m \\
# & 0,  if \ j < m 
# \end{aligned}
# $$ 
# 
# and 
# $$\frac{\partial z_j}{\partial \beta} = x_i $$
# 
# Then $\mathcal{L}(\eta, \beta)$ could be re-written as: 
# 
# $$
# \mathcal{L}(\eta, \beta) = - \sum_{i=1}^{n} \{  \sum_{j=1}^{k} [y_{i,j}\log(\phi(z_j) - \phi(z_{j-1})] \} 
# $$
# Where we introduced that $\phi(z_0) = 0, \ \phi(z_k) = 1 $
# 
# 

# Then: 
# 
# Denote $M_j = \phi(z_j)' =  \phi(z_j)(1-\phi(z_j))$, and $M_0 = \phi(z_0)' = 0' = 0, M_k = \phi(z_k)' = 1' = 0$, we have: 
# 
# $$
# \frac{\partial \mathcal{L}(\eta, \beta)}{\partial \eta_m} \\
# = - \sum_{i=1}^{n} \{  \sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} (\frac{\partial \phi(z_j)}{\partial \eta_m} - \frac{\partial \phi(z_{j-1})}{\partial \eta_m}) ]\} \\
# = - \sum_{i=1}^{n} \{  \sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} (\phi(z_j)'\frac{\partial z_j}{\partial \eta_m} - \phi(z_{j-1})'\frac{\partial z_{j-1}}{\partial \eta_m} )]\} \\
# = - \sum_{i=1}^{n} \{  \sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_j \frac{\partial z_j}{\partial \eta_m}] -\sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_{j-1}\frac{\partial z_{j-1}}{\partial \eta_m}] \}  \\
# = - \sum_{i=1}^{n} \{  \sum_{j=m}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_j] -\sum_{j={m+1}}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_{j-1}] \} \\
# = - \sum_{i=1}^{n} \{ \frac{y_{i,m}M_m}{\phi(z_m) - \phi(z_{m-1})}  + \sum_{j={m+1}}^{k}[\frac{y_{i,j} (M_j - M_{j-1})}{\phi(z_j)- \phi(z_{j-1})} ]\} \\
# = - \sum_{i=1}^{n} \{  \sum_{j={m}}^{k}[\frac{y_{i,j} (M_j - M_{j-1})}{\phi(z_j)- \phi(z_{j-1})}] +  \frac{y_{i,m}M_{m-1}}{\phi(z_m) - \phi(z_{m-1})}\}
# $$
# 
# 
# For numerical reason, we will calculate above term as follows: 
# 
# $$
# \frac{\partial \mathcal{L}(\eta, \beta)}{\partial \eta_m} \\
# = - \sum_{i=1}^{n} \{  \sum_{j=m}^{k-1}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_j] -\sum_{j={m+1}}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_{j-1}]  + \frac{y_{i,k}}{\phi(z_j) - \phi(z_{k-1})} M_k\}
# $$
# 
# And we denote: 
# $$
# Term1 = \sum_{j=m}^{k-1}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_j] \\
# Term2 = \sum_{j={m+1}}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} M_{j-1}]  , \  and \\
# Term3 = \frac{y_{i,k}}{\phi(z_j) - \phi(z_{k-1})} M_k
# $$

# and 
# $$\frac{\partial \mathcal{L}(\eta, \beta)}{\partial \beta} =  - \sum_{i=1}^{n} \{  \sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} (\frac{\partial \phi(z_j)}{\partial \beta} - \frac{\partial \phi(z_{j-1})}{\partial \beta}) \} \\
# = - \sum_{i=1}^{n} \{  \sum_{j=1}^{k}[\frac{y_{i,j}}{\phi(z_j) - \phi(z_{j-1})} (\phi(z_j)'\frac{\partial z_j}{\partial \beta} - \phi(z_{j-1})'\frac{\partial z_{j-1}}{\partial \beta}) ] \} \\
# = - \sum_{i=1}^{n} \{ x_i \sum_{j=1}^{k}[\frac{y_{i,j} (M_j - M_{j-1})}{\phi(z_j) - \phi(z_{j-1})} ] \}
# $$

# In[545]:


# Option 1
from scipy.misc import logsumexp


def grad_negloglik(params, X=X, Y=Y):
     
    proba = predict_proba(params, X=X)    # = phi(z_j) -  phi(z_{j-1}) 
    assert Y.shape == proba.shape
    
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]
    
    p = n_features
    k = n_classes
       
    phi = np.cumsum(proba, axis=1) # i.e. phi(z1), ..., phi(zk)
    Mj = phi * (1 - phi)           # phi(z_j)'
    
    zerocl = np.zeros((Mj.shape[0],1))
    Mjminus = np.column_stack((zerocl, Mj[:,0:k-1]))  # Mjminus
   
    Term1 = Y[:,0:k-1] * Mj[:,0:k-1] / (proba[:,0:k-1] + np.finfo('float').eps)      # m = 1, 2, ..., k-1 
    Term2 = Y[:,1:k] * Mj[:,0:k-1] / (proba[:,1:k]  + np.finfo('float').eps)  
    Term3 = Y[:,-1] * Mj[:,-1] /  (proba[:,-1] + np.finfo('float').eps) 
    Term3_dim_expansion = np.repeat(Term3[:, np.newaxis],  Term1.shape[1], axis=1)
    ThreeTerms = Term1 - Term2 + Term3_dim_expansion  
    
    addend_inbigBracket = np.flip(np.cumsum(np.flip(ThreeTerms, 1), axis=1), 1)
    delta_eta = -np.sum(addend_inbigBracket, axis=0)

    # calculating the derivatives for beta     
    yijMjdp = Y * (Mj - Mjminus) / (proba + np.finfo('float').eps)
    pro_yijMjdp = np.sum(yijMjdp, axis=1)[:,np.newaxis]
    XyijM = X * pro_yijMjdp
    betas = -np.sum(XyijM, axis=0)
  
    delta = np.concatenate([delta_eta,betas])

    return delta


# In[546]:


# Option 2

import autograd.numpy as np
from autograd import grad

def negloglik_autograd(params, X=X, Y=Y):
    """Compute the negative log-likelihood

    Parameters
    ----------
    params : ndarray, shape (p + k - 1,)
        The parameters. The first k-1 values are the etas
        and the remaining p ones correspond to beta.
    X : ndarray, shape (n, p)
        Design matrix.
    Y : ndarray, shape (n, k)
        The target after one-hot encoding.

    Returns
    -------
    nlk : float
        The negative log-likelihood to be minimized.
    """
    # TODO
    
    proba = predict_proba(params, X=X)
    assert Y.shape == proba.shape 
    -np.sum(np.sum(np.log(proba ^ Y + np.finfo('float').eps),axis=1)) 
    return  -np.sum(np.sum(np.log(proba ^ Y),axis=1)) 


grad_negloglik_auto = grad(negloglik_autograd)


# In[547]:


from scipy.optimize import check_grad
rng = np.random.RandomState(7)
x0 = rng.randn(p + k - 1)
x0[1:k - 1] = np.abs(x0[1:k - 1])
# WARNING: check_grad is likely to return a quite high value
# due to numerical instability with exp and log with tiny
# probability values. Don't be surprised as long as your
# solvers below converge.
check_grad(negloglik, grad_negloglik, x0=x0)


# Now plug your gradient into L-BFGS and check the result:

# In[548]:


x0 = np.zeros(p + k - 1)
x0[:k - 1] = np.arange(k - 1)  # initiatlizing with etas all equal to zero is a bad idea!


bounds = [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p
x_hat, _, _ = fmin_l_bfgs_b(negloglik, fprime=grad_negloglik,
                            x0=x0, bounds=bounds)
Y_proba = predict_proba(x_hat)
y_pred = np.argmax(Y_proba, axis=1)

for j in range(k):
    Xj = X[y_pred == j]
    plt.plot(Xj[:, 0], Xj[:, 1], 'o', label='y = %d' % j, alpha=0.5)
plt.legend();


# In[551]:


# x_hat
# beta


# In[552]:


#  [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p
# l2_beta_grad = 2 * beta 
# l2_grad = np.zeros((len(params), 1))
# l2_grad[-n_features:] = l2_beta_grad[0]
# # l2_grad = [0] * (len(params) - n_features) 
# l2_grad


# <div class="alert alert-success">
#     <b>QUESTION 9:</b>
#     <ul>
#     <li>
#         Wrap this into a function of X, y and lbda that implements
#         the function proportional_odds_lbfgs_l2 that will be
#         used to get a good value of x_min (minimum of the L2 regularized
#         model).
#     </li>
#     </ul>
# </div>

# To help you we give you the code of the objective to minimize
# in case you use $\ell_1$ or $\ell_2$ penalty.

# In[553]:


def pobj_l1(params, X=X, Y=Y, lbda=1.):
    n_features = X.shape[1]
    beta = params[-n_features:]
    n_thresh = Y.shape[1] - 1
    eta = params[:n_thresh]
    if np.any(eta[1:] < 0):
        return np.inf
    return negloglik(params, X=X, Y=Y) + lbda * np.sum(np.abs(beta))


def pobj_l2(params, X=X, Y=Y, lbda=1.):
    n_features = X.shape[1]
    beta = params[-n_features:]
    n_thresh = Y.shape[1] - 1
    eta = params[:n_thresh]
    if np.any(eta[1:] < 0):
        return np.inf
    return negloglik(params, X=X, Y=Y) + lbda / 0.5 * np.dot(beta, beta)


# In[554]:


def grad_l2(params, X=X, Y=Y, lbda=1.):

    n_features = X.shape[1]
    beta = params[-n_features:]
    l2_beta_grad = lbda * beta
    l2_grad = np.zeros((len(params)))
    l2_grad[-n_features:] = l2_beta_grad
    negloglik_term_grad = grad_negloglik(params, X, Y)
    
    return l2_grad + negloglik_term_grad
    

def proportional_odds_lbfgs_l2(X, y, lbda):
    Y = binarize(y)
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]
    
    p = n_features
    k = n_classes
    # TODO
    x0 = np.zeros(p + k - 1)
    x0[:k - 1] = np.arange(k - 1) 
    
 
    bounds = [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p
    x_min, _, _ = fmin_l_bfgs_b(pobj_l2, fprime=grad_l2,args=(X, Y, lbda),
                            x0=x0, bounds=bounds)
    

    # END TODO
    return x_min

x_min = proportional_odds_lbfgs_l2(X, y, lbda=1.)


# Check that `x_min` is ok.

# In[555]:


Y_proba = predict_proba(x_min)
y_pred = np.argmax(Y_proba, axis=1)

for j in range(k):
    Xj = X[y_pred == j]
    plt.plot(Xj[:, 0], Xj[:, 1], 'o', label='y = %d' % j, alpha=0.5)

plt.legend();


# Now that we have a gradient of the negative loglikelihood term we can implement other solvers. Namely you are going to implement:
# 
# - Proximal Gradient Descent (PGD aka ISTA)
# - Accelerated Proximal Gradient Descent (APGD aka FISTA)
# 
# Before this we are going to define the `monitor` class previously used in the second lab as well as plotting functions useful to monitor convergence.

# In[556]:


class monitor(object):
    def __init__(self, algo, obj, x_min, args=()):
        self.x_min = x_min
        self.algo = algo
        self.obj = obj
        self.args = args
        if self.x_min is not None:
            self.f_min = obj(x_min, *args)

    def run(self, *algo_args, **algo_kwargs):
        t0 = time.time()
        print (*algo_args, **algo_kwargs)
        _, x_list = self.algo(*algo_args, **algo_kwargs)
        self.total_time = time.time() - t0
        self.x_list = x_list
        if self.x_min is not None:
            self.err = [linalg.norm(x - self.x_min) for x in x_list]
            self.obj = [self.obj(x, *self.args) - self.f_min for x in x_list]
        else:
            self.obj = [self.obj(x, *self.args) for x in x_list]


def plot_epochs(monitors, solvers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for monit in monitors:
        ax1.semilogy(monit.obj, lw=2)
        ax1.set_title("Objective")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("objective")

    ax1.legend(solvers)

    for monit in monitors:
        if monit.x_min is not None:
            ax2.semilogy(monit.err, lw=2)
            ax2.set_title("Distance to optimum")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("$\|x_k - x^*\|_2$")

    ax2.legend(solvers)


def plot_time(monitors, solvers):
    for monit in monitors:
        objs = monit.obj
        plt.semilogy(np.linspace(0, monit.total_time, len(objs)), objs, lw=2)
        plt.title("Loss")
        plt.xlabel("Timing")
        plt.ylabel("$f(x_k) - f(x^*)$")

    plt.legend(solvers)


# <div class="alert alert-success">
#     <b>QUESTION 8a:</b>
#     <ul>
#     <li>
#         Implement the proximal gradient descent (PGD) method
#     </li>
#     </ul>
# </div>
# 
# The parameter `step` is the size of the gradient step.

# In[557]:


def pgd(x_init, grad, prox, n_iter=100, step=1., store_every=1,
        grad_args=(), prox_args=()):
    """Proximal gradient descent algorithm."""
    x = x_init.copy()
    x_list = []
    for i in range(n_iter):
        
        ### TODO 
        # grad should be the gradient of negloglik
        x = prox(x - step * grad(x, *grad_args), *prox_args)       

        
        ### END TODO
        if i % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


# <div class="alert alert-success">
#     <b>QUESTION 8b:</b>
#     <ul>
#     <li>
#         Using the monitor class and the plot_epochs function, display the convergence.
#     </li>
#     </ul>
# </div>
# 
# NOTE: You will have to provide a `step` value, which should be theoretially less than `1 / lipschitz_constant`. You will propose a value for it but you are not expected to provide a mathematical proof, unless you think it's a moral duty to give one...

# To help you we give you the proximal operator functions for $\ell_1$ and $\ell_2$ regularized models.

# In[558]:


def prox_l1(params, step, lbda, n_classes):
    return prox_f1(params, reg=step * lbda, n_classes=n_classes)

def prox_l2(params, step, lbda, n_classes):
    return prox_f2(params, reg=step * lbda, n_classes=n_classes)


# In[560]:


# from numpy.linalg import norm
# def lipschitz_constant(A,  lbda):
#     n = A.shape[0]
#     return norm(A, ord=2) ** 2 / n + lbda

# step = 1. / lipschitz_constant(X,  lbda)
# step


# In[561]:


# TODO

# lips_const = np.linalg.norm(X) ** 2 / 4
# step = 1 /  lips_const 
# step


# In[562]:


x_init = np.zeros(p + k - 1)
x_init[:k - 1] = np.arange(k - 1)
n_iter = 1000
lbda = .1

# lbda = 1. / n ** (0.5)
# step = 1. / lipschitz_constant(X,  lbda)
# step = 0.0000005
# TODO

lips_const = np.linalg.norm(X) ** 2 / 4
step = 1 /  lips_const 

monitor_pgd_l2 =  monitor(pgd, pobj_l2, x_min, (X, Y, lbda))
monitor_pgd_l2.run(x_init, grad_negloglik, prox_l2, n_iter, step,
                   grad_args=(X, Y), prox_args=(step, lbda, k))

monitors = [monitor_pgd_l2]
solvers = ["PGD"]
plot_epochs(monitors, solvers)


# In[563]:


# xlist = monitor_pgd_l2.x_list
# print (np.sum((x_min - xlist[-1]) ** 2))
# print (xlist[-1], '\n', x_min)
# n_iter = 10000


# Now for the $\ell_1$ regularization:

# In[565]:


lbda = 1.

# Run PGD for L1
monitor_pgd_l1 = monitor(pgd, pobj_l1, x_min=None, args=(X, Y, lbda))
monitor_pgd_l1.run(x_init, grad_negloglik, prox_l1, n_iter, step,
                   grad_args=(X, Y), prox_args=(step, lbda, k))

monitors = [monitor_pgd_l1]
solvers = ["PGD"]
plot_epochs(monitors, solvers)


# In[566]:



# xlist = monitor_pgd_l1.x_list
# print (np.sum((x_min - xlist[-1]) ** 2))
# print (xlist[-1], '\n', x_min)
# n_iter = 10000


# <div class="alert alert-success">
#     <b>QUESTION 9:</b>
#     <ul>
#     <li>
#         Implement the accelerated proximal gradient descent (APGD) and add this solver to the monitoring plots.
#     </li>
#     </ul>
# </div>

# In[567]:


def apgd(x_init, grad, prox, n_iter=100, step=1., store_every=1,
        grad_args=(), prox_args=()):
    """Accelerated proximal gradient descent algorithm."""
    x = x_init.copy()
    y = x_init.copy()
    t = 1.
    x_list = []
    
    t_new = 1
    b = 1 
    for i in range(n_iter):
        ### TODO
        x_new = prox(y - step * grad(y, *grad_args), *prox_args)
        b_new =  (1. + (1. + 4. * b ** 2.) ** 0.5) / 2.
        y = x_new + (b - 1.) / b_new * (x_new - x)
        x = x_new
        b = b_new

        ### END TODO
        if i % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


# In[568]:


lbda = 0.1

# TODO
monitor_apgd_l2 =  monitor(apgd, pobj_l2, x_min, args=(X, Y, lbda))
monitor_apgd_l2.run(x_init, grad_negloglik, prox_l2, n_iter, step,
                   grad_args=(X, Y), prox_args=(step, lbda, k))


# END TODO

monitors = [monitor_pgd_l2, monitor_apgd_l2]
solvers = ["PGD", "APGD"]
plot_epochs(monitors, solvers)


# In[569]:


# xlist = monitor_apgd_l2.x_list
# print (x_min, xlist[1000], xlist[3000], xlist[5000], xlist[6000], xlist[6500],xlist[7000],xlist[8000],xlist[9000],xlist[9999],)
# type(xlist)
# xlist[6800:8000]


# In[570]:


lbda = 1.


# TODO
monitor_apgd_l1 =  monitor(apgd, pobj_l1, x_min, args=(X, Y, lbda))
monitor_apgd_l1.run(x_init, grad_negloglik, prox_l1, n_iter, step,
                   grad_args=(X, Y), prox_args=(step, lbda, k))
# END TODO

monitors = [monitor_pgd_l1, monitor_apgd_l1]
solvers = ["PGD", "APGD"]
plot_epochs(monitors, solvers)


# # Part 2: Application
# 
# You will now apply your solver to the `wine quality` dataset. Given 11 features
# that describe certain wines (our samples), the objective it to predict the quality of the wine,
# encoded by integers between 3 and 8. Rather than using a multiclass classification
# model we're going to use a proportional odds model.
# 
# Let's first inspect the dataset:

# In[571]:


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


# In order to facilitate our experiment we're going to write a full scikit-learn estimator.
# 
# <div class="alert alert-success">
#     <b>QUESTION 10:</b>
#     <ul>
#     <li>
#         Implement the `fit` method from the estimator in the next cell
#     </li>
#     </ul>
# </div>

# In[588]:


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
        
        
        if self.penalty == 'l1': 
            objf = pobj_l1
            proxf = prox_l1
        elif  self.penalty == 'l2':   
            objf = pobj_l1
            proxf = prox_l2
            
        if  self.solver == "lbfgs": 
            if self.penalty == 'l2':
                bounds = [(None, None)] + [(0, np.inf) for j in range(k-2)] + [(None, None)] * p
                

                
            
        elif self.solver == "pgd" or  self.solver == "apgd":
            monitor_solver =  monitor(solver, objf, x_min=None, args=(X, Y, lbda))
#             monitor_solver.run(x_init, grad_negloglik, proxf, n_iter=self.max_iter, step=step, grad_args=(X, Y), prox_args=(step, lbda, k))
          
        x = x_init  
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


# In[582]:


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
params = x_init
proba = predict_proba(params, X=X)


# In[583]:


proba


# In[584]:


Y.shape


# In[585]:


proba.shape


# In[261]:


# # Option 1
# from scipy.misc import logsumexp

# def grad_negloglik(params, X=X, Y=Y):
#     # TODO
     
#     proba = predict_proba(params, X=X)    # = phi(z_j) -  phi(z_{j-1}) 
#     assert Y.shape == proba.shape
       
#     phi = np.cumsum(proba, axis=1) # i.e. phi(z1), ..., phi(zk)
#     Mj = phi * (1 - phi)           # phi(z_j)'
    
#     zerocl = np.zeros((Mj.shape[0],1))
#     Mjminus = np.column_stack((zerocl, Mj[:,0:k-1]))  # Mjminus
    
# #     YijdM_dprob = Y * (Mj - Mjminus) / (proba + np.finfo('float').eps)
    
# #     YimMm_dprob_m = Y[:,0:k-1] * Mj[:,0:k-1] /(proba[:,0:k-1] + np.finfo('float').eps)                                     # m = 1, 2, ..., k-1
    
# #     sum_upto_k_YijdM_dprob = np.sum(YijdM_dprob, axis=1)
    
# #     expandDim_sum_upto_k_YijdM_dprob = np.repeat(sum_upto_k_YijdM_dprob[:,np.newaxis],YimMm_dprob_m.shape[1], axis=1)
# #     delta_etas = -np.sum(YimMm_dprob_m + expandDim_sum_upto_k_YijdM_dprob, axis=0)
    
     
# #     yijMj_dp =  Y * Mj / (proba + np.finfo('float').eps)
# #     yijMminus_dp = Y * Mjminus / (proba + np.finfo('float').eps)
# #     sum_m_k = np.sum(yijMj_dp - yijMminus_dp, axis=1)
    
# #     yijDiffM_dp = Y * (Mj- Mjminus) / (proba + np.finfo('float').eps)
       
# #     sum_m_k = np.sum(yijDiffM_dp, axis=1)
# #     sum_in_bracket = yijMminus_dp  + np.repeat(sum_m_k[:, np.newaxis],yijMminus_dp.shape[1], axis=1)
    
# #     delta_etas = -np.sum(sum_in_bracket[:,0:k-1], axis=0)
    
    
    
    
#     Term1 = Y[:,0:k-1] * Mj[:,0:k-1] / (proba[:,0:k-1] + np.finfo('float').eps)      # m = 1, 2, ..., k-1
    
#     Term2 = Y[:,1:k] * Mj[:,0:k-1] / (proba[:,1:k]  + np.finfo('float').eps)
    
#     Term3 = Y[:,-1] * Mj[:,-1] /  (proba[:,-1] + np.finfo('float').eps)
    
#     Term3_dim_expansion = np.repeat(Term3[:, np.newaxis],  Term1.shape[1], axis=1)

    
#     ThreeTerms = Term1 - Term2 + Term3_dim_expansion  
    
#     addend_inbigBracket = np.flip(np.cumsum(np.flip(ThreeTerms, 1), axis=1), 1)
    
#     delta_eta = -np.sum(addend_inbigBracket, axis=0)

    
# #     print (delta_etas, delta_eta)

    

#     # calculating the derivatives for beta     
#     yijMjdp = Y * (Mj - Mjminus) / (proba + np.finfo('float').eps)
#     pro_yijMjdp = np.sum(yijMjdp, axis=1)[:,np.newaxis]
#     XyijM = X * pro_yijMjdp
#     betas = -np.sum(XyijM, axis=0)

    
    
        
#     delta = np.concatenate([delta_eta,betas])

#     # END TODO
#     return delta


# In[ ]:


# np.random.seed(100)
# x0 = rng.randn(p + k - 1)
# x0[1:k - 1] = np.abs(x0[1:k - 1])

# # x0
# # Y[:,0:k-1] 

# import autograd.numpy as np
# from autograd import grad

# aug1 = grad(negloglik)
# aug1(x0)

# print (grad_negloglik(x0, X, Y), '\n',  aug1(x0))



# X

# x1 = np.ones((X.shape[0], 1))

# print (np.shape(X), np.shape(x1))

# X * x1

#  right gradient array([ 213.43015163, -343.50994237, 4586.3534711 , 2139.84054958])

# x1 = x0.T
# np.shape(x1)
# x2 = x1[:,np.newaxis]
# x2
# x3 = np.hstack((x1, x1))
# x3

# k = 2
# b = np.zeros((len(x1),k))
# b[:,0] = x1

# b = np.repeat(x1[:, np.newaxis], 2, axis=1)
# b

#array([ 213.43015163, -343.50994237, 4586.3534711 , 2139.84054958]


# proba = predict_proba(x0, X=X)
# proba.shape

# Mj = proba
# k = 3
# zcl = np.zeros((Mj.shape[0],1))
# Mjminus = np.column_stack((zcl, Mj[:,0:k-1]))


# $$
# \mathcal{L}(\eta, \beta) =
# - \sum_{i=1}^{n} \left [ y_{i1} \log \left ( \phi(\eta_1 + \beta^T x_i) \right )
# + \sum_{j=2}^{k-1} y_{ij} \log \left ( \phi(\sum_{l=1}^j \eta_l + \beta^T x_i) - \phi(\sum_{l=1}^{j-1} \eta_l + \beta^T x_i) \right ) + y_{ik} \log \left ( 1 - \phi(\sum_{l=1}^{k-1} \eta_l + \beta^T x_i) \right ) \right ] .
# $$

