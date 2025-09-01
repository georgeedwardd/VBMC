#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
from scipy.stats import multivariate_normal
from matplotlib import cm
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import logsumexp


# ## MC Integration

# In[3]:


def f(x):
    return (np.e)**x -4*x**2 + x**3 + 1

true_value = 1.634948495125712
n = 1000
#Normal MC
x1 = np.random.uniform(0,1,n)
mc = np.mean(f(x1))
error1 = np.abs(true_value - mc)
StDev1 = np.sqrt(np.var(f(x1))/n)

#MC with Antithetic Variables
x2 = 1 - x1
mc_at = 0.5*np.mean(f(x1)) + 0.5*np.mean(f(x2))
error2 = np.abs(true_value - mc_at)
StDev2 = np.sqrt(np.var(0.5*f(x1) + 0.5*f(x2))/n)

print(f"True value: {true_value}")
print()
print("Normal Monte Carlo:")
print(f"Value: {mc:.6f}, Absolute Error: {error1:.6f}, StDev:{StDev1:.6f}")
print()
print("Monte Carlo with Antithetic Variables:")
print(f"Value: {mc_at:.6f}, Absolute Error: {error2:.6f}, StDev:{StDev2:.6f}")


# ## Gaussian Process

# <div class="alert alert-block alert-info">
# 
# The partitioned Gaussian vector:
# $$
# \begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} \right),
# $$
# 
# $$
# \boldsymbol{\mu}_{2|1} = \boldsymbol{\mu}_2 + \Sigma_{21} \Sigma_{11}^{-1} (\mathbf{x}_1 - \boldsymbol{\mu}_1),
# $$
# $$
# \Sigma_{2|1} = \Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}.
# $$
# 
# 
# Let:
# $$
# \mathbf{x}_1 = \mathbf{f}, \quad \mathbf{x}_2 = \mathbf{f}_*, \quad \boldsymbol{\mu}_1 = \mathbf{0}, \quad \boldsymbol{\mu}_2 = \mathbf{0},
# $$
# $$
# \Sigma_{11} = K(X, X), \quad \Sigma_{12} = K(X, X_*), \quad \Sigma_{21} = K(X_*, X), \quad \Sigma_{22} = K(X_*, X_*).
# $$
# 
# Then:
# $$
# p\left( \begin{bmatrix} \mathbf{f} \\ \mathbf{f}_* \end{bmatrix} \middle| X, X_* \right) = \mathcal{N}\left( \begin{bmatrix} \mathbf{0} \\ \mathbf{0} \end{bmatrix}, \begin{bmatrix} K(X, X) & K(X, X_*) \\ K(X_*, X) & K(X_*, X_*) \end{bmatrix} \right),
# $$
# 
# ---
# 
# The posterior $\textbf{mean}$ is:
# $$
# \begin{aligned}
# \boldsymbol{\mu}_{\mathbf{f}_* | \mathbf{f}} &= \boldsymbol{\mu}_2 + \Sigma_{21} \Sigma_{11}^{-1} (\mathbf{x}_1 - \boldsymbol{\mu}_1)\\ &= K(X_*, X) K(X, X)^{-1} \mathbf{f}.
# \end{aligned}
# $$
# 
# The posterior $\textbf{covariance}$ is:
# $$
# \begin{aligned}
# \Sigma_{\mathbf{f}_* | \mathbf{f}} &= \Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}\\ &= K(X_*, X_*) - K(X_*, X) K(X, X)^{-1} K(X, X_*).
# \end{aligned}
# $$
#   
# </div>

# #### 1D Gaussian Process

# In[4]:


def gp_predict1d(xs,ys,x_eval,kernel, s=1, l=1):
    
    x_eval = x_eval.reshape(-1,1)
    ys = ys.reshape(-1,1)
    
    K_xx = kernel(xs, xs, s, l)
    K_xs = kernel(x_eval, xs, s, l)
    K_sx = kernel(xs, x_eval, s, l)
    K_xx_eval = kernel(x_eval, x_eval, s, l)

    mu = K_xs@np.linalg.solve(K_xx, ys)
    sig = K_xx_eval - K_xs@np.linalg.solve(K_xx, K_sx)

    return mu.ravel(), sig + 1e-9*np.eye(sig.shape[0])

#Kernel 1
def rbf1d(x1,x2,s=10, l=0.2):
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(1,-1)
    dists = (x1-x2)**2/(2*l**2)
    return s**2*np.exp(-dists)

#Kernel 2
def matern1d(x1,x2,rho=1/5):
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(1,-1)
    return (1. + np.sqrt(3.)*np.abs(x1-x2)/rho)*np.exp(-np.sqrt(3.)*np.abs(x1-x2)/rho)


# In[5]:


x = np.linspace(0,1,8)
y = f(x)
s = 10
l = 0.2

x_eval = np.linspace(0,1,100)
mu_eval,var_eval = gp_predict1d(x,y,x_eval,kernel=rbf1d, s=s, l=l)
sig_diag = np.sqrt(np.diag(var_eval))

plt.plot(x_eval,mu_eval,label='Fit')
plt.plot(x,y,'+',label='Samples')
plt.fill_between(x_eval,mu_eval-3*sig_diag,mu_eval+3*sig_diag,alpha=0.1,label='3-sigma')
plt.legend()
plt.grid()
plt.show()


# In[6]:


#sample paths of the GP posterior
n_samples = 100000
samples = np.random.multivariate_normal(mu_eval, var_eval, size=n_samples)

#use trapezoidal rule weights for integration
dx = np.diff(x_eval)
weights = np.zeros_like(x_eval)
weights[1:-1] = (dx[:-1] + dx[1:])/2
weights[0] = dx[0]/2
weights[-1] = dx[-1]/2

#compute integrals and find 1% with lowest values
integrals = samples@weights
n_bottom = int(0.01*n_samples)
bottom_idx = np.argsort(integrals)[:n_bottom]
bottom_mean_curve = samples[bottom_idx].mean(axis=0)

#plot results
plt.plot(x_eval, mu_eval, label='Posterior mean', color='C0')
std_eval = np.sqrt(np.diag(var_eval))
plt.fill_between(
    x_eval,
    mu_eval - 3 * std_eval,
    mu_eval + 3 * std_eval,
    alpha=0.1,
    label='3-sigma band'
)
plt.plot(x_eval, bottom_mean_curve, color='C3')
plt.plot(x,y,'+',label='Samples')
plt.legend()
plt.grid()
plt.show()



# #### 2D Gaussian Process

# In[7]:


def gp_predict2d(xs, ys, x_eval, kernel, l = 5, s = 5):

    ys = ys.reshape(-1,1)

    K = kernel(xs, xs, l, s) + 1e-6*np.eye(xs.shape[0])
    K_s = kernel(xs, x_eval, l, s)
    K_ss = kernel(x_eval, x_eval, l, s)


    # Predictive mean
    mu = K_s.T@np.linalg.solve(K, ys)

    # Predictive covariance
    v = np.linalg.solve(K, K_s)
    sig = K_ss - K_s.T@v

    return mu.ravel(), sig


def rbf2d(X1, X2, l=2, s=1):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    diff = X1[:, None, :] - X2[None, :, :]
    sqdist = np.sum(diff**2, axis=2)
    return s**2*np.exp(-0.5/l**2*sqdist)


# Relative Humidity is a function of temperature and dew point.
# 
# $$
# e_s(T) = 6.1121 \, \exp\left(\left(18.678 - \frac{T}{234.5}\right) \cdot \left(\frac{T}{257.14 + T}\right)\right) \\
# 
# e = 6.1121 \, \exp\left(\left(18.678 - \frac{T_d}{234.5}\right) \cdot \left(\frac{T_d}{257.14 + T_d}\right)\right) \\
# $$
# $$
# RH = 100 \times \frac{e}{e_s(T)}
# $$

# In[8]:


def RH(T, Td):
    eT = 6.1121*np.exp((18.678 - T/234.5)*(T/(257.14 + T)))
    eTd = 6.1121*np.exp((18.678 - Td/234.5)*(Td/(257.14 + Td)))
    return 100*eTd/eT


# In[9]:


#training data points
Temp = np.linspace(0,50,5)
DewP = np.linspace(-10,35,4)
T, D = np.meshgrid(Temp, DewP)
TD = np.column_stack([T.ravel(), D.ravel()])
rh = RH(TD[:,0], TD[:,1])


#evaluation points
Temp_eval = np.linspace(0,50,50)
DewP_eval = np.linspace(-10,35,50)
T_eval, D_eval = np.meshgrid(Temp_eval, DewP_eval)
TD_eval = np.column_stack([T_eval.ravel(), D_eval.ravel()])

#fit GP
l = 120
s = 27000
mu_grid, var_grid = gp_predict2d(TD, rh, TD_eval, kernel=rbf2d, l=l, s=s)
mu_grid = mu_grid.reshape(T_eval.shape)
mu_grid = np.where(D_eval > T_eval, np.nan, mu_grid)


#calculate relative humidity at random point
T = np.random.uniform(0,50,1)[0]
D = np.random.uniform(-10,T,1)[0]
relhum = RH(T,D)
#predict at same random point
gp_est, gp_var = gp_predict2d(TD, rh, (T,D), kernel=rbf2d, l=l, s=s)
low = gp_est - 1.96*np.sqrt(gp_var)
high = gp_est + 1.96*np.sqrt(gp_var)
error = np.abs(relhum - gp_est)


print(f'For Temp: {T:.1f}°C and DewP: {D:.1f}°C')
print('--------------------------------------')
print(f'Relative humidity1 is: {relhum:.4f}%')
print(f'The gp estimate is: {gp_est[0]:.4f}%')

print(f'Error: {error[0]:.4f}')
print(f'95% CI: [{low[0][0]:.4f}%, {high[0][0]:.4f}%]')
print()

#plot of GP surface
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_eval, D_eval, mu_grid, cmap=cm.coolwarm, edgecolor='k')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Dew Point (°C)')
ax.set_zlabel('RH (%)')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()


# ## Bayesian Quadrature

# $$
# F(x) := \int_a^x f(t)\,dt
# $$
# 
# where $f \sim \mathcal{GP}(m(x), k(x, x'))$,
# 
# ---
# 
# Mean function:
# 
# $$
# \mathbb{E}[F(x)] = \mathbb{E}\left[\int_a^x f(t) dt\right] = \int_a^x \mathbb{E}[f(t)] dt = \int_a^x m(t)\,dt
# $$
# 
# 
# $$
# M(x) = \int_a^x m(t) dt
# $$
# 
# Covariance function:
# 
# 
# $$
# K(x, x') := \operatorname{Cov}(F(x), F(x')) = \operatorname{Cov} \left( \int_a^x f(t) dt,\ \int_a^{x'} f(s) ds \right)
# $$
# 
# Using Fubini's Theorem:
# $$
# K(x, x') = \int_a^x \int_a^{x'} \operatorname{Cov}(f(t), f(s))\, ds\, dt
# = \int_a^x \int_a^{x'} k(t, s)\, ds\, dt
# $$
# 
# 
# $$
# K(x, x') = \iint_{[a,x]\times[a,x']} k(t, s)\, dt\, ds
# $$
# 
# ---
# 
# $$
# F(x) \sim \mathcal{GP}\left( M(x),\ K(x, x') \right)
# $$
# 
# where:
# 
# * $M(x) = \int_a^x m(t)\,dt$
# * $K(x, x') =  \int_a^x \int_a^{x'} k(t, s)\,ds\,dt$
# 
# 
# $$
# E[F|y] = M(x) + \textbf{k}_F^T K^{-1} \left( y-m(X) \right)
# $$
# 
# $$
# Var(F|y) = K(x,x) - \textbf{k}_F^T K^{-1} \textbf{k}_F
# $$
# 
# 
# where $\textbf{k}_F = \left( \int_a^x k(t, x_1)dt, \int_a^x k(t, x_2)dt, \cdots , \int_a^x k(t, x_n)dt \right)$
# 
# ---

# Lets start by deriving the mean of the integral
# 
# Using RBF Kernel $k(y,x) = \sigma_f^2\exp \left(-\frac{1}{2l^2} |x-y|^2 \right)$:
# $$
# \begin{aligned}
# v[k] &= \int_a^x k(t,s) dt \\
#      &= \int_a^x \sigma_f^2\exp \left(-\frac{1}{2l^2} |s-t|^2 \right) dt \\
# \end{aligned}
# $$
# Let $l = \sigma$ hereon and factor out $\sigma_f^2 \cdot \sigma \sqrt{2\pi}$:
# $$
# \begin{aligned}
# v[k] &= \sigma_f^2 \cdot \sigma \sqrt{2\pi} \int_a^x \frac{1}{\sigma\sqrt{2\pi}}\exp \left(-\frac{(s-t)^2}{2\sigma^2} \right) dt \\
#      &= \sigma_f^2 \cdot \sigma \sqrt{2\pi} \int_a^x k^*(t,s) dt \\
# \end{aligned}
# $$
# Now working with just $k^*(t,s)$:
# $$
# \begin{aligned}
# v^*[k] &= \int_a^x \frac{1}{\sigma\sqrt{2\pi}}\exp \left(-\frac{(s-t)^2}{2\sigma^2} \right) dt \\
#      &= \frac{1}{2} \left[1+\text{erf} \left(\frac{x-s}{\sigma\sqrt{2}} \right) \right] - \frac{1}{2} \left[1+\text{erf} \\ \left(\frac{a-s}{\sigma\sqrt{2}} \right) \right] \\
#      &= \frac{1}{2} \left[ \text{erf} \left(\frac{x-s}{\sigma\sqrt{2}} \right) - \text{erf}\left(\frac{a-s}{\sigma\sqrt{2}} \right) \right]
# \end{aligned}
# $$
# 
# ---
# 
# For the uncertainty we have to integrate again. 
# 
# $$
# \begin{aligned}
# vv[k] &= \int_a^x \int_a^x k(t,s) dsdt \\
#       &= \sigma_f^2 \cdot \sigma \sqrt{2\pi} \int_a^x \int_a^x k^*(t,s) dtds \\
#       &= \sigma_f^2 \cdot \sigma \sqrt{2\pi} \int_a^x \frac{1}{2}\text{erf} \left(\frac{x-s}{\sigma\sqrt{2}} \right) - \frac{1}{2}\text{erf} \left(\frac{a-s}{\sigma\sqrt{2}} \right)ds \\
# \end{aligned}
# $$
# 
# Focusing on just $k^*(t,s)$, let $u_1 = \frac{x-s}{\sigma\sqrt{2}}$ and $u_2 = \frac{a-s}{\sigma\sqrt{2}}$  
# 
# $\implies$ $du_1 = du_2 = du = \frac{-1}{\sigma\sqrt{2}} \cdot ds$ 
# 
# $\implies$ $ds = -\sigma\sqrt{2} \cdot du$
# 
# $$
# \begin{aligned}
# vv^*[k] &= \int_a^x \frac{1}{2}\text{erf}(u_1)ds - \int_a^x\frac{1}{2}\text{erf}(u_2)ds \\
#        &= \int_{\frac{x-a}{\sigma\sqrt{2}}}^0 -\frac{1}{2}\text{erf}(u_1) \cdot \sigma\sqrt{2} \cdot du_1 - \int_0^{\frac{a-x}{\sigma\sqrt{2}}} - \frac{1}{2}\text{erf}(u_2) \cdot \sigma\sqrt{2} \cdot du_2 \\
#        &= \int_0^{\frac{x-a}{\sigma\sqrt{2}}} \frac{1}{2}\text{erf}(u) \cdot \sigma\sqrt{2} \cdot du - \int_{\frac{a-x}{\sigma\sqrt{2}}}^0 \frac{1}{2}\text{erf}(u) \cdot \sigma\sqrt{2} \cdot du \\
#        &= \frac{\sigma\sqrt{2}}{2} \left(\int_0^{\frac{x-a}{\sigma\sqrt{2}}} \text{erf}(u)du + \int_0^{\frac{x-a}{\sigma\sqrt{2}}} \text{erf}(u)du \right) \\
#        &= \sigma\sqrt{2} \left(\int_0^{\frac{x-a}{\sigma\sqrt{2}}} \text{erf}(u)du \right) \\
# \end{aligned}
# $$
# 
# Using $\int_0^\theta \text{erf}(u)du = \theta \, \text{erf}(\theta) + \frac{e^{-\theta^2} - 1}{\sqrt{\pi}}$, Let $\theta = \frac{x-a}{\sigma\sqrt{2}}$:
# 
# $$
# \begin{aligned}
# vv^*[k] &= \sigma\sqrt{2} \left(\theta \, \text{erf}(\theta) + \frac{e^{-\theta^2} - 1}{\sqrt{\pi}} \right) \\
#         &= (x-a) \, \text{erf} \left( \frac{x-a}{\sigma \sqrt{2}} \right) + \sigma\sqrt{2} \left( \frac{\exp{\left(-{\frac{(x-a)^2}{2 \sigma^2}}\right)} - 1}{\sqrt{\pi}} \right)\\
# \end{aligned}
# $$
# 
# ---
# 
# So we have:
# $$
# \begin{align*}
# v[k] &= \left( \sigma_f^2 \cdot \sigma \sqrt{2\pi} \right) v^*[k] \\
#      &= \left( \sigma_f^2 \cdot \sigma \sqrt{2\pi} \right) \, \left[\frac{1}{2} \text{erf} \left(\frac{x-s}{\sigma\sqrt{2}} \right) - \frac{1}{2} \text{erf}\left(\frac{a-s}{\sigma\sqrt{2}} \right) \right] \\
#      \\
# vv[k] &= \left( \sigma_f^2 \cdot \sigma \sqrt{2\pi} \right) vv^*[k] \\
#       &= \left( \sigma_f^2 \cdot \sigma \sqrt{2\pi} \right) \left[(x-a) \, \text{erf} \left( \frac{x-a}{\sigma \sqrt{2}} \right) + \sigma\sqrt{2} \left( \frac{\exp{\left(-{\frac{(x-a)^2}{2 \sigma^2}}\right)} - 1}{\sqrt{\pi}} \right) \right] \\
#       &= \sigma_f^2 \left[ \sigma \sqrt{2\pi} \, (x-a) \, \text{erf} \left( \frac{x-a}{\sigma \sqrt{2}} \right) + 2 \sigma^2 \left( e^{-{\frac{(x-a)^2}{2 \sigma^2}}} - 1 \right) \right] \\
# \end{align*}
# $$
# 
# 

# #### 1D Example

# In[10]:


def vk(input,lower, upper, s=1,l=1):
    return s**2*l*np.sqrt(2*np.pi)*(0.5*erf((upper-input)/(l*np.sqrt(2))) - 0.5*erf((lower-input)/(l*np.sqrt(2))))

def vvk(lower, upper, s=1, l=1):
    return s**2*(l*np.sqrt(2*np.pi)*(upper-lower)*erf((upper-lower)/(l*np.sqrt(2))) + 2*l**2*(np.exp(-(upper-lower)**2/(2*l**2)) - 1))

def integrate_rbf1d(x_eval,y_eval, lower, upper, s=1,l=1):
    
    x_eval = x_eval.reshape(-1,1)
    y_eval = y_eval.reshape(-1,1)
    
    kxx = rbf1d(x_eval,x_eval.T,s,l)
    
    mu = vk(x_eval,lower,upper,s,l).reshape(1,-1)@np.linalg.solve(kxx,y_eval)
    var = vvk(lower,upper,s,l) - vk(x_eval,lower,upper,s,l).reshape(1,-1)@np.linalg.solve(kxx,vk(x_eval,lower,upper,s,l).reshape(-1,1))
    
    return mu, var


# In[11]:


def f(x):
    return (np.e)**x -4*x**2 + x**3 + 1

def F(x):
    return (np.e)**x + (1/4)*x**4 -(4/3)*x**3 + x

#integral of f(x) from lower to upper
lower = -3
upper = 5
true_value = F(upper) - F(lower)

#kernel hyperparameters
s = 25
l = 0.6
#sample n points
n = 10
x = np.linspace(lower,upper,n)
y = f(x)

#points to plot gp
x_eval = np.linspace(lower,upper,100)
mu_eval,var_eval = gp_predict1d(x,y,x_eval,kernel=rbf1d,s=s,l=l)
sig_diag = np.sqrt(np.diag(var_eval))

#real curve
y1 = f(x_eval)

#plot gp
plt.plot(x_eval,mu_eval,label='Fit')
plt.plot(x_eval,y1,label='True f(x)')
plt.plot(x,y,'+',label='Samples')
plt.fill_between(x_eval,mu_eval-3*sig_diag,mu_eval+3*sig_diag,alpha=0.1,label='3-sigma')
plt.legend()
plt.grid()
plt.show()


# In[12]:


print(f"True value: {true_value}")
print()
print("Bayesian Quadrature")
for n in [1,3,5,8,12,20,50,75]:
    x = np.linspace(lower,upper,n)
    y = f(x)
    I_mu, I_var = integrate_rbf1d(x,y,lower,upper,s,l)
    error = np.abs(true_value - I_mu[0][0])
    print(f"{n} pts:")
    print(f"Estimate: {I_mu[0][0]:.8f} , StDev: {np.sqrt(I_var[0][0]):.10f}")
    print(f"Absolute Error: {error:.8f}")
    print()


# #### 2D Example

# 

# ## Run of VBMC

# In[13]:


def plot_GP_surrogate(X, y, X_new, y_pred_mean, y_pred_std, samples):
    plt.figure(figsize=(10, 6))

    plt.plot(X_new, y_pred_mean, lw=2, color='gray', label='GP Mean')
    plt.fill_between(X_new.ravel(),
                    y_pred_mean.ravel() - 2*y_pred_std.ravel(),
                    y_pred_mean.ravel() + 2*y_pred_std.ravel(),
                    color='lightgray', label='95% CI')


    plt.scatter(X, y, c='red', s=50, label='Observed')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Log GP Surrogate")
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.show()


# In[14]:


#In practice we wont know the the value of the parameter/s (obviously) and likelihoods will be difficult or expensive
def expensive_log_likelihood(mu1, mu2, sigma1, sigma2, obs):
    gauss1 = (1/(np.sqrt(2*np.pi)*sigma1))*np.exp(-0.5*((obs - mu1)/sigma1)**2)
    gauss2 = (1/(np.sqrt(2*np.pi)*sigma2))*np.exp(-0.5*((obs - mu2)/sigma2)**2)
    

    likelihood = 0.5*gauss1 + 0.5*gauss2 
    log_likelihood = np.log(likelihood)
    return log_likelihood

# def expensive_log_likelihood(mu1, mu2, sigma1, sigma2, obs):
#     return -(obs-mu1)**2


# In[15]:


n = 50
mu1_true=-7
mu2_true=4
sigma1_true=3
sigma2_true=1
lower = min(mu1_true, mu2_true) - 7*max(sigma1_true, sigma2_true)
upper = max(mu1_true, mu2_true) + 7*max(sigma1_true, sigma2_true)
l1 = min(mu1_true, mu2_true) - 6*max(sigma1_true, sigma2_true)
u1 = max(mu1_true, mu2_true) + 6*max(sigma1_true, sigma2_true)


#vbmc traditionally samples much smarter than randomly but here we sample 3 random points and infer a likelihood
#x = np.sort(np.random.uniform(lower,upper,n)).reshape(-1,1)
x = np.linspace(lower,upper,n).reshape(-1,1)
y = expensive_log_likelihood(mu1_true,mu2_true,sigma1_true,sigma2_true,x.ravel()).reshape(-1,1)

#hyperparameters for gp
s = 1
l = 1

#plot gp
x_eval= np.linspace(lower,upper,100).reshape(-1,1)
mu_eval,var_eval = gp_predict1d(x,y,x_eval,kernel=rbf1d,s=s,l=l)
sig_diag = np.sqrt(np.diag(var_eval))

plot_GP_surrogate(x, y, x_eval, mu_eval, sig_diag, samples=[])


x_vals = np.linspace(lower, upper, 100).ravel().reshape(-1,1)
log_likelihood = expensive_log_likelihood(mu1_true,mu2_true,sigma1_true,sigma2_true,x_vals).reshape(-1,1)

plt.plot(x_vals, log_likelihood, label='True Log Likelihood', color='orange')
plt.ylim(-15,0)
plt.legend()
plt.show()


# ### Find variational posterior

# $$
# \log p(\mathbf{x}) = \underbrace{\mathbb{E}_q[\log p(\mathbf{z},\mathbf{x})] - \mathbb{E}_q[\log q(\mathbf{z})]}_{= ELBO} + \underbrace{\text{KL}(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x}))}_{\ge 0}
# $$
# 
# $$
# \text{ELBO} = \underbrace{\mathbb{E}_q[\log p(\mathbf{z},\mathbf{x})]}_{=\text{Expected Log Joint}} + \underbrace{\left(-\mathbb{E}_q[\log q(\mathbf{z})]\right)}_{=\text{Entropy of Q}}
# $$

# 
# ---
# To find the entropy, we use Monte Carlo with a control variable.
# 
# 
# We want to find $-\mathbb{E}_q[\log q(\mathbf{z})]$, i.e the mean of $h(x) = -\log p(x)$. We introduce a control variable $g(x)$ whose expected value $\mathbb{E}[g]$ is known, so that we can use:
# 
# $$
# \hat{h}_{\text{cv}} = \frac{1}{n}\sum_{i=1}^n \left(h(x_i) - c \cdot \left[g(x_i) - \mathbb{E}[g] \right]\right)
# $$
# 
# since $\frac{1}{n} \left[ g(x_i) - \mathbb{E}[g] \right]  \rightarrow 0$ as $n \rightarrow \infty$.
# 
# It can be shown that $c$ is chosen optimally as $c = \frac{\text{Cov}(h,g)}{\text{Var}(g)}$.
# 
# Using regular monte carlo:
# 
# $$
# h(x) = -\log p_{\text{mixture}}(x) 
# $$
# 
# is sampled over points from the mixture.
# 
# Introducing control $g(x)$, the negative log-density of a single gaussian with the same mixture mean and variance has a known expected value:
# 
# $$
# g(x) = -\log \mathcal{N}(x; \mu_{\text{mix}}, \sigma_{\text{mix}}^2)
# $$
# 
# $$
# \mathbb{E}[g] = \frac{1}{2} \log(2 \pi e \sigma_{\text{mix}}^2)
# $$
# 
# Using the sampled covariance:
# 
# $$
# c_{\text{opt}} = \frac{\text{Cov}(h,g)}{\text{Var}(g)}
# $$
# 
# This ensures the variance of $h - c(g - E[g])$ is minimized.
# 
# ---
# 
# #### For control variables to work:
# 
# $\mathbb{E}[g]$ must be available exactly, otherwise we can’t subtract it correctly.
# 
# If $\text{Cov}(h,g) = 0$, the control variate gives no benefit.
# If $|\text{Corr}(h,g)| > 0$, then using the optimal $c = \frac{\text{Cov}(h,g)}{\text{Var}(g)}$ strictly reduces variance.
# 
# The variance of the adjusted estimator is:
# 
# $$
# \text{Var}(\hat{\theta}_{cv}) = \frac{1}{n}\Big(\text{Var}(h) - \frac{\text{Cov}(h,g)^2}{\text{Var}(g)}\Big).
# $$
# 
# So variance reduction is proportional to the square of the correlation.

# In[16]:


def mixture_pdf(x, mu1, mu2, s1, s2):
    term1 = 0.5*(1/(np.sqrt(2*np.pi)*s1))*np.exp(-0.5*((x - mu1)/s1)**2)
    term2 = 0.5*(1/(np.sqrt(2*np.pi)*s2))*np.exp(-0.5*((x - mu2)/s2)**2)
    return term1 + term2

def normal_logpdf(x, mu, s):
    return -0.5*np.log(2*np.pi*s*s) - 0.5*((x - mu)/s)**2

def bimodal_entropy2(mu1, mu2, s1, s2, n_samples=10000):
    #sample from mixture
    mix = np.random.choice([0,1], size=n_samples, p=[0.5,0.5])
    z = np.random.standard_normal(n_samples)
    samples = np.where(mix==0, mu1 + s1*z, mu2 + s2*z)

    #compute pdf at sampled points
    pdf = mixture_pdf(samples, mu1, mu2, s1, s2)
    h = -np.log(pdf)
    
    #introduce control variate, a normal with the underlyings mixture mean and variance
    mixed_mean = 0.5*(mu1 + mu2)
    mixed_var = 0.5*(s1**2 + (mu1 - mixed_mean)**2) + 0.5*(s2**2 + (mu2 - mixed_mean)**2)

    #g is control variate with same expected value
    g = -normal_logpdf(samples, mixed_mean, np.sqrt(mixed_var))
    E_g = 0.5*np.log(2*np.pi*np.e*mixed_var)
    
    #sample covariance and optimal coefficient c
    cov_hg = np.cov(h, g, ddof=1)[0,1]
    var_g  = np.var(g, ddof=1)
    c_opt = cov_hg/var_g if var_g > 0 else 0.0

    
    est = h.mean() - c_opt*(g.mean() - E_g)
    #estimate standard error using sample variance of the adjusted quantities
    adjusted = h - c_opt*(g - E_g)
    se = adjusted.std(ddof=1)/np.sqrt(n_samples)
    return est


#MC estimate of entropy of a bimodal Gaussian
def bimodal_entropy(mu1, mu2, s1, s2, n_samples=10000):
    #sample from mixture
    mix = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    samples = np.where(mix == 0, np.random.normal(mu1, s1, n_samples),
                                 np.random.normal(mu2, s2, n_samples))
    
    #compute pdf at sampled points
    pdf = 0.5*(1/(np.sqrt(2*np.pi)*s1))*np.exp(-0.5*((samples - mu1)/s1)**2) + \
          0.5*(1/(np.sqrt(2*np.pi)*s2))*np.exp(-0.5*((samples - mu2)/s2)**2)


    h = -np.log(pdf)
    se = h.std(ddof=1)/np.sqrt(n_samples)
    #return entropy
    return np.mean(h)


# In[17]:


def elbo(params, x, y, lower, upper, s=1, l=1):
    mu1, mu2, log_s1, log_s2 = params
    
    #ensure log_s1 and log_s2 are within reasonable bounds
    s1_q = np.exp(np.clip(log_s1, -3, 3))
    s2_q = np.exp(np.clip(log_s2, -3, 3))

    #compute expected log joint
    z = np.linspace(lower, upper, 100).reshape(-1,1)
    mu_gp, _ = gp_predict1d(x, y, z, kernel=rbf1d, s=s, l=l)
    q_pdf = (0.5*(1/(np.sqrt(2*np.pi)*s1_q))*np.exp(-0.5*((z - mu1)/s1_q)**2) + \
             0.5*(1/(np.sqrt(2*np.pi)*s2_q))*np.exp(-0.5*((z - mu2)/s2_q)**2))
    
    integrand = mu_gp.reshape(-1,1)*q_pdf
    expected_log_joint = np.trapezoid(integrand.ravel(), z.ravel())

    #entropy of Gaussian q
    entropy_q = bimodal_entropy2(mu1=mu1, mu2=mu2, s1=s1_q, s2=s2_q)

    #return elbo
    return (-expected_log_joint - entropy_q)


# In[ ]:


import jax.numpy as jnp
from jax import grad
from jax import jit
import torch


# In[25]:


import torch

# convert numpy arrays to torch tensors if needed
x_t = torch.tensor(x, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# initial parameters
params = torch.tensor([-1.0, 1.0, 0.0, 0.0], requires_grad=True)

optimizer = torch.optim.Adam([params], lr=0.01)

# helper functions need to be rewritten using torch
def elbo_torch(params, x, y, lower, upper, s=1, l=1):
    mu1, mu2, log_s1, log_s2 = params
    s1_q = torch.exp(torch.clamp(log_s1, -3, 3))
    s2_q = torch.exp(torch.clamp(log_s2, -3, 3))

    z = torch.linspace(lower, upper, 100).reshape(-1,1)

    # assume gp_predict1d can be rewritten or replaced with torch operations
    mu_gp, _ = gp_predict1d_torch(x, y, z, kernel=rbf1d, s=s, l=l)

    q_pdf = 0.5*(1/(torch.sqrt(2*torch.pi)*s1_q))*torch.exp(-0.5*((z - mu1)/s1_q)**2) + \
            0.5*(1/(torch.sqrt(2*torch.pi)*s2_q))*torch.exp(-0.5*((z - mu2)/s2_q)**2)

    integrand = mu_gp.reshape(-1,1)*q_pdf
    expected_log_joint = torch.trapz(integrand.ravel(), z.ravel())

    entropy_q = bimodal_entropy2_torch(mu1, mu2, s1_q, s2_q)

    return -expected_log_joint - entropy_q

# optimisation loop
for i in range(3000):
    optimizer.zero_grad()
    loss = elbo_torch(params, x_t, y_t, lower, upper, s=1, l=1)
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
        mu1_opt, mu2_opt, log_s1_opt, log_s2_opt = params.detach().numpy()
        print(f"Iter {i+1}: mu1={mu1_opt:.4f}, mu2={mu2_opt:.4f}, s1={np.exp(log_s1_opt):.4f}, s2={np.exp(log_s2_opt):.4f}, ELBO={-loss.item():.4f}")


# In[ ]:


#intial parameters
params = [-1.0, 1.0, 0.0, 0.0]
n_iter =500

params = torch.tensor(params, requires_grad=True)
optimizer = torch.optim.Adam([params], lr=0.01)

for i in range(n_iter):
    optimizer.zero_grad()
    loss = -elbo(params, x, y, l1, u1, s, l)
    loss.backward()
    optimizer.step()

print("Optimised params:", params.detach().numpy())



# In[21]:


#This should theoretically be the highest ELBO we can get
-elbo([mu1_true,mu2_true,np.log(sigma1_true),np.log(sigma2_true)],x,y,s,l)


# In[22]:


x_min = min(mu1_opt - 5*s1_opt, mu2_opt - 5*s2_opt)
x_max = max(mu1_opt + 5*s1_opt, mu2_opt + 5*s2_opt)
x_vals = np.linspace(x_min, x_max, 1000)

#estimated mixture PDF
pdf_vals = 0.5*norm.pdf(x_vals, loc=mu1_opt, scale=s1_opt) + \
           0.5*norm.pdf(x_vals, loc=mu2_opt, scale=s2_opt)

#true mixture PDF
true_vals = 0.5*norm.pdf(x_vals, loc=mu1_true, scale=sigma1_true) + \
            0.5*norm.pdf(x_vals, loc=mu2_true, scale=sigma2_true)

#compare estimated and true distributions
plt.figure(figsize=(8,5))
plt.plot(x_vals, pdf_vals, label='Estimated Posterior')
plt.plot(x_vals, true_vals, label='True Distribution')
plt.axvline(mu1_true, color='r', linestyle='--', label=f'True μ1 = {mu1_true:.3f}')
plt.axvline(mu2_true, color='m', linestyle='--', label=f'True μ2 = {mu2_true:.3f}')
plt.title('Variational Approximate Posterior Distribution over Parameter')
plt.xlabel('Parameter')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

