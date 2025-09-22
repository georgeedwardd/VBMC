import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import jax
from jax import grad
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import cho_solve
from utils import *


#Kernel 1
def rbf(x1, x2, s=1, L=1.0):
    x1 = jnp.atleast_2d(x1)
    x2 = jnp.atleast_2d(x2)
    d, n1, n2 = x1.shape[1], x1.shape[0], x2.shape[0]
    if L.size == 1:
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(1, -1)
        dists = (x1 - x2)**2/(2*L**2)
        return s**2*jnp.exp(-dists)

    L = L@L.T
    L_chol = jnp.linalg.cholesky(L)

    diff = x1[:, None, :] - x2[None, :, :]
    diff_flat = diff.reshape(-1, d).T
    y = jnp.linalg.solve(L_chol, diff_flat).T.reshape(n1, n2, d)  
    sqdist = jnp.sum(y**2, axis=2)
    return s**2*jnp.exp(-0.5*sqdist)

#mean and covariance matrix of the posterior predictive distribution
def gp_predict(xs, ys, x_eval, kernel, s, L):
    xs = jnp.asarray(xs)
    x_eval = jnp.asarray(x_eval)
    ys = jnp.asarray(ys)
    L = jnp.asarray(L)

    # Ensure 2D arrays
    if xs.ndim == 1:
        xs = xs[:, None]
    if x_eval.ndim == 1:
        x_eval = x_eval[:, None]
    if ys.ndim == 1:
        ys = ys[:, None]

    if L.ndim == 1:
        L = jnp.diag(L)

    K = kernel(xs, xs, s, L) + 1e-6*jnp.eye(xs.shape[0])
    K_s = kernel(xs, x_eval, s, L)
    K_ss = kernel(x_eval, x_eval, s, L)

    alpha = jnp.linalg.solve(K, ys)
    mu = K_s.T @ alpha
    v = jnp.linalg.solve(K, K_s)
    sig = K_ss - K_s.T @ v + 1e-9*jnp.eye(K_ss.shape[0])

    return mu.ravel(), sig


def neg_gp_likelihood(params, xs, ys, kernel, sigma_e=0):
    s = params[0]
    L_flat = params[1:]
    xs = jnp.reshape(xs, (-1, xs.shape[-1] if xs.ndim > 1 else 1))
    L = jnp.diag(L_flat)

    n = xs.shape[0]
    K_xx = kernel(xs, xs, s, L) + (sigma_e + 1e-6)*jnp.eye(n)

    Lc = jnp.linalg.cholesky(K_xx)
    alpha = cho_solve((Lc, True), ys)

    term1 = 0.5*ys@alpha
    term2 = jnp.sum(jnp.log(jnp.diag(Lc)))
    term3 = 0.5*n*jnp.log(2*jnp.pi)

    return term1 + term2 + term3


##############################################################################
#1d quadrature

def vk1(input,s,L,mu_p,s_p):
    t1 = jnp.sqrt(L**2/(L**2 + s_p**2))
    t2 = jnp.exp(-(input-mu_p)**2/(2*(L**2 + s_p**2)))
    return s**2*t1*t2

def vvk1(s,L,s_q):
    t1 = jnp.sqrt(L**2/(L**2 + 2*s_q**2))
    return s**2*jnp.sqrt(L**2/(L**2 + 2*s_q**2))

def integrate_gaussian_1(x_eval, y_eval, s, L, mu_p, s_p):
    x_eval = x_eval.reshape(-1,1)
    y_eval = y_eval.reshape(-1,1)
    
    #kernel matrix
    Kxx = rbf(x_eval, x_eval, s, L)
    Kxx += 1e-10*jnp.eye(len(Kxx))
    
    #cross-covariances
    kF = vk1(x_eval, s, L, mu_p, s_p).reshape(-1,1)
    vF = vvk1(s, L, s_p)
    
    #posterior mean and variance
    mu = (kF.T@jnp.linalg.solve(Kxx, y_eval)).item()
    var = vF - (kF.T@jnp.linalg.solve(Kxx, kF)).item()
    var = jnp.maximum(var, 0.0)
    
    return mu, var[0][0]


def Iij(i, j, s, L, mu_params, s_params):
    t1 = s**2*jnp.sqrt(L**2 / (L**2 + s_params[i]**2 + s_params[j]**2))
    t2 = jnp.exp(- (mu_params[i] - mu_params[j])**2/(2*(L**2 + s_params[i]**2 + s_params[j]**2)))
    return t1*t2

##############################################################################
#nd quadrature

def vk2(x_eval, s, L, mu_p, S_p):
    S_p = jnp.array(S_p)
    L = jnp.atleast_2d(jnp.array(L)) 
    x_eval = jnp.atleast_2d(x_eval).T
    mu_p = jnp.atleast_2d(mu_p).T

    L2 = L@L.T
    t1 = jnp.linalg.det(S_p + L2)
    t1 = jnp.linalg.det(L)/jnp.sqrt(t1)

    inv_term = jnp.linalg.inv(S_p + L2)
    diff = x_eval - mu_p
    t2 = jnp.exp(-0.5*diff.T@inv_term@diff)

    return s**2*t1*t2


def vvk2(s, L, S_p):
    S_p = jnp.array(S_p)
    L = jnp.atleast_2d(jnp.array(L)) 
    L2 = L@L.T
    det_term = jnp.linalg.det(2*S_p + L2)
    t1 = jnp.linalg.det(L)/jnp.sqrt(det_term)
    return s**2*t1


def integrate_gaussian_n(x_eval, y_eval, s, L, mu_p, S_p):
    L = jnp.asarray(L)
    if L.ndim == 1:
        L = jnp.diag(L) 

    x_eval = jnp.atleast_2d(x_eval).astype(jnp.float64)
    y_eval = jnp.atleast_2d(y_eval).astype(jnp.float64)
    n, _ = x_eval.shape

    # Kernel matrix
    Kxx = rbf(x_eval, x_eval, s, L)
    Kxx += 1e-10*jnp.eye(n)

    # Cross-covariances with Gaussian measure
    kF = jnp.array([vk2(x_eval[i], s, L, mu_p, S_p) for i in range(n)]).reshape(-1,1)

    # Variance of the integral
    vF = vvk2(s, L, S_p)

    # Posterior mean and variance of the integral
    mu = float((kF.T@jnp.linalg.solve(Kxx, y_eval)).item())
    var = float(jnp.maximum(vF - (kF.T@jnp.linalg.solve(Kxx, kF)).item(), 0.0))

    return mu, var



##########################################################################################################
#VBMC 1d

#multimodal gaussian mixture pdf
def mixture_pdf(x, mu_params, s_params, weights=None):
    x = jnp.atleast_1d(x)
    mu_params = jnp.array(mu_params)
    s_params = jnp.array(s_params)

    n_components = mu_params.shape[0]

    # default equal weights
    if weights is None:
        weights = jnp.ones(n_components)/n_components
    else:
        weights = jnp.array(weights)/jnp.sum(weights)

    t1 = 1.0/(jnp.sqrt(2*jnp.pi)*s_params)
    t2 = jnp.exp(-0.5*((x[:, None] - mu_params[None, :])/s_params[None, :])**2)
    terms = t1[None, :]*t2

    return jnp.sum(weights[None, :]*terms, axis=1)

#MC estimate of entropy of a multimodal gaussian
def entropy(params, weights=None,key=None, random=True):
    mu_params = jnp.array(params[0])
    s_params = jnp.array(params[1])
    n_components = len(mu_params)
    key1, key2 = jax.random.split(key)

    #assume equal weights
    if weights is None:
        weights = jnp.ones_like(mu_params)/len(mu_params)
    else:
        weights = jnp.array(weights)/jnp.sum(weights)

    n_samples = 100 if random else 100_000
        
    #sample from mixture
    mix = jax.random.choice(key1, n_components, shape=(n_samples,), p=weights)
    z = jax.random.normal(key2, shape=(n_samples,))
    samples = mu_params[mix] + s_params[mix]*z

    #compute pdf at sampled points
    pdf_vals = jax.vmap(lambda x: mixture_pdf(x, mu_params, s_params, weights))(samples)

    #return entropy
    h = -jnp.log(pdf_vals + 1e-12)
    return jnp.mean(h)


def log_joint(x_eval, y_eval, s, L, params, weights=None, n_s=10, key=None, random=True):
    mu_params = jnp.array(params[0])
    s_params = jnp.array(params[1])
    n_components = len(mu_params)
    
    x_eval = x_eval.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)
    
    #assume equal weights
    if weights is None:
        weights = jnp.ones(n_components)/n_components
    else:
        weights = jnp.array(weights)/jnp.sum(weights)

    #kernel matrix
    Kxx = rbf(x_eval, x_eval.T, s, L)
    Kxx += 1e-10*jnp.eye(len(Kxx))
    
    #cross-covariances for all components
    kFs = jnp.stack([vk1(x_eval, s, L, mu_params[i], s_params[i]).reshape(-1, 1)
                    for i in range(n_components)], axis=0)
    kF = jnp.sum(weights[:, None, None]*kFs, axis=0).reshape(-1, 1)
    
    
    #posterior mean
    mu = kF.T@jnp.linalg.solve(Kxx, y_eval)
    mu = jnp.squeeze(mu)

    #posterior variance
    vv_mix = jnp.sum(weights[:, None]*weights[None, :]*
                jnp.array([[Iij(i, j, s, L, mu_params, s_params) for j in range(n_components)] 
                            for i in range(n_components)]))
    var = vv_mix - (kF.T @ jnp.linalg.solve(Kxx, kF))
    var = jnp.maximum(var, 0.0)
    var = jnp.squeeze(var) 

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey)

    return mu + random*jnp.sqrt(var/n_s)*z


#ELBO computation for trimodal guassian
def elbo(params, x, y, s=1, L=1, key=None, random=True):
    #convert parameters to appropriate form
    mixture_params = (params[0], jnp.exp(params[1]))

    #compute expected log joint
    expected_log_joint = log_joint(x, y, s, L, mixture_params, key=key, random=random)

    #entropy of Gaussian q
    entropy_q = entropy(mixture_params, weights=None,key=key, random=random)

    #return negative elbo
    return -expected_log_joint - entropy_q

def vbmc1(x, y, params, grad_elbo, s=5, L=5, lr=0.005, n1=10, n2=20_000, key=None):
    elbo_history = []
    best_elbo = -jnp.inf
    best_params = params.copy()
    for j in range(n1):
        for i in range(n2):
            key, _ = jax.random.split(key)
            grads = grad_elbo(params, x, y, s, L, key=key)
            params = params - lr*grads

            if (i+1) % (n2/2) == 0:
                current_elbo = -elbo(params, x, y, s, L, key=key, random=False)
                elbo_history.append(current_elbo)
                if current_elbo > best_elbo:
                    best_elbo = current_elbo
                    best_params = params.copy()
                print(f"Iter {j+1}, {i+1}: ELBO = {current_elbo}")
    return best_params, best_elbo, elbo_history

#returns new gp points
def active_sample(x, y, proposed_points, params_true, params_q, s, L, alpha=1,beta=1,gamma=1,n_points=1):

    mu_q = params_q[0]
    s_q = params_q[1]
    acquisition_values = jnp.zeros((int(n_points), int(proposed_points.shape[0])))
    for i in range(n_points):
        grad_fn = grad(neg_gp_likelihood)
        s_init = s
        l_init = L
        init_params = jnp.concatenate([jnp.ravel(s_init), l_init.ravel()])
        res = minimize(
            fun=lambda p: float(neg_gp_likelihood(p, x, y, rbf)),
            x0=init_params,
            jac=lambda p: jnp.array(grad_fn(p, x, y, rbf)),
            method="L-BFGS-B",
            bounds=[(1, 12)]*len(init_params)
            )
        #kernel hyperparameters
        s = res.x[0]
        L = jnp.diag(res.x[1:])
        
        mu, var = gp_predict(x, y, proposed_points, kernel=rbf, s=s, L=L)
        var = jnp.diag(var)
        q = mixture_pdf(proposed_points, mu_q, s_q)
        f_exp = jnp.exp(mu)
        
        acquisition_values = acquisition_values.at[i].set(var**alpha*q**beta*f_exp**gamma)
        
        #pick next point
        best_idx = jnp.argmax(acquisition_values[i])
        x_new = proposed_points[best_idx].reshape(-1,1)
        y_new = expensive_log_likelihood(x_new, params_true)
        x = jnp.vstack([x, x_new])
        y = jnp.concatenate([y, y_new])
        sorted_idx = jnp.argsort(x.flatten())
        x = x[sorted_idx]
        y = y[sorted_idx]
        
    return x, y, s, L, acquisition_values

def vbmc2(x, y, params, grad_elbo, s=5, L=5, alpha=1, 
          beta=1, gamma=1, lr=0.005, n_points=1, 
          n1=10, n2=20_000, plots=True, 
          real_log_likelihood=None, key=None):
    elbo_history = []
    best_elbo = -jnp.inf
    best_params = params.copy()
    x_eval = jnp.linspace(lower,upper,200)
    for j in range(n1):
        for i in range(n2):
            key, _ = jax.random.split(key)
            grads = grad_elbo(params, x, y, s, L, key=key)
            params = params - lr*grads

            if (i+1) % (n2/2) == 0:
                current_elbo = -elbo(params, x, y, s, L, key=key, random=False)
                elbo_history.append(current_elbo)
                if current_elbo > best_elbo:
                    best_elbo = current_elbo
                    best_params = params.copy()
                print(f"Iter {j+1}, {i+1}: ELBO = {current_elbo}")
    
        s = jnp.atleast_1d(s)
        x, y, s, L, acquisition_values = active_sample(x, y, x_eval, params_true, best_params, 
                                                       s, L, alpha, beta, gamma, n_points)

        if plots:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            for i in range(acquisition_values.shape[0]):
                axes[0].plot(x_eval, acquisition_values[i], label=f'Iteration {i+1}')
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("Acquisition Value")
            axes[0].set_title("Acquisition Functions")
            axes[0].legend()

            #plot gp
            mu_eval,var_eval = gp_predict(x,y,x_eval,kernel=rbf,s=s,L=L)
            sig_diag = jnp.sqrt(np.diag(var_eval))
            axes[1].plot(x_eval, mu_eval, lw=2, color='gray', label='GP Mean')
            axes[1].fill_between(x_eval.ravel(),
                                mu_eval.ravel() - 2*sig_diag.ravel(),
                                mu_eval.ravel() + 2*sig_diag.ravel(),
                                color='lightgray', label='95% CI')
            axes[1].scatter(x, y, c='red', s=20, label='Observed')
            axes[1].plot(x_eval, real_log_likelihood, label='True Log Likelihood')
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("f(x)")
            axes[1].set_title("Log GP Surrogate")
            axes[1].legend(loc='upper left')

            #current mixture
            mu_best, s_best = best_params
            mixture_vals = mixture_pdf(x_eval, mu_best, s_best)
            axes[2].plot(x_eval, mixture_vals, lw=2, color='blue', label='Current Mixture')
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("Density")
            axes[2].set_title("Current Mixture PDF")
            axes[2].legend(loc='upper left')

            plt.tight_layout()
            plt.show()

        
    return best_params, best_elbo, elbo_history