import jax.numpy as jnp
import jax
from scipy.special import erf
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.linalg import cho_solve
from gp_utils import *

##############################################################################
#univariate no measure
def vk(input,lower, upper, s=1,L=1):
    t1 = jnp.sqrt(2*jnp.pi)
    t2 = (0.5*erf((upper-input)/(L*jnp.sqrt(2))) - 0.5*erf((lower-input)/(L*jnp.sqrt(2))))
    return s**2*L*t1*t2

def vvk(lower, upper, s=1, L=1):
    t1 = L*jnp.sqrt(2*jnp.pi)*(upper-lower)
    t2 = erf((upper-lower)/(L*jnp.sqrt(2)))
    t3 = jnp.exp(-(upper-lower)**2/(2*L**2)) - 1
    return s**2*(t1*t2 + 2*L**2*t3)

def integrate_rbf(x_eval,y_eval, lower, upper, s=1,L=1):
    x_eval = x_eval.reshape(-1,1)
    y_eval = y_eval.reshape(-1,1)
    
    kxx = rbf(x_eval,x_eval.T,s,L)
    kxx += 1e-8*jnp.eye(len(kxx))
    kxx = (kxx + kxx.T)/2
    
    kf = vk(x_eval,lower,upper,s,L)

    mean = kf.reshape(1,-1)@jnp.linalg.solve(kxx,y_eval)
    var = vvk(lower,upper,s,L) - kf.reshape(1,-1)@jnp.linalg.solve(kxx,kf.reshape(-1,1))
    var = jnp.maximum(var, 0.0)
    return mean, var


##############################################################################
#univariate

def vk1(input,s,L,mu_p,s_p):
    t1 = jnp.sqrt(L**2/(L**2 + s_p**2))
    t2 = jnp.exp(-(input-mu_p)**2/(2*(L**2 + s_p**2)))
    return s**2*t1*t2

#########################
#single gaussian

def vvk1(s,L,s_q):
    t1 = jnp.sqrt(L**2/(L**2 + 2*s_q**2))
    return s**2*jnp.sqrt(L**2/(L**2 + 2*s_q**2))

def integrate_gaussian_1(x_eval, y_eval, s, L, mu, sigma, m0=None, x_m=None, w=None):    
    x_eval = x_eval.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)

    #kernel matrix
    Kxx = rbf(x_eval, x_eval.T, s, L)
    Kxx += 1e-10*jnp.eye(len(Kxx))
    
    #cross-covariances for all components
    kF = vk1(x_eval, s, L, mu, sigma).reshape(-1, 1)

    #mean function integral
    if m0 is None:
        m0 = jnp.mean(y_eval)
        x_m = jnp.mean(x_eval, axis=0)
        w = (jnp.max(x_eval, axis=0) - jnp.min(x_eval, axis=0))/2.0
    m = m0 - 0.5*((x_eval - x_m)**2)/w**2
    M = m0 - 0.5*((mu - x_m)**2 + sigma**2)/w**2

    #posterior mean
    mean = M + kF.T@jnp.linalg.solve(Kxx, y_eval - m)
    mean = jnp.squeeze(mean)

    #posterior variance
    vv_mix = vvk1(s, L, sigma).reshape(-1, 1)
    adjustment = (kF.T @ jnp.linalg.solve(Kxx, kF))
    var = vv_mix - adjustment
    var = jnp.maximum(var, 0.0)
    var = jnp.squeeze(var) 

    return mean, var


#########################
#mixture


def Iij_1(i, j, s, L, mu_params, s_params):
    t1 = s**2*jnp.sqrt(L**2 / (L**2 + s_params[i]**2 + s_params[j]**2))
    t2 = jnp.exp(- (mu_params[i] - mu_params[j])**2/(2*(L**2 + s_params[i]**2 + s_params[j]**2)))
    return t1*t2

def integrate_mixture_1(x_eval, y_eval, s, L, params, weights=None,m0=None,x_m=None,w=None):
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
    
    #mean function integral
    m = m0 - 0.5*((x_eval - x_m)** 2)/w**2
    M_components = m0 - 0.5*((mu_params - x_m)** 2 + s_params**2)/w**2
    M = jnp.sum(weights*M_components)

    #posterior mean
    mu = M + kF.T@jnp.linalg.solve(Kxx, y_eval - m)
    mu = jnp.squeeze(mu)

    # #posterior variance
    I_matrix = jnp.array([[Iij_1(i, j, s, L, mu_params, s_params) 
                        for j in range(n_components)] 
                        for i in range(n_components)])
    vv_mix = jnp.sum(weights[:, None] * weights[None, :] * I_matrix)

    # Step 2: compute vk for each mixture component
    vk_list = [vk1(x_eval, s, L, mu_params[i], s_params[i]).reshape(-1, 1) 
            for i in range(n_components)]

    # Step 3: adjustment term using full double sum over mixture components
    adjustment = jnp.sum(weights[:, None] * weights[None, :] * 
                        jnp.array([[vk_list[i].T @ jnp.linalg.solve(Kxx, vk_list[j])
                                    for j in range(n_components)]
                                    for i in range(n_components)]))

    # Step 4: posterior variance
    var = vv_mix - adjustment
    var = jnp.maximum(var, 0.0)
    var = jnp.squeeze(var)

    return mu, var




##############################################################################
#multi variate

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

#########################
#single gaussian

def vvk2(s, L, S_p):
    S_p = jnp.array(S_p)
    L = jnp.atleast_2d(jnp.array(L)) 
    L2 = L@L.T
    det_term = jnp.linalg.det(2*S_p + L2)
    t1 = jnp.linalg.det(L)/jnp.sqrt(det_term)
    return s**2*t1


def integrate_gaussian_n(x_eval, y_eval, s, L, mu_p, S_p, m0=None, x_m=None, w=None):
    L = jnp.diag(L) 
    S_p = jnp.array([jnp.diag(S_p)])
                    
    x_eval = jnp.atleast_2d(x_eval).astype(jnp.float64)
    y_eval = jnp.atleast_2d(y_eval).reshape(-1, 1)
    n, _ = x_eval.shape

    # Kernel matrix
    Kxx = rbf(x_eval, x_eval, s, L)
    Kxx += 1e-10*jnp.eye(n)

    # Cross-covariances with Gaussian measure
    kF = jnp.array([vk2(x_eval[i], s, L, mu_p, S_p) for i in range(n)]).reshape(-1,1)

    # Variance of the integral
    vF = vvk2(s, L, S_p)

    #mean function integral
    if m0 is None:
        m0 = jnp.mean(y_eval)
        x_m = jnp.mean(x_eval, axis=0)
        w = (jnp.max(x_eval, axis=0) - jnp.min(x_eval, axis=0)) / 2.0

    m = m0 - 0.5 * jnp.sum(((x_eval - x_m) / w) ** 2, axis=1, keepdims=True)
    M = m0 - 0.5 * (jnp.sum(((mu_p - x_m) / w) ** 2) + jnp.sum(S_p.flatten()/(w**2)))

    # Posterior mean and variance of the integral
    mu = float((M + kF.T@jnp.linalg.solve(Kxx, y_eval - m)).item())
    var = float(jnp.maximum(vF - (kF.T@jnp.linalg.solve(Kxx, kF)).item(), 0.0))

    return mu, var
#########################
#mixture


def Iij_n(i, j, s, L, params):
    mu_params = jnp.array(params[0])
    S_params = jnp.array([jnp.diag(S) for S in params[1]])
    mu_i, mu_j = mu_params[i], mu_params[j]
    Sigma_i, Sigma_j = S_params[i], S_params[j]
    
    # Compute combined covariance
    cov_sum = Sigma_i + Sigma_j + L @ L
    
    # Determinant factor
    det_factor = jnp.linalg.det(cov_sum)**(-0.5)
    
    # Exponential factor
    diff = mu_i - mu_j
    exp_factor = jnp.exp(-0.5 * diff.T @ jnp.linalg.inv(cov_sum) @ diff)
    
    # Determinant of L
    det_L = jnp.linalg.det(L)
    
    return s**2 * det_L * det_factor * exp_factor

##accept new x,y points and take in old kxx inverse.
def integrate_mixture_n(x, y, m_params, hp_params, weights=None, chol_k=None):
    mu_params = jnp.array(m_params[0])
    S_params  = jnp.array([jnp.diag(S) for S in m_params[1]])
    s, L, m0, x_m, w, sigma_e = hp_params

    n_components, d = mu_params.shape

    #x = jnp.atleast_2d(x).astype(jnp.float64)
    #y = jnp.atleast_2d(y).reshape(-1, 1)
    n_eval = x.shape[0]

    L = jnp.diag(L)


    # cross-covariances
    def comp_vals(mu_j, S_j):
        return jax.vmap(lambda xi: vk2(xi, s, L, mu_j, S_j))(x).reshape(-1)
    
    kFs = jnp.stack([comp_vals(mu_params[j], S_params[j])
                     for j in range(n_components)], axis=1)


    kF = (weights[None, :] * kFs).sum(axis=1).reshape(-1, 1)


    m = m0 - 0.5 * jnp.sum(((x - x_m) / w) ** 2, axis=1, keepdims=True)
    M_components = jnp.array([
        m0 - 0.5 * (jnp.sum(((mu_params[j] - x_m) / w) ** 2) + jnp.sum(S_params[j] / (w ** 2)))
        for j in range(n_components)
    ])
    M = jnp.sum(weights * M_components)


    if chol_k is None:
        # GP covariance matrix
        Kxx = rbf(x, x, s, L)
        Kxx += 1e-10*jnp.eye(n_eval)
        chol_k = jnp.linalg.cholesky(Kxx)

    alpha = jax.scipy.linalg.cho_solve((chol_k, True), y - m)
    beta = jax.scipy.linalg.cho_solve((chol_k, True), kF)
    mu = M + kF.T @ alpha
    mu = mu.squeeze()

    #variance term
    vv_mix = 0
    for i in range(n_components):
        for j in range(n_components):
            vv_mix += weights[i]*weights[j]*Iij_n(i,j,s,L, m_params)

    #variance adjustment term
    adjustment = (kF.T @ beta).squeeze()

    var = vv_mix - adjustment
    
    return mu, var