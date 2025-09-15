import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

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

def vk1(input,sigma,L,mu_p,s_p):
    t1 = jnp.sqrt(L**2/(L**2 + s_p**2))
    t2 = jnp.exp(-(input-mu_p)**2/(2*(L**2 + s_p**2)))
    return sigma**2*t1*t2

def vvk1(sigma,L,s_q):
    t1 = jnp.sqrt(L**2/(L**2 + 2*s_q**2))
    return sigma**2*jnp.sqrt(L**2/(L**2 + 2*s_q**2))

def integrate_gaussian_1(x_eval, y_eval, sigma, L, mu_p, s_p):
    x_eval = x_eval.reshape(-1,1)
    y_eval = y_eval.reshape(-1,1)
    
    #kernel matrix
    Kxx = rbf(x_eval, x_eval, sigma, L)
    Kxx += 1e-10*jnp.eye(len(Kxx))
    
    #cross-covariances
    kF = vk1(x_eval, sigma, L, mu_p, s_p).reshape(-1,1)
    vF = vvk1(sigma, L, s_p)
    
    #posterior mean and variance
    mu = (kF.T@jnp.linalg.solve(Kxx, y_eval)).item()
    var = vF - (kF.T@jnp.linalg.solve(Kxx, kF)).item()
    var = jnp.maximum(var, 0.0)
    
    return mu, var[0][0]

##############################################################################
#nd quadrature

def vk2(input, sigmaf, L, mu_p, S_p):
    S_p = jnp.array(S_p)
    L = jnp.atleast_2d(jnp.array(L)) 
    input = jnp.atleast_2d(input).T
    mu_p = jnp.atleast_2d(mu_p).T

    L2 = L@L.T
    t1 = jnp.linalg.det(S_p + L2)
    t1 = jnp.linalg.det(L)/jnp.sqrt(t1)

    inv_term = jnp.linalg.inv(S_p + L2)
    diff = input - mu_p
    t2 = jnp.exp(-0.5*diff.T@inv_term@diff)

    return sigmaf**2*t1*t2

def vvk2(sigmaf, L, S_p):
    S_p = jnp.array(S_p)
    L = jnp.atleast_2d(jnp.array(L)) 
    L2 = L@L.T
    det_term = jnp.linalg.det(2*S_p + L2)
    t1 = jnp.linalg.det(L)/jnp.sqrt(det_term)
    return sigmaf**2*t1


def integrate_gaussian_n(x_eval, y_eval, sigmaf, L, mu_p, S_p):
    L = jnp.asarray(L)
    if L.ndim == 1:
        L = jnp.diag(L) 

    x_eval = jnp.atleast_2d(x_eval).astype(jnp.float64)
    y_eval = jnp.atleast_2d(y_eval).astype(jnp.float64)
    n, _ = x_eval.shape

    # Kernel matrix
    Kxx = rbf(x_eval, x_eval, sigmaf, L)
    Kxx += 1e-10*jnp.eye(n)

    # Cross-covariances with Gaussian measure
    kF = jnp.array([vk2(x_eval[i], sigmaf, L, mu_p, S_p) for i in range(n)]).reshape(-1,1)

    # Variance of the integral
    vF = vvk2(sigmaf, L, S_p)

    # Posterior mean and variance of the integral
    mu = float((kF.T@jnp.linalg.solve(Kxx, y_eval)).item())
    var = float(jnp.maximum(vF - (kF.T@jnp.linalg.solve(Kxx, kF)).item(), 0.0))

    return mu, var