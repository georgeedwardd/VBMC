import jax.numpy as jnp

#multimodal gaussian mixture pdf
def mixture_pdf_1(x, mu_params, s_params, weights=None):
    x = jnp.ravel(jnp.atleast_1d(x))
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

def comp_pdf(x, mu, Sigma):
    det = jnp.linalg.det(Sigma)
    diff = x - mu
    sol = jnp.linalg.solve(Sigma, diff.T)
    exponent = -0.5*jnp.sum(diff*sol.T, axis=1)
    norm_const = 1.0/(2*jnp.pi*jnp.sqrt(det))
    return norm_const*jnp.exp(exponent)

def mixture_pdf(x, mu_params, s_params, weights=None):
    x = jnp.atleast_2d(x)
    mu_params = jnp.array(mu_params)
    s_params = jnp.array(s_params)
    n_components = mu_params.shape[0]

    #assume equal weights
    if weights is None:
        weights = jnp.ones(n_components)/n_components
    else:
        weights = jnp.array(weights)/jnp.sum(weights)

    if s_params.ndim == 2:
        s_params = jnp.array([jnp.diag(s) for s in s_params])

    terms = jnp.stack([comp_pdf(x, mu_params[i], s_params[i]) for i in range(n_components)], axis=1)
    return jnp.sum(weights[None, :] * terms, axis=1)

import numpy as np

def mc_nd(func, mus, covs, weights=None, n=100_000):
    mus = jnp.atleast_2d(mus)
    covs = np.array([np.diag(c) for c in covs])
    n_components, d = mus.shape

    # default to equal weights
    if weights is None:
        weights = np.ones(n_components)/n_components
    else:
        weights = np.array(weights)/np.sum(weights)

    counts = np.random.multinomial(n, weights)
    samples = np.empty((n, d))
    start = 0
    for i, n_i in enumerate(counts):
        if n_i > 0:
            samples[start:start+n_i] = np.random.multivariate_normal(mus[i], covs[i], n_i)
            start += n_i

    # evaluate function
    values = func(*samples.T)
    mean_est = np.mean(values)
    se_est = np.std(values, ddof=1)/np.sqrt(n)

    return mean_est, se_est


def sample_from_mixture(mu, covs, weights, n_samples=5000, key=None):
    K = len(weights)
    # choose components according to mixture weights
    component_choices = np.random.choice(K, size=n_samples, p=np.array(weights))
    samples = []
    for k in range(K):
        n_k = np.sum(component_choices == k)
        if n_k > 0:
            samp = np.random.multivariate_normal(np.array(mu[k]), np.array(covs[k]), size=n_k)
            samples.append(samp)
    return np.vstack(samples)

from scipy.stats import multivariate_normal

def log_mixture_density(x, means, covs, weights):
    """Compute log p(x) for a Gaussian mixture model."""
    K = len(weights)
    pdf_vals = np.zeros((x.shape[0], K))
    for k in range(K):
        pdf_vals[:, k] = weights[k] * multivariate_normal.pdf(x, mean=means[k], cov=covs[k])
    return np.log(np.sum(pdf_vals, axis=1) + 1e-300)  # stability

def estimate_elbo(params_q, params_p, n_samples=50000):
    """Monte Carlo estimate of ELBO = E_q[log p(z) - log q(z)]"""
    mu_q, vars_q, weights_q = params_q
    mu_p, covs_p, weights_p = params_p
    weights_q = jnp.exp(weights_q) / jnp.sum(jnp.exp(weights_q))

    
    covs_q = [np.eye(mu_q.shape[1]) * v for v in np.exp(vars_q)]
    
    # Sample from q
    K = len(weights_q)
    comp_idx = np.random.choice(K, size=n_samples, p=np.array(weights_q))
    samples = []
    for k in range(K):
        n_k = np.sum(comp_idx == k)
        if n_k > 0:
            s = np.random.multivariate_normal(mu_q[k], covs_q[k], size=n_k)
            samples.append(s)
    z_samples = np.vstack(samples)
    
    # Compute log p(z) and log q(z)
    log_p = log_mixture_density(z_samples, mu_p, covs_p, weights_p)
    log_q = log_mixture_density(z_samples, mu_q, covs_q, weights_q)
    
    return np.mean(log_p - log_q)