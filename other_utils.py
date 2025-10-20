import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#multimodal gaussian mixture pdf
def mixture_pdf_1(x, mu_params, s_params, weights=None):
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