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
import cma
from quad_utils import *
from other_utils import *

##########################################################################################################
#VBMC multivariate

#MC estimate of entropy of a multivariate gaussian mixture
def entropy(params, weights, key=None, random=True):
    mu_params = jnp.array(params[0])
    s_params = jnp.array(params[1])
    n_components = len(mu_params)
    key1, key2 = jax.random.split(key)

    s_params = jnp.array([jnp.diag(s) for s in s_params])

    n_samples = 100 if random else 250_000

    #sample from mixture
    mix = jax.random.choice(key1, n_components, shape=(n_samples,), p=weights)
    z = jax.random.normal(key2, shape=(n_samples, 2))
    #transform samples
    Ls = jnp.linalg.cholesky(s_params)
    samples = jax.vmap(lambda L, mu, z: mu + L@z)(Ls[mix], mu_params[mix], z)

    #compute pdf at sampled points
    pdf_vals = jax.vmap(lambda x: mixture_pdf(x, mu_params, s_params, weights))(samples)

    #return entropy
    h = -jnp.log(pdf_vals + 1e-12)
    return jnp.mean(h)


def elbo(params, x, y, s=1, L=1, chol_k=None, key=None, random=True):
    #convert parameters to appropriate form
    mixture_params = (params[0], jnp.exp(params[1]))
    logitweights = params[2]
    mixture_weights = jax.nn.softmax(logitweights)

    #compute expected log joint
    expected_log_joint, var, chol_k = integrate_mixture_n(x, y, s, L, 
                        mixture_params, mixture_weights, chol_k=chol_k)

    #entropy of Gaussian q
    entropy_q = entropy(mixture_params,key=key, random=random, weights=mixture_weights)

    #if random, return average of 20 samples
    #key, subkey = jax.random.split(key)
    #z = jax.random.normal(subkey)
    #expected_log_joint += random*jnp.sqrt(var/20)*z

    #return negative elbo
    return -expected_log_joint - entropy_q, var, chol_k


def plot_plots(x,y,s,L,n_points, params, params_true=None, ll=None):
    real_ll = ll is not None

    mu_real = params_true[0]
    sigma_real = jnp.diagonal(params_true[1], axis1=1, axis2=2)

    mu = params[0]
    sigma = jnp.exp(params[1])
    logitweights = params[2]
    weights = jax.nn.softmax(logitweights)
    
    #surrogate and variation distribution are first 2 plots
    n_subplots = 2

    #real ll and dist if ll is supplied
    if ll is not None:
        n_subplots += 2

    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots, 6))
    subplot_idx = 0
    
    #grid to plot for mixture
    lower1 = jnp.min(mu_real[:, 0] - 2.33*sigma_real[:, 0])
    upper1 = jnp.max(mu_real[:, 0] + 2.33*sigma_real[:, 0])
    lower2 = jnp.min(mu_real[:, 1] - 2.33*sigma_real[:, 1])
    upper2 = jnp.max(mu_real[:, 1] + 2.33*sigma_real[:, 1])

    #grid for evaluation
    x1_eval = jnp.linspace(lower1, upper1, 40)
    x2_eval = jnp.linspace(lower2, upper2, 40)
    X1_eval, X2_eval = jnp.meshgrid(x1_eval, x2_eval)
    points = jnp.column_stack([X1_eval.ravel(), X2_eval.ravel()])

    #Variational Mixture PDF
    mixture_vals = jax.vmap(lambda x: mixture_pdf(x, mu, sigma, weights))(points)
    mixture_grid = mixture_vals.reshape(X1_eval.shape)
    axes[subplot_idx].contourf(X1_eval, X2_eval, mixture_grid, levels=30, cmap='Oranges')
    axes[subplot_idx].contour(X1_eval, X2_eval, mixture_grid, levels=30, colors='black', linewidths=0.5)
    axes[subplot_idx].set_title("Current Mixture PDF")
    axes[subplot_idx].set_xlabel("x1")
    axes[subplot_idx].set_ylabel("x2")
    subplot_idx += 1

    #True Dist
    if real_ll:
        ll = ll(points, params_true).ravel()
        mixture_true = jnp.exp(ll)
        mixture_true = mixture_true.reshape(X1_eval.shape)

        axes[subplot_idx].contourf(X1_eval, X2_eval, mixture_true.reshape(X1_eval.shape),
                                levels=30, cmap='Oranges')
        axes[subplot_idx].contour(X1_eval, X2_eval, mixture_true.reshape(X1_eval.shape),
                                levels=30, colors='black', linewidths=0.5)
        axes[subplot_idx].set_title("True Mixture")
        axes[subplot_idx].set_xlabel("x1")
        axes[subplot_idx].set_ylabel("x2")
        subplot_idx += 1


    #grid to plot for ll
    lower1 = jnp.min(mu_real[:, 0] - 5*sigma_real[:, 0])
    upper1 = jnp.max(mu_real[:, 0] + 5*sigma_real[:, 0])
    lower2 = jnp.min(mu_real[:, 1] - 5*sigma_real[:, 1])
    upper2 = jnp.max(mu_real[:, 1] + 5*sigma_real[:, 1])


    #grid for evaluation
    x1_eval = jnp.linspace(lower1, upper1, 40)
    x2_eval = jnp.linspace(lower2, upper2, 40)
    X1_eval, X2_eval = jnp.meshgrid(x1_eval, x2_eval)
    points = jnp.column_stack([X1_eval.ravel(), X2_eval.ravel()])


    #GP surrogate mean
    last_n = len(x) - n_points

    mu_eval, var_eval, _ = gp_predict(x[:last_n, :], y[:last_n], points, kernel=rbf, s=s, L=L)
    mu_eval_grid = mu_eval.reshape(X1_eval.shape)
    axes[subplot_idx].contourf(X1_eval, X2_eval, mu_eval_grid, levels=30, cmap='Oranges')
    axes[subplot_idx].contour(X1_eval, X2_eval, mu_eval_grid, levels=30, colors='black', linewidths=0.5)
    # Plot all but last n_points in red
    axes[subplot_idx].scatter(x[:last_n, 0], x[:last_n, 1], c='black', s=20, label='Observed')
    axes[subplot_idx].scatter(x[last_n:, 0], x[last_n:, 1], c='blue', marker='x', s=80, label='New Points')
    axes[subplot_idx].set_title("GP Surrogate Mean")
    axes[subplot_idx].set_xlabel("x1")
    axes[subplot_idx].set_ylabel("x2")
    subplot_idx += 1


    #True ll
    if real_ll:
        axes[subplot_idx].contourf(X1_eval, X2_eval, ll.reshape(X1_eval.shape),
                                levels=30, cmap='Oranges')
        axes[subplot_idx].contour(X1_eval, X2_eval, ll.reshape(X1_eval.shape),
                                levels=30, colors='black', linewidths=0.5)
        axes[subplot_idx].set_title("True Log Likelihood")
        axes[subplot_idx].set_xlabel("x1")
        axes[subplot_idx].set_ylabel("x2")

    plt.tight_layout()
    plt.show()

