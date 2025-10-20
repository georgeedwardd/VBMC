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


def gp_predict(xs, ys, x_eval, kernel, s, L, m0=None, x_m=None,w=None, chol_k=None):
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

    # mean function at training and evaluation points
    if m0 is None:
        m0 = jnp.mean(ys)
        x_m = jnp.mean(xs, axis=0)
        w = (jnp.max(xs, axis=0) - jnp.min(xs, axis=0))/5.0

    m = m0 - 0.5*jnp.sum(((xs - x_m) ** 2) / (w ** 2), axis=1)
    M = m0 - 0.5*jnp.sum(((x_eval - x_m) ** 2) / (w ** 2), axis=1)

    K = kernel(xs, xs, s, L) + 1e-6*jnp.eye(xs.shape[0])
    K_s = kernel(xs, x_eval, s, L)
    K_ss = kernel(x_eval, x_eval, s, L)

    if chol_k is not None:
        n = chol_k.shape[0]
        K12 = K[:n, n:]
        K21 = K[n:, :n]
        K22 = K[n:, n:]

        S = K22 - K21 @ jax.scipy.linalg.cho_solve((chol_k, True), K12)
        def solve_S(b):
            return jnp.linalg.solve(S, b)

        top_left = jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(n)) + \
                jax.scipy.linalg.cho_solve((chol_k, True), K12) @ \
                solve_S(K21 @ jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(n)))

        top_right = -jax.scipy.linalg.cho_solve((chol_k, True), K12) @ solve_S(jnp.eye(K22.shape[0]))
        bottom_left = -solve_S(K21 @ jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(n)))
        bottom_right = solve_S(jnp.eye(K22.shape[0]))

        #Assemble updated "inverse"
        K_inv = jnp.vstack([jnp.hstack([top_left, top_right]),
                            jnp.hstack([bottom_left, bottom_right])])

    else:
        chol_k = jnp.linalg.cholesky(K + 1e-10 * jnp.eye(K.shape[0]))
        K_inv = jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(K.shape[0]))

    # Compute mean and covariance
    ym = ys - m[:, None]
    mu = M + (K_s.T @ K_inv @ ym).ravel()
    sig = K_ss - K_s.T @ K_inv @ K_s + 1e-9 * jnp.eye(K_ss.shape[0])

    return mu.ravel(), sig, chol_k


def neg_gp_likelihood(params, xs, ys, kernel, sigma_e=0):
    s = params[0]
    L_flat = params[1:]
    xs = jnp.reshape(xs, (-1, xs.shape[-1] if xs.ndim > 1 else 1))
    L = jnp.diag(L_flat)

    n = xs.shape[0]
    K_xx = kernel(xs, xs, s, L) + (sigma_e + 1e-6)*jnp.eye(n)

    chol_k = jnp.linalg.cholesky(K_xx)

    alpha = cho_solve((chol_k, True), ys)

    term1 = 0.5*ys@alpha
    term2 = jnp.sum(jnp.log(jnp.diag(chol_k)))
    term3 = 0.5*n*jnp.log(2*jnp.pi)

    return term1 + term2 + term3
