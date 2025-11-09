import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import cma
import corner
from scipy.stats import qmc
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
    d = params[0].shape[1]


    s_params = jnp.array([jnp.diag(s) for s in s_params])

    n_samples = 100 if random else 100_000

    #sample from mixture
    mix = jax.random.choice(key1, n_components, shape=(n_samples,), p=weights)
    z = jax.random.normal(key2, shape=(n_samples, d))
    #transform samples
    Ls = jnp.linalg.cholesky(s_params)
    samples = jax.vmap(lambda L, mu, z: mu + L@z)(Ls[mix], mu_params[mix], z)

    #compute pdf at sampled points
    pdf_vals = jax.vmap(lambda x: mixture_pdf(x, mu_params, s_params, weights))(samples)

    #return entropy
    h = -jnp.log(pdf_vals + 1e-12)
    return jnp.mean(h)


def elbo(params, x, y, hp_samples, chol_ks=None, key=None, random=True, fullsig=True):
    d = params[0].shape[1]
    if fullsig:
        s_params = jnp.exp(params[1])
    else:
        s_params = jnp.array([jnp.ones(d)*jnp.exp(s) for s in params[1]])
    
    mixture_params = (params[0], s_params)
    mixture_weights = jax.nn.softmax(params[2])

    s_hp, L_hp, m0_hp, x_m_hp, w_hp, sigma_e_hp = hp_samples

    entropy_q = entropy(mixture_params, key=key, random=random, weights=mixture_weights)

    def elbo_single(hp_i, chol_k_i):
        expected_log_joint, var = integrate_mixture_n(x, y, mixture_params, hp_i, mixture_weights, chol_k_i)
        elbo_i = -expected_log_joint - entropy_q
        return elbo_i, var

    hp_stacked = (s_hp, L_hp, m0_hp, x_m_hp, w_hp, sigma_e_hp)
    elbos, vars_ = jax.vmap(elbo_single, in_axes=((0,0,0,0,0,0), 0))(hp_stacked, chol_ks)


    return jnp.mean(elbos), jnp.mean(vars_) + jnp.var(elbos)


def initialise(params_true, ll, d, lower, upper, n=10, k=2, key=None, fullsig=True):

    x = jnp.empty((0, d))
    y = jnp.empty((0,))

    # ---- Sobol sampling ----
    sobol = qmc.Sobol(d=d, scramble=True)
    u = sobol.random(n)
    points = qmc.scale(u, lower, upper)
    points = jnp.array(points)

    new_y_points = ll(points, params_true)

    x = jnp.vstack([x, points])
    y = jnp.hstack([y, new_y_points])

    # Initialise mixture parameters near the centre region
    key, subkey = jax.random.split(key)
    mu_params = jax.random.uniform(subkey, (k, d)) -0.5 + (upper+lower)/2
    if fullsig:
        s_params = jnp.ones_like(mu_params)*jnp.max(upper-lower)/8
    else:
        s_params = jnp.ones(k)*jnp.max(upper-lower)/8

    logitweights = jnp.zeros(k)

    params = (mu_params, jnp.log(s_params), logitweights)

    return x, y, params

def remove(params):

    print("Current Weights:")
    print(jax.nn.softmax(params[2]))
    
    #take away component
    k = params[0].shape[0]
    current_weights = jax.nn.softmax(params[2])
    mask = current_weights*k >= 0.15
    if jnp.sum(mask) != k:
        print(f"Removing {k-jnp.sum(mask)} components")
    mu_params = params[0][mask]
    logsigma_params = params[1][mask]
    logitweights = params[2][mask]

    # Update params and k
    params = (mu_params, logsigma_params, logitweights)
    k = mu_params.shape[0]
    return params, k

def add(params, key, fullsig=True):
    print("Adding new component.")
    
    k = params[0].shape[0]

    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (), 0, k)
    
    key, subkey = jax.random.split(key)
    jitter1 = 0.5*jax.random.normal(subkey, params[0][idx].shape)
    key, subkey = jax.random.split(subkey)
    jitter2 = 0.1*jax.random.normal(subkey, params[0][idx].shape)
    new_mu = params[0][idx] + jitter1
    new_s = params[1][idx] + 0.1 + jitter2
    
    # Add new component to mixture
    mu_params = jnp.vstack([params[0], new_mu])
    if fullsig:
        s_params = jnp.vstack([params[1], new_s])
    else:
        s_params = jnp.concatenate([params[1], jnp.atleast_1d(new_s)])
    logitweights = jnp.hstack([params[2], params[2][idx]])
    params = (mu_params, s_params, logitweights)

    k += 1

    return params, k

def jitter(nfast, params, x1, y1, hp_samples, max_range, current_elbo, chol_ks, key,warmup, fullsig=True):

    mu, log_s, logitw = params
    k, d = mu.shape
    best_elbo = current_elbo
    best_params = params

    jitter_scale = jnp.where(warmup, 
                             max_range/6, 
                             jnp.maximum(10/k+1, 2.0))
    for i in range(k):
        key, subkey = jax.random.split(key)
        noise = jax.random.uniform(subkey, (nfast, d), minval=-1, maxval=1)*jitter_scale
        key, subkey = jax.random.split(key)
        if fullsig:
            noise2 = jax.random.uniform(subkey, (nfast,1), minval=-0.15, maxval=0.2)*0
        else:
            noise2 = jax.random.uniform(subkey, (nfast,), minval=-0.15, maxval=0.2)*0
        
        mu_candidates = mu[i] + noise
        s_candidates = log_s[i] + noise2
        
        def eval_candidate(mu_i, s_i):
            new_mu = mu.at[i].set(mu_i)
            new_log_s = log_s.at[i].set(s_i)
            return -elbo((new_mu, new_log_s, logitw), x1, y1, hp_samples, 
                         chol_ks=chol_ks, key=key, random=False, fullsig=fullsig)[0]

        elbos = jax.vmap(eval_candidate)(mu_candidates, s_candidates)
        best_idx = jnp.argmax(elbos)
        if elbos[best_idx] > best_elbo:
            best_elbo = elbos[best_idx]
            best_params = (mu.at[i].set(mu_candidates[best_idx]), log_s.at[i].set(s_candidates[best_idx]), logitw)

    if best_elbo > current_elbo:
        print("Jittered Candidate Selected")

    return best_params



def plot_plots(params, params_true, x_points, n_new,
               obs=None, likelihood_fn=None, fullsig=True):

    means = params[0]
    d = params[0].shape[1]
    if fullsig:
        vars_ = jnp.exp(params[1])
    else:
        vars_ = jnp.array([jnp.ones(d)*jnp.exp(s) for s in params[1]])
    weights = jnp.exp(params[2])
    weights /= jnp.sum(weights)

    n_samples = 1_000_000
    dim = means.shape[1]

    # --- Sample from variational mixture ---
    components = np.random.choice(len(weights), size=n_samples, p=np.array(weights))
    samples_var = np.zeros((n_samples, dim))
    for i in range(len(weights)):
        mask = (components == i)
        n_i = np.sum(mask)
        if n_i > 0:
            samples_var[mask] = np.random.normal(
                loc=np.array(means[i]),
                scale=np.sqrt(np.array(vars_[i])),
                size=(n_i, dim)
            )

    labels = [fr"$x_{i+1}$" for i in range(dim)]

    # --- Decide number of subplots ---
    if d == 2:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    else:
        fig = plt.figure(figsize=(7, 6))
        gs = fig.add_gridspec(1, 1)

    # Left panel: corner plot
    ax_corner = fig.add_subplot(gs[0])
    fig_corner = corner.corner(
        samples_var,
        bins=40,
        labels=labels,
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.3, 0.67, 0.95, 0.997),
        alpha=0.5,
        color='green'
    )

    # Overlay points
    if x_points is not None and len(x_points) > 0:
        axes = np.array(fig_corner.axes).reshape((dim, dim))
        if n_new is not None and n_new > 0 and n_new < len(x_points):
            x_old = x_points[:-n_new]
            x_new = x_points[-n_new:]
        else:
            x_old = x_points
            x_new = None

        for i in range(1, dim):
            for j in range(i):
                ax = axes[i, j]
                if len(x_old) > 0:
                    ax.scatter(x_old[:, j], x_old[:, i], c='black', s=4, alpha=0.6)
                if x_new is not None:
                    ax.scatter(x_new[:, j], x_new[:, i], c='red', marker='x', s=12)

    # Convert corner figure to image and place into ax_corner
    fig_corner.canvas.draw()
    corner_img = np.asarray(fig_corner.canvas.buffer_rgba())[:, :, :3]
    ax_corner.imshow(corner_img)
    ax_corner.axis("off")
    ax_corner.set_title("Variational Mixture (corner)")
    plt.close(fig_corner)

    # Right panel: likelihood surface only if d == 2
    if d == 2:
        ax_ll = fig.add_subplot(gs[1])
        grid_size = 100
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x, y)
        obs_grid = np.stack([X.ravel(), Y.ravel()], axis=-1)

        Z = np.exp(likelihood_fn(obs_grid, params_true)).reshape(grid_size, grid_size)

        contour = ax_ll.contourf(X, Y, Z, levels=3, cmap='Blues')
        fig.colorbar(contour, ax=ax_ll)
        ax_ll.set_title("Exp(Log-Likelihood) Surface")
        ax_ll.set_xlabel("x₁")
        ax_ll.set_ylabel("x₂")

    plt.tight_layout()
    plt.show()


def active_sample(x,y,hp_params,K_inv,
                  alpha, beta, gamma, params_true,
                  params,ll,  n_points, lower, upper, fullsig=True):
    

    for iteration in range(n_points):
        def acquistion_objective(obj):

            obj = jnp.atleast_2d(obj)
            #gp predictions at some obj
            mu_gp, var_gp = gp_predict_marginal(x,y,obj, hp_params, K_invs=K_inv)

            #mixture pdf values at some obj
            mu_params = params[0]
            if fullsig:
                sigma_params = jnp.exp(params[1])
            else:
                d = params[0].shape[1]
                sigma_params = jnp.array([jnp.ones(d)*jnp.exp(s) for s in params[1]])
            logitweights = params[2]
            weights = jax.nn.softmax(logitweights)

            #term 1, GP predictive variance at obj        
            t1 = var_gp
            t1 = jnp.clip(t1, 1e-12, jnp.inf)

            #term 2, variational mixture value at obj
            t2 = mixture_pdf(obj, mu_params, sigma_params, weights)
            t2 = jnp.clip(t2, 1e-100, jnp.inf)

            #term 3, GP predictive mean obj
            t3 = mu_gp
            t3 = jnp.clip(t3, -400, 400)

            acq = (t1**alpha)*(t2**beta)*jnp.exp(t3*gamma)

            v_reg = 1e-3
            reg_term1 = jnp.exp(-3*(v_reg/t1 - 1)*(t1<v_reg))

            acq = jnp.clip(acq*reg_term1, 1e-100, 1e100)
            return -float(acq.item())
        
        d = x.shape[1]
        x0 = (upper+lower)/2
        sigma0 = jnp.max(upper-lower)/6
        opts = {'verb_disp': 0,
                'verb_log': 0,
                'maxiter': int(25*(d-1) + 15),
                'bounds': [lower, upper]}

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(acquistion_objective)

        new_x_point = jnp.array(es.result.xbest).reshape(1, -1)
        new_y_point = ll(new_x_point, params_true).reshape(-1,1)


        x = jnp.vstack([x, new_x_point])
        y = jnp.vstack([y, new_y_point])

        print(f"Point {iteration+1} of {n_points}")
        if iteration < n_points-1:
            K_inv = rank_one_update(x, K_inv, hp_params)


    return x, y
    

def trim(x,y,hp_samples):
    #prune after warmup
    d = x.shape[1]
    threshold = jnp.max(y) - 8*d
    mask = (y >= threshold).ravel()
    num_keep = jnp.sum(mask)

    if num_keep < y.shape[0]:
        print(f"Trimming {y.shape[0] - num_keep} points")
        x = x[mask, :]
        y = y[mask]

    x1,y1 = reshape(x, y)
    chol_k, K_inv = kinv(x1, hp_samples=hp_samples)

    warmup = False
    print()
    print("Warm up phase complete:")
    print()
    return x1,y1, chol_k, K_inv