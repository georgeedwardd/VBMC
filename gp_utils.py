import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np
import emcee
import jax
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



def common_components(x, y, L, m0=None, x_m=None, w=None):

    x = jnp.atleast_2d(x).astype(jnp.float64)
    if x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    y = jnp.atleast_2d(y).reshape(-1, 1)
    n = x.shape[0]

    L = jnp.diag(L)

    if m0 is None:
        m0 = jnp.mean(y) - 10
        x_m = jnp.mean(x, axis=0)
    if w is None:
        w = (jnp.max(x, axis=0) - jnp.min(x, axis=0))/7


    x_m = jnp.asarray(x_m).reshape(-1)
    w = jnp.asarray(w).reshape(-1)
    
    return x,y,L,m0,x_m,w


def kinv1(x,s,L, sigma_e=0):
    Kxx = rbf(x, x, s, L) + (sigma_e**2 + 1e-4)*jnp.eye(x.shape[0])
    chol_k = jnp.linalg.cholesky(Kxx)
    K_inv = jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(chol_k.shape[0]))
    return chol_k, K_inv
    


def neg_gp_likelihood(params, xs, ys, kernel, sigma_e=0):
    s = params[0]
    L_flat = params[1:]
    xs = jnp.reshape(xs, (-1, xs.shape[-1] if xs.ndim > 1 else 1))
    L = jnp.diag(L_flat)

    n = xs.shape[0]
    K_xx = kernel(xs, xs, s, L) + (sigma_e + 1e-4)*jnp.eye(n)

    chol_k = jnp.linalg.cholesky(K_xx)

    alpha = cho_solve((chol_k, True), ys)

    term1 = 0.5*ys@alpha
    term2 = jnp.sum(jnp.log(jnp.diag(chol_k)))
    term3 = 0.5*n*jnp.log(2*jnp.pi)

    return term1 + term2 + term3

#no mean function
def gp_predict_1(xs, ys, x_eval, kernel, s, L, m0=None, x_m=None,w=None):
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
    sig = K_ss - K_s.T @ v + 1e-8*jnp.eye(K_ss.shape[0])

    return mu.ravel(), sig

#negative quadratic mean
def gp_predict(
        xs, ys, x_eval, 
        s,L,m0,x_m,w, sigma_e=0.0,
        K_inv=None
        ):
    x_eval = x_eval.reshape(-1, x_m.shape[0])
    n_eval = x_eval.shape[0]

    M = m0 - 0.5 * jnp.sum(((x_eval - x_m)/w)**2, axis=1)
    m = m0 - 0.5 * jnp.sum(((xs - x_m)/w)**2, axis=1, keepdims=True)

    K_s = rbf(xs, x_eval, s, L)
    K_ss = rbf(x_eval, x_eval, s, L) + 1e-4*jnp.eye(n_eval)

    # Compute mean and covariance
    ym = ys - m
    mu = M + (K_s.T @ K_inv @ ym).ravel()
    sig = K_ss - K_s.T @ K_inv @ K_s

    return mu.ravel(), sig


def rank_one_update(xs, K_invs, hp_samples):

    n_samples = len(hp_samples[0])
    n_old = K_invs.shape[1]
    n_new = xs.shape[0] - n_old

    def update_one(sample_idx, K_inv_old, s_i, L_i, sigma_e=0.0):
        K12 = rbf(xs[:n_old], xs[n_old:], s_i, jnp.diag(L_i))
        K21 = K12.T
        K22 = rbf(xs[n_old:], xs[n_old:], s_i, jnp.diag(L_i)) + (sigma_e**2 + 1e-4)*jnp.eye(n_new)

        # Schur complement
        S = K22 - K21 @ K_inv_old @ K12
        chol_S = jnp.linalg.cholesky(S + 1e-10*jnp.eye(n_new))

        def solve_S(b):
            return jax.scipy.linalg.cho_solve((chol_S, True), b)

        top_left = K_inv_old + K_inv_old @ K12 @ solve_S(K21 @ K_inv_old)
        top_right = -K_inv_old @ K12 @ solve_S(jnp.eye(n_new))
        bottom_left = -solve_S(K21 @ K_inv_old)
        bottom_right = solve_S(jnp.eye(n_new))

        K_inv_updated = jnp.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        return K_inv_updated

    K_inv_new= jax.vmap(update_one, in_axes=(0, 0, 0, 0, None))(
        jnp.arange(n_samples),
        K_invs,
        hp_samples[0],
        hp_samples[1],
        0.0
    )

    return K_inv_new

def gp_predict_marginal(x, y, obj, hp_samples, K_invs):
    mu_list = []
    var_list = []

    n_hp = len(hp_samples[0])
    for i in range(n_hp):
        mu_i, var_i = gp_predict(
            x, y, obj,
            s=hp_samples[0][i],
            L=jnp.diag(hp_samples[1][i]),
            m0=hp_samples[2][i],
            x_m=hp_samples[3][i],
            w=hp_samples[4][i],
            K_inv=K_invs[i],
            sigma_e=0.0#hp_samples[5][i]
        )
        mu_list.append(mu_i)
        var_list.append(var_i)

    mu = jnp.mean(jnp.stack(mu_list), axis=0)
    var = jnp.mean(jnp.stack(var_list), axis=0) + jnp.var(jnp.stack(mu_list), axis=0)  

    return mu, var



def reshape(x, y):
    x = jnp.atleast_2d(x).astype(jnp.float64)
    if x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
    y = jnp.atleast_2d(y).reshape(-1, 1)
    return x,y

def kinv(x, hp_samples=None):
    n_samples = len(hp_samples[0])
    n_points = x.shape[0]

    chol_ks = jnp.zeros((n_samples, n_points, n_points))
    K_invs = jnp.zeros((n_samples, n_points, n_points))

    n_samples = len(hp_samples[0])
    for i in range(n_samples):
        s_i = hp_samples[0][i]
        L_i = jnp.diag(hp_samples[1][i])
        #sigma_e_i = hp_samples[2][i] #introduce later
        sigma_e_i = 0
        
        Kxx = rbf(x, x, s_i, L_i) + (sigma_e_i**2 + 1e-4)*jnp.eye(x.shape[0])
        chol_k = jnp.linalg.cholesky(Kxx)
        K_inv = jax.scipy.linalg.cho_solve((chol_k, True), jnp.eye(chol_k.shape[0]))
        
        chol_ks = chol_ks.at[i].set(chol_k)
        K_invs = K_invs.at[i].set(K_inv)
        
    return chol_ks, K_invs


def gp_hp(xs,ys, prev_sampler=None,n_samples=1, chainlength=800, discard=100):
    D = xs.shape[1] if xs.ndim > 1 else 1
    ndim = 3 * D + 3
    nwalkers = 2*ndim

    if prev_sampler is not None:
        prev_chain = prev_sampler.get_chain()
        p0 = prev_chain[-1]
        p0 += 1e-4*np.random.randn(*p0.shape)
    else:
        p0 = prior_sample(xs, ys, n_samples=nwalkers)
        p0 += 1e-5*np.random.randn(*p0.shape)



    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(jnp.array(xs), jnp.array(ys.ravel())))
    sampler.run_mcmc(p0, chainlength, progress=True)
    samples = sampler.get_chain(discard=discard, flat=True)

    idx = np.random.choice(samples.shape[0], size=n_samples, replace=False)
    draws = samples[idx]
    
    #unpack
    s_hp = jnp.exp(draws[:, 0])
    L_hp = jnp.exp(draws[:,1:1+D])
    m0_hp = draws[:,1+D]
    x_m_hp = draws[:,2+D:2+D+D]
    w_hp = jnp.exp(draws[:,2+D+D:2+D+D+D])
    sigma_e_hp = jnp.exp(draws[:, -1])

    hp_samples = (s_hp, L_hp, m0_hp, x_m_hp, w_hp, sigma_e_hp)

    return sampler, hp_samples


def log_gp_likelihood(params, xs, ys):

    #unpack parameters
    D = xs.shape[1] if xs.ndim > 1 else 1
    
    s = jnp.exp(params[0])
    L = jnp.diag(jnp.exp(params[1:1+D]))
    m0 = params[1+D]
    x_m = params[2+D:2+D+D]
    w = jnp.exp(params[2+D+D:2+D+D+D])
    sigma_e = jnp.exp(params[-1])

    xs = jnp.reshape(xs, (-1, xs.shape[-1] if xs.ndim > 1 else 1))
    n = xs.shape[0]

    # Compute mean function at training points
    m = m0 - 0.5*jnp.sum(((xs - x_m)/w)**2, axis=1, keepdims=True)

    # Kernel matrix
    K_xx = rbf(xs, xs, s, L) + (sigma_e**2)*jnp.eye(n)
    chol_k = jnp.linalg.cholesky(K_xx)

    # Residuals
    ym = ys - m.ravel()

    # Solve for alpha
    alpha = cho_solve((chol_k, True), ym)

    # Log marginal likelihood
    term1 = -0.5 * ym @ alpha
    term2 = -jnp.sum(jnp.log(jnp.diag(chol_k)))

    ll = term1 + term2 # -0.5*n*jnp.log(2*jnp.pi)

    return ll

def log_prior(params, xs, ys):

    D = xs.shape[1] if xs.ndim > 1 else 1

    log_s = params[0]
    log_L = params[1:1+D]
    m0 = params[1+D]
    x_m = params[2+D:2+D+D]
    log_w = params[2+D+D:2+D+D+D]
    log_sigma_e = params[-1]

    y_threshold = jnp.quantile(ys, 0.2)
    hpd_mask = (ys >= y_threshold).ravel()
    ys = ys[hpd_mask]
    xs = xs[hpd_mask, :]


    ymax = jnp.max(ys)
    diamy = jnp.max(ys) - jnp.min(ys)
    if D == 1:
        SDx = jnp.std(xs)
        diamx = jnp.max(xs) - jnp.min(xs)
        xmin = jnp.min(xs)
        xmax = jnp.max(xs)
    else:
        SDx = jnp.std(xs, axis=0)
        diamx = jnp.max(xs, axis=0) - jnp.min(xs, axis=0)
        xmin = jnp.min(xs, axis=0)
        xmax = jnp.max(xs, axis=0)

    logp = 0.0

    #log(L)~N(log SDx, log(diamx/2SDx))
    mu_L = jnp.log(SDx + 1e-8)
    sigma_L = jnp.maximum(jnp.log((diamx + 1e-8)/(2*SDx + 1e-8)),2)
    logp += jnp.sum(-0.5*((log_L - mu_L)/sigma_L)**2 - jnp.log(sigma_L))# - 0.5*jnp.log(2*jnp.pi))

    #s_e~N(log(0.001), 0.5])
    mu_sigma_e = jnp.log(0.001)
    sigma_sigma_e = 0.5
    logp += -0.5*((log_sigma_e - mu_sigma_e)/sigma_sigma_e)**2 - jnp.log(sigma_sigma_e)# - 0.5*jnp.log(2*jnp.pi))

    #m0~N(max(y), diamy)
    mu_m0 = ymax
    sigma_m0 = diamy + 1e-8
    logp += -0.5*((m0 - mu_m0)/sigma_m0)**2 - jnp.log(sigma_m0)# - 0.5*jnp.log(2*jnp.pi)

    #x_m~U(xmin, xmax)
    if jnp.any(x_m < xmin) or jnp.any(x_m > xmax):
        logp += -jnp.inf

    #log(w)~U(-10*diam, 10*diam)
    if jnp.any(log_w < -jnp.log(5*diamx)) or jnp.any(log_w > jnp.log(5*diamx)):
        logp += -jnp.inf

    return logp

def log_posterior(params, xs, ys):
    lp = log_prior(params, xs, ys)
    if not jnp.isfinite(lp):
        return -jnp.inf
    ll = log_gp_likelihood(params, xs, ys)


    return lp + ll


def prior_sample(xs, ys, n_samples):
    D = xs.shape[1] if xs.ndim > 1 else 1
    samples = []

    y_threshold = jnp.quantile(ys, 0.2)
    hpd_mask = (ys >= y_threshold).ravel()
    ys = ys[hpd_mask]
    xs = xs[hpd_mask, :]

    ymax = jnp.max(ys)
    diamy = jnp.max(ys) - jnp.min(ys)
    SDx = jnp.std(xs, axis=0) if D > 1 else jnp.std(xs)
    diamx = jnp.max(xs, axis=0) - jnp.min(xs, axis=0)
    xmin = jnp.min(xs, axis=0)
    xmax = jnp.max(xs, axis=0)

    mu_L = jnp.log(SDx + 1e-8)
    sigma_L = jnp.maximum(jnp.log((diamx + 1e-8)/(2*SDx + 1e-8)), 2.0)

    for _ in range(n_samples):
        log_s = np.random.normal(1,1)
        log_L = np.random.normal(mu_L, sigma_L)
        m0 = np.random.normal(ymax, diamy + 1e-8)
        x_m = np.random.uniform(xmin, xmax)
        log_w = np.random.uniform(-jnp.log(5*diamx), jnp.log(5*diamx))
        log_sigma_e = np.random.normal(jnp.log(0.001), 0.5)

        params = jnp.concatenate([jnp.atleast_1d(jnp.array(log_s)),
                                  jnp.atleast_1d(jnp.array(log_L)),
                                  jnp.atleast_1d(jnp.array(m0)),
                                  jnp.atleast_1d(jnp.array(x_m)),
                                  jnp.atleast_1d(jnp.array(log_w)),
                                  jnp.atleast_1d(jnp.array(log_sigma_e))])

        samples.append(params)

    return jnp.array(samples)
