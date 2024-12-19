import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
from scipy.special import factorial
from scipy.signal import correlate
import matplotlib.cm as cm
import copy


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i,idx[i]] = 1
    return G


def probit(v):
    return (v > 0).astype(int)


def predict_t(samples):
    phi = norm.cdf(samples)
    return np.mean(phi, axis=1)


def predict_t_true(samples):
    return np.mean(probit(samples), axis=1)


def autocorrelation(samples, max_lag):
    N = samples.shape[0]
    num_iters = samples.shape[1]
    autocorrelation = np.zeros((N, N, max_lag))
    for i in range(N):
        for j in range(N):
            autocorrelation[i,j,:] = ((correlate(samples[i, :]-np.mean(samples[i, :]), samples[j, :]-np.mean(samples[j, :]), mode='full')[num_iters-1:])/np.arange(num_iters, 0, -1)/(np.std(samples[i, :])*np.std(samples[j, :])))[:max_lag]
    return autocorrelation

###--- Density functions ---###

def log_prior(u, K_inverse):
    return -0.5 * u.T @ K_inverse @ u # + constant


def log_continuous_likelihood(u, v, G):
    return -0.5 * (v - G@u).T @ (v - G@u) # + constant


def log_probit_likelihood(u, t, G):
    phi = norm.cdf(G @ u)
    return np.sum(t*np.log(phi) + (1-t)*np.log(1-phi))


def log_poisson_likelihood(u, c, G):
    theta = np.exp(G @ u)
    return np.sum(-theta + c * np.log(theta) - np.log(factorial(c)))# TODO: Return likelihood p(c|u)


def log_continuous_target(u, y, K_inverse, G):
    return log_prior(u, K_inverse) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K_inverse, G):
    return log_prior(u, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K_inverse, G):
    return log_prior(u, K_inverse) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###

def grw(log_target, u0, K, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        K - prior covariance
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    acc = 0
    u_prev = u0
    N = len(u0)
    X = np.zeros((N, n_iters))

    # Cholesky computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    lt_prev = log_target(u_prev)

    for i in range(n_iters):

        u_new = u_prev + beta*Kc@np.random.randn(N)

        lt_new = log_target(u_new)

        log_alpha = np.min(lt_new - lt_prev, 0)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_u < log_alpha
        if accept:
            acc += 1
            X[:,i] = u_new
            u_prev = u_new
            lt_prev = lt_new
        else:
            X[:,i] = u_new

    return X, acc / n_iters


def pcn(log_likelihood, u0, K, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        K - prior covariance
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    acc = 0
    u_prev = u0
    N = len(u0)
    X = np.zeros((N, n_iters))

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    ll_prev = log_likelihood(u_prev)

    for i in range(n_iters):

        u_new = np.sqrt(1-beta**2) * u_prev + beta*Kc@np.random.randn(N)

        ll_new = log_likelihood(u_new)

        log_alpha = np.min(ll_new - ll_prev, 0)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_u < log_alpha
        if accept:
            acc += 1
            X[:,i] = u_new
            u_prev = u_new
            ll_prev = ll_new
        else:
            X[:,i] = u_new

    return X, acc / n_iters


###--- Plotting ---###

def plot_3D(u, x, y, title=None):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    fig.tight_layout()
    if title:  plt.title(title)
    return fig


def plot_2D(counts, xi, yi, title=None, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    fig.colorbar(im)
    fig.tight_layout()
    if title:  plt.title(title)
    return fig


def plot_result(u, data, x, y, x_d, y_d, title=None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax.scatter(x_d, y_d, data, marker='x', color='r')
    fig.tight_layout()
    if title:  plt.title(title)
    return fig
