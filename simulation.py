import numpy as np
import matplotlib.pyplot as plt
from functions import *
from tqdm import tqdm
from pathlib import Path
import time

def generate_data(D, l, subsample_factor):
    Dx = D
    Dy = D
    N = Dx * Dy                                                                    # Total number of coordinates
    coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
    x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists
    
    idx = subsample(N, subsample_factor)
    M = len(idx)   
    
    K = GaussianKernel(coords, l)
    z = np.random.randn(N, )
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    u = Kc @ z

    ### Observation model: v = G(u) + e,   e~N(0,I)
    G = get_G(N, idx)
    v = G @ u + np.random.randn(M) 
    
    return x, y, u, v, K, G, idx, N, M

def process_mc_results(x, y, u, u_samples, fig_folder, run):
    E_u = np.mean(u_samples, axis=1)
    fig = plot_3D(E_u, x, y)
    fig.savefig(fig_folder / f'u_mean_{run}.pdf')

def prior_sample_plots(fig_folder):
    D = 16
    ls = [0.1, 0.3, 1]
    subsample_factor = 4
    for l in ls:
        x, y, u, *_  = generate_data(D, l, subsample_factor)
        fig = plot_3D(u, x, y)
        fig.savefig(fig_folder / f'prior_l={l}.pdf')        

def log_marginal_likelihood(log_likelihood, Kc, N, T):
    z = np.random.randn(N, T)
    u = Kc @ z
    log_likelihoods = np.zeros(T)
    for i in range(T):
        log_likelihoods[i] = log_likelihood(u[:, i])
    return np.log(np.exp(log_likelihoods).mean())
    
def beta_effect(fig_folder):
    l = 0.3
    subsample_factor = 4
    T = 100000
    max_lag = 1000

    betas = np.linspace(0.01, 1, 30)
    Ds = np.array([4,16])
    
    acc_grw = np.zeros((len(Ds), len(betas)))
    acc_pcn = np.zeros((len(Ds), len(betas)))
    autocorrelation_coefficient_grw = np.zeros((len(Ds), len(betas)))
    autocorrelation_coefficient_pcn = np.zeros((len(Ds), len(betas)))

    for w, D in enumerate([4,16]):
        x, y, u, v, K, G, idx, N, M = generate_data(D, l, subsample_factor)

        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        Kc_inverse = np.linalg.inv(Kc)
        K_inverse = Kc_inverse.T @ Kc_inverse
        u0 = Kc@np.random.randn(N)
        
        for i, beta in enumerate(tqdm(betas, desc='Beta Sweep')):
            u_samples_grw, acc_grw[w, i] = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
            u_samples_pcn, acc_pcn[w, i] = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
            
            auto_grw = autocorrelation(u_samples_grw, max_lag)
            for j in range(N):
                if (auto_grw[j,:] < 0.1).any():
                    k = np.where(auto_grw[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_grw[w, i] += auto_grw[j,:k].sum()
                
            auto_pcn = autocorrelation(u_samples_pcn, max_lag)
            for j in range(N):
                if (auto_pcn[j,:] < 0.1).any():
                    k = np.where(auto_pcn[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_pcn[w, i] += auto_pcn[j,:k].sum()
                
    fig, ax = plt.subplots()
    for i, D in enumerate(Ds):
        ax.plot(betas, autocorrelation_coefficient_grw[i,:], label=f'GRW, D={D}')
        ax.plot(betas, autocorrelation_coefficient_pcn[i,:], label=f'PCN, D={D}')
    ax.set_xlabel('beta')
    ax.set_ylabel('Autocorrelation coefficient')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'autocorrelation.pdf')
    
    fig, ax = plt.subplots()
    for i, D in enumerate(Ds):
        ax.plot(betas, acc_grw[i,:]*100, label=f'GRW, D={D}')
        ax.plot(betas, acc_pcn[i,:]*100, label=f'PCN, D={D}')
    ax.set_xlabel('beta')
    ax.set_ylabel('Acceptance rate (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'acceptance_rate.pdf')

def main(fig_folder):
    np.random.seed(0)
    
    # Prior sample plots
    prior_sample_plots(fig_folder)
    
    # Generate data
    D = 16
    l = 0.3
    subsample_factor = 1
    
    x, y, u, v, K, G, idx, N, M = generate_data(D, l, subsample_factor)
    xi, yi = np.meshgrid(np.arange(D, dtype=np.uint32), np.arange(D, dtype=np.uint32))                               # Get x, y index lists
    xi, yi = xi.flatten(), yi.flatten()
    
    # Plot data
    fig = plot_result(u, v, x, y, x[idx], y[idx])
    fig.savefig(fig_folder / 'data.pdf')

    # MCMC initialisation
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse.T @ Kc_inverse
    u0 = Kc@np.random.randn(N)

    # Beta/D sweep
    T = 100000
    betas = np.linspace(0.01, 1, 30)
    Ds = np.array([4,16])
    acc_grw = np.zeros((len(Ds), len(betas)))
    acc_pcn = np.zeros((len(Ds), len(betas)))
    autocorrelation_coefficient_grw = np.zeros((len(Ds), len(betas)))
    autocorrelation_coefficient_pcn = np.zeros((len(Ds), len(betas)))

    max_lag = 1000
    for w, d in enumerate([4,16]):
        x, y, u, v, K, G, idx, N, M = generate_data(d, l, subsample_factor)

        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        Kc_inverse = np.linalg.inv(Kc)
        K_inverse = Kc_inverse.T @ Kc_inverse
        u0 = Kc@np.random.randn(N)
        
        for i, beta in enumerate(tqdm(betas, desc='Beta Sweep')):
            u_samples_grw, acc_grw[w, i] = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
            u_samples_pcn, acc_pcn[w, i] = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
            
            auto_grw = autocorrelation(u_samples_grw, max_lag)
            for j in range(N):
                if (auto_grw[j,:] < 0.1).any():
                    k = np.where(auto_grw[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_grw[w, i] += auto_grw[j,:k].sum()
                
            auto_pcn = autocorrelation(u_samples_pcn, max_lag)
            for j in range(N):
                if (auto_pcn[j,:] < 0.1).any():
                    k = np.where(auto_pcn[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_pcn[w, i] += auto_pcn[j,:k].sum()
        
    
    fig, ax = plt.subplots()
    ax.plot(betas, autocorrelation_coefficient_grw[0,:], label='GRW, D=4')
    ax.plot(betas, autocorrelation_coefficient_grw[1,:], label='GRW, D=16')
    ax.plot(betas, autocorrelation_coefficient_pcn[0,:], label='PCN, D=4')
    ax.plot(betas, autocorrelation_coefficient_pcn[1,:], label='PCN, D=16')
    ax.set_xlabel('beta')
    ax.set_ylabel('Autocorrelation coefficient')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'autocorrelation.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(betas, acc_grw[0,:]*100, label='GRW, D=4')
    ax.plot(betas, acc_grw[1,:]*100, label='GRW, D=16')
    ax.plot(betas, acc_pcn[0,:]*100, label='PCN, D=4')
    ax.plot(betas, acc_pcn[1,:]*100, label='PCN, D=16')
    ax.set_xlabel('beta')
    ax.set_ylabel('Acceptance rate (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'acceptance_rate.pdf')

    # Mean field inference
    T = 1_000_000
    beta = 0.2
    
    start = time.time()
    u_samples_grw, acc_grw = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
    total_time_grw = time.time() - start
    print(f'GRW - Acceptance rate: {acc_grw*100}%, Time per iteration: {total_time_grw/T}s')
    process_mc_results(x, y, u, u_samples_grw, fig_folder, 'grw')
    
    start = time.time()
    u_samples_pcn, acc_pcn = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
    total_time_pcn = time.time() - start
    print(f'PCN - Acceptance rate: {acc_pcn*100}%, Time per iteration: {total_time_pcn/T}s')
    process_mc_results(x, y, u, u_samples_pcn, fig_folder, 'pcn')

    # Probit observation
    t = probit(v)       # Probit transform of data
    t_true = probit(u)  # Probit transform of latent field
    
    fig = plot_2D(t, xi[idx], yi[idx])
    fig.savefig(fig_folder / 'probit_observations.pdf')
    fig = plot_2D(t_true, xi, yi)
    fig.savefig(fig_folder / 'probit_true.pdf')

    # Probit MCMC
    T = 10_000
    beta = 0.2
    
    u_samples_probit, _ = pcn(lambda u: log_probit_likelihood(u, t, G), u0, K, T, beta)
    posterior_t = predict_t(u_samples_probit)
    posterior_t_true = predict_t_true(u_samples_probit)
    
    # Plotting
    fig = plot_2D(posterior_t, xi, yi)
    fig.savefig(fig_folder / 'probit_predictive_posterior.pdf')
    fig = plot_2D(posterior_t_true, xi, yi)
    fig.savefig(fig_folder / 'probit_true_posterior.pdf')
    fig = plot_2D(posterior_t > 0.5, xi, yi)
    fig.savefig(fig_folder / 'probit_predictive_assignments.pdf')
    fig = plot_2D(posterior_t_true > 0.5, xi, yi)
    fig.savefig(fig_folder / 'probit_true_assignments.pdf')
    
    # length scale sweep
    T = 10_000
    ls = np.linspace(0.01, 2, 100)
    error = np.zeros_like(ls)
    error_smooth = np.zeros_like(ls)
    error_data_only = np.zeros_like(ls)
    log_marginal_likelihoods = np.zeros_like(ls)

    for i, l in enumerate(tqdm(ls, desc='Length Scale Sweep')):
        K = GaussianKernel(np.stack([x, y], axis=1), l)
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        Kc_inverse = np.linalg.inv(Kc)
        K_inverse = Kc_inverse.T @ Kc_inverse
        u0 = Kc@np.random.randn(N)
        
        u_samples_probit, _ = pcn(lambda u: log_probit_likelihood(u, t, G), u0, K, T, beta)
        posterior_t = predict_t(u_samples_probit)
        posterior_t_true = predict_t_true(u_samples_probit)
        
        assignments_t_true = (posterior_t_true > 0.5).astype(np.uint8)
        
        error[i] = np.mean((assignments_t_true - t_true)**2)
        error_smooth[i] = np.mean((posterior_t_true - t_true)**2)
        error_data_only[i] = np.mean((posterior_t[idx] - t)**2)
        log_marginal_likelihoods[i] = log_marginal_likelihood(lambda u: log_probit_likelihood(u, t, G), Kc, N, 100000)

        
    fig, ax = plt.subplots()
    ax.plot(ls, error)
    ax.plot(ls, error_smooth)
    ax.plot(ls, error_data_only)
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.plot(ls, log_marginal_likelihoods)
    fig.tight_layout()


if __name__ == '__main__':
    main(Path('figures/simulation'))
    plt.show()
