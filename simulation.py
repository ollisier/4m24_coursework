import numpy as np
import matplotlib.pyplot as plt
from functions import *
from tqdm import tqdm
from pathlib import Path

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

def main(fig_folder):
    np.random.seed(0)
    
    # Prior sample plots
    prior_sample_plots(fig_folder)
    
    # Generate data
    D = 16
    l = 0.3
    subsample_factor = 4
    
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

    # # Beta effect
    # T = 10000
    # betas = np.linspace(0, 1, 20)
    # acc_grw = np.zeros_like(betas)
    # acc_pcn = np.zeros_like(betas)
    
    # plt.figure()
    # for i, beta in enumerate(tqdm(betas, desc='Beta Sweep')):
    #     _, acc_grw[i] = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
    #     _, acc_pcn[i] = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
        
    # fig, ax = plt.subplots()
    # ax.plot(betas, acc_grw*100, label='GRW')
    # ax.plot(betas, acc_pcn*100, label='PCN')
    # ax.set_xlabel('beta')
    # ax.set_ylabel('Acceptance rate (%)')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(fig_folder / 'acceptance_rate.pdf')

    # # Mean field inference
    # T = 1_000_000
    # beta = 0.2
    
    # u_samples_grw, acc_grw = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
    # print('Acceptance rate GRW: ', acc_grw)
    # process_mc_results(x, y, u, u_samples_grw, fig_folder, 'grw')
      
    # u_samples_pcn, acc_pcn = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
    # print('Acceptance rate PCN: ', acc_pcn)
    # process_mc_results(x, y, u, u_samples_pcn, fig_folder, 'pcn')

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
        
    fig, ax = plt.subplots()
    ax.plot(ls, error)
    ax.plot(ls, error_smooth)
    ax.plot(ls, error_data_only)
    fig.tight_layout()


if __name__ == '__main__':
    main(Path('figures/simulation'))
    plt.show()
