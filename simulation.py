import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    fig = plot_3D(E_u-u, x, y)
    fig.savefig(fig_folder / f'u_error_{run}.pdf')
    

def prior_sample_plots(fig_folder):
    D = 16
    ls = [0.1, 0.3, 1]
    subsample_factor = 4
    for l in ls:
        x, y, u, *_  = generate_data(D, l, subsample_factor)
        fig = plot_3D(u, x, y)
        fig.savefig(fig_folder / f'prior_l={l}.pdf')        
    
def beta_effect(fig_folder):
    l = 0.3
    subsample_factor = 4
    T = 100_000
    max_lag = 1000

    betas = np.linspace(0.01, 1, 30)
    Ds = np.array([2, 4, 8, 16])
    
    acc_grw = np.zeros((len(betas), len(Ds)))
    acc_pcn = np.zeros((len(betas), len(Ds)))
    autocorrelation_coefficient_grw = np.zeros((len(betas), len(Ds)))
    autocorrelation_coefficient_pcn = np.zeros((len(betas), len(Ds)))

    for w, D in enumerate(tqdm(Ds, desc='Dimension Sweep')):
        x, y, u, v, K, G, idx, N, M = generate_data(D, l, subsample_factor)

        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        Kc_inverse = np.linalg.inv(Kc)
        K_inverse = Kc_inverse.T @ Kc_inverse
        u0 = Kc@np.random.randn(N)
        
        for i, beta in enumerate(tqdm(betas, desc='Beta Sweep', leave=False)):
            u_samples_grw, acc_grw[i, w] = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, beta)
            u_samples_pcn, acc_pcn[i, w] = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, beta)
            
            auto_grw = autocorrelation(u_samples_grw, max_lag)
            for j in range(N):
                if (auto_grw[j,:] < 0.1).any():
                    k = np.where(auto_grw[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_grw[i, w] += auto_grw[j,:k].sum()
                
            auto_pcn = autocorrelation(u_samples_pcn, max_lag)
            for j in range(N):
                if (auto_pcn[j,:] < 0.1).any():
                    k = np.where(auto_pcn[j,:] < 0.1)[0].min()
                else:
                    k = max_lag
                autocorrelation_coefficient_pcn[i, w] += auto_pcn[j,:k].sum()
                
    data = {
        'beta': betas[:,np.newaxis,np.newaxis].repeat(len(Ds),1).repeat(2,2).flatten(),
        'autocorrelation': np.concatenate([autocorrelation_coefficient_grw[:,:,np.newaxis], autocorrelation_coefficient_pcn[:,:,np.newaxis]], axis=2).flatten(),
        'Dimension': Ds[np.newaxis,:,np.newaxis].repeat(len(betas),0).repeat(2,2).flatten(),
        'Algorithm': np.array(['GRW-MH', 'PCN'])[np.newaxis,np.newaxis,:].repeat(len(betas),0).repeat(len(Ds),1).flatten()
    }
    ax = sns.lineplot(data, x='beta', y='autocorrelation', hue='Dimension', style='Algorithm')
    ax.set(xlabel='$\\beta$', ylabel='Total Autocorrelation')
    ax.set(yscale='log')
    fig = ax.get_figure()
    fig.savefig(fig_folder / 'autocorrelation.pdf')
    
    fig, ax = plt.subplots()
    optimal_betas_grw = betas[np.argmin(autocorrelation_coefficient_grw, axis=0)]
    optimal_betas_pcn = betas[np.argmin(autocorrelation_coefficient_pcn, axis=0)]
    ax.plot(Ds, optimal_betas_grw, label='GRW', linestyle='-')
    ax.plot(Ds, optimal_betas_pcn, label='PCN', linestyle='--')
    ax.set_xlabel('D')
    ax.set_ylabel('Optimal $\\beta$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'optimal_beta.pdf')
    
    fig, ax = plt.subplots()
    data = {
        'beta': betas[:,np.newaxis,np.newaxis].repeat(len(Ds),1).repeat(2,2).flatten(),
        'acceptance_rate': np.concatenate([acc_grw[:,:,np.newaxis], acc_pcn[:,:,np.newaxis]], axis=2).flatten()*100,
        'Dimension': Ds[np.newaxis,:,np.newaxis].repeat(len(betas),0).repeat(2,2).flatten(),
        'Algorithm': np.array(['GRW-MH', 'PCN'])[np.newaxis,np.newaxis,:].repeat(len(betas),0).repeat(len(Ds),1).flatten()
    }
    ax = sns.lineplot(data, x='beta', y='acceptance_rate', hue='Dimension', style='Algorithm')
    ax.set(xlabel='$\\beta$', ylabel='Acceptance rate (%)')
    fig = ax.get_figure()
    fig.savefig(fig_folder / 'acceptance_rate.pdf')

def mesh_refinement(fig_folder):
    D = 3
    l = 0.3 
    subsample_factor = 1
    
    x, y, u, v, K, G, idx, N, M = generate_data(D, l, subsample_factor)
    
    mesh_refinements = np.array([1, 2, 4, 8, 16])
    betas = np.logspace(-3, 0, 30, base=10)
    
    acc_grw = np.zeros((len(betas), len(mesh_refinements)))
    acc_pcn = np.zeros((len(betas), len(mesh_refinements)))
    
    T = 10_000
    for i, mesh_refinement in enumerate(tqdm(mesh_refinements, desc='Mesh Refinement Sweep')):
        Dx = (D - 1) * mesh_refinement + 1
        Dy = Dx
        N_r = Dx * Dy                                                                  # Total number of coordinates
        coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
        x_r, y_r = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])  # Get x, y coordinate lists
        
        K_r = GaussianKernel(coords, l)
        Kc_r = np.linalg.cholesky(K_r + 1e-6 * np.eye(N_r))
        Kc_inverse_r = np.linalg.inv(Kc_r)
        K_inverse_r = Kc_inverse_r.T @ Kc_inverse_r
        u0 = Kc_r@np.random.randn(N_r)
        
        idx_r = np.zeros_like(idx)
        
        for j in range(M):
            measurement_x = x[idx[j]]
            measurement_y = y[idx[j]]
            idx_r[j] = np.where((x_r == measurement_x) & (y_r == measurement_y))[0][0]

        G_r = get_G(N_r, idx_r)
        
        for j, beta in enumerate(tqdm(betas, desc='Beta Sweep', leave=False)):
            _, acc_grw[j, i] = grw(lambda u: log_continuous_target(u, v, K_inverse_r, G_r), u0, K_r, T, beta)
            _, acc_pcn[j, i] = pcn(lambda u: log_continuous_likelihood(u, v, G_r), u0, K_r, T, beta)
            
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i, mesh_refinement in enumerate(mesh_refinements):
        ax1.semilogx(betas, acc_grw[:,i]*100, label=f'Mesh refinement: {mesh_refinement}')
        ax2.semilogx(betas, acc_pcn[:,i]*100, label=f'Mesh refinement: {mesh_refinement}')
    ax1.set_xlabel('$\\beta$')
    ax2.set_xlabel('$\\beta$')
    ax1.set_ylabel('Acceptance rate (%)')
    ax2.set_ylabel('Acceptance rate (%)')
    ax1.legend()
    ax2.legend()
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(fig_folder / 'acceptance_rate_mesh_refinement_grw.pdf')
    fig2.savefig(fig_folder / 'acceptance_rate_mesh_refinement_pcn.pdf')
    
    fig, ax = plt.subplots()
    data = {
        'beta': betas[:,np.newaxis,np.newaxis].repeat(len(mesh_refinements),1).repeat(2,2).flatten(),
        'acceptance_rate': np.concatenate([acc_grw[:,:,np.newaxis], acc_pcn[:,:,np.newaxis]], axis=2).flatten()*100,
        'Mesh Refinement': mesh_refinements[np.newaxis,:,np.newaxis].repeat(len(betas),0).repeat(2,2).flatten(),
        'Algorithm': np.array(['GRW-MH', 'PCN'])[np.newaxis,np.newaxis,:].repeat(len(betas),0).repeat(len(mesh_refinements),1).flatten()
    }
    ax = sns.lineplot(data, x='beta', y='acceptance_rate', hue='Mesh Refinement', style='Algorithm')
    ax.set(xlabel='$\\beta$', ylabel='Acceptance rate (%)')
    ax.set(xscale='log')
    fig = ax.get_figure()
    fig.savefig(fig_folder / 'acceptance_rate_mesh_refinement.pdf')

def main(fig_folder):
    
    ### Algorithm Analysis
    # Prior sample plots
    # prior_sample_plots(fig_folder)
    
    # Beta/D sweep
    # beta_effect(fig_folder)
    
    # Mesh refinement
    # mesh_refinement(fig_folder)

    ### Simulated inference
    np.random.seed(2)
    
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

    # Mean field inference
    T = 1_000_000
    
    start = time.time()
    u_samples_grw, acc_grw = grw(lambda u: log_continuous_target(u, v, K_inverse, G), u0, K, T, 0.2)
    total_time_grw = time.time() - start
    print(f'GRW - Acceptance rate: {acc_grw*100}%, Time per iteration: {total_time_grw/T}s')
    process_mc_results(x, y, u, u_samples_grw, fig_folder, 'grw')
    
    start = time.time()
    u_samples_pcn, acc_pcn = pcn(lambda u: log_continuous_likelihood(u, v, G), u0, K, T, 0.4)
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
    T = 1_000_000
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
        log_marginal_likelihoods[i] = log_marginal_likelihood(lambda u: log_probit_likelihood(u, t, G), Kc, N, int(np.ceil(10000/l)))

        
    fig, ax = plt.subplots()
    ax.plot(ls, error, label='Mean Error')
    ax.plot(ls, error_smooth, label='Mean Smooth Error')
    ax.plot(ls, error_data_only, label='Mean Smooth Error to Observations')
    ax.set_xlabel('$\\ell$')
    ax.set_ylabel('Error')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_folder / 'l_sweep_error.pdf')
    
    fig, ax = plt.subplots()
    ax.plot(ls, log_marginal_likelihoods)
    ax.set_xlabel('$\\ell$')
    ax.set_ylabel('Log Marginal Likelihood')
    fig.tight_layout()
    fig.savefig(fig_folder / 'l_sweep_log_marginal_likelihood.pdf')


if __name__ == '__main__':
    main(Path('figures/simulation'))
    plt.show()
