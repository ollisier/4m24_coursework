import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
from tqdm import tqdm
from pathlib import Path


def main(fig_folder):
    ### Read data
    df = pd.read_csv('data.csv')
    
    data = np.array(df["bicycle.theft"])
    xi = np.array(df['xi'])
    yi = np.array(df['yi'])
    x = np.array(df['x'])*7800
    y = np.array(df['y'])*8800
    N = len(data)
    coords = [(x[i],y[i]) for i in range(N)]

    ### Subsample the original data set
    subsample_factor = 4
    idx = subsample(N, subsample_factor, seed=42)
    M = len(idx)
    G = get_G(N, idx)
    c = G @ data
    
    fig = plot_2D(data, xi, yi)                   # Plot bike theft count data
    fig.savefig(fig_folder / 'bike_theft_data.pdf')
    fig = plot_2D(c, xi[idx], yi[idx])      # Plot subsampled data
    fig.savefig(fig_folder / 'bike_theft_data_subsampled.pdf')
    
    ###--- MCMC ---####
    ls = np.logspace(2, 4, 100)
    
    # log_marginal_likelihoods = np.zeros(len(ls))
    # G_subsampled = np.eye(M)
    # coords_subsampled = [(x[i],y[i]) for i in idx]
    # for i, l in enumerate(tqdm(ls)):
    #     K = GaussianKernel(coords_subsampled, l)
    #     Kc = np.linalg.cholesky(K + 1e-6 * np.eye(M))
    #     log_marginal_likelihoods[i] = log_marginal_likelihood(lambda u: log_poisson_likelihood(u, c, G_subsampled), Kc, M, int(np.ceil(100000000000/l**2)))
        
    # fig, ax = plt.subplots()
    # ax.semilogx(ls, log_marginal_likelihoods)
    # ax.set_xlabel('$\\ell$')
    # ax.set_ylabel('Log Marginal Likelihood')
    # fig.tight_layout()
    # fig.savefig(fig_folder / 'log_marginal_likelihood.pdf')

    T = 10_000
    beta = 0.2
    mse = np.zeros(len(ls))
    
    # for i, l in enumerate(tqdm(ls, desc='MSE')):
    #     K = GaussianKernel(coords, l)
    #     Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        
    #     u0 = Kc@np.random.randn(N)
    #     u_samples, acc = pcn(lambda u: log_poisson_likelihood(u, c, G), u0, K, T, beta)
        
    #     E_c = np.mean(np.exp(u_samples), axis=1)
    #     mse[i] = np.mean((E_c - data)**2)
        
    # fig, ax = plt.subplots()
    # ax.semilogx(ls, mse)
    # ax.set_xlabel('$\\ell$')
    # ax.set_ylabel('Mean Squared Error')
    # fig.tight_layout()
    # fig.savefig(fig_folder / 'mse.pdf')

    ls = [100, 1.5e3, 10e3]
    T = 1_000_000
    
    for l in tqdm(ls, desc='Posterior'):
        K = GaussianKernel(coords, l)
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

        u0 = Kc@np.random.randn(N)
        u_samples, _ = pcn(lambda u: log_poisson_likelihood(u, c, G), u0, K, T, beta)
        
        fig = plot_2D(np.mean(np.exp(u_samples), axis=1), xi, yi)
        fig.savefig(fig_folder / f'posterior_mean_field_l={int(l)}.pdf')
        fig = plot_2D(np.mean(np.exp(u_samples), axis=1)-data, xi, yi, clim_lower=np.min(np.mean(np.exp(u_samples), axis=1)-data))       
        fig.savefig(fig_folder / f'error_field_l={int(l)}.pdf')
        
        

if __name__ == '__main__':
    main(Path('figures/spatial'))
    plt.show()