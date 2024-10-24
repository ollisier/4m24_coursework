import numpy as np
import matplotlib.pyplot as plt
from functions import *


###--- Data Generation ---###

### Inference grid defining {ui}i=1,Dx*Dy
Dx = 16
Dy = 16
N = Dx * Dy     # Total number of coordinates
points = [(x, y) for y in np.arange(Dx) for x in np.arange(Dy)]                # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])    # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)
M = len(idx)                                                                   # Total number of data points

### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
l = 0.3
K = GaussianKernel(coords, l)
z = np.random.randn(N, )
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
u = Kc @ z

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)


###--- MCMC ---####

### Set MCMC parameters
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

### Sample from prior for MCMC initialisation

# TODO: Complete Simulation questions (a), (b).
# u_samples_grw, acc_rate_grw = grw(log_target=log_target, u0=u*0, y=v, K=K, G=G, n_iters=n, beta=beta)
# u_samples_pcn, acc_rate_pcn = pcn(log_likelihood=log_likelihood, u0=u*0, y=v, K=K, G=G, n_iters=n, beta=beta)

# u_samples_mean_grw = sum(u_samples_grw)/n
# u_samples_mean_pcn = sum(u_samples_pcn)/n

# print(f'Acc% GRW: {acc_rate_grw}')
# print(f'Acc% PCN: {acc_rate_pcn}')

# error_grw = u_samples_mean_grw - u
# error_pcn = u_samples_mean_pcn - u

# ### Plotting examples
# plot_3D(u, x, y)                                      # Plot original u surface
# plot_result(u, v, x, y, x[idx], y[idx])               # Plot original u with data v
# plot_3D(u_samples_mean_pcn, x, y)
# plot_3D(error_grw, x, y)
# plot_3D(error_pcn, x, y)


###--- Probit transform ---###
t = probit(v)       # Probit transform of data

ls = np.linspace(0.01, 1, 10)
errors = np.zeros(10)

# TODO: Complete Simulation questions (c), (d).
for i, l in enumerate(ls):
    K = GaussianKernel(coords, l)
    u_samples_pcn, acc_rate_pcn = pcn(log_likelihood=log_probit_likelihood, u0=u*0, y=t, K=K, G=G, n_iters=n, beta=beta)

    predictions = predict_t(u_samples_pcn)
    classifications = probit(predictions-0.5)

    errors[i] = sum(np.abs(classifications - probit(u)))/N
    
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(ls, errors)
plt.show()

### Plotting examples
plot_2D(probit(u), xi, yi, title='Original Data')     # Plot true class assignments
plot_2D(t, xi[idx], yi[idx], title='Probit Data')     # Plot data
plot_2D(predictions, xi, yi, title='Predictions')     # Plot data
plot_2D(classifications, xi, yi, title='Predictions')     # Plot data
