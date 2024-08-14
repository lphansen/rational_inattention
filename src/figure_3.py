import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 2  # Number of states
M = 3  # Number of actions
xi = 200.0
Lam = 10.0

# Create alpha array
alpha_min = 0.0
alpha_max = 1.0
alpha_increment = (alpha_max - alpha_min) / 50
alpha_set = np.arange(alpha_min, alpha_max, alpha_increment)
T = len(alpha_set)

# Step sizes
q_step = 0.5
p_step = 0.5
mu_step = 0.5

# Baseline prior
hat_mu = np.full(N, 0.5)  # Initial uniform prior

eps = 1e-8  # Error tolerance

# Define utility function
def utility_function(payoff, alpha):
    if alpha != 1:
        return (payoff ** (1 - alpha) - 1) / (1 - alpha)
    else:
        return np.log(payoff)

# Define payoffs
r = 7.5
payoff_set = np.array([[0.0, 2 * r], [r, 0.0], [5.0, 5.0]])

def utility(x, a, alpha):
    return utility_function(payoff_set[a, x], alpha)

# Initialize arrays
p = np.ones((T, M, N)) / M  # Initial value of p(a|x)
q = np.zeros((T, M))  # Initial value of q(a)
v = np.zeros((T, N))  # Initial value of G(p*,q*)(x)
mu = np.zeros((T, N))  # Initial mu(x)
post = np.zeros((T, N, M))  # Posterior distribution of p(a|x)
p_old = np.zeros((T, M, N))
q_old = np.zeros((T, M))
mu_old = np.zeros((T, N))

# Main loop over alpha values
for t, alpha in enumerate(alpha_set):
    mu[t, :] = hat_mu
    p_old[t, :, :] = p[t, :, :]
    q_old[t, :] = q[t, :]
    mu_old[t, :] = mu[t, :]

    Iter_err = float('inf')

    while Iter_err > eps:
        # Update q
        for a in range(M):
            q[t, a] = (1 - q_step) * q_old[t, a] + q_step * sum(mu[t, x] * p[t, a, x] for x in range(N))
        
        # Update p
        for x in range(N):
            den = sum(q[t, a] * np.exp(utility(x, a, alpha) / Lam) for a in range(M))
            for a in range(M):
                p[t, a, x] = (1 - p_step) * p_old[t, a, x] + p_step * q[t, a] * np.exp(utility(x, a, alpha) / Lam) / den
        
        # Update v
        for x in range(N):
            v[t, x] = Lam * np.log(sum(q[t, a] * np.exp(utility(x, a, alpha) / Lam) for a in range(M)))
        
        # Update mu
        den = sum(np.exp(-v[t, x] / xi) * hat_mu[x] for x in range(N))
        for x in range(N):
            mu[t, x] = (1 - mu_step) * mu_old[t, x] + mu_step * np.exp(-v[t, x] / xi) * hat_mu[x] / den
        
        # Compute error
        Iter_err = (np.linalg.norm(p[t, :, :] - p_old[t, :, :], 'fro') +
                    np.linalg.norm(q[t, :] - q_old[t, :]) +
                    np.linalg.norm(mu[t, :] - mu_old[t, :]))
        
        # Update old values
        p_old[t, :, :] = p[t, :, :]
        q_old[t, :] = q[t, :]
        mu_old[t, :] = mu[t, :]

    # Update posterior
    for a in range(M):
        for x in range(N):
            if q[t, a] > 1e-4:
                post[t, x, a] = p[t, a, x] * mu[t, x] / q[t, a]

# Calculate R and I
R = np.sum(np.log(mu / hat_mu) * mu, axis=1)
I = np.array([np.sum(np.sum(np.log(p[i, :, :] / q[i, :, None]) * p[i, :, :], axis=0) * mu[i]) for i in range(T)])

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
index = alpha_set

# Subplot 1
axs[0, 0].plot(index, p[:, 0, 0], '-b', linewidth=2, label=r'$p^*(1|x_1)$')
axs[0, 0].plot(index, p[:, 1, 0], '-r', linewidth=2, label=r'$p^*(2|x_1)$', linestyle='dashed')
axs[0, 0].plot(index, p[:, 2, 0], '-k', linewidth=2, label=r'$p^*(3|x_1)$', linestyle='dotted')
axs[0, 0].set_xlabel(r'$\alpha$')
axs[0, 0].set_ylabel(r'$p$')
axs[0, 0].set_title('State 1 decision rule')
axs[0, 0].legend(loc='right')

# Subplot 2
axs[0, 1].plot(index, p[:, 0, 1], '-b', linewidth=2, label=r'$p^*(1|x_2)$')
axs[0, 1].plot(index, p[:, 1, 1], '-r', linewidth=2, label=r'$p^*(2|x_2)$', linestyle='dashed')
axs[0, 1].plot(index, p[:, 2, 1], '-k', linewidth=2, label=r'$p^*(3|x_2)$', linestyle='dotted')
axs[0, 1].set_xlabel(r'$\alpha$')
axs[0, 1].set_ylabel(r'$p$')
axs[0, 1].set_title('State 2 decision rule')
axs[0, 1].legend()

# Subplot 3
axs[1, 0].plot(index, I, '-k', linewidth=2)
axs[1, 0].set_xlabel(r'$\alpha$')
axs[1, 0].set_ylabel(r'$I$')
axs[1, 0].set_title('Mutual information $I$')

# Subplot 4
axs[1, 1].plot(index, q[:, 0], '-b', linewidth=2, label=r'$q^*(1)$')
axs[1, 1].plot(index, q[:, 1], '-r', linewidth=2, label=r'$q^*(2)$', linestyle='dashed')
axs[1, 1].plot(index, q[:, 2], '-k', linewidth=2, label=r'$q^*(3)$', linestyle='dotted')
axs[1, 1].set_xlabel(r'$\alpha$')
axs[1, 1].set_ylabel(r'$q$')
axs[1, 1].set_title('$q$')
axs[1, 1].legend(loc='right')

for ax in axs.flatten():
    ax.set_ylim(-0.01, 1.1)

axs[1, 0].set_ylim(-0.001, 0.1)
plt.tight_layout()

# Save plots and results
results = {'p': p, 'mu': mu, 'q': q, 'R': R, 'I': I}
plt.savefig("../plots/figure_3.png", dpi=300)
np.savez("../results/figure_3.npz", results)

plt.close()
