import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 2  # Number of states
M = 3  # Number of actions
Lam_array = [200]  # Approximation for infinite lambda
alpha_array = [0.0]  # Set of alpha values

# Create xi_array
xi_minstart = 1
xi_max = 55
xi_increment = (xi_max - xi_minstart) / 50
xi_set = np.arange(xi_minstart, xi_max, xi_increment)
T = len(xi_set)  # Number of time steps

# Time-step increments
q_step = 0.1
p_step = 0.1
mu_step = 0.1

# Baseline prior
hat_mu = np.array([0.5, 0.5])  # Initial prior distribution
eps = 1e-10  # Error tolerance

# Utility function
def utility_function(payoff, alpha):
    if alpha != 1:
        return 1 / (1 - alpha) * (payoff**(1 - alpha) - 1) + 1
    else:
        return np.log(payoff + 1e-7) if payoff > 0 else np.log(payoff + 1e-7)

# Payoff utility calculation
def utility(x, a, alpha, r):
    payoff_set = np.array([[0.0, r * 2], [r, 0.0], [5.0, 5.0]])
    return utility_function(payoff_set[a, x], alpha)

# Initialize results dictionary
results = {}
r_array = [7.0, 7.5]

for r in r_array:
    utility_r = r
    for Lam in Lam_array:
        for alpha in alpha_array:
            p = np.ones((T, M, N)) / M  # Initial probability p(a|x)
            q = np.zeros((T, M))  # Initial probability q(a)
            v = np.zeros((T, N))  # Value function
            mu = np.zeros((T, N))  # Worst-case prior
            post = np.zeros((T, N, M))  # Posterior distribution

            for t in range(T):
                xi = xi_set[t]
                mu[t, :] = hat_mu
                p_old, q_old, mu_old = p.copy(), q.copy(), mu.copy()

                Iter_err = float('inf')
                while Iter_err > eps:
                    # Update q
                    for a in range(M):
                        q[t, a] = (1 - q_step) * q_old[t, a] + q_step * sum(mu[t, x] * p[t, a, x] for x in range(N))

                    # Update p
                    for x in range(N):
                        den = sum(q[t, a] * np.exp(utility(x, a, alpha, utility_r) / Lam) for a in range(M))
                        for a in range(M):
                            p[t, a, x] = (1 - p_step) * p_old[t, a, x] + p_step * q[t, a] * np.exp(utility(x, a, alpha, utility_r) / Lam) / den

                    # Update v
                    for x in range(N):
                        v[t, x] = Lam * np.log(sum(q[t, a] * np.exp(utility(x, a, alpha, utility_r) / Lam) for a in range(M)))

                    # Update mu
                    den = sum(np.exp(-v[t, x] / xi) * hat_mu[x] for x in range(N))
                    for x in range(N):
                        mu[t, x] = (1 - mu_step) * mu_old[t, x] + mu_step * np.exp(-v[t, x] / xi) * hat_mu[x] / den

                    # Compute iteration error
                    Iter_err = (np.linalg.norm(p[t, :, :] - p_old[t, :, :], 'fro') +
                                np.linalg.norm(q[t, :] - q_old[t, :]) +
                                np.linalg.norm(mu[t, :] - mu_old[t, :]))

                    p_old, q_old, mu_old = p.copy(), q.copy(), mu.copy()

                # Update posterior distribution
                for a in range(M):
                    for x in range(N):
                        if q[t, a] > 1e-4:
                            post[t, x, a] = p[t, a, x] * mu[t, x] / q[t, a]

            # Calculate R (relative entropy)
            R = np.sum(np.log(mu / hat_mu) * mu, axis=1)

            # Calculate I (mutual information)
            I = np.zeros(T)
            for i in range(T):
                q_array = q[i].reshape(-1, 1)
                q_array_2 = np.repeat(q_array, N, axis=1)
                I[i] = np.sum(np.sum(np.log(p[i, :, :] / q_array_2) * p[i, :, :], axis=0) * mu[i])

            # Store results for current r
            results[r] = (p, mu, q, R, I)

# Plot results
index = xi_set
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot configurations
configurations = [
    (axs[0, 0], r_array[0], 1, "Decision rule", 'right'),
    (axs[0, 1], r_array[1], 1, "Decision rule", 'best'),
    (axs[1, 0], r_array[0], 0, "Worst-case prior $\mu^*$", 'best'),
    (axs[1, 1], r_array[1], 0, "Worst-case prior $\mu^*$", 'best')
]

# Plot each subplot
for ax, r, plot_type, title, legend_loc in configurations:
    if plot_type == 1:
        ax.plot(index, results[r][0][:, 0, 1], '-b', linewidth=2, label=r'$p^*(1)$')
        ax.plot(index, results[r][0][:, 1, 1], '-r', linewidth=2, label=r'$p^*(2)$', linestyle='dashed')
        ax.plot(index, results[r][0][:, 2, 1], '-k', linewidth=2, label=r'$p^*(3)$', linestyle='dotted')
        ax.set_ylabel(r'$p$')
    else:
        ax.plot(index, results[r][1][:, 0], '-k', linewidth=2, label=r'$\mu^*(x_1)$')
        ax.plot(index, results[r][1][:, 1], '-r', linewidth=2, label=r'$\mu^*(x_2)$', linestyle='dashed')
        ax.set_ylabel(r'$\mu$')
    ax.set_xlabel(r'$\xi$')
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    ax.set_ylim(-0.01, 1.1)
    ax.set_xlim(0, 50)

# Finalize and save the figure
plt.tight_layout()
plt.savefig("../plots/figure_1.png", dpi=300)
np.savez("../results/figure_1.npz", results)

# Close the plot to free up resources
plt.close()