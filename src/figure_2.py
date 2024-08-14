import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 2  # Number of states
M = 3  # Number of actions
Lam_array = [10]  # Lambda values to explore
alpha_array = [0.0]  # Alpha values to explore

# Create xi_set array
xi_minstart = 1
xi_max = 55
xi_increment = (xi_max - xi_minstart) / 50
xi_set = np.arange(xi_minstart, xi_max, xi_increment)
T = len(xi_set)

# Time step parameters
q_step = 0.1
p_step = 0.1
mu_step = 0.1

# Baseline prior
hat_mu = np.array([0.5, 0.5])

eps = 1e-8  # Error tolerance

# Utility function based on payoff and alpha
def utility_function(payoff, alpha):
    if alpha != 1:
        return 1 / (1 - alpha) * (payoff**(1 - alpha) - 1)
    elif alpha == 1:
        return np.log(payoff + 1e-7) if payoff >= 0 else ValueError("Negative log utility")

# Payoff set definition
r = 7.5
payoff_set = np.array([[0.0, r * 2], [r, 0.0], [5.0, 5.0]])

# Utility calculation for different actions and states
def utility(x, a, alpha):
    return utility_function(payoff_set[a, x], alpha)

# Main loop: Calculate decision rules and related metrics for each alpha and lambda
for l in Lam_array:
    for alph in alpha_array:
        p = np.ones((T, M, N)) / M  # Initialize p(a|x) with equal probabilities
        q, v, mu = np.zeros((T, M)), np.zeros((T, N)), np.zeros((T, N))
        post = np.zeros((T, N, M))  # Posterior distribution
        p_old, q_old, mu_old = np.zeros_like(p), np.zeros_like(q), np.zeros_like(mu)

        for t in range(T):
            xi = xi_set[t]
            mu[t, :] = hat_mu
            p_old[t, :, :], q_old[t, :], mu_old[t, :] = p[t, :, :], q[t, :], mu[t, :]

            Iter_err, Iter = float('inf'), 0

            while Iter_err > eps:
                Iter += 1
                # Update q
                for a in range(M):
                    q[t, a] = (1 - q_step) * q_old[t, a] + q_step * sum(mu[t, x] * p[t, a, x] for x in range(N))

                # Update p
                for x in range(N):
                    den = sum(q[t, a] * np.exp(utility(x, a, alph) / l) for a in range(M))
                    for a in range(M):
                        p[t, a, x] = (1 - p_step) * p_old[t, a, x] + p_step * (q[t, a] * np.exp(utility(x, a, alph) / l) / den)

                # Update v
                for x in range(N):
                    tot = sum(q[t, a] * np.exp(utility(x, a, alph) / l) for a in range(M))
                    v[t, x] = l * np.log(tot)

                # Update mu
                den = sum(np.exp(-v[t, x] / xi) * hat_mu[x] for x in range(N))
                for x in range(N):
                    mu[t, x] = (1 - mu_step) * mu_old[t, x] + mu_step * np.exp(-v[t, x] / xi) * hat_mu[x] / den

                # Check for convergence
                Iter_err = (np.linalg.norm(p[t, :, :] - p_old[t, :, :], 'fro') +
                            np.linalg.norm(q[t, :] - q_old[t, :]) +
                            np.linalg.norm(mu[t, :] - mu_old[t, :]))
                p_old[t, :, :], q_old[t, :], mu_old[t, :] = p[t, :, :], q[t, :], mu[t, :]

            # Update posterior distribution
            for a in range(M):
                for x in range(N):
                    if q[t, a] > 1e-4:
                        post[t, x, a] = p[t, a, x] * mu[t, x] / q[t, a]

        # Calculate relative entropy R
        R = np.sum(np.log(mu / hat_mu) * mu, axis=1)

        # Calculate mutual information I
        I = np.zeros(T)
        for i in range(T):
            q_array = np.array(q[i]).reshape(-1, 1)
            q_array_2 = np.repeat(q_array, N, axis=1)
            I[i] = np.sum(np.sum(np.log(p[i, :, :] / q_array_2) * p[i, :, :], axis=0) * mu[i])

        # Calculate baseline q
        q_baseline = np.sum(0.5 * p, axis=2)

# Plotting results
index = xi_set
fig, axs = plt.subplots(3, 2, figsize=(10, 12))

# Subplot configurations
configurations = [
    (axs[0, 0], p[:, 0, 0], p[:, 1, 0], p[:, 2, 0], 'State 1 decision rule', r'$p^*(1|x_1)$', r'$p^*(2|x_1)$', r'$p^*(3|x_1)$'),
    (axs[0, 1], p[:, 0, 1], p[:, 1, 1], p[:, 2, 1], 'State 2 decision rule', r'$p^*(1|x_2)$', r'$p^*(2|x_2)$', r'$p^*(3|x_2)$'),
    (axs[1, 0], I, None, None, 'Mutual information $I$', None, None, None),
    (axs[1, 1], R, None, None, 'Relative entropy $R$', None, None, None),
    (axs[2, 0], mu[:, 0], mu[:, 1], None, 'Worst case prior $\mu^*$', r'$\mu^*(x_1)$', r'$\mu^*(x_2)$', None),
    (axs[2, 1], q[:, 0], q[:, 1], None, 'Worst case $q^*$', r'$q^*(x_1)$', r'$q^*(x_2)$', None)
]

# Plot each subplot
for ax, data1, data2, data3, title, label1, label2, label3 in configurations:
    ax.plot(index, data1, '-b', linewidth=2, label=label1)
    if data2 is not None:
        ax.plot(index, data2, '-r', linewidth=2, linestyle='dashed', label=label2)
    if data3 is not None:
        ax.plot(index, data3, '-k', linewidth=2, linestyle='dotted', label=label3)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$p$')
    ax.set_title(title)
    if label1 is not None:
        ax.legend(loc='best')
    ax.set_ylim(-0.01, 1.1)
    ax.set_xlim(0, 50)

# Adjust ylim for specific plots
axs[1, 0].set_ylim(-0.01, 0.15)
axs[1, 1].set_ylim(-0.01, 0.15)

# Finalize and save the figure
plt.tight_layout()
results = {'p': p, 'mu': mu, 'q': q, 'R': R, 'I': I}
plt.savefig("../plots/figure_2.png", dpi=300)
np.savez("../results/figure_2.npz", results)

# Close the plot to free up resources
plt.close()
