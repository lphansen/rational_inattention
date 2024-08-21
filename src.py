import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pdb


# def inner_objective(M, N, xi, Lam, alpha, r, hat_mu, eps, q_step, p_step, mu_step,payoff_set):
def solve(xi, Lam, alpha, params):

    M, N, hat_mu, eps, q_step, p_step, mu_step,payoff_set = params

    # Takes in payoff and calculates utility for that payoff.
    def utility_function(payoff, alpha):
        if alpha != 1:
            return 1 / (1 - alpha) * (payoff**(1 - alpha) - 1) + 1
        else:
            return np.log(payoff + 1e-7) if payoff > 0 else np.log(payoff + 1e-7)

    # Calculates utility for a given state, action, alpha and r value.
    # You can change the payoff set here.
    def utility(x, a, alpha, payoff_set):
        return utility_function(payoff_set[a, x], alpha)
    
    #Initial guesses
    p = np.ones((M, N)) / M  # Initial value of p(a|x)
    q = np.zeros((M))  # Initial value of q(a)
    v = np.zeros((N))  # Initial value of G(p*,q*)(x)
    mu = np.zeros((N))  # Initial mu(x)
    post = np.zeros((N, M))  # Posterior distribution of p(a|x)

    #Containers
    p_old = np.zeros((M, N))
    q_old = np.zeros((M))
    mu_old = np.zeros((N))

    mu[:] = hat_mu
    p_old[:, :] = p[:, :]
    q_old[:] = q[:]
    mu_old[:] = mu[:]
    Iter_err = float('inf')
    Iter = 0

    while Iter_err > eps:
        Iter += 1
        # Update q
        for a in range(M):
            q[a] = (1-q_step)*q_old[a] + q_step*sum(mu[x] * p[a, x] for x in range(N))
        
        # Update p
        for x in range(N):
            den = sum(q[a] * np.exp(utility(x, a, alpha,payoff_set) / Lam) for a in range(M))
            for a in range(M):
                p[a, x] = (1-p_step)*p_old[a,x] + (p_step)*(q[a] * np.exp(utility(x, a, alpha,payoff_set) / Lam) / den)

        # Update v
        for x in range(N):
            v[x] = Lam * np.log(sum(q[a] * np.exp(utility(x, a, alpha, payoff_set) / Lam) for a in range(M)))

        # Update mu
        den = sum(np.exp(-v[x] / xi) * hat_mu[x] for x in range(N))
        # Replace zeros or near-zeros in 'den' with a small positive value
        den = np.where(den == 0, np.finfo(float).eps, den)
        for x in range(N):
            mu[x] = (1 - mu_step) * mu_old[x] + mu_step * np.exp(-v[x] / xi) * hat_mu[x] / den
           
        Iter_err = (np.linalg.norm(p[:, :] - p_old[:, :], 'fro') +
                        np.linalg.norm(q[:] - q_old[:]) +
                        np.linalg.norm(mu[:] - mu_old[:]))
        p_old[:, :] = p[:, :]
        q_old[:] = q[:]
        mu_old[:] = mu[:]

    # Define ranges for x and a
    x_values = np.arange(0, N)  # Assuming x can be 0 or 1
    a_values = np.arange(0, M)  # Assuming a can be 0, 1, or 2

    # Create an empty 2D array to store the results
    utility_values = np.zeros((len(a_values), len(x_values)))

    # Evaluate the utility function for each pair of (x, a)
    for i, a in enumerate(a_values):
        for j, x in enumerate(x_values):
            utility_values[i,j] = utility(x, a, alpha,payoff_set)
    
    q_array= np.array(q[:]).reshape(-1,1)
    for w in range(M):
        if q_array[w][0] == 0:
            q_array[w][0] = 1e-30
    q_array = np.repeat(q_array, N, axis=1)
    entropy = np.sum(np.log(mu[:]/hat_mu) * mu[:])
    information_term = p[:,:]/q_array
    for i in range(M):
        for j in range(N):
            if information_term[i,j] == 0:
                information_term[i,j] = 1e-30
            else:
                information_term[i,j] = information_term[i,j]   
    information = np.sum(np.sum(np.log(information_term) * p[:,:],axis=0)*mu[:])
    total_utility = np.sum(np.sum(utility_values*p[:,:],axis=0)*mu[:])
    objective = total_utility+xi*entropy-Lam*information

    return {
            'objective': objective,
            'information': information,
            'entropy': entropy,
            'p': p,
            'q': q,
            'mu': mu
            }

def save_results(name, results, alpha_array, Lam_array, xi_array, params,**kwargs):
    # Ensure the directory exists
    kappa_array = kwargs.get('kappa_array', None)
    Lam_grid = kwargs.get('Lam_grid', None)
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the data using pickle
    filepath = os.path.join(directory, f'{name}.pkl')
    with open(filepath, 'wb') as file:
        pickle.dump({
            'results': results,
            'alpha_array': alpha_array,
            'Lam_array': Lam_array,
            'xi_array': xi_array,
            'params': params,
            'kappa_array': kappa_array,
            'Lam_grid': Lam_grid
        }, file)

def plot(results, alpha_array, Lam_array, xi_array, axis, fixed_values, plot_name, index_labels=[r'$\alpha$', r'$\lambda$', r'$\xi$']):
    # Determine the shape of the results array
    shape = results.shape
    assert len(shape) == 3, "results should be a 3D array"
    
    # Unpack index labels for axes
    x_label = index_labels[axis]

    # Prepare the axes to hold constant
    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    arrays = [alpha_array, Lam_array, xi_array]

    # Extract the indices for the fixed values
    fixed_indices = [None, None]
    for i, ax in enumerate(other_axes):
        if fixed_values[i] < shape[ax]:
            fixed_indices[i] = fixed_values[i]
        else:
            raise ValueError(f"Fixed value index {fixed_values[i]} is out of bounds for axis {ax}")

    # Extract the data for plotting
    if axis == 0:
        p = np.array([results[i, fixed_indices[0], fixed_indices[1]]['p'] for i in range(shape[0])])
        I = np.array([results[i, fixed_indices[0], fixed_indices[1]]['information'] for i in range(shape[0])])
        R = np.array([results[i, fixed_indices[0], fixed_indices[1]]['entropy'] for i in range(shape[0])])
        mu = np.array([results[i, fixed_indices[0], fixed_indices[1]]['mu'] for i in range(shape[0])])
        q = np.array([results[i, fixed_indices[0], fixed_indices[1]]['q'] for i in range(shape[0])])
        index = alpha_array
    elif axis == 1:
        p = np.array([results[fixed_indices[0], i, fixed_indices[1]]['p'] for i in range(shape[1])])
        I = np.array([results[fixed_indices[0], i, fixed_indices[1]]['information'] for i in range(shape[1])])
        R = np.array([results[fixed_indices[0], i, fixed_indices[1]]['entropy'] for i in range(shape[1])])
        mu = np.array([results[fixed_indices[0], i, fixed_indices[1]]['mu'] for i in range(shape[1])])
        q = np.array([results[fixed_indices[0], i, fixed_indices[1]]['q'] for i in range(shape[1])])
        index = Lam_array
    else:  # axis == 2
        p = np.array([results[fixed_indices[0], fixed_indices[1], i]['p'] for i in range(shape[2])])
        I = np.array([results[fixed_indices[0], fixed_indices[1], i]['information'] for i in range(shape[2])])
        R = np.array([results[fixed_indices[0], fixed_indices[1], i]['entropy'] for i in range(shape[2])])
        mu = np.array([results[fixed_indices[0], fixed_indices[1], i]['mu'] for i in range(shape[2])])
        q = np.array([results[fixed_indices[0], fixed_indices[1], i]['q'] for i in range(shape[2])])
        index = xi_array

    # Create the subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 17))

    # Subplot configurations
    configurations = [
        (axs[0, 0], p[:, 0, 0], p[:, 1, 0], p[:, 2, 0], 'State 1 decision rule', r'$p^*(1|x_1)$', r'$p^*(2|x_1)$', r'$p^*(3|x_1)$'),
        (axs[0, 1], p[:, 0, 1], p[:, 1, 1], p[:, 2, 1], 'State 2 decision rule', r'$p^*(1|x_2)$', r'$p^*(2|x_2)$', r'$p^*(3|x_2)$'),
        (axs[1, 0], I, None, None, 'Mutual information $I$', None, None, None),
        (axs[1, 1], R, None, None, 'Relative entropy $R$', None, None, None),
        (axs[2, 0], mu[:, 0], mu[:, 1], None, 'Worst case prior $\mu^*$', r'$\mu^*(x_1)$', r'$\mu^*(x_2)$', None),
        (axs[2, 1], q[:, 0], q[:, 1], q[:, 2], 'Worst case $q^*$', r'$q^*(x_1)$', r'$q^*(x_2)$', r'$q^*(x_3)$')
    ]

    # Plot each subplot
    for ax, data1, data2, data3, title, label1, label2, label3 in configurations:
        if data3 is not None:
            ax.plot(index, data1, '-b', linewidth=2, label=label1)
            ax.plot(index, data2, '-r', linewidth=2, linestyle='dashed', label=label2)
            ax.plot(index, data3, '-k', linewidth=2, linestyle='dotted', label=label3)
        elif data2 is not None:
            ax.plot(index, data1, '-k', linewidth=2, label=label1)
            ax.plot(index, data2, '-r', linewidth=2, linestyle='dashed', label=label2)
        else:
            ax.plot(index, data1, '-k', linewidth=2, label=label1)
        
        ax.set_xlabel(x_label)
        # ax.set_ylabel('Value')
        ax.set_title(title)
        if label1 is not None:
            ax.legend(loc='best')
        ax.set_ylim(-0.01, 1.1)
        ax.set_xlim(left=0)
    
    plt.suptitle(rf'{index_labels[other_axes[0]]}={arrays[0][fixed_values[0]]}, {index_labels[other_axes[1]]}={arrays[1][fixed_values[1]]}',fontsize=16)
    plt.tight_layout()
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig("plots/"+plot_name+".png")

    
    plt.show()

def find_min_Lam(alpha,kappa,xi,params,Lam_grid):
    obj = 0
    best_objective = float('inf')
    best_Lam = 0
    for Lam in Lam_grid:
        res = solve(xi,Lam,alpha,params)
        obj = (res['objective'] + Lam*kappa)
        if obj < best_objective and res['information']<kappa:
            best_objective = obj
            best_Lam = Lam
    if best_Lam==0:
        raise ValueError('Optimal solution exceeds information constraint for (alpha,xi,kappa) = ({},{},{})'.format(alpha,xi,kappa))
        return None
    return best_Lam