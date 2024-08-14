import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 2  # Number of states
M = 3  # Number of actions

# #Time step
q_step = 0.01
p_step = 0.01
mu_step = 1

# Baseline prior
hat_mu = np.zeros(N)  
hat_mu[0] = 0.5
hat_mu[1] = 0.5

eps = 1e-15  # Error tolerance


#Define utility function
def utility_function(payoff, alpha):
    if alpha!=1:
        return 1/(1-alpha)*(payoff**(1-alpha)-1)
    elif alpha==1:
        if payoff>0:
            return np.log(payoff)
        elif payoff==0:
            return np.log(payoff+1e-7)
        else:
            ValueError("Negative log utility")



def utility(x, a,alpha,r):
    #Define payoffs
    payoff_set = np.array([[0.0, 2*r],[r, 0.0], [5.0, 5.0]])
    if a == 0:  # action 1
        return utility_function(payoff_set[0,0],alpha) if x==0 else utility_function(payoff_set[0,1],alpha)
    elif a == 1:  # aciton 2
        return utility_function(payoff_set[1,0],alpha) if x == 0 else utility_function(payoff_set[1,1],alpha)
    elif a == 2:  # action 3
        return utility_function(payoff_set[2,0],alpha) if x == 0 else utility_function(payoff_set[2,1],alpha)

# Create alpha array
alpha_array = np.arange(0,0.95,0.05)
A = len(alpha_array)


p_final = np.zeros((A, M, N)) / M  # Initial value of p(a|x)
q_final = np.zeros((A, M))  # Initial value of q(a)
mu_final = np.zeros((A, N))  # Initial mu(x)
R_final = np.zeros((A))
I_final = np.zeros((A))
xi_final = np.zeros((A))
lambda_final = np.zeros((A))
objective_final = np.zeros((A))

def inner_objective(xi, Lam,alpha,r):

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
    R = 0.0
    I = 0.0
    q_baseline = np.zeros((M))

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
            den = sum(q[a] * np.exp(utility(x, a, alpha,r) / Lam) for a in range(M))
            for a in range(M):
                p[a, x] = (1-p_step)*p_old[a,x] + (p_step)*(q[a] * np.exp(utility(x, a, alpha,r) / Lam) / den)
           
        Iter_err = (np.linalg.norm(p[:, :] - p_old[:, :], 'fro') +
                        np.linalg.norm(q[:] - q_old[:]) +
                        np.linalg.norm(mu[:] - mu_old[:]))
        p_old[:, :] = p[:, :]
        q_old[:] = q[:]
        mu_old[:] = mu[:]

    # Define ranges for x and a
    x_values = np.arange(0, 2)  # Assuming x can be 0 or 1
    a_values = np.arange(0, 3)  # Assuming a can be 0, 1, or 2

    # Create an empty 2D array to store the results
    utility_values = np.zeros((len(a_values), len(x_values)))

    # Evaluate the utility function for each pair of (x, a)
    for i, a in enumerate(a_values):
        for j, x in enumerate(x_values):
            utility_values[i,j] = utility(x, a, alpha,r)

    def calculate_objective(p,q):
        q_array= np.array(q[:]).reshape(-1,1)
        for w in range(M):
            if q_array[w][0] == 0:
                q_array[w][0] = 1e-30
        q_array = np.repeat(q_array, 2, axis=1)
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

        return objective, information, entropy
    
    main_objective = calculate_objective(p,q)
    p_constrained = np.array([[0,0],[0,0],[1,1]]).astype(np.float64)
    q_constrained = np.array([0,0,1]).astype(np.float64)
    objective_constrained = calculate_objective(p_constrained,q_constrained)

    if main_objective[0] > objective_constrained[0]:

        objective = main_objective[0]
        information = main_objective[1]
        entropy = main_objective[2]
        return objective, p[:,:], q[:], mu[:], information, entropy
    else:
        print('Constrained option for alpha-Lam:',{alpha,Lam})
        objective = objective_constrained[0]
        information = objective_constrained[1]
        entropy = objective_constrained[2]
        return objective, p_constrained,q_constrained, mu[:], information, entropy
        
        


kappa_array = [0.1]
# kappa_array = [0.7]
eta_array = [0.04]
r_array= [6.0, 6.5, 7.0, 7.5]
r_array= [7.0]
# r_array=[7.0]
for r in range(len(r_array)):
    utility_r = r_array[r]
    for k in range(len(kappa_array)):
        kappa = kappa_array[k]
        for e in range(len(eta_array)):
            eta = eta_array[e]
            for a in range(len(alpha_array)):
                alpha = alpha_array[a]

                Lam_values = np.linspace(1,10,100)
                    
                
                xi = 10

                def find_min_Lam(xi):
                    obj = 0
                    best_objective = float('inf')
                    best_Lam = 0
                    for Lam in Lam_values:
                        res = inner_objective(xi,Lam,alpha,utility_r)
                        obj = (res[0] + Lam*kappa)
                        if obj < best_objective and res[-2]<kappa:
                            best_objective = obj
                            best_Lam = Lam
                    return best_Lam
                
                Lam = find_min_Lam(xi)

                
                xi_final[a] = xi
                lambda_final[a] = Lam
                objective_final[a],p_final[a,:,:],q_final[a,:],mu_final[a,:],I_final[a],R_final[a] = inner_objective(xi_final[a],lambda_final[a],alpha,utility_r)

            # Plotting
            index = alpha_array

            fig, axs = plt.subplots(2, 2, figsize=(10, 8))



            # Subplot 3
            axs[0, 0].plot(index, p_final[:, 0, 0], '-b', linewidth=2, label=r'$p^*(1|x_1)$')
            axs[0, 0].plot(index, p_final[:, 1, 0], '-r', linewidth=2, label=r'$p^*(2|x_1)$', linestyle='dashed')
            axs[0, 0].plot(index, p_final[:, 2, 0], '-k', linewidth=2, label=r'$p^*(3|x_1)$', linestyle='dotted')
            axs[0, 0].set_xlabel(r'$\alpha$')
            axs[0, 0].set_ylabel(r'$p$')
            axs[0, 0].set_title('State 1 decision rule')
            axs[0, 0].legend(loc='upper right')

            # Subplot 4
            axs[0, 1].plot(index, p_final[:, 0, 1], '-b', linewidth=2, label=r'$p^*(1|x_2)$')
            axs[0, 1].plot(index, p_final[:, 1, 1], '-r', linewidth=2, label=r'$p^*(2|x_2)$', linestyle='dashed')
            axs[0, 1].plot(index, p_final[:, 2, 1], '-k', linewidth=2, label=r'$p^*(3|x_2)$', linestyle='dotted')
            axs[0, 1].set_xlabel(r'$\alpha$')
            axs[0, 1].set_ylabel(r'$p$')
            axs[0, 1].set_title('State 2 decision rule')
            axs[0, 1].legend()

            # Subplot 5
            axs[1,  0].plot(index, lambda_final[:], '-k', linewidth=2, label=r'$\lambda$')
            axs[1,  0].set_xlabel(r'$\alpha$')
            axs[1,  0].set_ylabel(r'$\lambda$')
            axs[1,  0].set_title(r'$\lambda$')
            axs[1,  0].legend()
            axs[1,  0].legend(loc='upper right')


            # Subplot 2
            axs[1,  1].plot(index, q_final[:, 0], '-b', linewidth=2, label=r'$q^*(1)$')
            axs[1,  1].plot(index, q_final[:, 1], '-r', linewidth=2, label=r'$q^*(2)$', linestyle='dashed')
            axs[1,  1].plot(index, q_final[:, 2], '-k', linewidth=2, label=r'$q^*(3)$', linestyle='dotted')
            axs[1,  1].set_xlabel(r'$\alpha$')
            axs[1,  1].set_ylabel(r'$q$')
            axs[1,  1].set_title('$q$ worst-case')
            axs[1,  1].legend(loc='upper right')


            for ax in axs.flatten():
                ax.set_ylim(0,1)
            axs[1,0].autoscale()
            plt.tight_layout()

            results = {'p':p_final, 'mu':mu_final,'q':q_final,'R':R_final,'I':I_final}
            plt.savefig("../plots/figure_4.png",dpi=300)
            np.savez("../results/figure_4.npz",results)
            # plt.savefig(f"../plots/alpha/gridsearch/kappa_{kappa_array[0]}_constrained_lambdalen_{len(Lam_values)}_alphalen_{len(alpha_array)}_tol_{eps}_q_step{q_step}_r_{utility_r}_eta_{eta}.png",dpi=300)
            # np.savez(f"../results/alpha/gridsearch/kappa_{kappa_array[0]}_constrained_lambdalen_{len(Lam_values)}_alphalen_{len(alpha_array)}_tol_{eps}_q_step{q_step}_r_{utility_r}_eta_{eta}.png.npz",
            #             p = p_final, q = q_final, mu = mu_final, R = R_final, I = I_final, alpha_array = alpha_array,\
            #                 Lam_values = Lam_values, xi_final = xi_final, lambda_final = lambda_final, eps = eps)
            # else:
            #     plt.savefig(f"../plots/alpha/gridsearch/xi_infty_lam_infty_{Lam_vary}_r_{utility_r}_eta_{eta}.png",dpi=300)
            #     np.savez(f"../results/alpha/gridsearch/xi_infty_lam_infty_{Lam_vary}_r_{utility_r}_eta_{eta}.png.npz",
            #                p = p_final, q = q_final, mu = mu_final, R = R_final, I = I_final, alpha_array = alpha_array)
            plt.close()
            