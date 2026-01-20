import numpy as np
from itertools import product
from Assignment2Tools import prob_vector_generator, markov_matrix_generator
import matplotlib.pyplot as plt
import seaborn as sns

def q_func(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, s, z, b, p, a):
    
    if p == 0:
        wind_diff_sq = (Swind[s] - Swind[z])**2
        s_next_vals = np.arange(len(Swind))[:, None]              
        d_vals = np.arange(Delta+1)[None, :]
        b_indices = np.clip(b + d_vals, 0, B)                    

        # Compute V values for both continuation and reset
        V_slice = gamma * V[s_next_vals, z, b_indices, tau] + \
          (1 - gamma) * V[s_next_vals, z, b_indices, 0]  

        # Weight by transition probabilities and solar distribution
        q_val = wind_diff_sq + beta * np.sum(P[s, :, None] * alpha[None, :] * V_slice)

    elif p != 0 and a == -1:
        wind_diff_sq = (Swind[s] - Swind[z])**2
        s_next_vals = np.arange(len(Swind))[:, None]               
        d_vals = np.arange(Delta+1)[None, :]                           
        b_indices = np.clip(b + d_vals, 0, B)                        

        # Gather the relevant slice of V
        V_slice = V[s_next_vals, z, b_indices, p - 1]                

        # Multiply with P and alpha using broadcasting
        q_val = wind_diff_sq + beta * np.sum(P[s, :, None] * alpha[None, :] * V_slice)

    else:
        wind_diff_sq1 = (Swind[s] - Swind[z])**2
        wind_diff_sq2 = (Swind[s] - Swind[a])**2

        s_next_vals = np.arange(len(Swind))[:, None]                     
        d_vals = np.arange(Delta+1)[None, :]                            
        b_indices = np.clip(b + d_vals - eta, 0, B)                      

        V_z = V[s_next_vals, z, b_indices, p - 1]                        
        V_a = V[s_next_vals, a, b_indices, p - 1]                        

        P_s = P[s, :, None]                                              
        alpha_d = alpha[None, :]                                         

        sum1 = beta * np.sum(P_s * alpha_d * V_z)
        sum2 = beta * np.sum(P_s * alpha_d * V_a)

        q_val1 = wind_diff_sq1 + sum1
        q_val2 = wind_diff_sq2 + sum2

        q_val = (1 - lmbda) * q_val1 + lmbda * q_val2

    return q_val

def action_space_func(b,p,eta,Swind):
    
    if p!=0 and b>=eta:
        return np.concatenate((np.array([-1]),np.arange(len(Swind)).astype(int)))
    if p == 0:
        return [-1]
    if p != 0 and b <= eta:
        return [-1]

def value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):

    # value function initialization
    V = np.zeros((len(Swind),len(Swind), B+1, tau+1 ))
    V_new = np.zeros((len(Swind),len(Swind), B+1, tau+1 ))

# Estimating the optimal value function
    Ss = np.arange(len(Swind)).astype(int)
    Sz = np.arange(len(Swind)).astype(int)  
    Sb = np.arange(B+1).astype(int)
    Sp = np.arange(tau+1).astype(int)
    
    D = np.inf
    iteration = 1
    while D > theta or iteration <= Kmin:
        for s,z,b,p in product(Ss,Sz,Sb,Sp):
            action_space = action_space_func(b,p,eta,Swind)
            min_q = np.inf
            for a in action_space:
                q_val = q_func(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, s, z, b, p, a)
                if q_val<min_q:
                    min_q = q_val
            V_new[s,z,b,p] = min_q

        D = np.max(np.abs(V_new - V))
        V = np.copy(V_new)

        print(iteration, D)
        iteration+=1                

    # Estimating the optimal policy using optimal value function
    
    policy = np.zeros((len(Swind),len(Swind), B+1, tau+1 ))

    for s,z,b,p in product(Ss,Sz,Sb,Sp):
        action_space = action_space_func(b,p,eta,Swind)
        min_q = np.inf
        for a in action_space:
            q_val = q_func(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, s, z, b, p, a)
            if q_val<min_q:
                min_q = q_val
                min_action = a
        policy[s,z,b,p] = min_action

    return V, policy

# System parameters (set to default values)
Swind = np.linspace(0, 1, 21)                      # The set of all possible normalized wind speed.
mu_wind = 0.3                                      # Mean wind speed. You can vary this between 0.2 to 0.8.
z_wind = 0.5                                       # Z-factor of the wind speed. You can vary this between 0.25 to 0.75.
                                                   # Z-factor = Standard deviation divided by mean.
                                                   # Higher the Z-factor, the more is the fluctuation in wind speed.
stddev_wind = z_wind*np.sqrt(mu_wind*(1-mu_wind))  # Standard deviation of the wind speed.
retention_prob = 0.9                               # Retention probability is the probability that the wind speed in the current and the next time slot is the same.
                                                   # You can vary the retention probability between 0.05 to 0.95.
                                                   # Higher retention probability implies lower fluctuation in wind speed.
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)  # Markovian probability matrix governing wind speed.

lmbda = 0.7  # Probability of successful transmission.

B = 10         # Maximum battery capacity.
eta = 2        # Battery power required for one transmission.
Delta = 3      # Maximum solar power in one time slot.
mu_delta = 2   # Mean of the solar power in one time slot.
z_delta = 0.5  # Z-factor of the slower power in one time slot. You can vary this between 0.25 to 0.75.                  
stddev_delta = z_delta*np.sqrt(Delta*(Delta-mu_delta))  # Standard deviation of the solar power in one time slot.
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)  # Probability distribution of solar power in one time slot.

tau = 4       # Number of time slots in active phase.
gamma = 1/15  # Probability of getting chance to transmit. It can vary between 0.01 to 0.99.

beta = 0.95   # Discount factor.
theta = 0.01  # Convergence criteria: Maximum allowable change in value function to allow convergence.
Kmin = 10     # Convergence criteria: Minimum number of iterations to allow convergence.


# Call value iteration function.
V_optimal_vi, policy_optimal_vi = value_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

#visualize
def plot_value_function(V, Swind, b=5, p=2):
    """
    Plots the value function for a fixed battery state (b) and time slot (p).
    """
    V_slice = V[:, :, b, p]  # Extract V for specific battery level b and time slot p

    plt.figure(figsize=(10, 8))
    sns.heatmap(V_slice, xticklabels=np.round(Swind, 2), yticklabels=np.round(Swind, 2), cmap="coolwarm", annot=False)
    plt.xlabel("Wind Speed State (z)")
    plt.ylabel("Wind Speed State (s)")
    plt.title(f"Value Function Heatmap (b={b}, p={p})")
    plt.show()

def plot_policy(policy, Swind, b=5, p=2):
    """
    Plots the optimal policy for a fixed battery state (b) and time slot (p).
    """
    policy_slice = policy[:, :, b, p]  # Extract policy for b, p

    plt.figure(figsize=(10, 8))
    sns.heatmap(policy_slice, xticklabels=np.round(Swind, 2), yticklabels=np.round(Swind, 2), cmap="viridis", annot=False)
    plt.xlabel("Wind Speed State (z)")
    plt.ylabel("Wind Speed State (s)")
    plt.title(f"Optimal Policy Heatmap (b={b}, p={p})")
    plt.show()

# Call the visualization functions
plot_value_function(V_optimal_vi, Swind)
plot_policy(policy_optimal_vi, Swind)