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

def action_space_func(b, p, eta, Swind):
    if p != 0 and b >= eta:
        return np.concatenate((np.array([-1]), np.arange(len(Swind)).astype(int)))
    return [-1]

def policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    V = np.zeros((len(Swind), len(Swind), B+1, tau+1))
    policy = np.zeros((len(Swind), len(Swind), B+1, tau+1), dtype=int)

    Ss, Sz, Sb, Sp = np.arange(len(Swind)), np.arange(len(Swind)), np.arange(B+1), np.arange(tau+1)
    policy_stable = False
    iteration = 0

    while not policy_stable:
        iteration += 1
        print(f"Policy Iteration Step: {iteration}")

        # Policy Evaluation
        eval_iter = 0
        while True:
            eval_iter += 1
            delta = 0
            V_new = np.copy(V)
            for s, z, b, p in product(Ss, Sz, Sb, Sp):
                a = policy[s, z, b, p]
                q_val = q_func(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, s, z, b, p, a)
                delta = max(delta, abs(V_new[s, z, b, p] - q_val))
                V_new[s, z, b, p] = q_val
            V = np.copy(V_new)
            print(f"\tEvaluation Iteration {eval_iter}, Delta = {delta:.5f}")
            if delta < theta and eval_iter > Kmin:
                print("\tPolicy Evaluation Converged.")
                break

        # Policy Improvement
        policy_stable = True
        changes = 0
        for s, z, b, p in product(Ss, Sz, Sb, Sp):
            old_action = policy[s, z, b, p]
            actions = action_space_func(b, p, eta, Swind)
            best_action = old_action
            min_q = float('inf')
            for a in actions:
                q_val = q_func(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, V, s, z, b, p, a)
                if q_val < min_q:
                    min_q = q_val
                    best_action = a
            policy[s, z, b, p] = best_action
            if old_action != best_action:
                policy_stable = False
                changes += 1
        print(f"\tPolicy Improvement: {changes} changes made")

    return V, policy

# System parameters
Swind = np.linspace(0, 1, 21)
mu_wind = 0.3
z_wind = 0.5
stddev_wind = z_wind * np.sqrt(mu_wind * (1 - mu_wind))
retention_prob = 0.9
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)

lmbda = 0.7
B = 10
eta = 2
Delta = 3
mu_delta = 2
z_delta = 0.5
stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)
tau = 4
gamma = 1 / 15
beta = 0.95
theta = 0.01
Kmin = 10

# Run policy iteration
V_optimal_pi, policy_optimal_pi = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

# Visualization
def plot_value_function(V, Swind, b=2, p=3):
    V_slice = V[:, :, b, p]
    plt.figure(figsize=(10, 8))
    sns.heatmap(V_slice, xticklabels=np.round(Swind, 2), yticklabels=np.round(Swind, 2), cmap="coolwarm", annot=False)
    plt.xlabel("Wind Speed State (z)")
    plt.ylabel("Wind Speed State (s)")
    plt.title(f"Value Function Heatmap (b={b}, p={p})")
    plt.show()

def plot_policy(policy, Swind, b=2, p=3):
    policy_slice = policy[:, :, b, p]
    plt.figure(figsize=(10, 8))
    sns.heatmap(policy_slice, xticklabels=np.round(Swind, 2), yticklabels=np.round(Swind, 2), cmap="viridis", annot=False)
    plt.xlabel("Wind Speed State (z)")
    plt.ylabel("Wind Speed State (s)")
    plt.title(f"Optimal Policy Heatmap (b={b}, p={p})")
    plt.show()

# Plot results
plot_value_function(V_optimal_pi, Swind)
plot_policy(policy_optimal_pi, Swind)
