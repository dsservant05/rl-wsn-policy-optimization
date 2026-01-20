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

def evaluate_policy(policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, num_episodes=100, episode_length=100):
    """
    Evaluate a policy by running simulations and calculating average discounted reward.
    """
    total_rewards = 0
    
    for _ in range(num_episodes):
        # Random initial state
        s = np.random.choice(len(Swind))
        z = np.random.choice(len(Swind))
        b = np.random.randint(0, B+1)
        p = np.random.randint(0, tau+1)
        
        episode_reward = 0
        discount = 1.0
        
        for _ in range(episode_length):
            # Get action from policy
            a = policy[s, z, b, p]
            
            # Calculate immediate reward
            if a == -1:  # No transmission
                reward = (Swind[s] - Swind[z])**2
            else:  # Transmission
                if np.random.rand() < lmbda:  # Successful transmission
                    reward = (Swind[s] - Swind[a])**2
                    z = a  # Update last successful transmission
                else:  # Failed transmission
                    reward = (Swind[s] - Swind[z])**2
            
            # Add to total reward with discount
            episode_reward += discount * reward
            discount *= beta
            
            # State transition
            # Solar energy arrival
            d = np.random.choice(len(alpha), p=alpha)
            
            # Battery update
            if a != -1:  # If transmitted, consume energy
                new_b = min(b + d - eta, B)
            else:
                new_b = min(b + d, B)
            
            # Phase transition
            if p == 0:  # Passive phase
                if np.random.rand() < gamma:  # Switch to active
                    new_p = tau
                else:
                    new_p = 0
            else:  # Active phase
                new_p = p - 1
            
            # Wind speed transition
            s = np.random.choice(len(Swind), p=P[s])
            
            # Update state for next step
            b, p = new_b, new_p
        
        total_rewards += episode_reward
    
    return total_rewards / num_episodes

def get_greedy_policy(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta):
    """
    Compute greedy policy that minimizes immediate reward.
    """
    policy = np.zeros((len(Swind), len(Swind), B+1, tau+1), dtype=int)
    
    for s, z, b, p in product(range(len(Swind)), range(len(Swind)), range(B+1), range(tau+1)):
        actions = action_space_func(b, p, eta, Swind)
        min_reward = float('inf')
        best_action = -1
        
        for a in actions:
            if a == -1:  # No transmission
                reward = (Swind[s] - Swind[z])**2
            else:  # Transmission
                # Expected reward considering transmission success probability
                reward = lmbda * (Swind[s] - Swind[a])**2 + (1-lmbda) * (Swind[s] - Swind[z])**2
            
            if reward < min_reward:
                min_reward = reward
                best_action = a
        
        policy[s, z, b, p] = best_action
    
    return policy

def run_comparative_study():
    # Fixed parameters
    Swind = np.linspace(0, 1, 21)
    B = 10
    eta = 2
    Delta = 3
    tau = 4
    gamma = 1/15
    beta = 0.95
    theta = 0.01
    Kmin = 10
    lmbda = 0.7
    retention_prob = 0.9
    
    # Vary mean solar power (mu_delta)
    mu_delta_values = np.linspace(1, Delta-0.5, 5)
    stddev_wind_values = np.linspace(0.05, 0.2, 5)
    
    # Results storage
    results_solar = []
    results_wind = []
    
    # Study 1: Vary mean solar power (fixed wind stddev = 0.1)
    mu_wind = 0.3
    stddev_wind = 0.1
    P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)
    
    for mu_delta in mu_delta_values:
        # Generate solar distribution
        z_delta = 0.5
        stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
        alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)
        
        # Get optimal policy
        V_opt, policy_opt = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
        
        # Get greedy policy
        policy_greedy = get_greedy_policy(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        
        # Evaluate both policies
        perf_opt = evaluate_policy(policy_opt, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        perf_greedy = evaluate_policy(policy_greedy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        
        results_solar.append((mu_delta, perf_opt, perf_greedy))
    
    # Study 2: Vary wind speed stddev (fixed mean solar power = 2)
    mu_delta = 2
    z_delta = 0.5
    stddev_delta = z_delta * np.sqrt(Delta * (Delta - mu_delta))
    alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)
    
    for stddev_wind in stddev_wind_values:
        # Generate wind transition matrix
        P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)
        
        # Get optimal policy
        V_opt, policy_opt = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
        
        # Get greedy policy
        policy_greedy = get_greedy_policy(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        
        # Evaluate both policies
        perf_opt = evaluate_policy(policy_opt, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        perf_greedy = evaluate_policy(policy_greedy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta)
        
        results_wind.append((stddev_wind, perf_opt, perf_greedy))
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Solar power variation
    plt.subplot(1, 2, 1)
    mu_deltas, opt_solar, greedy_solar = zip(*results_solar)
    plt.plot(mu_deltas, opt_solar, 'b-', label='Optimal Policy')
    plt.plot(mu_deltas, greedy_solar, 'r--', label='Greedy Policy')
    plt.xlabel('Mean Solar Power (μ_δ)')
    plt.ylabel('Average Discounted Cost')
    plt.title('Performance vs Mean Solar Power')
    plt.legend()
    plt.grid(True)
    
    # Wind stddev variation
    plt.subplot(1, 2, 2)
    stddevs, opt_wind, greedy_wind = zip(*results_wind)
    plt.plot(stddevs, opt_wind, 'b-', label='Optimal Policy')
    plt.plot(stddevs, greedy_wind, 'r--', label='Greedy Policy')
    plt.xlabel('Wind Speed Standard Deviation')
    plt.ylabel('Average Discounted Cost')
    plt.title('Performance vs Wind Speed Variability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_solar, results_wind

# Run the comparative study
results_solar, results_wind = run_comparative_study()