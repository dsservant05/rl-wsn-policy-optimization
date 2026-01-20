import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def entropy_func(theta):
    return np.sum(theta*np.log(np.clip(theta+1e-12,1e-12, 1)))
    
def entropy_func_gradient(theta):
    return np.log(np.clip(theta+1e-12, 1e-12, 1))+1

def prob_vector_generator(support, mu, stddev):
    
    N = len(support)
    
    bounds = Bounds(np.zeros(N), np.ones(N))
    
    A = np.zeros((3,N))
    A[0,:] = 1
    A[1,:] = support
    A[2,:] = support**2
    
    b_lb = np.array([1,mu,mu**2+stddev**2])
    b_ub = np.array([1,mu,mu**2+stddev**2])
    
    linear_constraint = LinearConstraint(A, b_lb, b_ub)
    
    x_start = (1/(N))*np.ones(N)
    result = minimize(entropy_func, x_start, method='SLSQP', jac=entropy_func_gradient, constraints=linear_constraint, bounds=bounds)
    
    theta = np.clip(result.x, 0, 1)
    theta = theta/np.sum(theta)
    
    return theta


def markov_ob_func(theta, p_stnry, N, var):
    return np.sum((np.diag(np.reshape(theta, (N,N)))-var)**2)
    # return 0


def markov_ob_func_gradient(theta, p_stnry, N, var):  
    return np.reshape(np.diag(2*(np.diag(np.reshape(theta, (N,N)))-var)), -1)


def markov_matrix_generator(support, mu, stddev, var):
    
    N = len(support)
    
    p_stnry = prob_vector_generator(support, mu, stddev)
    
    bounds = Bounds(np.zeros(N**2), np.ones(N**2))
    
    A = np.zeros((2*N, N**2))
    b_lb = np.zeros(2*N)
    b_ub = np.zeros(2*N)
    
    ix = np.arange(0,N**2, N)
    for i in range(N):
        A[i,i*N:(i+1)*N] = 1
        b_lb[i] = 1
        b_ub[i] = 1
        
        A[i+N,ix+i] = p_stnry
        b_lb[i+N] = p_stnry[i]
        b_ub[i+N] = p_stnry[i]
    
    
    linear_constraint = LinearConstraint(A, b_lb, b_ub)
    
    # x_start = (1/(N))*np.ones(N**2)
    x_start = np.array(list(p_stnry)*N)
    result = minimize(markov_ob_func, x_start, args=(p_stnry,N,var), method='SLSQP', jac=markov_ob_func_gradient, constraints=linear_constraint, bounds=bounds)
    
    theta = np.clip(result.x, 0, 1)
    theta = np.reshape(theta, (N,N))
    for i, val in enumerate(theta):
        theta[i,:] = val/np.sum(val)
        
    return theta