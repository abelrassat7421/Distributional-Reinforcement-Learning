import numpy as np
import matplotlib.pyplot as plt

# Environment
n_states = 5
reward_bounds = (-1, 1)
#timestep = 0.01 

# Hyperparameters
lr = 0.1  # Learning rate
num_gamma = 50 # Number of gammas
num_sensitivities = 10 # Number of sensitivities
alpha = 0.1  # Regularization parameter - study its effects TODO 
num_it = 10

# Initialize the state value function V(s) with zeros for a simple environment with 5 states
V = np.zeros((num_sensitivities, num_gamma, n_states))

# Going to have a buffer system -> check how it's implemented for the expectile code and how I can reuse this system? TODO
# Example transitions (state, reward, next_state)
# Assuming a small environment with 5 states and some transitions for illustration
transitions = [(0, -2, 1), (0, 2, 1), (1, -2, 2), (1, 2, 2), (3, -1, 4), (3, 1, 4), (3, -1, 4), (4, 1, "done"), (4, -1, "done")] # (current state, reward, next state) # how  can I make an absorbing state?

# Following theoretical results we want gammas to be equidistant in the 1/(log(gamma)) space
start = 1 / np.log(0.01)  
end = 1 / np.log(0.99)  
transformed_gammas = np.linspace(start, end, num_gamma)
gammas = np.exp(1 / transformed_gammas)
# assume equidistant reward sensitivities (akin to categorical distribution)
sensitivities = np.linspace(reward_bounds[0], reward_bounds[1], num_sensitivities)

# TD Learning with different discount factors and sensitivities (eq. 10)
for h in sensitivities:
    for gamma_idx, gamma in enumerate(gammas):
        for _ in range(num_it):
            for current_state, reward, next_state in transitions:
                td_target = tf.math.sigmoid(reward - h) + gamma * V[h][gamma_idx][next_state] # could change to Heaviside step function NOTE
                td_error = td_target - V[h][gamma_idx][current_state]
                V[h][gamma_idx][current_state] += lr * td_error
                # how do we check convergence? TODO
                # how do we ensure that every state action pair is visited? -> you espilon greedy policy? TODO
                print(f"Updated values: {V}")

# Final Value Function
print("Final Value Function:", V, V.shape)
#plt.plot( )


# Translation-invariant Linear Decoder based Post's approximation to the inverse Laplace transform (inspiration from ref. 35)
# This Linear decoder is more noise-resilient than the one in ref. 35 (used for NMR data in ref. 36) 
def SVD_discrete_linear_decoder(y_h_vect, gammas, sensitivities, time_horizon):
    """
    SVD-based Discrete Linear Decoder
    :param V_h_gamma: Value function for a specific sensitivity and discount factor
    :param sensitivities: Sensitivities
    :return: Expectile
    """
    assert y_h_vect.shape[0] == len(sensitivities)
    assert y_h_vect.shape[1] == len(gammas)

    # iterate over all differences of neighboring sensitivities 
    p_h_diff_vect = np.zeros((len(sensitivities)-1, len(time_horizon+1)))
    y_h_diff_vect = np.diff(y_h_vect, axis=0)
    
    for i in range(len(sensitivities)-1):
        p_h_diff_vect[i] = linear_decoder_least_square_solution(y_h_diff_vect[i], gammas, time_horizon)
    
    # Normalization step - crucial for long temporal horizons, since regularization causes the overall 
    # magnitude of the recovered τ -space to decrease as τ increases
    normalization_cst = np.sum(p_h_diff_vect, axis=0)
    norm_p_h_diff_vect = p_h_diff_vect / normalization_cst 
    
    return norm_p_h_diff_vect
     

def linear_decoder_least_square_solution(y_h_diff, gammas, time_horizon, alpha):
    # might want to change the temporal resolution to smaller than timestep of the MP TODO
    # (for better regularization by smoothing the τ -space across τ) 
    # for continuous MPs - need to choose discretization of time
    F = np.zeros((len(gammas), len(time_horizon)))
    for i in range(len(gammas)):
        for j in range(0, time_horizon+1):
            F[i][j] = gammas[i] ** j
    
    U, S, V = np.linalg.svd(F, full_matrices=True)
    
    # least squares solution of expression A.5 (Tikhonov-regularized solution)
    p_h_diff = np.sum([S[i]**2 / (S[i]**2 + alpha**2) * np.dot(U[:, i].T, y_h_diff) * V[i] for i in range(len(S))], axis=0)

    assert p_h_diff.shape == (len(time_horizon+1), 1)
    print("TEST1: ", p_h_diff.shape) 

    return p_h_diff


# alternative to SVD_discrete_linear_decoder presented in the supplementary material A.2
def trained_linear_decoder():
    pass

