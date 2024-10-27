import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

################## INTIAL DATA ################################
X = np.array([[1, 0], [0, 2], [3, -1]])
Mu = np.array([[2, -1], [1, 1]])
Sigma = np.array([[[4, 1], [1, 4]], [[2, 0], [0, 2]]])
Pi = np.array([0.5, 0.5])
###############################################################

def round_if(a, places: int | None):
    return np.round(a, places) if places is not None else a

def em(X: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray, Pi: np.ndarray, iterations: int, round: int = None) -> None:
    '''
    Runs the Expectation-Maximization Algorithm on multivariate data of any size, number of observations and clusters \\
    `X`     - Training data \\
    `Mu`    - Array of multivariate means for each cluster \\
    `Sigma` - Array of covariance matrices for each cluster \\
    `Pi`    - Mixture proportions of each cluster \\
    `iterations` - Number of iterations to run. Should be >= 1 \\
    `round`      - Specifies the number of decimal places to round each intermediary step. Defaults as None (No rounding) \\
    '''

    # First Initialization
    Mu_cur = Mu
    Sigma_cur = Sigma
    Pi_cur = Pi
    Normal_cur = np.array([multivariate_normal(mu, sigma, allow_singular=True) for mu, sigma in zip(Mu_cur, Sigma_cur)])

    print(f'> Initial Conditions:')
    print(f'Mu: {Mu_cur}')
    print('----------')
    print(f'Sigma: {Sigma_cur}')
    print('----------')
    print(f'Pi: {Pi_cur}')
    print('\n')

    if (iterations <= 0):
        raise ValueError('Please run at least 1 iteration >:(')
    
    for i in range(iterations):
        # E-step
        P = np.array([normal.pdf(X) * pi for normal, pi in zip(Normal_cur, Pi_cur)])
        Gamma = round_if(P / np.sum(P.T, axis=1), round)

        # M-step
        N = round_if(np.sum(Gamma, axis=1), round)
        Mu_cur = round_if(np.dot(Gamma, X) / N[:, np.newaxis], round)

        X_minus_Mu = round_if(X - Mu_cur[:, np.newaxis, :], round)
        Sigma_cur = round_if(np.matmul(X_minus_Mu.transpose(0, 2, 1), Gamma[:, :, np.newaxis] * X_minus_Mu) / N[:, np.newaxis, np.newaxis], round)
        Pi_cur = round_if(N / np.sum(N), round)

        # Report Iteration
        print(f'> Iteration {i + 1}:')
        print(f'Mu: {Mu_cur}')
        print('----------')
        print(f'Sigma: {Sigma_cur}')
        print('----------')
        print(f'Pi: {Pi_cur}')
        print('----------')
        print(f'Gamma: {Gamma}')
        print('----------')
        print(f'N: {N}')
        print('\n')

        # Initialize multivariate normal dist for each cluster for next iteration
        Normal_cur = np.array([multivariate_normal(mu, sigma, allow_singular=True) for mu, sigma in zip(Mu_cur, Sigma_cur)])

# Runs 2 epochs of the EM algrotithm (round 3 decimal places)
em(X, Mu, Sigma, Pi, 2)