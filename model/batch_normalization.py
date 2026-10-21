import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        # x is a matrix in R^NxD with N number of examples in the batch and D number of features per example
        # gamma is the "scale"
        # beta is the "shift"
        # eps is a small constant to avoid division by zero 
        x_numpy = np.array(x)
        mu_B = np.mean(x_numpy, axis = 0)
        var_B = np.var(x_numpy, axis = 0)
        gamma_numpy, beta_numpy = np.array(gamma), np.array(beta)
        running_mean_numpy, running_var_numpy = np.array(running_mean), np.array(running_var)
        if training: 
            Y = beta_numpy + gamma_numpy * (x_numpy - mu_B) / (np.sqrt(var_B + eps))
            running_mean_numpy = (1-momentum) * running_mean_numpy + momentum * mu_B
            running_var_numpy = (1-momentum) * running_var_numpy + momentum * var_B
        else:
            Y = beta_numpy + gamma_numpy * (x_numpy - running_mean_numpy) / (np.sqrt(running_var_numpy + eps))

        Y, running_mean_numpy, running_var_numpy = np.round(Y, 4), np.round(running_mean_numpy, 4), np.round(running_var_numpy, 4)
        Y, running_mean_numpy, running_var_numpy = Y.tolist(), running_mean_numpy.tolist(), running_var_numpy.tolist()
        return (Y, running_mean_numpy, running_var_numpy)

    
