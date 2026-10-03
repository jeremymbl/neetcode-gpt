import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        for i in range(len(z)):
            z[i] = 1/(1+np.exp(-z[i]))
            z[i] = round(z[i], 5)
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        for i in range(len(z)):
            z[i] = max(0, z[i])
        return z
