import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float
    ) -> float:
        # z = x.w + b
        z = x @ w + b

        # y_hat = sigmoid(z)
        y_hat = self.sigmoid(z)

        return float(y_hat)

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float
    ) -> Tuple[NDArray[np.float64], float]:

        # Forward pass
        z = x @ w + b
        y_hat = self.sigmoid(z)

        # Loss:
        # L = 0.5 * (y_hat - y_true)^2

        # dL/dy_hat
        dL_dy_hat = y_hat - y_true

        # dy_hat/dz
        dy_hat_dz = y_hat * (1 - y_hat)

        # dL/dz
        dL_dz = dL_dy_hat * dy_hat_dz

        # dz/dw = x
        dL_dw = dL_dz * x

        # dz/db = 1
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)