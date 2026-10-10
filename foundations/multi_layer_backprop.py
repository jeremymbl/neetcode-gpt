
import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert inputs to NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        a1 = W1 @ x + b1
        z = np.maximum(0, a1)
        y_hat = W2 @ z + b2

        # Loss (MSE)
        loss = np.mean((y_hat - y_true) ** 2)

        # Backward pass
        n = len(y_hat)

        # Output layer gradients
        delta2 = (2 / n) * (y_hat - y_true)

        dW2 = np.outer(delta2, z)
        db2 = delta2

        # Hidden layer gradients
        delta1 = (W2.T @ delta2) * (a1 > 0)

        dW1 = np.outer(delta1, x)
        db1 = delta1

        # Return results
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
