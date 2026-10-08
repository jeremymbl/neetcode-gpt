import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eps = 10**(-7)
        y_pred_clipped = np.clip(y_pred, eps, 1-eps)
        entropy = y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped)
        res = -np.mean(entropy)
        return np.round(res, 4)




    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        eps = 10**(-7)
        y_pred_clipped = np.clip(y_pred, eps, 1-eps)
        entropy = np.sum(y_true*np.log(y_pred_clipped), axis=1)
        res = -np.mean(entropy)
        return np.round(res, 4)
