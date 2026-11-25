import numpy as np
from numpy.typing import NDArray


class Solution:

    CLIP_MIN, CLIP_MAX = 1e-7, 1-1e-7

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred_clipped = y_pred.clip(self.CLIP_MIN, self.CLIP_MAX)
        L = y_true*np.log(y_pred_clipped) + (1-y_true)*np.log(1-y_pred_clipped)
        L = -L.mean()
        return np.round(L, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred_clipped = y_pred.clip(self.CLIP_MIN, self.CLIP_MAX)
        L = (y_true*np.log(y_pred_clipped)).sum(axis=1)
        L = -L.mean(axis=0)
        return np.round(L, 4)
