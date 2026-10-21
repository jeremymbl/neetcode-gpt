import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_np = np.array(x)
        mean_sq = np.mean(x_np**2)
        RMS_x = np.sqrt(mean_sq + eps)
        x_np = x_np / RMS_x
        y = np.round(gamma * x_np, 4)
        return y.tolist()

