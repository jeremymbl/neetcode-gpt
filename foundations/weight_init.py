
import torch
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        sigma = math.sqrt(2 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * sigma

        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        sigma = math.sqrt(2 / fan_in)
        weights = torch.randn(fan_out, fan_in) * sigma

        return torch.round(weights, decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        weights = []

        # Generate all weight matrices first
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                sigma = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == "kaiming":
                sigma = math.sqrt(2 / fan_in)
            else:
                sigma = 1.0

            W = torch.randn(fan_out, fan_in) * sigma
            weights.append(W)

        # Generate random input
        x = torch.randn(input_dim)

        # Forward pass through all layers
        res = []

        for W in weights:
            x = torch.relu(x @ W.T)
            res.append(round(torch.std(x).item(), 2))

        return res
