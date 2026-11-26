import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.W_K = nn.Linear(
            in_features = embedding_dim,
            out_features = attention_dim,
            bias=False)
        self.W_Q = nn.Linear(
            in_features = embedding_dim,
            out_features = attention_dim,
            bias=False)
        self.W_V = nn.Linear(
            in_features = embedding_dim,
            out_features = attention_dim,
            bias=False)
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.W_K(embedded)
        Q = self.W_Q(embedded)
        V = self.W_V(embedded)
        scores = (Q @ K.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        context_dim = embedded.shape[1]
        mask = torch.tril(torch.ones(context_dim, context_dim))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = torch.nn.functional.softmax(scores, dim = 2)
        scores = scores @ V
        return torch.round(input = scores, decimals = 4)

