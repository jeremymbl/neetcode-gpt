import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:

        K = self.W_K(embedded)
        Q = self.W_Q(embedded)
        V = self.W_V(embedded)

        scores = (Q @ K.transpose(-2, -1)) / self.attention_dim**0.5

        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T, device=embedded.device))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        return torch.round(output, decimals=4)