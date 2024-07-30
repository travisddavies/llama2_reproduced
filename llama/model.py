import math
from dataclass import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear
)
from torch import nn


@dataclass
class ModelArgs:
    # the dims for the embeddings
    dim: int = 4096
    # how many stacked transformers
    n_layers: int = 32
    # the number of key and value heads
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    # the epsilon value for RMS to ensure it's not zero
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # return the norm along the -1 axis
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        # multiply the calculated norm with the learnable weights
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # We first want to get the indices of the frequency values
    # to be computed. In this case, it will be even numbers, 0 -> dim-2
    freqs = torch.arange(0, dim, 2)
    # We then want to slice it to ensure that it has the correct number of
    # elements for the given dimension
    freqs = freqs[: (dim // 2)]
    freqs = freqs.float() / dim
    freqs = 1.0 / (theta ** (freqs))
    # Our timesteps
    t = torch.arange(end, device=freqs.device)
    # We then do an outer dot product to get our actual frequencies for
    # every time step. Should be shape(len(t), len(freqs))
    freqs = torch.outer(t, freqs).float()
    # We then get the complex numbers of the frequencies.
    # Magnitude is 1 (torch.ones_like) and phase is freqs
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
