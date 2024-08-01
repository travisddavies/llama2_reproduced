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
    # Note: the formula for polar is magnitude * cos(phase) + magnitude * sin(phase) * j
    # This means that it will include both the sine and the cos in this freqs
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the
    target tensor 'x' for the purpose of broadcasting the frequency tensor
    during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected
        number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1
             for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key
    'xk' tensors using the provided frequency tensor 'freqs_cis'. The input
    tensors are reshaped as complex numbers, and the frequency tensor is
    reshaped for broadcasting compatibility. The resulting tensors contain
    rotary embeddings and are returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
        exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and
        key tensor with rotary embeddings.
    """
    # Split the embedding into 2 on the last dim to consider the 2 sin
    # operations in RoPE. Cos has already been considered in the embedding with
    # the polar function used initially
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Reshape the frequencies array for matching dimensions
    # Essentially we should have a shape of (1, seq_len, 1, 1, 2)
    # Now this can be broadcasted along the query and key embeddings
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # Perform the pointwise multiplications and flatten the embeddings back
    # to their normal shape
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # This converts x to the following dimensions:
        # (bs, slen, n_kv_heads, 1, head_dim)
        x[:, :, :, None, :]
        # New dimensions:
        # (bs, slen, n_kv_heads, n_rep, head_dim)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # New dimensions:
        # (bs, slen, n_kv_heads * n_rep, head_dim)
        # We therefore concatenated together the 3rd and 4th column
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
        """
        super().__init__()
        self.n_kv_heads = args.n_heads \
            if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # This will create a tensor that is parallelised across the columns of
        # the tensor. The dimensions will be (args.dim, args.n_heads * self.head_dim)
        # and determined by how many GPUs are available for partitioning the
        # tensor, each GPU will process only x number of columns

        # These are the weights for the query embeddings
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_head=lambda x: x,
        )
        # These are the weights for the key embeddings
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # These are the weights for the value embeddings
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # Maybe for the final linear layer once the heads are concatenated
        # together again? [MAKE SURE TO CHECK ON THIS]
        self.wo = RowParallelLinear(
            args.n_head * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # bsz is batch_size, seqlen is the length of the input tokens
        bsz, seqlen, _ = x.shape
        # This is passing each model through the parallelised linear layers
        # for the key, query and value embeddings
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # We now break the inputs into sub-embeddings for multi-headed
        # attention.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # We apply positonal encoding to the query and key embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # This is now storing the key and value embeddings, with padding at
        # the end of the sequence
        self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv

        keys = self.cache_k[:bsz, :start_pos+seqlen]
        values = self.cache_v[:bsz, :start_pos+seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # Create our grouped queries
        keys = repeat_kv(keys, self.n_rep)
        # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)
        # (bs, n_local_heads, seqlen, head_dim)

        # We do this so that now a query sub-embedding can query every
        # sub-embedding in in the keys embeddings (remember that the dot product
        # will go across the next dimension too, idea the words in the sequence)
        xq = xq.transpose(1, 2)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        keys = keys.transpose(1, 2)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)
        # Here we transpose keys in this way since the last two dims need to
        # be transposed - simple linear algebra rule for matrix multiplicaitons
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (bs, n_local_heads, seqlen, cache_len + seqlen)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (bs, n_local_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float]
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=True,
            init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))
