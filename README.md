# Llama 2 Reproduced
## Introduction
This is a deconstruction and reproduction of the LLama 2 code. The code provided
is a copy of the original [source code](https://github.com/meta-llama/llama), but I just wrote it out by hand again
to have a stronger understanding of the code. The explanations below are largely
based off [this article](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up),
however I have tried to add extra detail and code snippets for better understanding
of the implementations.

## Table of Contents
1. [Prompting](#Prompting)
2. [Tokenisation](#Tokenisation)
3. [Chunking and Caching](#Chunking-and-Caching)
4. [Input and Output](#Input-and-Output)
5. [RoPE](#RoPE)
6. [RMSNorm](#RMSNorm)
7. [SwiGLU](#SwiGLU)
8. [Transfomer](#Transformer)
9. [GQA](#GQA)
10. [Training](#Training)

## Prompting
## Tokenisation
The tokeniser for this model is [SentencePiece](https://github.com/google/sentencepiece),
a common tokeniser that is used for transformers. I won't go into too much detail
of how this tokeniser works (largely because I haven't put in _too much_ time
into understanding it yet), but a good explanation for how the model works can
be found [here](https://colabdoge.medium.com/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08).
Deconstructing this tokeniser will likely be another project!
Meta has provided a pretrained model for the tokeniser for Llama.

## Chunking and Caching
The LLM works with a combination of chunking and caching. Basically what that
means is that we will, for each prompt, only feed in a chunk of the tokens
from the prompt. This is to enable the model to remember the chunks that came
before it. For LLama, the chunking is set to the length of tokens in a list of
prompts, and the starting position of each chunk is sent as input with the
tokens to the Llama model. This starting point is used as an index for where
to store a chunk in a cache. In LLama, only the key and value tokens are
cached.

An example of this is shown below, at line 205 in llama/generation.py:
```python
logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
```

Here we are sending in a chunk of tokens, along with the starting point for
where these tokens should be cached.

An example of how the caching works is shown below, at line 337 in
llama/model.py:

```python
self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv
```

Here we are essentially caching the key and value embeddings of the tokens
in a dedicated tensor, keeping in mind what position in the prompt they are
located.

So why do we do this? It is so we can include the cache in the attention
mechanism, providing further context of what came before the current chunk.
This is shown below, with a code chunk from llama/model.py:
```python
# The key and value embeddings from the beginning of the prompt
keys = self.cache_k[:bsz, :start_pos+seqlen]
values = self.cache_v[:bsz, :start_pos+seqlen]
# Repeating keys to match the number of query keys (explained in GQA)
keys = repeat_kv(keys, self.n_rep)
values = repeat_kv(values, self.n_rep)
# The linearly transformed query embeddings
xq = xq.transpose(1, 2)
keys = keys.transpose(1, 2)
values = values.transpose(1, 2)
# Attention matrix multiplication scores
scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
```
## Input and Output
## RoPE
## RMSNorm
The normalisation formula employed for this transformer stack is Root Mean
Square Normalisation (RMSNorm), this is generally just supposed to help
with the stability of the model when feeding in data. The formula is shown below:

<p align="center">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbaeeee5e-8cba-426f-9fa4-8666a7e76155_1080x138.png"
  alt="RMSNorm"
  width="600">
</p>

Below shows the code block for how RMSNorm is used in the model. It is important
to note that an epsilon and weights are also included in the calculation,
largely to help with numerical stability and to learn a scaling parameter (as
mentioned in the comments of the code).

```python
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
```

## SwiGLU
SwiGLU is the activation function used on the feed-forward layer of each
transformer block. The formula for SwiGLU is as follows:

<p align="center">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9aec7e3e-45bc-42f1-9e0c-0248bee25263_982x244.png"
  alt="Equation"
  width="300">
</p>

How does this look in code? Like the following code block:
```python
def forward(self, x):
    # This is the SwiGLU activation function
    # Two linear transformations with pointwise multiplication
    # Then linear transformation from w2 multiplied by the activation
    # function of Swish (F.silu in this case)
    return self.w2(F.silu(self.w1(x) * self.w3(x)))
```

## Transformer
For the transformer architecture, it's just a general stacked architecture of
transformer blocks, all decoder architecture. If you want a detailed explanation
of how a typical transformer architecture works, you can look at my other
repo explaining it [here](https://github.com/travisddavies/the_annotated_transformer). However, one thing that is
different in this architecture is that inputs are normalised before being
fed into each transformer block. As mentioned above, the normalisation
used for Llama is RMSNorm. So in essence, data is first normalised before
going into the attention layer, and then normalised again before being fed
into the last feed-forward layer.

<p align="center">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3b6fab4d-7825-47ec-a59e-979d9dfa0384_1256x938.png" alt="Equation">
</p>


The code block below shows how this works
within the transformer block.
```python
def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    """
    Perform a forward pass through the TransformerBlock.

    Args:
        x (torch.Tensor): Input tensor.
        start_pos (int): Starting position for attention caching.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after applying attention and feedforward layers.

    """
    # We've included an additional norm at the beginning of each attention
    # layer, something unique to this architecture
    h = x + self.attention(
        self.attention_norm(x), start_pos, freqs_cis, mask
    )
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
```
## GQA
Grouped-Query Attention (GQA) is a more efficient way of applying the self-
attention mechanism. The way it works is that we can maintain the same level
of performance as vanilla multi-head attention, but with more memory efficiency
reducing the number of heads in the values and keys heads.


<p align="center">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a7dc1e2-e66c-4a30-a0a7-518ae7e3a566_1536x596.png" alt="GQA">
</p>



Where does it save the memory? In the caching of previous chunks, not in the
actual attention matrix multiplication itself, since the dimensions will need
to be returned back to their original state before the matrix multiplication.
This concept took me a while to understand, so just remember - it is making
the amount of memory required for caching much more efficient!

So how do we achieve this? We first reduce the dimensionality of the key and
value heads with the initial linear transformation section, as shown in the
code block below:
```python
# Linear transformations
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

# We now break the inputs into sub-embeddings for multi-headed
# attention.
xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
```

The output dimensins of the linear transformations are different for the
query embeddings to the key and value embeddings, which is shown below:
```python
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
```
The query embeddings will have an output dim of `self.n_heads * self.head_dim`
whereas the value and key embeddings will have an output dim of
`self.n_kv_heads * self.head_dim`, where `self.n_kv_heads` will be a reduced
number.

Since the number of heads will be different to the query heads, before we
perform the matrix multiplication for the attention scores, we need to repeat
each head `k` times to match the dimensionality of the key heads. This is
performed with the following code:
```python
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
```
What this essentially ends up with is the a group of `k` query heads will perform
matrix multiplication on the same key and value heads, since each key and value
head will be repeated `k` times.
## Training

