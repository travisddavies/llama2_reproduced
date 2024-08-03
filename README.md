## Llama 2 Reproduced
This is a just reproduction code to understand how Llama 2 works. May give further explanations for
my findings later
### Prompting
### Tokenisation
### Chunking and Caching
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
### Input and Output
### RoPE
### RMSNorm
### SwiGLU
### Transformer
For the transformer architecture, it's just a general stacked architecture of
transformer blocks, all decoder architecture. However, one thing that is
different in this architecture is that inputs are normalised before being
fed into each transformer block. As mentioned above, the normalisation
used for Llama is RMSNorm. So in essence, data is first normalised before
going into the attention layer, and then normalised again before being fed
into the last feed-forward layer.

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
### GQA
### Training

