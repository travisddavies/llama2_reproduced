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
### Input and Output
### RoPE
### RMSNorm
### SwiGLU
### Transformer
### GQA
### Training

