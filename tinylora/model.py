# model.py
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding, RMSNorm
import tinygrad.nn as nn
import math

class Attention:
    def __init__(self, dim, n_heads):
        self.c_attn = Linear(dim, 3*dim)
        self.c_proj = Linear(dim, dim)
        self.n_heads = n_heads
        self.dim = dim

    def __call__(self, x:Tensor) -> Tensor:
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(x.shape[0], x.shape[1], self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(x.shape[0], x.shape[1], self.n_heads, self.dim // self.n_heads).permute(0, 2, 1, 3)
        
        # NOTE: GPT-2 uses standard attention, Pythia/Llama use RoPE.
        attn = q.scaled_dot_product_attention(k, v, is_causal=True)
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim)
        return self.c_proj(attn)

class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim)
        self.c_proj = Linear(hidden_dim, dim)

    def __call__(self, x:Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())

class Block:
    def __init__(self, dim, n_heads):
        self.ln_1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln_2 = RMSNorm(dim)
        self.mlp = FeedForward(dim, 4*dim)

    def __call__(self, x:Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT:
    def __init__(self, dim=128, n_heads=4, n_layers=6, vocab_size=10000):
        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(2048, dim) # Max context size
        self.h = [Block(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = RMSNorm(dim) # Final norm
        self.lm_head = Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens:Tensor) -> Tensor:
        b, t = tokens.shape
        pos = Tensor.arange(t).reshape(1, t)
        
        x = self.wte(tokens) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)