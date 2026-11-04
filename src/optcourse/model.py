"""A small, byte-level causal Transformer, without external model downloads."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [item.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for item in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.projection(y))


class Block(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width, bias=False), nn.GELU(),
                                 nn.Linear(4 * width, width, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, heads = config["n_embd"], config["n_head"]
        if width % heads:
            raise ValueError("Embedding width must be divisible by attention heads")
        self.block_size = config["block_size"]
        self.token_embedding = nn.Embedding(256, width)
        self.position_embedding = nn.Embedding(self.block_size, width)
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.Sequential(*[Block(width, heads, config["dropout"])
                                     for _ in range(config["n_layer"])])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 256, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.block_size or tokens.shape[1] < 1:
            raise ValueError("Input must have shape (batch, length) with 1 <= length <= block_size")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        return self.head(self.final_norm(self.blocks(x)))
