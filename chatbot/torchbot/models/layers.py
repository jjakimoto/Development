import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .utils import clones, attention


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, size, drop_rate):
        super().__init__()
        self.size = size
        self.dropout = nn.Dropout(drop_rate)
        self.norm = LayerNorm(size)

    def forward(self, x, layer):
        next_x = layer(self.norm(x))
        return x + self.dropout(next_x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, drop_rate):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.h_dim = d_model // h
        self.dropout = nn.Dropout(drop_rate)
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.o_linear = nn.Linear(h * self.h_dim, d_model)

    def forward(self, query, key, value, mask):
        inputs = (query, key, value)
        projs = list()
        for x, lin in zip(inputs, self.linears):
            x = lin(x)
            x = torch.cat(torch.chunk(x, self.h, dim=-1), dim=0)
            projs.append(x)
        proj_q, proj_k, proj_v = projs
        mask = torch.cat([mask,] * self.h, dim=0)
        output, self.attn = attention(proj_q, proj_k, proj_v, mask)
        output = torch.cat(torch.chunk(output, self.h, dim=0), dim=-1)
        return self.o_linear(output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_rate=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin2(x)


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.embeds = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embeds(x) * np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_rate, maxlen=50000):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        pe = torch.zeros(maxlen, d_model)
        position = torch.arange(0, maxlen).unsqueeze(1)
        const = 1e4
        div_term = torch.exp((-1) * torch.arange(0, d_model, 2)\
                             * np.log(const) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

