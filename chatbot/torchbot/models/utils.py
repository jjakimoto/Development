import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


def clones(layer, N):
    """Produce identical N layers as ModuleList"""
    return nn.ModuleList([deepcopy(layer) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # qurey.size == (n_batch, input_length, key_dim)
    # key.size == (n_batch, memory_length, key_dim)
    # value.size == (n_batch, memory_length, value_dim)
    q_k = torch.matmul(query, key.transpose(-1, -2))
    key_dim = query.size()[-1]
    scores = q_k / np.sqrt(key_dim)
    if mask is not None:
        # .masked_fill fills indices whose values are 1 with given value
        scores = scores.masked_fill(mask, -1e9)
    # attention.size == (n_batch, input_length, memory_length)
    attention = F.softmax(scores)
    if dropout is not None:
        attention = dropout(attention)
    # output.size == (n_batch, input_length, value_dim)
    output = torch.matmul(attention, value)
    return output, attention


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask)
