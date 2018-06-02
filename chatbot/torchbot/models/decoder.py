import torch
import torch.nn as nn

from .utils import clones
from .layers import LayerNorm, SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, drop_rate, attn, self_attn, feed_forward):
        super().__init__()
        self.sublayers = clones(SublayerConnection(size, drop_rate), 3)
        self.attn = attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.size = size

    def forward(self, memory, tgt, src_mask, tgt_mask):
        x = tgt
        m = memory
        x = self.sublayers[0](x, lambda x_: self.self_attn(x_, x_, x_, tgt_mask))
        x = self.sublayers[1](x, lambda x_: self.attn(x_, m, m, src_mask))
        x = self.sublayers[2](x, self.feed_forward)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, memory, tgt, src_mask, tgt_mask):
        x = tgt
        for layer in self.layers:
            x = layer(memory, x, src_mask, tgt_mask)
        return self.norm(x)