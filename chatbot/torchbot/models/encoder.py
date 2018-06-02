import torch
import torch.nn as nn

from .utils import clones
from .layers import LayerNorm, SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, size, drop_rate, self_attn, feed_forward):
        super().__init__()
        self.sublayers = clones(SublayerConnection(size, drop_rate), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x_: self.self_attn(x_, x_, x_, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return self.norm(x)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
