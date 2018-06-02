import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
