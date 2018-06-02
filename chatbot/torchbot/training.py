import torch
import torch.nn as nn
from torch.autograd import Variable
import time

from .models.utils import subsequent_mask


class Batch(object):
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # Mask shape has to be (n_batch, tgt_length, memory_length)
        self.src_mask = (src == pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        length = tgt.size(-1)
        tgt_mask = (tgt == pad).unsqueeze(-1)
        seq_mask = Variable(subsequent_mask(length).type_as(tgt_mask.data))
        tgt_mask = tgt_mask | seq_mask
        return tgt_mask


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, tgt):
        # x.size == (n_batch * length, vocab)
        # tgt.size == (n_batch * length)
        assert x.size(-1) == self.size
        true_dist = x.clone()
        # Subtract 2 because of padding and EOS index
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, tgt.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        # index equivalent to padding_idx, shape=(num, 1)
        mask = torch.nonzero(tgt == self.padding_idx)
        if mask.dim() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute(object):
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        norm = float(norm)
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm


def run_epoch(data_iter, model, loss_compute, log_freq=50):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss.item()
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()
        if i % log_freq == 0:
            elapsed = time.time() - start
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            print(i, loss, tokens / elapsed)
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
