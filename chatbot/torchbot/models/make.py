import torch.nn as nn
from copy import deepcopy

from .layers import MultiHeadedAttention, PositionwiseFeedForward,\
    PositionalEncoding, Embeddings
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .generator import Generator
from .model import EncoderDecoder


def make_model(src_vocab, tgt_vocab, N=6, d_model=512,
               d_ff=2048, h=8, drop_rate=0.1):
    attn = MultiHeadedAttention(h, d_model, drop_rate)
    ff = PositionwiseFeedForward(d_model, d_ff, drop_rate)
    pe = PositionalEncoding(d_model, drop_rate)
    c = deepcopy
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, drop_rate, c(attn), c(ff)), N),
        Decoder(DecoderLayer(d_model, drop_rate, c(attn), c(attn), c(ff)), N),
        nn.Sequential(Embeddings(src_vocab, d_model), c(pe)),
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(pe)),
        Generator(d_model, tgt_vocab)
    )

    # Xavier initializer
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal(p)
    return model