import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        decoded = self.decode(memory, tgt, src_mask, tgt_mask)
        return self.generator(decoded)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, memory, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        return self.decoder(memory, tgt, src_mask, tgt_mask)
