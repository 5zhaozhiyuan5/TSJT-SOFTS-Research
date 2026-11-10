import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 288
        self.pred_len = 36
        self.output_attention = False #configs.output_attention
        self.use_norm = False

        self.e_layers = 3 # 

        self.d_model = 512
        self.d_ff = 512
        self.factor = 1
        self.n_heads = 8
        self.activation = 'gelu'
        #self.output_attention = False
        self.dropout = 0.0
        self.embed = 'timeF'
        self.freq = 'h'# help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h'
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        self.class_strategy = 'projection' #configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        #self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

        self.projection1 = nn.Linear(self.d_model, 128, bias=True)
        self.projection2 = nn.Linear(128, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Normalization from Non-stationary Transformer
        if self.use_norm: # not used
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out = self.projection1(enc_out)
        dec_out = self.projection2(enc_out) # [B, seq_len, N] 

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        #print('enc_out.shape, dec_out.shape', enc_out.shape, dec_out.shape) 
        # enc_out.shape, dec_out.shape torch.Size([16, 207, 128]) torch.Size([16, 207, 36])
        return enc_out, dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        enc_out, dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return enc_out, dec_out #dec_out[:, -self.pred_len:, :]  # [B, L, D]
