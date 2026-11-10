import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Embed import DataEmbedding_inverted
from .layers.Transformer_EncDec import Encoder, EncoderLayer

class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, None


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 288 #configs.seq_len
        self.pred_len = 36 #configs.pred_len
        self.d_model = 512
        self.d_core = 512
        self.dropout = 0.0
        self.d_ff = 512 #self.d_model * 4
        self.e_layers = 3 
        
        # Embedding
        #self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, 512, self.dropout)
        self.use_norm = False 
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(self.d_model, self.d_core),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation='gelu',
                ) for l in range(self.e_layers)
                #'''
                #EncoderLayer(
                #    STAR(configs.d_model, configs.d_core),
                #    configs.d_model,
                #    configs.d_ff,
                #    dropout=configs.dropout,
                #    activation=configs.activation,
                #) for l in range(configs.e_layers)
                #'''
            ],
        )

        # Decoder
        self.projection1 = nn.Linear(self.d_model, 128, bias=True)
        self.projection2 = nn.Linear(128, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Normalization from Non-stationary Transformer
        if self.use_norm:  # not used
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
