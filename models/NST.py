import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


class Model(nn.Module):
    """
    Non-stationary Transformers
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # De-stationary Factors
        self.factors = DestationaryFactors(configs.seq_len, configs.enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DestationaryAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DestationaryAttention(True, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False, apply_delta=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        DestationaryAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Normalization
        x_enc_stationarized, mu_x, sigma_x = normalization(x_enc)
        tau, delta = self.factors(x_enc, mu_x, sigma_x)

        # Encoder with De-stationary Attention
        enc_out = self.enc_embedding(x_enc_stationarized, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        # Decoder with De-stationary Attention
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-Normalization
        dec_out = denormalization(dec_out, mu_x, sigma_x)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


def normalization(x):
    # See Algorithm 1: Series Stationarization - Normalization.
    mu_x = torch.mean(x, dim=1, keepdim=True)
    sigma_x = torch.std(x, dim=1, keepdim=True) + 1e-5  # Adding a small value to avoid division by zero
    x = (1 / sigma_x) * (x - mu_x)

    return x, mu_x, sigma_x


def denormalization(y, mu_x, sigma_x):
    # See Algorithm 2: Series Stationarization - Denormalization.
    y = sigma_x * (y + mu_x)

    return y


class DestationaryFactors(nn.Module):
    # See Algorithm 4Non-stationary Transformers - Overall Architecture. Line 2 and 3
    def __init__(self, input_len, n_var, d_hidden=128):
        super(DestationaryFactors, self).__init__()
        # Input: x [S, C], mu [C], sigma [C]
        # S: Input length
        # C: number of variables
        input = n_var * (1 + input_len)
        self.tau_mlp = nn.Sequential(nn.Linear(input, d_hidden),
                                     nn.Linear(d_hidden, 1))
        # tau [B, 1] from R+
        self.delta_mlp = nn.Sequential(nn.Linear(input, d_hidden),
                                       nn.Linear(d_hidden, input_len))
        # delta [B, S]

    def forward(self, x, mu, sigma):
        tau_input = torch.cat([sigma, x], dim=1).reshape(x.size(0), -1)
        tau = self.tau_mlp(tau_input)
        # tau = self.tau_mlp(sigma.reshape(x.size(0), -1)) # better results without x as input but different to paper

        # TODO Scaling after mlp
        # Paper does not state exp, pseudo code does. Paper asumes a positive tau, so some kind of scaling is needed.
        # tau = torch.exp(tau)  # -> tau has inf values
        # without scaling -> bad performance
        # tau = tau ** 2 # -> bad performance
        # tau = torch.exp(tau / (x.size(-1) * x.size(-2))) # -> works but same performance as without factors
        tau = torch.exp(tau / sqrt(x.size(-1) * x.size(-2)))  # -> works but same performance as without factors

        # manually computation -> same performance as without factors
        # tau = sigma.mean(dim=-1).reshape(x.size(0), -1) ** 2


        delta_input = torch.cat([mu, x], dim=1).reshape(x.size(0), -1)
        # delta_input = mu.reshape(x.size(0), -1) # better result without x as input but different to paper
        delta = self.delta_mlp(delta_input)

        return tau, delta


class DestationaryAttention(nn.Module):
    # See Algorithm 3: De-stationary Attention.
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 apply_delta=True):
        super(DestationaryAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.apply_delta = apply_delta

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # De-stationary factors
        scores = tau.unsqueeze(1).unsqueeze(1) * scores
        if self.apply_delta:
            # delta is not applied in the self-attention layer of the decoder
            scores = scores + delta.unsqueeze(1).unsqueeze(1)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
