import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
import numpy as np
# For Sparse Tansformer
from math import sqrt
from utils.masking import TriangularCausalMask
from entmax import sparsemax, entmax15, entmax_bisect, entmax_bisect


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        # TODO: Check pos embedding from AST
        if configs.pos_emb:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        else:
            # CHANGE: Switching to Embedding without positional encoding (from Autoformer)
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        # Encoder
        # CHANGE: Added Sparse Attention from entmax library
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        SparseAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention,
                                        attention_type=configs.attention_type),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        # CHANGE: Added Sparse Attention from entmax library
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        SparseAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False,
                                        attention_type=configs.attention_type),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        SparseAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False,
                                        attention_type=configs.attention_type),
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

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = dec_out[:, -self.pred_len:, :]

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


class SparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 attention_type=None):
        super(SparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_type = attention_type
        self.alpha = None

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if self.attention_type == 'softmax':
            p_attn = F.softmax(scale * scores, dim=-1)
        elif self.attention_type == 'sparsemax':
            p_attn = sparsemax(scale * scores, dim=-1)
        elif self.attention_type == 'entmax15':
            p_attn = entmax15(scale * scores, dim=-1)
        elif self.attention_type == 'entmax':
            p_attn = entmax_bisect(scale * scores, self.alpha, n_iter=25)
        else:
            raise Exception

        A = self.dropout(p_attn)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# NOTE: Not using the generator for now (not using q90)
class Generator(nn.Module):
    def __init__(self, d_model, c_out):
        super(Generator, self).__init__()
        self.q50 = nn.Linear(d_model, c_out)
        self.q90 = nn.Linear(d_model, c_out)

    def forward(self, x):
        q50 = self.q50(x)
        q90 = self.q90(x)
        return q50, q90


def quantile_loss(mu, labels, quantile=torch.tensor(0.5)):
    I = (labels >= mu).float()
    diff = torch.sum(quantile * ((labels - mu) * I) + (1 - quantile) * (mu - labels) * (1 - I))
    # Note: TFT Paper reports normalization with number if instances, TFT code uses no normalization.
    q_loss = diff  # / torch.numel(labels)
    return q_loss


def normalized_quantile_loss(mu, labels, quantile=torch.tensor(0.5)):
    I = (labels >= mu).float()
    diff = 2 * torch.sum(quantile * ((labels - mu) * I) + (1 - quantile) * (mu - labels) * (1 - I))
    denom = torch.sum(torch.abs(labels))
    q_loss = diff / denom
    return q_loss
