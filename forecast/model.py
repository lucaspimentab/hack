#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DemandTransformer(nn.Module):
    def __init__(self, in_dim=3, d_model=64, nhead=4, num_layers=3, dropout=0.1,
                 num_pdv=1, num_prod=1, emb_dim=32):
        super().__init__()
        self.in_linear = nn.Linear(in_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4, dropout=dropout,
                                                   activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.emb_pdv = nn.Embedding(num_pdv, emb_dim)
        self.emb_prod = nn.Embedding(num_prod, emb_dim)
        self.ctx_proj = nn.Linear(d_model + emb_dim*2, d_model)
        self.head_act = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Sigmoid()
        )
        self.head_size = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    def forward(self, x, pdv_id, prod_id):
        h = self.in_linear(x); h = self.posenc(h); h = self.encoder(h)
        h_last = h[:, -1, :]
        emb = torch.cat([self.emb_pdv(pdv_id), self.emb_prod(prod_id)], dim=-1)
        h_c = self.ctx_proj(torch.cat([h_last, emb], dim=-1))
        p_active = self.head_act(h_c).squeeze(-1)
        y_log = self.head_size(h_c).squeeze(-1)
        return p_active, y_log
