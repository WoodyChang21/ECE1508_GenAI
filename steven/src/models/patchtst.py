"""Minimal PatchTST benchmark: patch embedding -> transformer encoder -> flatten-and-
project head. Context-only input (no horizon, no mask channels) -- this model never
receives the masked region in any form, matching a plain forecasting benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_pipeline import MAX_CONTEXT, N_FEATURE_CHANNELS, N_PATCHES, PATCH_LEN


class PatchTST(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        patch_dim = PATCH_LEN * N_FEATURE_CHANNELS  # 7 bars x 7 channels = 49
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path for padding masks
        # isn't implemented on MPS (aten::_nested_tensor_from_mask_left_aligned).
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(N_PATCHES * d_model, 15)

    def forward(
        self, context: torch.Tensor, patch_key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """context: (B, 70, 7). patch_key_padding_mask: (B, 10) bool, True = ignore
        (fully-padded patch). Returns (price_components (B,3,4), volume (B,3))."""
        B = context.shape[0]
        assert context.shape[1] == MAX_CONTEXT, context.shape

        patches = context.reshape(B, N_PATCHES, PATCH_LEN * N_FEATURE_CHANNELS)
        x = self.patch_embed(patches) + self.pos_embed
        x = self.encoder(x, src_key_padding_mask=patch_key_padding_mask)
        x = x.reshape(B, -1)
        raw = self.head(x)

        price_raw = raw[:, :12].reshape(B, 3, 4)
        open_ret = price_raw[..., 0]
        body_ret = price_raw[..., 1]
        upper_wick = F.softplus(price_raw[..., 2])
        lower_wick = F.softplus(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)

        volume = raw[:, 12:15]
        return price, volume
