"""Minimal PatchTST benchmark: patch embedding -> transformer encoder -> flatten-and-
project head. Context-only input (no horizon, no mask channels) -- this model never
receives the masked region in any form, matching a plain forecasting benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data_pipeline import HORIZON, MAX_CONTEXT, MAX_LOG_RETURN, N_FEATURE_CHANNELS, N_PATCHES, PATCH_LEN

N_PRICE = HORIZON * 4  # open_ret, body_ret, upper_wick, lower_wick, per horizon bar
N_TARGET = N_PRICE + HORIZON  # + volume, per horizon bar


class PatchTST(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        n_feature_channels: int = N_FEATURE_CHANNELS,
    ):
        super().__init__()
        self.n_feature_channels = n_feature_channels
        patch_dim = PATCH_LEN * n_feature_channels  # 7 bars x n_feature_channels
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
        self.head = nn.Linear(N_PATCHES * d_model, N_TARGET)

    def forward(
        self, context: torch.Tensor, patch_key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """context: (B, 70, n_feature_channels). patch_key_padding_mask: (B, 10) bool,
        True = ignore (fully-padded patch). Returns (price_components (B,HORIZON,4),
        volume (B,HORIZON))."""
        B = context.shape[0]
        assert context.shape[1] == MAX_CONTEXT, context.shape

        # self.n_feature_channels, NOT the module-level N_FEATURE_CHANNELS import above --
        # that's bound once at import time and would go stale if data_pipeline's own
        # N_FEATURE_CHANNELS gets monkey-patched afterward (e.g. by momentum_pipeline.py's
        # channel-layout patching), same staleness risk flagged for N_PATCHES/PATCH_LEN
        # everywhere else this model is discussed.
        patches = context.reshape(B, N_PATCHES, PATCH_LEN * self.n_feature_channels)
        x = self.patch_embed(patches) + self.pos_embed
        x = self.encoder(x, src_key_padding_mask=patch_key_padding_mask)
        x = x.reshape(B, -1)
        raw = self.head(x)

        price_raw = raw[:, :N_PRICE].reshape(B, HORIZON, 4)
        # Bounded to +-MAX_LOG_RETURN (open_ret/body_ret) or [0, MAX_LOG_RETURN] (wicks)
        # so an undertrained/unstable head can't blow up into unrealistic price moves.
        open_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 0])
        body_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 1])
        upper_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 2])
        lower_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)

        volume = raw[:, N_PRICE:N_TARGET]
        return price, volume
