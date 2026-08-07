"""HF PatchTSTModel-backed replacement for the hand-rolled PatchTST in patchtst.py.

patchtst.py patches the input and self-attends across patches, but does early channel
fusion -- all 7 feature channels are flattened into one linear patch embedding before any
attention happens. It has no channel-independence and no way to toggle channel-mixing.

This model uses the real HF PatchTSTModel backbone instead, which implements
channel-independent patching (each channel patched/embedded separately) with an optional
channel_attention sublayer to let channels cross-inform each other -- see
docs/experiments.md for the comparison. A deterministic, bounded head reproduces
patchtst.py's forward() output exactly (same 15-dim layout, same tanh/sigmoid x
MAX_LOG_RETURN bounding), so weighted_mse_loss/unpack_y from losses.py work unmodified.

Deliberate simplification vs. patchtst.py: fixed context_length only (see
train_patchtst.py), not the variable 2-10 day curriculum -- HF PatchTSTModel patchifies
one fixed context_length per model.

Also defines PatchTSTHFPriceOnly, a variant of PatchTSTHF with volume dropped entirely
(see its own docstring below) -- the checkpoint produced by
train_patchtst_hf_channel_attention_{True,False}.ipynb's current version.

Also defines RevIN and PatchTSTHFRevIN, a variant that predicts raw OHLC price directly
instead of the anchored log-return decomposition every other model in this file uses --
the checkpoint produced by train_patchtst_revin_channel_attention_{True,False}.ipynb (see
its own docstring below).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from src.data_pipeline import HORIZON, MAX_LOG_RETURN, N_FEATURE_CHANNELS

# Channels dropped for the price-only variant below: data_pipeline.py's FEATURE_COLS is
# [open_ret, body_ret, upper_wick, lower_wick, log_volume_norm, time_gap_norm,
# day_bar_index_norm] -- index 4 is volume. Selecting everything except it turns a
# 7-channel context slice into the 6-channel one PatchTSTHFPriceOnly expects.
PRICE_ONLY_CHANNEL_IDX = [0, 1, 2, 3, 5, 6]
N_FORECAST_CHANNELS = 4   # open_ret, body_ret, upper_wick, lower_wick (no volume)


class PatchTSTHF(nn.Module):
    def __init__(
        self,
        context_length: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        channel_attention: bool = False,
        patch_length: int = 7,
        patch_stride: int = 7,
    ):
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=N_FEATURE_CHANNELS,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            head_dropout=head_dropout,
            channel_attention=channel_attention,
            # patchtst.py has no per-window instance normalization (its open_ret/body_ret/
            # wicks are raw log-returns, log_volume_norm is already globally normalized) --
            # disabled here so the comparison is about the encoder architecture, not
            # confounded by an extra normalization layer only one side has.
            scaling=None,
        )
        self.backbone = PatchTSTModel(config)
        # Only channels 0-4 (open_ret, body_ret, upper_wick, lower_wick, log_volume_norm)
        # are ever forecast -- channels 5/6 (time_gap_norm, day_bar_index_norm) are
        # context-only inputs, matching data_pipeline.py's PRICE_VOL_IDX restriction.
        self.head = nn.Linear(5 * d_model, 15)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, past_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """past_values: (B, context_length, 7). Returns (price (B,3,4), volume (B,3)),
        same output contract as patchtst.py's PatchTST.forward()."""
        base_out = self.backbone(past_values=past_values)
        pooled = base_out.last_hidden_state.mean(dim=2)       # mean-pool across patches: (B, 7, d_model)
        pooled = self.dropout(pooled[:, :5, :])                 # only the 5 forecastable channels
        raw = self.head(pooled.reshape(pooled.shape[0], -1))    # (B, 15)

        price_raw = raw[:, :12].reshape(-1, 3, 4)
        # Exact same bounding as patchtst.py's forward() -- keeps this model's output
        # space identical to the target space weighted_mse_loss/unpack_y expect.
        open_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 0])
        body_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 1])
        upper_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 2])
        lower_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)

        volume = raw[:, 12:15]
        return price, volume


class PatchTSTHFPriceOnly(nn.Module):
    """Price-only variant of PatchTSTHF: volume dropped entirely, from both the context
    input (6 channels instead of 7, see PRICE_ONLY_CHANNEL_IDX) and the output (12-dim,
    4 price components x HORIZON, instead of 15) -- see docs/experiments.md. Volume's
    target (log_volume_norm) is z-scored to ~unit variance while price targets are raw
    log-returns (~1e-4 variance), so a combined price+volume loss was overwhelmingly
    dominated by the volume term; volume predictions were also never used downstream
    (not in the backtest, not in confidence scoring). Same fused/flattened head as
    PatchTSTHF (not the channel-independent head tried and reverted -- also
    docs/experiments.md)."""

    def __init__(
        self,
        context_length: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        channel_attention: bool = False,
        patch_length: int = 7,
        patch_stride: int = 7,
    ):
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=len(PRICE_ONLY_CHANNEL_IDX),   # 6
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            head_dropout=head_dropout,
            channel_attention=channel_attention,
            scaling=None,   # see PatchTSTHF -- steven's model has no per-window normalization
        )
        self.backbone = PatchTSTModel(config)
        self.head = nn.Linear(N_FORECAST_CHANNELS * d_model, N_FORECAST_CHANNELS * HORIZON)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        """past_values: (B, context_length, 6) -- volume already excluded by the caller
        (see PRICE_ONLY_CHANNEL_IDX). Returns price (B, HORIZON, 4) only -- no volume
        output, unlike PatchTSTHF."""
        base_out = self.backbone(past_values=past_values)
        pooled = base_out.last_hidden_state.mean(dim=2)                 # (B, 6, d_model)
        pooled = self.dropout(pooled[:, :N_FORECAST_CHANNELS, :])         # only the 4 price channels
        raw = self.head(pooled.reshape(pooled.shape[0], -1))              # (B, 12)

        price_raw = raw.reshape(-1, HORIZON, N_FORECAST_CHANNELS)
        open_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 0])
        body_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 1])
        upper_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 2])
        lower_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)
        return price


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al. 2021), with the learnable
    affine_weight/affine_bias correction HF PatchTSTModel's built-in scaler
    (PatchTSTStdScaler) does not have (verified against the transformers source -- zero
    matches for 'affine' anywhere in modeling_patchtst.py). fit() must be called once per
    batch with the raw context window before normalize()/denormalize() -- both use
    whatever mean/std/affine params were last fit."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        self.mean = None
        self.std = None

    def fit(self, context: torch.Tensor):
        """context: (bs, T, C). Stores per-window, per-channel mean/std (detached --
        these are window statistics, not something to backprop through)."""
        self.mean = context.mean(dim=1, keepdim=True).detach()
        self.std = (context.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return x * self.affine_weight + self.affine_bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return x * self.std + self.mean


class PatchTSTHFRevIN(nn.Module):
    """Predicts raw OHLC price directly (4 channels: open/high/low/close), normalized
    with a real RevIN (see above) computed from each window's own context statistics --
    not the anchored log-return decomposition every other model in this file uses.

    Motivated by a decisive negative result: PatchTSTHFPriceOnly collapsed to predicting
    values so close to zero that 0 of 2397 test windows ever had all 3 predicted horizon
    bars agree on direction (confirmed via evaluate.py's backtest: zero trades at every
    confidence threshold). Root cause: open_ret/body_ret/wick targets have near-zero mean
    and tiny variance for a near-random-walk price series, so "predict ~0" is a cheap,
    loss-minimizing shortcut that destroys the directional signal. RevIN's zero-point is
    the context window's own trailing mean *price*, not "no future change" -- price can
    genuinely drift from its own trailing average over 3 bars, so a lazy collapse isn't as
    cheap an escape hatch here. This does not solve the underlying unpredictability, it
    just removes this specific shortcut -- see docs/experiments.md.

    Same fused/flattened head design as PatchTSTHF/PatchTSTHFPriceOnly. No tanh/sigmoid
    output bounding (that existed specifically for log-returns' natural cap; doesn't
    apply to RevIN-normalized price)."""

    def __init__(
        self,
        context_length: int,
        d_model: int = 64,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        channel_attention: bool = False,
        patch_length: int = 7,
        patch_stride: int = 7,
    ):
        super().__init__()
        self.revin = RevIN(num_channels=N_FORECAST_CHANNELS)   # 4: open, high, low, close
        config = PatchTSTConfig(
            num_input_channels=N_FORECAST_CHANNELS,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            head_dropout=head_dropout,
            channel_attention=channel_attention,
            scaling=None,   # RevIN is applied externally instead -- see RevIN docstring
        )
        self.backbone = PatchTSTModel(config)
        self.head = nn.Linear(N_FORECAST_CHANNELS * d_model, N_FORECAST_CHANNELS * HORIZON)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, past_values_raw: torch.Tensor) -> torch.Tensor:
        """past_values_raw: (B, context_length, 4) raw OHLC. Returns predicted future
        OHLC in RevIN-*normalized* space (B, HORIZON, 4) -- caller denormalizes via
        self.revin.denormalize() to get back to real price."""
        self.revin.fit(past_values_raw)
        past_values_norm = self.revin.normalize(past_values_raw)

        base_out = self.backbone(past_values=past_values_norm)
        pooled = base_out.last_hidden_state.mean(dim=2)
        pooled = self.dropout(pooled)
        raw = self.head(pooled.reshape(pooled.shape[0], -1))

        pred_norm = raw.reshape(-1, HORIZON, N_FORECAST_CHANNELS)
        return pred_norm
