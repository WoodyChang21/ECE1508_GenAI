"""HF PatchTSTModel-backed RevIN raw-price forecaster (no volume).

This is `hf_patchtst_revin_no_volume`'s architecture, extracted from
`steven/train_patchtst_hf_channel_attention.ipynb` (commit 7638b6e) into an importable
module so `evaluate.py` can reconstruct the exact same model from a saved checkpoint --
previously this whole notebook family kept its models notebook-local, which works for
training but not for a shared eval/backtest script that needs the class definition too.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from src.data_pipeline import HORIZON, MAX_CONTEXT

OHLC_COLS = ["open", "high", "low", "close"]
N_CHANNELS = len(OHLC_COLS)
CLOSE_IDX = OHLC_COLS.index("close")


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al. 2021) -- learnable per-channel
    affine_weight/affine_bias, not HF PatchTSTModel's built-in scaler (mean/std only, no
    affine). fit() must be called once per batch with the raw context window before
    normalize()/denormalize() -- both use whatever mean/std were last fit."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, context: torch.Tensor) -> None:
        self.mean = context.mean(dim=1, keepdim=True).detach()
        self.std = (context.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return x * self.affine_weight + self.affine_bias

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
        return x * self.std + self.mean


class PatchTSTOHLCVRevIN(nn.Module):
    """RevIN + HF PatchTSTModel backbone + fused/flattened head, raw OHLC target (no
    volume). Same design as every other notebook in this family's fused head -- all
    N_CHANNELS pooled representations concatenated before one joint linear projection.
    No tanh/sigmoid output bounding (RevIN-normalized values don't have log-returns'
    physical cap)."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.revin = RevIN(num_channels=N_CHANNELS)
        self.backbone = PatchTSTModel(config)
        self.head = nn.Linear(N_CHANNELS * config.d_model, N_CHANNELS * HORIZON)
        self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, past_values_raw: torch.Tensor) -> torch.Tensor:
        """past_values_raw: (bs, MAX_CONTEXT, 4) raw OHLC. Returns predicted future OHLC
        in RevIN-*normalized* space (bs, HORIZON, 4) -- caller denormalizes via
        self.revin.denormalize() to get back to real price."""
        self.revin.fit(past_values_raw)
        past_values_norm = self.revin.normalize(past_values_raw)

        base_out = self.backbone(past_values=past_values_norm)
        pooled = base_out.last_hidden_state.mean(dim=2)  # mean-pool across patches: (bs, 4, d_model)
        pooled = self.dropout(pooled)
        raw = self.head(pooled.reshape(pooled.shape[0], -1))  # (bs, N_CHANNELS*HORIZON)

        return raw.reshape(-1, HORIZON, N_CHANNELS)


def build_model(
    channel_attention: bool,
    d_model: int = 64,
    num_attention_heads: int = 4,
    num_hidden_layers: int = 3,
    dropout: float = 0.1,
    head_dropout: float = 0.0,
    patch_length: int = 7,
    patch_stride: int = 7,
    context_length: int = MAX_CONTEXT,
) -> PatchTSTOHLCVRevIN:
    config = PatchTSTConfig(
        num_input_channels=N_CHANNELS,
        context_length=context_length,
        prediction_length=HORIZON,
        patch_length=patch_length,
        patch_stride=patch_stride,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        # transformers no longer has a single unified "dropout" field on PatchTSTConfig
        # (split into these four sub-layer dropouts, see modeling_patchtst.py) -- broadcasting
        # the same rate to all four keeps this equivalent to a single nn.Dropout(p=dropout).
        attention_dropout=dropout,
        positional_dropout=dropout,
        path_dropout=dropout,
        ff_dropout=dropout,
        head_dropout=head_dropout,
        channel_attention=channel_attention,
        scaling=None,  # RevIN is applied externally instead -- see RevIN docstring
    )
    return PatchTSTOHLCVRevIN(config)
