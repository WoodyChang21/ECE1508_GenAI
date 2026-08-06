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
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from src.data_pipeline import MAX_LOG_RETURN, N_FEATURE_CHANNELS


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
