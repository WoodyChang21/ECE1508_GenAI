"""Conditional VAE inpainting model.

Two mechanisms make this behave like inpainting rather than forecasting-with-a-VAE-
label: (1) it consumes the full 73-slot padded tensor with padding_mask/target_mask
channels and horizon values zeroed, exactly like a masked image region, and (2) it
splits into a recognition network (sees the true, unmasked horizon -- training only)
and a context-conditioned prior network (sees only the masked input -- training and
inference). Only the prior + decoder run at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data_pipeline import MAX_LOG_RETURN, N_CHANNELS


class ConvEncoder(nn.Module):
    """1D-conv stack over the time axis -> pooled fixed-size representation. Doesn't
    need TOTAL_LEN (73) to divide evenly by anything, unlike the patch-based model."""

    def __init__(self, in_channels: int = N_CHANNELS, hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T=73, C=9) -> (B, out_dim)."""
        x = x.transpose(1, 2)  # (B, C, T) for Conv1d
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class CVAEInpainting(nn.Module):
    def __init__(
        self,
        hidden: int = 32,
        ctx_dim: int = 64,
        z_dim: int = 16,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.context_encoder = ConvEncoder(hidden=hidden, out_dim=ctx_dim)
        self.recognition_encoder = ConvEncoder(hidden=hidden, out_dim=ctx_dim)

        self.prior_head = nn.Linear(ctx_dim, 2 * z_dim)
        self.recognition_head = nn.Linear(ctx_dim, 2 * z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + ctx_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 15),
        )

    def encode_prior(self, masked_tensor: torch.Tensor):
        ctx_repr = self.context_encoder(masked_tensor)
        mu_p, logvar_p = self.prior_head(ctx_repr).chunk(2, dim=-1)
        return mu_p, logvar_p, ctx_repr

    def encode_recognition(self, full_tensor: torch.Tensor):
        rec_repr = self.recognition_encoder(full_tensor)
        mu_q, logvar_q = self.recognition_head(rec_repr).chunk(2, dim=-1)
        return mu_q, logvar_q

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, ctx_repr: torch.Tensor):
        raw = self.decoder(torch.cat([z, ctx_repr], dim=-1))
        price_raw = raw[:, :12].reshape(-1, 3, 4)
        # Bounded to +-MAX_LOG_RETURN (open_ret/body_ret) or [0, MAX_LOG_RETURN] (wicks)
        # so an undertrained/unstable head can't blow up into unrealistic price moves.
        open_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 0])
        body_ret = MAX_LOG_RETURN * torch.tanh(price_raw[..., 1])
        upper_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 2])
        lower_wick = MAX_LOG_RETURN * torch.sigmoid(price_raw[..., 3])
        price = torch.stack([open_ret, body_ret, upper_wick, lower_wick], dim=-1)
        volume = raw[:, 12:15]
        return price, volume

    def forward(self, masked_tensor: torch.Tensor, full_tensor: torch.Tensor):
        """Training forward. z ~ q(z | full window) via the recognition network;
        decoder conditions on the *masked* context encoding, never the recognition
        encoding directly -- only z carries information from the true horizon."""
        mu_p, logvar_p, ctx_repr = self.encode_prior(masked_tensor)
        mu_q, logvar_q = self.encode_recognition(full_tensor)
        z = self.reparameterize(mu_q, logvar_q)
        price, volume = self.decode(z, ctx_repr)
        return price, volume, mu_p, logvar_p, mu_q, logvar_q

    @torch.no_grad()
    def sample(self, masked_tensor: torch.Tensor, k: int = 5):
        """Inference: only the prior network + decoder run. Returns price (K,B,3,4),
        volume (K,B,3)."""
        mu_p, logvar_p, ctx_repr = self.encode_prior(masked_tensor)
        prices, volumes = [], []
        for _ in range(k):
            z = self.reparameterize(mu_p, logvar_p)
            price, volume = self.decode(z, ctx_repr)
            prices.append(price)
            volumes.append(volume)
        return torch.stack(prices), torch.stack(volumes)
