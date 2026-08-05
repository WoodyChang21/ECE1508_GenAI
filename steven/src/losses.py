"""Shared weighted-MSE loss used identically by PatchTST and the CVAE decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unpack_y(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y: (B, 15) -> price components (B, 3, 4) [open_ret, body_ret, upper_wick,
    lower_wick] per horizon bar, and volume (B, 3)."""
    price = y[:, :12].reshape(-1, 3, 4)
    volume = y[:, 12:15]
    return price, volume


def weighted_mse_loss(
    pred_price: torch.Tensor,
    pred_volume: torch.Tensor,
    true_y: torch.Tensor,
    w_price: float = 1.0,
    w_vol: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    true_price, true_volume = unpack_y(true_y)
    price_loss = F.mse_loss(pred_price, true_price)
    vol_loss = F.mse_loss(pred_volume, true_volume)
    total = w_price * price_loss + w_vol * vol_loss
    return total, {"price_loss": price_loss.item(), "vol_loss": vol_loss.item()}


def kl_diag_gaussians(
    mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
) -> torch.Tensor:
    """KL(N(mu_q,var_q) || N(mu_p,var_p)), per-dimension -- NOT KL to a standard normal,
    since the prior here is context-conditioned rather than N(0,I). Shape (B, z_dim)."""
    return 0.5 * (logvar_p - logvar_q) + (
        torch.exp(logvar_q) + (mu_q - mu_p) ** 2
    ) / (2 * torch.exp(logvar_p)) - 0.5


def cvae_loss(
    pred_price: torch.Tensor,
    pred_volume: torch.Tensor,
    true_y: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    w_price: float,
    w_vol: float,
    beta: float,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss, parts = weighted_mse_loss(pred_price, pred_volume, true_y, w_price, w_vol)

    kl = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    kl = torch.clamp(kl, min=free_bits)  # free-bits floor: guards against posterior collapse
    kl_loss = kl.sum(dim=-1).mean()

    total = recon_loss + beta * kl_loss
    parts = dict(parts)
    parts["kl_loss"] = kl_loss.item()
    parts["beta"] = beta
    return total, parts
