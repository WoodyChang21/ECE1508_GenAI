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
    price_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """price_scale: optional (4,) per-component [open_ret, body_ret, upper_wick,
    lower_wick] training-set std, broadcast over price's last dim. `log_volume_norm` is
    already z-scored to unit variance (see data_pipeline.apply_normalize) but the raw
    price components are not, so without this, volume's ~1e5x larger raw variance
    dominates price_loss regardless of w_price/w_vol -- see cvae_direction_collapse.md.
    Default None reproduces the original unscaled behavior exactly (PatchTST's call
    site never passes this)."""
    true_price, true_volume = unpack_y(true_y)
    if price_scale is not None:
        pred_price = pred_price / price_scale
        true_price = true_price / price_scale
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
    price_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon_loss, parts = weighted_mse_loss(pred_price, pred_volume, true_y, w_price, w_vol, price_scale)

    kl = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    kl = torch.clamp(kl, min=free_bits)  # free-bits floor: guards against posterior collapse
    kl_loss = kl.sum(dim=-1).mean()

    total = recon_loss + beta * kl_loss
    parts = dict(parts)
    parts["recon_loss"] = recon_loss.item()  # unweighted by beta -- see cyclical annealing note in train_cvae.py
    parts["kl_loss"] = kl_loss.item()
    parts["beta"] = beta
    return total, parts
