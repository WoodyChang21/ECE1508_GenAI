"""Shared weighted-MSE loss used identically by PatchTST and the CVAE decoder, plus an
optional NLL reconstruction loss (CVAE-only -- PatchTST has no predicted variance to train
against, see src/models/patchtst.py)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.data_pipeline import per_bar_close_return


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
    w_direction: float = 0.0,
    direction_temperature: float = 0.003,
) -> tuple[torch.Tensor, dict[str, float]]:
    """price_scale: optional (4,) per-component [open_ret, body_ret, upper_wick,
    lower_wick] training-set std, broadcast over price's last dim. `log_volume_norm` is
    already z-scored to unit variance (see data_pipeline.apply_normalize) but the raw
    price components are not, so without this, volume's ~1e5x larger raw variance
    dominates price_loss regardless of w_price/w_vol -- see cvae_direction_collapse.md.
    Default None reproduces the original unscaled behavior exactly (PatchTST's call
    site never passes this).

    w_direction/direction_temperature: optional auxiliary loss (default 0.0 = off, so
    PatchTST's call site -- which never passes these -- is unaffected). Plain MSE on
    body_ret turned out not to be enough gradient signal to stop CVAE's direction
    prediction from collapsing to a near-constant value (see cvae_direction_collapse.md's
    "architectural fix" re-check: collapsed to a *different* constant, not a fixed
    direction sense) -- so this adds a much more direct classification-style signal on
    top of it: binary cross-entropy on whether `per_bar_close_return` (open_ret+body_ret,
    the same close-anchored return PatchTST's own quality gate checks the sign of, and
    exactly what exit_price_from_components' take-profit target is built from) has the
    right sign, per horizon bar. direction_temperature rescales the raw return into a
    BCE logit -- calibrated to roughly real body_ret's own std (~0.003) so the loss is
    steep in the region real moves actually live in, not saturated flat near zero."""
    true_price, true_volume = unpack_y(true_y)

    direction_loss = pred_price.new_tensor(0.0)
    if w_direction > 0:
        pred_close_ret = per_bar_close_return(pred_price)  # (B, 3), real log-return scale
        true_close_ret = per_bar_close_return(true_price)
        target_up = (true_close_ret > 0).float()
        direction_loss = F.binary_cross_entropy_with_logits(
            pred_close_ret / direction_temperature, target_up
        )

    if price_scale is not None:
        pred_price = pred_price / price_scale
        true_price = true_price / price_scale
    price_loss = F.mse_loss(pred_price, true_price)
    vol_loss = F.mse_loss(pred_volume, true_volume)
    total = w_price * price_loss + w_vol * vol_loss + w_direction * direction_loss
    return total, {
        "price_loss": price_loss.item(),
        "vol_loss": vol_loss.item(),
        "direction_loss": direction_loss.item(),
    }


def laplace_nll(true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """-log p(true) for true ~ Laplace(mu, b), parametrized by logvar so Var = 2*b^2
    matches logvar's usual meaning (b = exp(0.5*(logvar - log(2)))). Used for open_ret/
    body_ret: short-horizon returns are heavier-tailed and roughly symmetric, a better fit
    than Gaussian (see cvae_direction_collapse.md's "generative pivot" discussion)."""
    b = torch.exp(0.5 * (logvar - math.log(2.0)))
    return (true - mu).abs() / b + torch.log(2 * b)


def gaussian_nll(true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """-log p(true) for true ~ Normal(mu, exp(0.5*logvar)). Used for wicks (pragmatic
    choice -- a rigorous fit would need a logit-space change-of-variables to handle the
    non-negative, frequently-exactly-zero support correctly; revisit only if calibration
    checks show this is actually a problem) and volume (already log+z-scored close to
    Gaussian-shaped by data_pipeline.apply_normalize, no need for anything fancier)."""
    return 0.5 * ((true - mu) ** 2 * torch.exp(-logvar) + logvar + math.log(2 * math.pi))


def weighted_nll_loss(
    pred_price: torch.Tensor,
    pred_price_logvar: torch.Tensor,
    pred_volume: torch.Tensor,
    pred_vol_logvar: torch.Tensor,
    true_y: torch.Tensor,
    w_price: float = 1.0,
    w_vol: float = 0.5,
    w_direction: float = 0.0,
    direction_temperature: float = 0.003,
) -> tuple[torch.Tensor, dict[str, float]]:
    """NLL counterpart to weighted_mse_loss -- same signature shape, same returned dict
    keys (so train_cvae.py's logging needs no special-casing), but the reconstruction
    terms are proper log-likelihoods under a per-component predicted variance instead of
    plain MSE. No price_scale here: a learned per-example variance is an adaptive version
    of what price_scale's single fixed train-set constant approximates (see
    weighted_mse_loss's own docstring) -- keeping both active would double-correct, so
    cvae_loss disables price_scale whenever reconstruction="nll" (logged explicitly, same
    as use_price_scale=false already is).

    open_ret/body_ret (price[...,:2]) use laplace_nll; wicks (price[...,2:]) and volume
    use gaussian_nll -- see those functions' docstrings for why."""
    true_price, true_volume = unpack_y(true_y)

    direction_loss = pred_price.new_tensor(0.0)
    if w_direction > 0:
        pred_close_ret = per_bar_close_return(pred_price)  # (B, 3), real log-return scale
        true_close_ret = per_bar_close_return(true_price)
        target_up = (true_close_ret > 0).float()
        direction_loss = F.binary_cross_entropy_with_logits(
            pred_close_ret / direction_temperature, target_up
        )

    open_body_nll = laplace_nll(true_price[..., :2], pred_price[..., :2], pred_price_logvar[..., :2])
    wick_nll = gaussian_nll(true_price[..., 2:], pred_price[..., 2:], pred_price_logvar[..., 2:])
    price_loss = torch.cat([open_body_nll, wick_nll], dim=-1).mean()
    vol_loss = gaussian_nll(true_volume, pred_volume, pred_vol_logvar).mean()

    total = w_price * price_loss + w_vol * vol_loss + w_direction * direction_loss
    return total, {
        "price_loss": price_loss.item(),
        "vol_loss": vol_loss.item(),
        "direction_loss": direction_loss.item(),
    }


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
    w_direction: float = 0.0,
    direction_temperature: float = 0.003,
    reconstruction: str = "mse",
    pred_price_logvar: torch.Tensor | None = None,
    pred_vol_logvar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """reconstruction: "mse" (default, exactly today's behavior -- every existing config
    is unaffected) or "nll" (see weighted_nll_loss). "nll" requires pred_price_logvar/
    pred_vol_logvar (CVAEInpainting.decode()'s new outputs) and ignores price_scale
    (see weighted_nll_loss's docstring for why)."""
    if reconstruction == "mse":
        recon_loss, parts = weighted_mse_loss(
            pred_price, pred_volume, true_y, w_price, w_vol, price_scale, w_direction, direction_temperature
        )
    elif reconstruction == "nll":
        if pred_price_logvar is None or pred_vol_logvar is None:
            raise ValueError("reconstruction='nll' requires pred_price_logvar and pred_vol_logvar")
        recon_loss, parts = weighted_nll_loss(
            pred_price, pred_price_logvar, pred_volume, pred_vol_logvar, true_y,
            w_price, w_vol, w_direction, direction_temperature,
        )
    else:
        raise ValueError(f"unknown reconstruction mode: {reconstruction!r}")

    kl = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    kl = torch.clamp(kl, min=free_bits)  # free-bits floor: guards against posterior collapse
    kl_loss = kl.sum(dim=-1).mean()

    total = recon_loss + beta * kl_loss
    parts = dict(parts)
    parts["recon_loss"] = recon_loss.item()  # unweighted by beta -- see cyclical annealing note in train_cvae.py
    parts["kl_loss"] = kl_loss.item()
    parts["beta"] = beta
    return total, parts
