import math

import pytest
import torch
import torch.nn.functional as F

from src.data_pipeline import HORIZON
from src.losses import cvae_loss, gaussian_nll, laplace_nll, weighted_mse_loss, weighted_nll_loss


def _make_y(price: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Inverse of losses.unpack_y: price (B,HORIZON,4), volume (B,HORIZON) -> y (B,HORIZON*5)."""
    return torch.cat([price.reshape(price.shape[0], -1), volume], dim=-1)


def test_direction_loss_off_by_default_matches_old_behavior():
    torch.manual_seed(0)
    pred_price = torch.randn(4, HORIZON, 4) * 0.01
    pred_volume = torch.randn(4, HORIZON)
    true_price = torch.randn(4, HORIZON, 4) * 0.01
    true_volume = torch.randn(4, HORIZON)
    true_y = _make_y(true_price, true_volume)

    total, parts = weighted_mse_loss(pred_price, pred_volume, true_y)
    assert parts["direction_loss"] == 0.0

    expected_total = F.mse_loss(pred_price, true_price) + 0.5 * F.mse_loss(pred_volume, true_volume)
    assert torch.isclose(total, expected_total)


def test_direction_loss_penalizes_wrong_sign_more_than_right_sign():
    """true close_ret (open_ret+body_ret) is +0.01 (up) on every bar; a prediction with
    the same sign should score a lower direction_loss than one with the opposite sign."""
    true_price = torch.zeros(1, HORIZON, 4)
    true_price[..., 1] = 0.01
    true_volume = torch.zeros(1, HORIZON)
    true_y = _make_y(true_price, true_volume)
    pred_volume = torch.zeros(1, HORIZON)

    pred_price_right = torch.zeros(1, HORIZON, 4)
    pred_price_right[..., 1] = 0.01
    pred_price_wrong = torch.zeros(1, HORIZON, 4)
    pred_price_wrong[..., 1] = -0.01

    _, parts_right = weighted_mse_loss(pred_price_right, pred_volume, true_y, w_direction=1.0)
    _, parts_wrong = weighted_mse_loss(pred_price_wrong, pred_volume, true_y, w_direction=1.0)
    assert parts_right["direction_loss"] < parts_wrong["direction_loss"]


def test_direction_loss_disabled_does_not_affect_total_even_with_wrong_sign():
    true_price = torch.zeros(1, HORIZON, 4)
    true_price[..., 1] = 0.01
    true_volume = torch.zeros(1, HORIZON)
    true_y = _make_y(true_price, true_volume)
    pred_volume = torch.zeros(1, HORIZON)
    pred_price_wrong = torch.zeros(1, HORIZON, 4)
    pred_price_wrong[..., 1] = -0.01

    total, parts = weighted_mse_loss(pred_price_wrong, pred_volume, true_y, w_direction=0.0)
    assert parts["direction_loss"] == 0.0
    assert torch.isclose(total, F.mse_loss(pred_price_wrong, true_price))


def test_cvae_loss_forwards_direction_args():
    torch.manual_seed(0)
    batch, z_dim = 2, 4
    pred_price = torch.randn(batch, HORIZON, 4) * 0.01
    pred_volume = torch.randn(batch, HORIZON)
    true_price = torch.randn(batch, HORIZON, 4) * 0.01
    true_volume = torch.randn(batch, HORIZON)
    true_y = _make_y(true_price, true_volume)
    mu_q = torch.zeros(batch, z_dim)
    logvar_q = torch.zeros(batch, z_dim)
    mu_p = torch.zeros(batch, z_dim)
    logvar_p = torch.zeros(batch, z_dim)

    _, parts = cvae_loss(
        pred_price, pred_volume, true_y, mu_q, logvar_q, mu_p, logvar_p,
        w_price=1.0, w_vol=0.5, beta=1.0, w_direction=1.0,
    )
    assert parts["direction_loss"] > 0.0


def test_gaussian_nll_known_value_at_zero_logvar():
    """true == mu, logvar == 0 (std=1) -> NLL reduces to 0.5*log(2*pi), a standard result."""
    true = torch.zeros(3)
    mu = torch.zeros(3)
    logvar = torch.zeros(3)
    expected = 0.5 * math.log(2 * math.pi)
    assert torch.allclose(gaussian_nll(true, mu, logvar), torch.full((3,), expected))


def test_gaussian_nll_penalizes_larger_error_more():
    mu = torch.zeros(2)
    logvar = torch.zeros(2)
    small_err = gaussian_nll(torch.tensor([0.1, 0.1]), mu, logvar)
    large_err = gaussian_nll(torch.tensor([1.0, 1.0]), mu, logvar)
    assert torch.all(large_err > small_err)


def test_laplace_nll_known_value_at_zero_error_logvar_log2():
    """true == mu, logvar == log(2) -> b == 1 -> NLL reduces to log(2), a standard result."""
    true = torch.zeros(3)
    mu = torch.zeros(3)
    logvar = torch.full((3,), math.log(2.0))
    expected = math.log(2.0)
    assert torch.allclose(laplace_nll(true, mu, logvar), torch.full((3,), expected))


def test_laplace_nll_penalizes_larger_error_more():
    mu = torch.zeros(2)
    logvar = torch.zeros(2)
    small_err = laplace_nll(torch.tensor([0.1, 0.1]), mu, logvar)
    large_err = laplace_nll(torch.tensor([1.0, 1.0]), mu, logvar)
    assert torch.all(large_err > small_err)


def test_weighted_nll_loss_matches_hand_computed_value_at_zero():
    """All-zero true/pred/logvar -- price_loss should be the mean of 2 laplace_nll(0,0,0)
    terms (open_ret/body_ret) and 2 gaussian_nll(0,0,0) terms (wicks), per bar; vol_loss
    should be gaussian_nll(0,0,0)."""
    batch = 2
    pred_price = torch.zeros(batch, HORIZON, 4)
    pred_price_logvar = torch.zeros(batch, HORIZON, 4)
    pred_volume = torch.zeros(batch, HORIZON)
    pred_vol_logvar = torch.zeros(batch, HORIZON)
    true_y = _make_y(torch.zeros(batch, HORIZON, 4), torch.zeros(batch, HORIZON))

    laplace_term = laplace_nll(torch.zeros(1), torch.zeros(1), torch.zeros(1)).item()
    gaussian_term = gaussian_nll(torch.zeros(1), torch.zeros(1), torch.zeros(1)).item()
    expected_price_loss = (2 * laplace_term + 2 * gaussian_term) / 4
    expected_vol_loss = gaussian_term

    _, parts = weighted_nll_loss(pred_price, pred_price_logvar, pred_volume, pred_vol_logvar, true_y)
    assert math.isclose(parts["price_loss"], expected_price_loss, rel_tol=1e-5)
    assert math.isclose(parts["vol_loss"], expected_vol_loss, rel_tol=1e-5)
    assert parts["direction_loss"] == 0.0  # w_direction defaults to 0.0


def test_weighted_nll_loss_forwards_direction_args():
    """Same direction-loss behavior as weighted_mse_loss -- wrong sign scores worse."""
    true_price = torch.zeros(1, HORIZON, 4)
    true_price[..., 1] = 0.01
    true_y = _make_y(true_price, torch.zeros(1, HORIZON))
    pred_volume = torch.zeros(1, HORIZON)
    pred_vol_logvar = torch.zeros(1, HORIZON)
    pred_price_logvar = torch.zeros(1, HORIZON, 4)

    pred_price_right = torch.zeros(1, HORIZON, 4)
    pred_price_right[..., 1] = 0.01
    pred_price_wrong = torch.zeros(1, HORIZON, 4)
    pred_price_wrong[..., 1] = -0.01

    _, parts_right = weighted_nll_loss(
        pred_price_right, pred_price_logvar, pred_volume, pred_vol_logvar, true_y, w_direction=1.0
    )
    _, parts_wrong = weighted_nll_loss(
        pred_price_wrong, pred_price_logvar, pred_volume, pred_vol_logvar, true_y, w_direction=1.0
    )
    assert parts_right["direction_loss"] < parts_wrong["direction_loss"]


def test_cvae_loss_nll_mode_plumbs_logvar_through():
    torch.manual_seed(0)
    batch, z_dim = 2, 4
    pred_price = torch.randn(batch, HORIZON, 4) * 0.01
    pred_price_logvar = torch.zeros(batch, HORIZON, 4)
    pred_volume = torch.randn(batch, HORIZON)
    pred_vol_logvar = torch.zeros(batch, HORIZON)
    true_price = torch.randn(batch, HORIZON, 4) * 0.01
    true_volume = torch.randn(batch, HORIZON)
    true_y = _make_y(true_price, true_volume)
    mu_q = torch.zeros(batch, z_dim)
    logvar_q = torch.zeros(batch, z_dim)
    mu_p = torch.zeros(batch, z_dim)
    logvar_p = torch.zeros(batch, z_dim)

    total, parts = cvae_loss(
        pred_price, pred_volume, true_y, mu_q, logvar_q, mu_p, logvar_p,
        w_price=1.0, w_vol=0.5, beta=1.0, reconstruction="nll",
        pred_price_logvar=pred_price_logvar, pred_vol_logvar=pred_vol_logvar,
    )
    expected_recon, expected_parts = weighted_nll_loss(
        pred_price, pred_price_logvar, pred_volume, pred_vol_logvar, true_y, w_price=1.0, w_vol=0.5
    )
    assert math.isclose(parts["price_loss"], expected_parts["price_loss"], rel_tol=1e-5)
    assert torch.isclose(total, expected_recon)  # beta*kl_loss == 0 here (mu_q==mu_p, logvar_q==logvar_p)


def test_cvae_loss_nll_mode_requires_logvar_args():
    batch, z_dim = 2, 4
    pred_price = torch.zeros(batch, HORIZON, 4)
    pred_volume = torch.zeros(batch, HORIZON)
    true_y = _make_y(torch.zeros(batch, HORIZON, 4), torch.zeros(batch, HORIZON))
    zeros_z = torch.zeros(batch, z_dim)

    with pytest.raises(ValueError):
        cvae_loss(
            pred_price, pred_volume, true_y, zeros_z, zeros_z, zeros_z, zeros_z,
            w_price=1.0, w_vol=0.5, beta=1.0, reconstruction="nll",
        )


def test_cvae_loss_unknown_reconstruction_mode_raises():
    batch, z_dim = 2, 4
    pred_price = torch.zeros(batch, HORIZON, 4)
    pred_volume = torch.zeros(batch, HORIZON)
    true_y = _make_y(torch.zeros(batch, HORIZON, 4), torch.zeros(batch, HORIZON))
    zeros_z = torch.zeros(batch, z_dim)

    with pytest.raises(ValueError):
        cvae_loss(
            pred_price, pred_volume, true_y, zeros_z, zeros_z, zeros_z, zeros_z,
            w_price=1.0, w_vol=0.5, beta=1.0, reconstruction="bogus",
        )
