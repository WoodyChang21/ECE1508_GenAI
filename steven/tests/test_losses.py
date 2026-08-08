import torch
import torch.nn.functional as F

from src.losses import cvae_loss, weighted_mse_loss


def _make_y(price: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Inverse of losses.unpack_y: price (B,3,4), volume (B,3) -> y (B,15)."""
    return torch.cat([price.reshape(price.shape[0], -1), volume], dim=-1)


def test_direction_loss_off_by_default_matches_old_behavior():
    torch.manual_seed(0)
    pred_price = torch.randn(4, 3, 4) * 0.01
    pred_volume = torch.randn(4, 3)
    true_price = torch.randn(4, 3, 4) * 0.01
    true_volume = torch.randn(4, 3)
    true_y = _make_y(true_price, true_volume)

    total, parts = weighted_mse_loss(pred_price, pred_volume, true_y)
    assert parts["direction_loss"] == 0.0

    expected_total = F.mse_loss(pred_price, true_price) + 0.5 * F.mse_loss(pred_volume, true_volume)
    assert torch.isclose(total, expected_total)


def test_direction_loss_penalizes_wrong_sign_more_than_right_sign():
    """true close_ret (open_ret+body_ret) is +0.01 (up) on every bar; a prediction with
    the same sign should score a lower direction_loss than one with the opposite sign."""
    true_price = torch.zeros(1, 3, 4)
    true_price[..., 1] = 0.01
    true_volume = torch.zeros(1, 3)
    true_y = _make_y(true_price, true_volume)
    pred_volume = torch.zeros(1, 3)

    pred_price_right = torch.zeros(1, 3, 4)
    pred_price_right[..., 1] = 0.01
    pred_price_wrong = torch.zeros(1, 3, 4)
    pred_price_wrong[..., 1] = -0.01

    _, parts_right = weighted_mse_loss(pred_price_right, pred_volume, true_y, w_direction=1.0)
    _, parts_wrong = weighted_mse_loss(pred_price_wrong, pred_volume, true_y, w_direction=1.0)
    assert parts_right["direction_loss"] < parts_wrong["direction_loss"]


def test_direction_loss_disabled_does_not_affect_total_even_with_wrong_sign():
    true_price = torch.zeros(1, 3, 4)
    true_price[..., 1] = 0.01
    true_volume = torch.zeros(1, 3)
    true_y = _make_y(true_price, true_volume)
    pred_volume = torch.zeros(1, 3)
    pred_price_wrong = torch.zeros(1, 3, 4)
    pred_price_wrong[..., 1] = -0.01

    total, parts = weighted_mse_loss(pred_price_wrong, pred_volume, true_y, w_direction=0.0)
    assert parts["direction_loss"] == 0.0
    assert torch.isclose(total, F.mse_loss(pred_price_wrong, true_price))


def test_cvae_loss_forwards_direction_args():
    torch.manual_seed(0)
    batch, z_dim = 2, 4
    pred_price = torch.randn(batch, 3, 4) * 0.01
    pred_volume = torch.randn(batch, 3)
    true_price = torch.randn(batch, 3, 4) * 0.01
    true_volume = torch.randn(batch, 3)
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
