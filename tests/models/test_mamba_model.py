import pytest
import torch

from scripts.models.mamba_model import MambaForecaster
from scripts.models.train_mamba import ForecastLoss


def test_mamba_forecaster_output_shape_and_gradient():
    model = MambaForecaster(
        n_features=5,
        d_model=8,
        state_size=4,
        num_layers=1,
        expand=1,
        conv_kernel=2,
        dropout=0.0,
    )
    values = torch.randn(2, 6, 5)

    predictions = model(values)
    predictions.sum().backward()

    assert predictions.shape == (2,)
    assert model.input_projection.weight.grad is not None


def test_mamba_forecaster_rejects_wrong_feature_count():
    model = MambaForecaster(n_features=3, d_model=8, num_layers=1)

    with pytest.raises(ValueError, match="expected 3 features"):
        model(torch.randn(2, 4, 2))


def test_mamba_forecaster_multi_horizon_output_shape():
    model = MambaForecaster(
        n_features=3,
        d_model=8,
        state_size=4,
        num_layers=1,
        expand=1,
        conv_kernel=2,
        dropout=0.0,
        output_size=3,
    )

    predictions = model(torch.randn(2, 5, 3))

    assert predictions.shape == (2, 3)


def test_forecast_loss_combines_path_horizon_and_direction_gradients():
    loss_fn = ForecastLoss(
        target_mean=0.0001,
        target_scale=0.004,
        forecast_horizon=3,
        cumulative_weight=1.0,
        direction_weight=0.05,
    )
    output = torch.zeros((2, 3), requires_grad=True)
    target = torch.tensor([[1.0, 0.5, -0.25], [-1.0, -0.5, 0.25]])

    loss = loss_fn(output, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
