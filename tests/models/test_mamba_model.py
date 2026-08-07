import pytest
import torch

from scripts.models.mamba_model import MambaForecaster


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
