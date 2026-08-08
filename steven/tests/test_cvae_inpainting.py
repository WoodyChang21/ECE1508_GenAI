import torch

from src.data_pipeline import N_CHANNELS, TOTAL_LEN
from src.models.cvae_inpainting import CVAEInpainting


def _dummy_batch(batch_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    masked_tensor = torch.randn(batch_size, TOTAL_LEN, N_CHANNELS)
    full_tensor = torch.randn(batch_size, TOTAL_LEN, N_CHANNELS)
    return masked_tensor, full_tensor


def test_default_decoder_ctx_dim_matches_old_behavior():
    """Old saved configs never had decoder_ctx_dim -- constructing without it (as
    CVAEInpainting(**checkpoint["config"]["model"]) does for such a checkpoint) must
    still produce a decoder whose input is z_dim + ctx_dim, unchanged."""
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32)
    assert model.decoder_ctx_dim == 16
    assert model.decoder[0].in_features == 4 + 16

    masked_tensor, full_tensor = _dummy_batch()
    price, volume, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
    assert price.shape == (3, 3, 4)
    assert volume.shape == (3, 3)


def test_decoder_ctx_dim_bottlenecks_decoder_input():
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32, decoder_ctx_dim=4)
    assert model.decoder_ctx_dim == 4
    assert model.decoder[0].in_features == 4 + 4

    masked_tensor, full_tensor = _dummy_batch()
    price, volume, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
    assert price.shape == (3, 3, 4)
    assert volume.shape == (3, 3)

    k = 5
    sample_price, sample_volume = model.sample(masked_tensor, k=k)
    assert sample_price.shape == (k, 3, 3, 4)
    assert sample_volume.shape == (k, 3, 3)


def test_prior_head_still_sees_full_ctx_dim_when_bottlenecked():
    """decoder_ctx_dim must only affect the decoder's own view of context -- the prior
    (prior_head) always sees the full, un-bottlenecked ctx_dim representation."""
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32, decoder_ctx_dim=4)
    assert model.prior_head.in_features == 16
    assert model.recognition_head.in_features == 16
