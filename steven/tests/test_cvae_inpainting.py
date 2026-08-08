import torch

from src.data_pipeline import MAX_LOG_RETURN, N_CHANNELS, TOTAL_LEN
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
    price, price_logvar, volume, vol_logvar, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
    assert price.shape == (3, 3, 4)
    assert price_logvar.shape == (3, 3, 4)
    assert volume.shape == (3, 3)
    assert vol_logvar.shape == (3, 3)


def test_decoder_ctx_dim_bottlenecks_decoder_input():
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32, decoder_ctx_dim=4)
    assert model.decoder_ctx_dim == 4
    assert model.decoder[0].in_features == 4 + 4

    masked_tensor, full_tensor = _dummy_batch()
    price, price_logvar, volume, vol_logvar, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
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


def test_decoder_output_width_doubled_for_mean_and_logvar():
    """30, not 15: one (mean, logvar) pair per component."""
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32)
    assert model.decoder[-1].out_features == 30


def test_price_logvar_stays_within_configured_range():
    """decode()'s bounded-sigmoid squash must hold regardless of input -- feed the raw
    decoder linear layer wildly large/small weights (via extreme z/ctx) and confirm the
    returned logvar never leaves price_logvar_range."""
    lo, hi = -10.0, -4.0
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32, price_logvar_range=(lo, hi))
    masked_tensor, _ = _dummy_batch(batch_size=8)
    with torch.no_grad():
        mu_p, logvar_p, ctx_repr = model.encode_prior(masked_tensor)
        z = torch.randn(8, 4) * 1000  # extreme input to stress the squash
        price, price_logvar, volume, vol_logvar = model.decode(z, ctx_repr)
    assert torch.all(price_logvar >= lo) and torch.all(price_logvar <= hi)


def test_vol_logvar_stays_within_configured_range():
    lo, hi = -5.0, 2.0
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32, vol_logvar_range=(lo, hi))
    masked_tensor, _ = _dummy_batch(batch_size=8)
    with torch.no_grad():
        mu_p, logvar_p, ctx_repr = model.encode_prior(masked_tensor)
        z = torch.randn(8, 4) * 1000
        price, price_logvar, volume, vol_logvar = model.decode(z, ctx_repr)
    assert torch.all(vol_logvar >= lo) and torch.all(vol_logvar <= hi)


def test_sample_price_stays_within_architectural_bounds_with_noise():
    """sample_noise=True adds post-squash noise -- must still be re-clamped so open_ret/
    body_ret never exceed +-MAX_LOG_RETURN and wicks never go negative, even with a
    wide-open logvar range that would otherwise blow well past the cap."""
    model = CVAEInpainting(
        hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32,
        price_logvar_range=(-2.0, -2.0),  # std ~0.37, deliberately way past MAX_LOG_RETURN
    )
    masked_tensor, _ = _dummy_batch(batch_size=8)
    price, volume = model.sample(masked_tensor, k=20, sample_noise=True)
    assert torch.all(price[..., 0].abs() <= MAX_LOG_RETURN + 1e-6)
    assert torch.all(price[..., 1].abs() <= MAX_LOG_RETURN + 1e-6)
    assert torch.all(price[..., 2] >= -1e-6) and torch.all(price[..., 2] <= MAX_LOG_RETURN + 1e-6)
    assert torch.all(price[..., 3] >= -1e-6) and torch.all(price[..., 3] <= MAX_LOG_RETURN + 1e-6)


def test_sample_noise_false_reproduces_z_only_diversity():
    """sample_noise=False should give the exact same price/volume as decode() itself,
    i.e. no noise added on top of the z-conditioned mean -- the old behavior, for
    ablating how much of sample()'s diversity comes from z vs. from decoder noise."""
    torch.manual_seed(0)
    model = CVAEInpainting(hidden=8, ctx_dim=16, z_dim=4, decoder_hidden=32)
    masked_tensor, _ = _dummy_batch(batch_size=4)
    with torch.no_grad():
        mu_p, logvar_p, ctx_repr = model.encode_prior(masked_tensor)
        torch.manual_seed(1)
        z = model.reparameterize(mu_p, logvar_p)
        expected_price, _, expected_volume, _ = model.decode(z, ctx_repr)

    torch.manual_seed(1)  # same z draw as above
    price, volume = model.sample(masked_tensor, k=1, sample_noise=False)
    assert torch.allclose(price[0], expected_price)
    assert torch.allclose(volume[0], expected_volume)
