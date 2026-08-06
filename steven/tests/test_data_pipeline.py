import numpy as np
import pandas as pd
import pytest

from src import data_pipeline as dp

BAR_TIMES = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def _make_synthetic_ohlcv(n_days: int = 30, seed: int = 0, start: str = "2010-01-04") -> pd.DataFrame:
    """Synthetic OHLCV honoring H >= max(O,C) and L <= min(O,C), 7 bars/business-day."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)

    rows = []
    close = 100.0
    for day in days:
        for t in BAR_TIMES:
            open_ = close * (1 + rng.normal(0, 0.001))
            close = open_ * (1 + rng.normal(0, 0.002))
            wick = abs(rng.normal(0, 0.001))
            high = max(open_, close) * (1 + wick)
            low = min(open_, close) * (1 - wick)
            volume = int(rng.integers(1000, 100000))
            rows.append(
                {
                    "datetime": pd.Timestamp(f"{day.date()} {t}:00"),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def _featurize_and_normalize(raw: pd.DataFrame) -> pd.DataFrame:
    feat_df = dp.compute_raw_features(raw)
    stats = dp.fit_normalize(feat_df, (0, len(feat_df)))
    return dp.apply_normalize(feat_df, stats)


# ---------------------------------------------------------------------------
# Reparametrization round-trip
# ---------------------------------------------------------------------------


def test_reconstruct_prices_round_trip():
    raw = _make_synthetic_ohlcv(n_days=20, seed=1)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    ctx_bars = 14
    start_idx = 0
    window = dp.build_window(feat, opens, closes, start_idx, ctx_bars)

    price_components = window["y"][:12].reshape(3, 4)
    reconstructed = dp.reconstruct_prices(price_components, window["close_0"])

    # compute_raw_features drops the very first row of `raw` -- compare against
    # feat_df (which retains the original OHLC columns), not raw, to stay aligned.
    hz_start = start_idx + ctx_bars
    true_ohlc = feat_df.iloc[hz_start : hz_start + 3][["open", "high", "low", "close"]].to_numpy()

    np.testing.assert_allclose(reconstructed, true_ohlc, rtol=1e-6, atol=1e-6)


def test_anchor_correction_matches_close_0_for_all_horizon_bars():
    """open_ret for every horizon bar should equal ln(open / close_0), including bar 1
    (chained) and bars 2-3 (explicitly corrected) -- they must agree since bar 1's
    chained reference already equals close_0."""
    raw = _make_synthetic_ohlcv(n_days=20, seed=2)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    ctx_bars = 21
    start_idx = 5
    window = dp.build_window(feat, opens, closes, start_idx, ctx_bars)

    hz_start = start_idx + ctx_bars
    close_0 = closes[hz_start - 1]
    expected_open_ret = np.log(opens[hz_start : hz_start + 3] / close_0)

    open_ret_from_y = window["y"][:12].reshape(3, 4)[:, 0]
    np.testing.assert_allclose(open_ret_from_y, expected_open_ret, rtol=1e-6, atol=1e-8)


def test_wick_components_non_negative():
    raw = _make_synthetic_ohlcv(n_days=30, seed=3)
    feat_df = dp.compute_raw_features(raw)
    assert (feat_df["upper_wick"] >= 0).all()
    assert (feat_df["lower_wick"] >= 0).all()


# ---------------------------------------------------------------------------
# Window sampler / masking
# ---------------------------------------------------------------------------


def test_build_window_shapes_and_masks():
    raw = _make_synthetic_ohlcv(n_days=30, seed=4)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    for ctx_bars in dp.CONTEXT_LENGTHS:
        window = dp.build_window(feat, opens, closes, start_idx=0, ctx_bars=ctx_bars)
        assert window["masked_tensor"].shape == (dp.TOTAL_LEN, dp.N_CHANNELS)
        assert window["full_tensor"].shape == (dp.TOTAL_LEN, dp.N_CHANNELS)
        assert window["y"].shape == (15,)

        pad_len = dp.MAX_CONTEXT - ctx_bars
        padding_mask = window["masked_tensor"][:, dp.N_FEATURE_CHANNELS]
        target_mask = window["masked_tensor"][:, dp.N_FEATURE_CHANNELS + 1]

        assert (padding_mask[:pad_len] == 0).all()
        assert (padding_mask[pad_len:] == 1).all()
        assert (target_mask[: dp.MAX_CONTEXT] == 0).all()
        assert (target_mask[dp.MAX_CONTEXT :] == 1).all()

        # price/volume channels zeroed at horizon in masked_tensor, present in full_tensor
        horizon_masked_pv = window["masked_tensor"][dp.MAX_CONTEXT :, dp.PRICE_VOL_IDX]
        assert (horizon_masked_pv == 0).all()
        horizon_full_pv = window["full_tensor"][dp.MAX_CONTEXT :, dp.PRICE_VOL_IDX]
        assert not np.allclose(horizon_full_pv, 0)

        # aux features (time_gap, day_bar_index) NOT masked at horizon -- known in advance
        aux_idx = [5, 6]
        horizon_masked_aux = window["masked_tensor"][dp.MAX_CONTEXT :, aux_idx]
        horizon_full_aux = window["full_tensor"][dp.MAX_CONTEXT :, aux_idx]
        np.testing.assert_allclose(horizon_masked_aux, horizon_full_aux)


def test_to_patchtst_input_patch_padding_mask():
    raw = _make_synthetic_ohlcv(n_days=30, seed=5)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    ctx_bars = 21  # 3 real patches, 7 padded patches
    window = dp.build_window(feat, opens, closes, start_idx=0, ctx_bars=ctx_bars)
    context, patch_key_padding_mask = dp.to_patchtst_input(window["masked_tensor"])

    assert context.shape == (dp.MAX_CONTEXT, dp.N_FEATURE_CHANNELS)
    assert patch_key_padding_mask.shape == (dp.N_PATCHES,)
    n_real_patches = ctx_bars // dp.PATCH_LEN
    assert patch_key_padding_mask.sum() == dp.N_PATCHES - n_real_patches
    # padded patches come first (left-padding)
    assert patch_key_padding_mask[: dp.N_PATCHES - n_real_patches].all()
    assert not patch_key_padding_mask[dp.N_PATCHES - n_real_patches :].any()


def test_window_sampler_unique_and_within_bounds():
    lo, hi = 0, 2000
    sampler = dp.WindowSampler(lo, hi)
    rng = np.random.default_rng(0)
    pairs = sampler.draw(900, rng)

    assert len(pairs) == len(set(pairs))  # no (start_idx, ctx_bars) collisions
    for start_idx, ctx_bars in pairs:
        assert ctx_bars in dp.CONTEXT_LENGTHS
        assert start_idx >= lo
        assert start_idx + ctx_bars + dp.HORIZON <= hi


def test_window_sampler_respects_split_boundary():
    """No sampled window may reach past hi -- i.e. cross into the next split."""
    lo, hi = 100, 200  # deliberately small range relative to max context+horizon
    sampler = dp.WindowSampler(lo, hi)
    rng = np.random.default_rng(0)
    pairs = sampler.draw(50, rng)

    for start_idx, ctx_bars in pairs:
        assert start_idx + ctx_bars + dp.HORIZON <= hi
        # long-context windows shouldn't even be possible in a range this small
        if ctx_bars > (hi - lo - dp.HORIZON):
            pytest.fail(f"ctx_bars={ctx_bars} should have had no valid starts in range [{lo},{hi})")


# ---------------------------------------------------------------------------
# Gap report
# ---------------------------------------------------------------------------


def test_gap_report_detects_missing_and_short_days():
    raw = _make_synthetic_ohlcv(n_days=15, seed=6)
    days = sorted(raw["datetime"].dt.normalize().unique())

    missing_day = days[5]
    short_day = days[8]

    raw = raw[raw["datetime"].dt.normalize() != missing_day].reset_index(drop=True)
    short_day_mask = raw["datetime"].dt.normalize() == short_day
    short_day_rows = raw[short_day_mask]
    drop_indices = short_day_rows.index[:3]  # remove 3 of 7 bars -> 4 remain
    raw = raw.drop(index=drop_indices).reset_index(drop=True)

    report = dp.build_gap_report(raw)

    assert report["n_missing_days"] == 1
    assert str(pd.Timestamp(missing_day).date()) in report["missing_days"]
    assert report["n_short_days"] == 1
    assert report["short_days"][str(pd.Timestamp(short_day).date())] == 4


# ---------------------------------------------------------------------------
# Long-only backtest helpers
# ---------------------------------------------------------------------------


def test_exit_price_from_components_matches_manual_average():
    raw = _make_synthetic_ohlcv(n_days=20, seed=7)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    start_idx, ctx_bars = 5, 21
    window = dp.build_window(feat, opens, closes, start_idx, ctx_bars)
    price_components = window["y"][:12].reshape(3, 4)

    exit_price = dp.exit_price_from_components(price_components, window["close_0"])

    hz_start = start_idx + ctx_bars
    true_bars = feat_df.iloc[hz_start : hz_start + 3][["open", "close"]].to_numpy()
    expected = true_bars.sum() / 6.0  # (o1+c1+o2+c2+o3+c3) / 6

    np.testing.assert_allclose(exit_price, expected, rtol=1e-5)


def test_exit_price_from_components_batched():
    """Same function, batched over a leading window dimension, should match the
    single-window computation elementwise."""
    raw = _make_synthetic_ohlcv(n_days=20, seed=8)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    windows = [dp.build_window(feat, opens, closes, i, 14) for i in (0, 5, 10)]
    components = np.stack([w["y"][:12].reshape(3, 4) for w in windows])
    close_0 = np.array([w["close_0"] for w in windows])

    batched = dp.exit_price_from_components(components, close_0)
    individual = np.array(
        [dp.exit_price_from_components(c, c0) for c, c0 in zip(components, close_0)]
    )
    np.testing.assert_allclose(batched, individual, rtol=1e-6)


def test_max_close_from_components_matches_manual_max():
    raw = _make_synthetic_ohlcv(n_days=20, seed=12)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    start_idx, ctx_bars = 5, 21
    window = dp.build_window(feat, opens, closes, start_idx, ctx_bars)
    price_components = window["y"][:12].reshape(3, 4)

    take_profit = dp.max_close_from_components(price_components, window["close_0"])

    hz_start = start_idx + ctx_bars
    true_closes = feat_df.iloc[hz_start : hz_start + 3]["close"].to_numpy()
    np.testing.assert_allclose(take_profit, true_closes.max(), rtol=1e-5)


def test_max_close_from_components_batched():
    raw = _make_synthetic_ohlcv(n_days=20, seed=13)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    windows = [dp.build_window(feat, opens, closes, i, 14) for i in (0, 5, 10)]
    components = np.stack([w["y"][:12].reshape(3, 4) for w in windows])
    close_0 = np.array([w["close_0"] for w in windows])

    batched = dp.max_close_from_components(components, close_0)
    individual = np.array(
        [dp.max_close_from_components(c, c0) for c, c0 in zip(components, close_0)]
    )
    np.testing.assert_allclose(batched, individual, rtol=1e-6)


def test_per_bar_close_return_sign_matches_open_close_direction():
    raw = _make_synthetic_ohlcv(n_days=20, seed=9)
    feat_df = _featurize_and_normalize(raw)
    feat, opens, closes = dp.extract_arrays(feat_df)

    start_idx, ctx_bars = 0, 14
    window = dp.build_window(feat, opens, closes, start_idx, ctx_bars)
    price_components = window["y"][:12].reshape(3, 4)

    close_ret = dp.per_bar_close_return(price_components)
    assert close_ret.shape == (3,)

    hz_start = start_idx + ctx_bars
    close_0 = window["close_0"]
    true_closes = feat_df.iloc[hz_start : hz_start + 3]["close"].to_numpy()
    expected_sign = np.sign(true_closes - close_0)

    np.testing.assert_array_equal(np.sign(close_ret), expected_sign)


# ---------------------------------------------------------------------------
# Post-hoc recalibration
# ---------------------------------------------------------------------------


def test_train_exit_return_bound_matches_manual_percentile():
    raw = _make_synthetic_ohlcv(n_days=30, seed=10)
    feat_df = _featurize_and_normalize(raw)
    _, opens, closes = dp.extract_arrays(feat_df)

    lo, hi = 0, len(closes)
    bound = dp.train_exit_return_bound(opens, closes, (lo, hi), percentile=99.0)

    anchors = np.arange(lo + 1, hi - dp.HORIZON + 1)
    close_0 = closes[anchors - 1]
    horizon_idx = anchors[:, None] + np.arange(dp.HORIZON)[None, :]
    open_ret = np.log(opens[horizon_idx] / close_0[:, None])
    close_ret = np.log(closes[horizon_idx] / close_0[:, None])
    pooled = np.abs(np.concatenate([open_ret, close_ret], axis=1))
    expected = float(np.percentile(pooled, 99.0))

    assert bound == pytest.approx(expected)
    assert bound > 0


def test_train_exit_return_bound_raises_when_range_too_narrow():
    with pytest.raises(ValueError):
        dp.train_exit_return_bound(np.ones(10), np.ones(10), (0, dp.HORIZON), percentile=99.0)


def test_shrink_components_preserves_small_values_and_bounds_large_ones():
    bound = 0.01
    small = np.full((3, 4), bound / 100)  # << bound: tanh(x/b)*b ~= x, ~unchanged
    shrunk_small = dp.shrink_components(small, bound)
    np.testing.assert_allclose(shrunk_small, small, rtol=1e-2)

    large = np.full((3, 4), bound * 100)  # >> bound: squashed toward +-bound (tanh saturates)
    shrunk_large = dp.shrink_components(large, bound)
    assert np.all(np.abs(shrunk_large) <= bound)
    assert np.all(np.abs(shrunk_large) > bound * 0.9)


def test_shrink_components_shape_and_sign_preserved():
    rng = np.random.default_rng(11)
    components = rng.normal(scale=0.02, size=(5, 3, 4))
    bound = 0.015

    shrunk = dp.shrink_components(components, bound)
    assert shrunk.shape == components.shape
    np.testing.assert_array_equal(np.sign(shrunk), np.sign(components))
