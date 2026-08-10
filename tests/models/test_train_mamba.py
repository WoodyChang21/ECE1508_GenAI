import pytest

from scripts.models.train_mamba import (
    CheckpointSelectionOptions,
    choose_epoch_metrics,
    choose_lookback_result,
    cosine_warmup_multiplier,
)


def selection_options(metric="fixed_strategy_net_return"):
    return CheckpointSelectionOptions(
        metric=metric,
        mae_tolerance=0.01,
        fixed_threshold_bps=2.0,
        minimum_trades=30,
        transaction_cost_bps=1.0,
        periods_per_year=1764,
        position_mode="long_only",
        require_all_steps_agree=True,
    )


def epoch_row(epoch, mae, correlation, net_return, trades):
    return {
        "epoch": epoch,
        "cumulative_val_mae": mae,
        "correlation": correlation,
        "directional_accuracy": 0.52,
        "directional_edge": -0.01,
        "prediction_std": 0.001,
        "actual_std": 0.007,
        "fixed_strategy_net_return": net_return,
        "fixed_strategy_sharpe": 0.5,
        "fixed_strategy_trades": trades,
        "train_loss": 0.6,
        "learning_rate": 1e-4,
    }


def test_epoch_selection_uses_profit_inside_mae_quality_gate():
    rows = [
        epoch_row(1, 0.004000, 0.01, 0.01, 40),
        epoch_row(2, 0.004020, 0.08, 0.05, 40),
        epoch_row(3, 0.004100, 0.20, 0.20, 40),
    ]

    selected, reason = choose_epoch_metrics(rows, selection_options())

    assert selected["epoch"] == 2
    assert reason == (
        "near_best_mae_then_fixed_strategy_net_return_then_correlation"
    )


def test_epoch_selection_breaks_identical_profit_tie_with_correlation():
    rows = [
        epoch_row(1, 0.004000, 0.01, 0.05, 40),
        epoch_row(2, 0.004020, 0.08, 0.05, 40),
        epoch_row(3, 0.004030, 0.06, 0.05, 40),
    ]

    selected, _ = choose_epoch_metrics(rows, selection_options())

    assert selected["epoch"] == 2


def test_epoch_selection_falls_back_to_best_mae_without_positive_signal():
    rows = [
        epoch_row(1, 0.004000, -0.01, 0.10, 40),
        epoch_row(2, 0.004020, None, 0.20, 40),
    ]

    selected, reason = choose_epoch_metrics(rows, selection_options())

    assert selected["epoch"] == 1
    assert reason == "fallback_best_mae"


def test_lookback_selection_prefers_shorter_near_tied_profitable_model():
    results = [
        {
            "lookback": 24,
            "best_cumulative_val_mae": 0.00400,
            "selected_cumulative_val_mae": 0.00402,
            "selected_correlation": 0.08,
            "selected_fixed_strategy_net_return": 0.096,
            "selected_fixed_strategy_trades": 40,
        },
        {
            "lookback": 120,
            "best_cumulative_val_mae": 0.00399,
            "selected_cumulative_val_mae": 0.00400,
            "selected_correlation": 0.10,
            "selected_fixed_strategy_net_return": 0.100,
            "selected_fixed_strategy_trades": 40,
        },
    ]

    selected, reason = choose_lookback_result(
        results, selection_options(), score_tolerance=0.005
    )

    assert selected["lookback"] == 24
    assert reason.endswith("shorter_tie_break")


def test_cosine_schedule_warms_up_then_decays():
    assert cosine_warmup_multiplier(0, total_steps=100, warmup_steps=20) == pytest.approx(
        0.05
    )
    assert cosine_warmup_multiplier(19, 100, 20) == pytest.approx(1.0)
    assert cosine_warmup_multiplier(20, 100, 20) == pytest.approx(1.0)
    assert cosine_warmup_multiplier(100, 100, 20) == pytest.approx(0.0)
