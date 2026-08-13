"""Regenerate the data-driven tables/images in v1.md from the latest evaluate.py run.

Reads steven/outputs/metrics.json + steven/outputs/sample_plots/samples.json (both written
by evaluate.py) and replaces the marker-delimited regions in v1.md -- the Results section's
per-sample subsections plus its two summary tables, and the random-sample backtest's
strategy-comparison and outcome-breakdown tables. Everything else in v1.md (prose,
headings, caveats) is left untouched. Run this after every evaluate.py run so the doc's
tables/images always match the latest checkpoints instead of a stale hand-transcribed run.

Usage:
    python steven/src/update_report.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Paragraphs that cite specific numbers but aren't auto-rewritten -- interpretation, not
# a table/image, so regenerating them would mean writing prose, not formatting data.
MANUAL_REVIEW_REMINDER = (
    "not auto-updated -- reread and edit by hand if the story changed: the historical "
    "narrative sections (Loss, Bounding, Trading criteria's stop-loss/quality-gate "
    "history, Strengths/weaknesses/next-steps) describe how the project got here and are "
    "left as written; only the sections that directly describe the CURRENT evaluation "
    "methodology (Results, Random-sample backtest, Key terms' Backtest definition, "
    "Workflow step 5, How to reproduce step 3) were updated for the switch away from "
    "walk-forward."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=str, default="steven/outputs/metrics.json")
    p.add_argument("--samples-json", type=str, default="steven/outputs/sample_plots/samples.json")
    p.add_argument("--doc", type=str, default="steven/v1.md")
    return p.parse_args()


def fmt_plain_pct(fraction: float, decimals: int) -> str:
    """0.558 -> '55.8%' -- no sign, for rates that are never negative (win/hit rate)."""
    return f"{fraction * 100:.{decimals}f}%"


def fmt_signed_pct_fraction(fraction: float, decimals: int) -> str:
    """0.000561 -> '+0.056%', -0.00343 -> '−3.430%' -- input is a fraction (backtest returns)."""
    sign = "+" if fraction >= 0 else "−"
    return f"{sign}{abs(fraction) * 100:.{decimals}f}%"


def fmt_signed_bold_pct_value(pct_value: float, decimals: int = 2) -> str:
    """-1.28 -> '**−1.28%**' -- input is already a percentage (samples.json's realized_return_pct)."""
    sign = "+" if pct_value >= 0 else "−"
    return f"**{sign}{abs(pct_value):.{decimals}f}%**"


def candle_oc(candle: list[float]) -> str:
    """candle: [open, high, low, close] -> 'open / close', matching the existing table style."""
    return f"{candle[0]:.2f} / {candle[3]:.2f}"


OUTCOME_LABEL = {"stop_loss": "STOP LOSS", "take_profit": "TAKE PROFIT", "expired": "EXPIRED"}

# Mirrors src/evaluate.py's CASE_LABELS/CASE_TITLES -- kept as a local literal rather than
# importing evaluate.py, since this script only ever reads the JSON it already wrote and
# has no other dependency on torch/models.
CASE_LABELS = ["win_take_profit", "win_expiry", "lose_expiry", "lose_stop_loss", "skipped", "no_trade"]
CASE_TITLES = {
    "win_take_profit": "Win — take-profit hit",
    "win_expiry": "Win — expiry (gain)",
    "lose_expiry": "Lose — expiry (loss)",
    "lose_stop_loss": "Lose — stop-loss hit",
    "skipped": "Skipped (return too small)",
    "no_trade": "No trade (target <= buy)",
}

# Mirrors src/evaluate.py's make_random_sample_plots outcome grouping -- these are the
# sample-plot categories (win_take_profit/win_expiry/loss), distinct from CASE_LABELS
# (the finer-grained outcome_breakdown categories, which still separate lose_expiry from
# lose_stop_loss).
SAMPLE_OUTCOME_TITLES = {
    "win_take_profit": "Win — take-profit hit",
    "win_expiry": "Win — expiry (gain)",
    "loss": "Loss (expiry or stop-loss)",
}
BUCKET_TITLES = {"narrow": "Narrow context (14–28 bars)", "wide": "Wide context (56–70 bars)"}


def trade_decision_cell(model: dict) -> str:
    return "**ENTER**" if model["would_enter"] else "**NO TRADE**"


def vs_real_cell(model: dict) -> str:
    """This category is selected by CVAE's own outcome (see evaluate.py's
    make_random_sample_plots) -- PatchTST is guaranteed to have been EVALUATED at the same
    window now, but not guaranteed to have TRADED it (its own quality gate/return
    threshold can independently say no even though CVAE said yes, or vice versa for the
    'loss' category)."""
    if not model["would_enter"]:
        return "—"
    status = f"**{OUTCOME_LABEL[model['outcome']]}**"
    return f"{status} → sell {model['realized_price']:.2f} ({fmt_signed_bold_pct_value(model['realized_return_pct'])})"


# PRE-PIVOT STRATEGY (steven4): hardcodes a 3-candle table, matching evaluate.py's
# pre-pivot samples.json format (see the PRE-PIVOT note atop evaluate.py). Left as-is --
# still valid for v1.md's existing report, which still describes the HORIZON=3 bracket-
# order strategy -- but will need a real redesign (not a shape patch) once the rolling
# hour-by-hour strategy (steven/rolling_hour_backtest.md) gets its own report section.
def sample_section(s: dict) -> str:
    gt, pt, cvae = s["ground_truth"], s["patchtst"], s["cvae"]
    buy = s["buy_price"]
    title = f"{SAMPLE_OUTCOME_TITLES[s['outcome']]} — {BUCKET_TITLES[s['bucket']]}"
    return "\n".join([
        f"### {title} (`ctx_bars={s['ctx_bars']}`, `start_idx={s['start_idx']}`)",
        "",
        f"![{s['outcome']}_{s['bucket']}](outputs/sample_plots/{s['file']})",
        "",
        "| | Candle 1 (open / close) | Candle 2 (open / close) | Candle 3 (open / close) | "
        "Buy price | Take-profit | Trade? | vs. real price |",
        "|---|---|---|---|---|---|---|---|",
        f"| Ground truth | {candle_oc(gt['candles'][0])} | {candle_oc(gt['candles'][1])} | "
        f"{candle_oc(gt['candles'][2])} | {buy:.2f} | — | — | — |",
        f"| PatchTST | {candle_oc(pt['candles'][0])} | {candle_oc(pt['candles'][1])} | "
        f"{candle_oc(pt['candles'][2])} | {buy:.2f} | {pt['sell_limit']:.2f} | "
        f"{trade_decision_cell(pt)} | {vs_real_cell(pt)} |",
        f"| CVAE | {candle_oc(cvae['candles'][0])} | {candle_oc(cvae['candles'][1])} | "
        f"{candle_oc(cvae['candles'][2])} | {buy:.2f} | {cvae['sell_limit']:.2f} | "
        f"{trade_decision_cell(cvae)} | {vs_real_cell(cvae)} |",
    ])


def results_samples_block(samples: list[dict]) -> str:
    return "\n\n".join(sample_section(s) for s in samples)


def hit_status(model: dict) -> str:
    if not model["would_enter"]:
        return "NO TRADE"
    return OUTCOME_LABEL[model["outcome"]]


def hit_summary_block(samples: list[dict]) -> str:
    lines = ["| Outcome | Context | PatchTST | CVAE |", "|---|---|---|---|"]
    for s in samples:
        lines.append(
            f"| {SAMPLE_OUTCOME_TITLES[s['outcome']]} | {s['bucket']} | "
            f"{hit_status(s['patchtst'])} | {hit_status(s['cvae'])} |"
        )
    return "\n".join(lines)


def spread_summary_block(samples: list[dict]) -> str:
    lines = ["| Outcome | Context | Ground truth (actual) | PatchTST | CVAE |", "|---|---|---|---|---|"]
    for s in samples:
        lines.append(
            f"| {SAMPLE_OUTCOME_TITLES[s['outcome']]} | {s['bucket']} | {s['ground_truth']['spread']:.2f} | "
            f"{s['patchtst']['spread']:.2f} | {s['cvae']['spread']:.2f} |"
        )
    return "\n".join(lines)


def outcome_breakdown_table(pt_breakdown: dict, cvae_breakdown: dict) -> str:
    """Population-level breakdown across all n_test_windows -- see
    src/evaluate.py's outcome_breakdown. Rows sum to 100% per column."""
    lines = ["| Case | PatchTST | CVAE |", "|---|---|---|"]
    for case in CASE_LABELS:
        lines.append(
            f"| {CASE_TITLES[case]} | {fmt_plain_pct(pt_breakdown[case], 1)} | "
            f"{fmt_plain_pct(cvae_breakdown[case], 1)} |"
        )
    return "\n".join(lines)


def random_sample_row(name: str, row: dict) -> str:
    if not row.get("n_trades") or row.get("win_rate") is None:
        return f"| {name} | {row.get('n_trades', 0)} | — | — | — |"
    tp_rate = row.get("take_profit_rate")
    tp_cell = fmt_plain_pct(tp_rate, 1) if tp_rate is not None else "—"
    return (
        f"| {name} | {row['n_trades']} | {fmt_plain_pct(row['win_rate'], 1)} | {tp_cell} | "
        f"{fmt_signed_pct_fraction(row['avg_return'], 3)} |"
    )


def random_sample_table(patchtst: dict, cvae: dict) -> str:
    header = "| Strategy | Trades | Win rate | Take-profit rate | Avg. return per trade |"
    sep = "|---|---|---|---|---|"
    rows = [
        random_sample_row("PatchTST", patchtst),
        random_sample_row("CVAE", cvae),
    ]
    return "\n".join([header, sep] + rows)


def replace_block(doc_text: str, name: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- AUTO:{re.escape(name)}:start -->\n).*?(\n<!-- AUTO:{re.escape(name)}:end -->)",
        re.DOTALL,
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + content + m.group(2), doc_text)
    if n == 0:
        raise ValueError(f"marker block {name!r} not found in doc -- was it removed or renamed?")
    return new_text


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text())
    samples = json.loads(Path(args.samples_json).read_text())

    rs = metrics["random_sample"]
    blocks = {
        "results-samples": results_samples_block(samples),
        "hit-summary": hit_summary_block(samples),
        "spread-summary": spread_summary_block(samples),
        "random-sample-strategies": random_sample_table(rs["patchtst"], rs["cvae"]),
        "random-sample-outcome-breakdown": outcome_breakdown_table(
            rs["patchtst"]["outcome_breakdown"], rs["cvae"]["outcome_breakdown"]
        ),
    }

    doc_path = Path(args.doc)
    text = doc_path.read_text()
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    doc_path.write_text(text)

    logger.info("updated %s: %s", doc_path, ", ".join(blocks))
    logger.info(MANUAL_REVIEW_REMINDER)


if __name__ == "__main__":
    main()
