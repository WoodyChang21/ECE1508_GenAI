"""Regenerate the data-driven tables/images in v1.md from the latest evaluate.py run.

Reads steven/outputs/metrics.json + steven/outputs/sample_plots/samples.json (both written
by evaluate.py) and replaces 6 marker-delimited regions in v1.md -- the Results section's
per-sample subsections plus its two summary tables, the population-level outcome-breakdown
table, and the Long-only backtest results' two threshold tables. Everything else in v1.md
(prose, headings, caveats) is left untouched. Run this after every evaluate.py run so the
doc's tables/images always match the latest checkpoints instead of a stale hand-transcribed
run.

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
    "not auto-updated -- reread and edit by hand if the story changed: the 'In plain "
    "terms' / 'A subtle but important point' interpretation paragraphs under Results, "
    "the 'pre-retrain checkpoints' caveats in Results and Long-only backtest results, "
    "and the 'Retrain both models' checkbox under Next steps."
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


OUTCOME_LABEL = {"take_profit": "TAKE PROFIT", "expired": "EXPIRED"}

# Mirrors src/evaluate.py's CASE_LABELS/CASE_TITLES -- kept as a local literal rather than
# importing evaluate.py, since this script only ever reads the JSON it already wrote and
# has no other dependency on torch/models.
CASE_LABELS = ["win_take_profit", "win_expiry", "no_trade", "lose_expiry"]
CASE_TITLES = {
    "win_take_profit": "Win — take-profit hit",
    "win_expiry": "Win — expiry (gain)",
    "no_trade": "No trade (target <= buy)",
    "lose_expiry": "Lose — expiry (loss)",
}


def vs_real_cell(model: dict) -> str:
    """No trade was entered, so there's no take-profit order to have resolved."""
    if not model["would_enter"]:
        return "—"
    status = f"**{OUTCOME_LABEL[model['outcome']]}**"
    return f"{status} → sell {model['realized_price']:.2f} ({fmt_signed_bold_pct_value(model['realized_return_pct'])})"


def trade_decision_cell(model: dict) -> str:
    return "**ENTER**" if model["would_enter"] else "**NO TRADE**"


def ground_truth_vs_real_cell(gt: dict, buy: float) -> str:
    """Ground truth has no predicted target/entry decision to hit or miss, so its
    'vs. real price' cell instead reports a buy-and-hold benchmark: sell at candle 3's
    real close, the same forced-close price a MISSED trade would have used."""
    close = gt["candles"][2][3]
    pct = (close - buy) / buy * 100
    return f"**BENCHMARK** → sell {close:.2f} ({fmt_signed_bold_pct_value(pct)})"


def sample_section(s: dict) -> str:
    gt, pt, cvae = s["ground_truth"], s["patchtst"], s["cvae"]
    buy = s["buy_price"]
    return "\n".join([
        f"### {CASE_TITLES[s['case']]} — {s['ctx_bucket']} context (`ctx_bars={s['ctx_bars']}`, "
        f"`start_idx={s['start_idx']}`)",
        "",
        f"![{s['case']}_{s['ctx_bucket']}](outputs/sample_plots/{s['file']})",
        "",
        "| | Candle 1 (open / close) | Candle 2 (open / close) | Candle 3 (open / close) | "
        "Buy price | Take-profit | Trade? | vs. real price |",
        "|---|---|---|---|---|---|---|---|",
        f"| Ground truth | {candle_oc(gt['candles'][0])} | {candle_oc(gt['candles'][1])} | "
        f"{candle_oc(gt['candles'][2])} | {buy:.2f} | — | — | "
        f"{ground_truth_vs_real_cell(gt, buy)} |",
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
    """NO TRADE takes priority over the outcome -- a model whose own target sits below
    buy never has a take-profit order placed in the first place, so how it would have
    resolved is moot."""
    if not model["would_enter"]:
        return "NO TRADE"
    return OUTCOME_LABEL[model["outcome"]]


def hit_summary_block(samples: list[dict]) -> str:
    lines = ["| Case | Context | PatchTST | CVAE |", "|---|---|---|---|"]
    for s in samples:
        lines.append(
            f"| {CASE_TITLES[s['case']]} | {s['ctx_bucket']} | "
            f"{hit_status(s['patchtst'])} | {hit_status(s['cvae'])} |"
        )
    return "\n".join(lines)


def spread_summary_block(samples: list[dict]) -> str:
    lines = ["| Case | Context | Ground truth (actual) | PatchTST | CVAE |", "|---|---|---|---|---|"]
    for s in samples:
        lines.append(
            f"| {CASE_TITLES[s['case']]} | {s['ctx_bucket']} | {s['ground_truth']['spread']:.2f} | "
            f"{s['patchtst']['spread']:.2f} | {s['cvae']['spread']:.2f} |"
        )
    return "\n".join(lines)


def outcome_breakdown_table(pt_breakdown: dict, cvae_breakdown: dict) -> str:
    """Population-level breakdown (independent of confidence threshold) -- see
    src/evaluate.py's outcome_breakdown. Rows sum to 100% per column."""
    lines = ["| Case | PatchTST | CVAE |", "|---|---|---|"]
    for case in CASE_LABELS:
        lines.append(
            f"| {CASE_TITLES[case]} | {fmt_plain_pct(pt_breakdown[case], 1)} | "
            f"{fmt_plain_pct(cvae_breakdown[case], 1)} |"
        )
    return "\n".join(lines)


def backtest_row(row: dict) -> str:
    if row["win_rate"] is None:
        return f"| {row['threshold']} | {row['n_trades']} | — | — | — | — |"
    return (
        f"| {row['threshold']} | {row['n_trades']} | {fmt_plain_pct(row['win_rate'], 1)} | "
        f"{fmt_plain_pct(row['take_profit_rate'], 1)} | "
        f"{fmt_signed_pct_fraction(row['avg_return'], 3)} | {fmt_signed_pct_fraction(row['total_return'], 2)} |"
    )


def buy_hold_block(bh: dict) -> str:
    lines = [
        "| Entry (date / price) | Exit (date / price) | Elapsed | Total return | Annualized return |",
        "|---|---|---|---|---|",
        f"| {bh['entry_date']} / {bh['entry_price']:.2f} | {bh['exit_date']} / {bh['exit_price']:.2f} | "
        f"{bh['elapsed_years']:.2f} yr | {fmt_signed_pct_fraction(bh['total_return'], 2)} | "
        f"{fmt_signed_pct_fraction(bh['annual_return'], 2)} |",
    ]
    return "\n".join(lines)


def backtest_table(rows: list[dict]) -> str:
    header = (
        "| Confidence threshold | Trades taken | Win rate | Take-profit rate | "
        "Avg. return per trade | Total return |"
    )
    sep = "|---|---|---|---|---|---|"
    return "\n".join([header, sep] + [backtest_row(r) for r in rows])


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

    blocks = {
        "results-samples": results_samples_block(samples),
        "hit-summary": hit_summary_block(samples),
        "spread-summary": spread_summary_block(samples),
        "outcome-breakdown": outcome_breakdown_table(
            metrics["overall"]["backtest"]["patchtst_outcome_breakdown"],
            metrics["overall"]["backtest"]["cvae_outcome_breakdown"],
        ),
        "backtest-patchtst": backtest_table(metrics["overall"]["backtest"]["patchtst"]),
        "backtest-cvae": backtest_table(metrics["overall"]["backtest"]["cvae"]),
        "buy-hold-benchmark": buy_hold_block(metrics["overall"]["backtest"]["buy_and_hold"]),
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
