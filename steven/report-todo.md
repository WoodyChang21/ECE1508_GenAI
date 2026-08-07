# Report TODO — ideas to work into the writeup later

Just jotting things down so I don't lose them. Not polished, not final — a grab-bag for the discussion
section when we actually write the report.

## Risk / product-fit

- **No stop-loss = unlimited downside within the 3-candle window** (see `v1.md`'s "Trading strategy
  rationale" section for why we dropped it). Worth spelling out clearly in the report: this isn't free —
  we're trading a *known* mechanical honesty problem (can't tell TP vs SL order within a candle) for
  *unbounded* per-trade downside. It happens to be a reasonable trade for SPY specifically.
- **Why SPY makes this tolerable**: it's an index — diversified, much lower single-name volatility than
  an individual stock, no risk of a single company blowing up to zero overnight (earnings gap, fraud,
  delisting, etc.). A single bad 3-candle window won't wipe out the account.
- **This probably does NOT transfer safely to individual stocks as-is.** Individual names are more
  volatile, gap harder around earnings/news, and are way more exposed to idiosyncratic shocks (a bad
  headline, a guidance cut) that a 3-hour-ahead model has no way to see coming. Before extending this
  strategy to single stocks, we'd want to at least: re-derive `MAX_LOG_RETURN`/`train_exit_return_bound`
  per-instrument instead of reusing SPY's calibration, and probably reconsider the no-stop-loss decision
  entirely, since the "it's fine because it's an index" argument doesn't hold once we're not on an index.
- **Market correction / crash scenario**: even for SPY, this hasn't been stress-tested against a real
  crash regime (test period is 2024–2025-05, no 2020-style event in it). Worth flagging as an open
  question rather than an assumption — "no stop-loss is fine for SPY" is based on *this* test window, not
  a guarantee across all regimes.
- **FOREX as a next product to try**: major currency pairs are arguably even more liquid/stable than SPY
  in some respects, so the same "index-like stability" argument might extend there. Differences worth
  digging into before assuming it transfers directly: FOREX trades ~24/5 (no overnight gap the way equities
  have, but also no clean "candle after market close" boundary the way SPY's 9:30–16:00 session gives us),
  spreads/pip costs matter a lot more at these tiny target sizes (~1–2% moves) than for SPY, and the
  feature engineering (log-returns anchored to close_0) should carry over fine, but the context-length /
  session-alignment assumptions (`day_bar_index_norm`, market-hours-only bars) are SPY-session-specific and
  would need rethinking.

## Other things worth a mention in the discussion section

- **The KL-collapse finding is a good "here's something that surprised us" moment for the writeup** — CVAE's
  KL pinned at exactly the free-bits floor across every run is a clean, falsifiable diagnostic (not just a
  vibe), and it directly qualifies how much to trust "CVAE's samples disagree with each other" as a
  meaningful signal vs. just replaying learned prior uncertainty. Good discussion-section material: what
  does "generative uncertainty" even mean if the latent isn't doing much?
- **PatchTST's ~1% take-profit rate is a good cautionary tale about backtest metrics lying to you.** Its
  win rate looks fine in isolation (~54–56%) but is basically just "what happens if you hold SPY for 3
  hours regardless," dressed up as a model-driven trade. Worth a sentence in the report about how a
  headline metric (win rate) can look reasonable while the mechanism behind it is doing almost nothing.
- **Neither model's confidence threshold behaves monotonically** (CVAE's win rate actually *drops* as
  threshold rises post-stop-loss-removal). Worth discussing directly rather than glossing over — "our
  confidence score isn't calibrated yet" is a more honest framing than implying the backtest numbers are
  final.
- **Transaction costs / slippage / bid-ask spread aren't modeled at all.** For SPY this is a small effect;
  for FOREX or less liquid names it could matter a lot more relative to the tiny (~1–2%) target sizes
  we're aiming for. Worth a limitations-section line if we go there.
- **The inpainting framing itself is worth revisiting in the discussion**: does treating this as "generate
  a plausible candle" actually buy us something over point forecasting? So far CVAE (generative) beats
  PatchTST (point forecast) on every reasonableness metric, but neither is well-calibrated yet — so the
  honest claim right now is "the generative framing is more promising," not "the generative framing
  works."
- **Overlapping test windows / not a real equity curve** (already in `v1.md`'s Limitations) — worth
  repeating in the report's limitations section since it undercuts how literally to read "total return."
