# CVAE slide scaffold (1 min, 1 page)

Planning doc only -- no script yet. Goal here is to lock down the *visual* structure first
(what's on the page, how much space each thing gets), then write the 60-second script to match
what's actually on the slide, not the other way around.

## Constraints

- **1 minute of talking**, **1 page** of visuals. That's roughly 120-150 spoken words if delivered
  at a normal pace -- not enough time to explain training, losses, or hyperparameters. This slide
  has to answer exactly one question: **"what is CVAE and what does it do here?"**
- Result plots are secondary. A small, single supporting image is enough -- this slide is not the
  results slide.
- No trading framing. This project's actual goal is generating diverse, plausible candles, not a
  trading strategy (see `cvae_direction_collapse.md`'s "generative pivot") -- keep the slide
  consistent with that, even if other parts of the deck talk about PatchTST/backtesting.

## Page layout (wireframe)

```
+--------------------------------------------------------------------------+
|  Title: "CVAE: generating plausible future candles"                      |
+------------------------------------------------+-------------------------+
|                                                 |                         |
|   MAIN ZONE (~75% of page)                     |  SECONDARY ZONE (~25%)  |
|   "How CVAE works" -- the architecture/         |  One result plot        |
|   concept diagram                               |  (or a crop of one)     |
|                                                 |                         |
|                                                 |                         |
+--------------------------------------------------------------------------+
```

Alternative if a wide diagram reads better: main zone on top (full width), secondary strip along
the bottom instead of the side. Pick whichever the actual diagram's aspect ratio favors once
drawn -- don't force it into a layout that squeezes the diagram.

## Main zone: "how CVAE looks / works" (the majority of the page)

Two candidate levels of detail -- pick ONE, don't show both (no time to explain two diagrams):

**Option A -- concept-first (recommended for 1 minute):**
A single visual metaphor: a candlestick chart where the last 3 candles are grayed out/masked,
with an arrow into a box labeled "CVAE", fanning out into 3-5 *different* colored candle
completions on the other side. This is the one idea worth spending the whole minute on: *one
context, several plausible futures* -- inpainting, not point prediction. Labels to put directly
on the diagram (not in a script, ON the image):
- "context (real candles)" -- pointing at the un-masked part
- "?" or grayed candles -- pointing at the masked region
- "CVAE samples multiple times" -- pointing at the fan-out
- 3-5 distinct completions, visually different from each other (this IS the point -- if they look
  near-identical, the diagram undersells the model)

**Option B -- light architecture block diagram:**
Boxes and arrows, minimal labels, no formulas:
`masked candle window -> [context encoder] -> [prior network] -> z (latent) -> [decoder] ->
predicted candles (mean + spread)`, with a small side note "sampled many times = many futures."
Only worth it if the audience needs to see it's a real model, not a black box -- otherwise Option
A communicates more in the same 60 seconds.

Either way, avoid: loss formulas, KL/free-bits/annealing terminology, training curves, code,
hyperparameter tables. All real and all interesting, all wrong for a 60-second, 1-page slot.

## Secondary zone: one result plot

Don't build anything new for this pass -- reuse an existing image:
- A single panel cropped from `steven/outputs/scenario_charts/uptrend_2.png` (or any
  `{label}_{n}.png}`) -- ideally just the "CVAE" panel next to "Ground truth", not the full
  3-panel PatchTST comparison (PatchTST isn't part of this slide's story).
- Or `steven/outputs/generative_plots/diversity_fan.png` -- arguably a better fit than the
  scenario charts here, since it visually IS "one context, several sampled futures overlaid,"
  the same idea as the main-zone diagram, just on real data instead of a schematic. Worth
  comparing both once we're picking the actual image.

One caption line under it, e.g. "Sampled completions for one real context window" -- no metrics,
no numbers on the slide itself.

## Asset checklist (before moving to script)

- [ ] Decide Option A vs. B for the main diagram
- [ ] Draw/produce the main diagram (hand-sketch is fine to start, then clean up)
- [ ] Pick the one secondary result image (crop `uptrend_2.png`'s CVAE panel, or use
      `diversity_fan.png` as-is)
- [ ] Confirm both images read clearly at 1-page, presentation-projector size (not just on a
      laptop screen) -- text/labels legible from the back of a room
- [ ] Only after both visuals exist: write the ~60-second script keyed to what's actually on the
      page (this doc stays the layout reference; script is a separate pass)

## Open questions for you

- Option A (concept fan-out) or Option B (light architecture box diagram) for the main zone?
- Side-by-side layout or stacked (diagram on top, result strip on bottom)?
- Diversity fan chart or a cropped scenario-chart panel for the secondary image?
