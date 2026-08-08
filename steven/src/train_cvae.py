"""Train the CVAE inpainting model.

Usage:
    python steven/src/train_cvae.py --config steven/configs/cvae.yaml
    python steven/src/train_cvae.py --config steven/configs/cvae.yaml \\
        --max-epochs 2 --train-windows-per-epoch 200 --windows-per-eval-set 60
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.data_pipeline as dp
import src.momentum_pipeline as mp
from src.data_pipeline import WindowDataset, WindowSampler, build_dataset, extract_arrays
from src.generative_metrics import epoch_diagnostics
from src.losses import cvae_loss
from src.models.cvae_inpainting import CVAEInpainting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="steven/configs/cvae.yaml")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--train-windows-per-epoch", type=int, default=None)
    p.add_argument("--windows-per-eval-set", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def kl_beta_schedule(epoch: int, max_epochs: int, n_cycles: int, ramp_fraction: float) -> float:
    """Cyclical KL annealing (Fu et al. 2019, "Cyclical Annealing Schedule") -- repeats a
    0->1 linear ramp `n_cycles` times across training instead of ramping once and staying
    at 1.0 forever after. Once a latent dimension collapses under a flat, maxed-out KL
    penalty, the gradient encouraging the decoder to start using it again is very weak;
    periodically relaxing the penalty back toward 0 gives the decoder repeated chances to
    learn z is worth using before the penalty clamps back down. `ramp_fraction` of each
    cycle ramps 0->1 linearly; the rest of the cycle holds at 1.0."""
    cycle_len = max(1, max_epochs // n_cycles)
    pos_in_cycle = epoch % cycle_len
    ramp_len = max(1, int(cycle_len * ramp_fraction))
    return min(1.0, (pos_in_cycle + 1) / ramp_len)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def collate(batch: list[dict]) -> dict:
    return {
        "masked_tensor": torch.stack([b["masked_tensor"] for b in batch]),
        "full_tensor": torch.stack([b["full_tensor"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def run_epoch(
    model, loader, optimizer, loss_cfg, beta, price_scale, device, train: bool, reconstruction: str = "mse",
) -> dict:
    """reconstruction: "mse" (default, unchanged behavior) or "nll" -- see
    cvae_loss/weighted_nll_loss. The model always emits price_logvar/vol_logvar now
    regardless of mode (decode()'s new outputs), so price_std/vol_std below are always
    logged as a sanity check -- during "mse"-mode epochs (including the nll_warmup_epochs
    window main() resolves before calling this) the logvar head gets no gradient at all,
    so those numbers just reflect wherever it was initialized until "nll" mode switches on."""
    model.train(mode=train)
    totals = {
        "loss": 0.0, "price_loss": 0.0, "vol_loss": 0.0, "direction_loss": 0.0,
        "recon_loss": 0.0, "kl_loss": 0.0, "price_std": 0.0, "vol_std": 0.0,
    }
    n = 0
    for batch in loader:
        masked_tensor = batch["masked_tensor"].to(device)
        full_tensor = batch["full_tensor"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            price, price_logvar, volume, vol_logvar, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
            loss, parts = cvae_loss(
                price, volume, y, mu_q, logvar_q, mu_p, logvar_p,
                loss_cfg["w_price"], loss_cfg["w_vol"], beta, loss_cfg["free_bits"],
                price_scale=price_scale,
                w_direction=loss_cfg["w_direction"], direction_temperature=loss_cfg["direction_temperature"],
                reconstruction=reconstruction,
                pred_price_logvar=price_logvar if reconstruction == "nll" else None,
                pred_vol_logvar=vol_logvar if reconstruction == "nll" else None,
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            # Not present before this -- NLL's exp(-logvar) term produces spikier
            # gradients than plain MSE ever did; worth having regardless of mode.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = y.shape[0]
        totals["loss"] += loss.item() * bs
        totals["price_loss"] += parts["price_loss"] * bs
        totals["vol_loss"] += parts["vol_loss"] * bs
        totals["direction_loss"] += parts["direction_loss"] * bs
        totals["recon_loss"] += parts["recon_loss"] * bs
        totals["kl_loss"] += parts["kl_loss"] * bs
        with torch.no_grad():
            totals["price_std"] += torch.exp(0.5 * price_logvar).mean().item() * bs
            totals["vol_std"] += torch.exp(0.5 * vol_logvar).mean().item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.train_windows_per_epoch is not None:
        cfg["train"]["train_windows_per_epoch"] = args.train_windows_per_epoch
    if args.windows_per_eval_set is not None:
        cfg["train"]["windows_per_eval_set"] = args.windows_per_eval_set
    if args.device is not None:
        cfg["device"] = args.device

    device = resolve_device(cfg["device"])
    logger.info("device: %s", device)

    # Window sampling below is already seeded (np.random.default_rng(cfg["seed"])), but without
    # this, model init + DataLoader shuffling draw from torch's global RNG unseeded -- two runs
    # of this same config produce different starting weights, not just different-looking noise.
    torch.manual_seed(cfg["seed"])

    # Opt-in (configs/cvae.yaml has no `momentum_features` key, so build_dataset's original
    # 9-channel behavior is exactly preserved for every existing config -- see
    # configs/cvae_hourly_momentum.yaml for the enabled case).
    momentum_cfg = cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = mp.build_momentum_dataset(cfg["data_path"], momentum_cfg["vix_data_path"])
        logger.info(
            "momentum features enabled: ema_cross/trend_position/rsi/vix, N_CHANNELS=%d",
            dp.N_CHANNELS,
        )
    else:
        df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    # CVAE-only fix for the volume-dominated loss (see cvae_direction_collapse.md):
    # feat's first 4 columns are exactly [open_ret, body_ret, upper_wick, lower_wick]
    # (FEATURE_COLS order), already raw log-returns -- unlike log_volume_norm, they're
    # never z-scored, so their ~1e-6 variance is dwarfed by volume's unit variance in
    # weighted_mse_loss unless rescaled here. train-set std only, same split discipline
    # as fit_normalize. use_price_scale=false reproduces the pre-fix loss scale exactly
    # (see configs/cvae.yaml's comment) -- a controlled comparison against commit
    # 2c4ad99's checkpoint, not a recommended setting.
    # reconstruction="nll" disables price_scale unconditionally (see weighted_nll_loss's
    # docstring): a learned per-example variance is an adaptive version of what
    # price_scale's single fixed train-set constant approximates, so keeping both active
    # would double-correct and muddy what the learned variance actually represents.
    reconstruction = cfg["loss"].get("reconstruction", "mse")
    train_lo, train_hi = bounds["train"]
    if reconstruction == "nll":
        price_scale = None
        logger.info("price_scale: disabled (reconstruction=nll -- learned variance replaces it)")
    elif cfg["loss"]["use_price_scale"]:
        price_scale = torch.tensor(
            feat[train_lo:train_hi, :4].std(axis=0), dtype=torch.float32, device=device
        )
        logger.info("price_scale (open_ret, body_ret, upper_wick, lower_wick): %s", price_scale.tolist())
    else:
        price_scale = None
        logger.info("price_scale: disabled (use_price_scale=false)")

    train_sampler = WindowSampler(*bounds["train"])
    val_sampler = WindowSampler(*bounds["val"])

    # train_ds is rebuilt fresh every epoch (new sampled windows), so no persistent_workers.
    loader_kwargs = {
        "num_workers": cfg["train"].get("num_workers", 0),
        "pin_memory": cfg["train"].get("pin_memory", False),
    }

    val_rng = np.random.default_rng(cfg["seed"])
    val_pairs = val_sampler.draw(cfg["train"]["windows_per_eval_set"], val_rng)
    val_ds = WindowDataset(feat, opens, closes, val_pairs)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate, **loader_kwargs
    )

    model = CVAEInpainting(**cfg["model"], in_channels=dp.N_CHANNELS).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_val_recon_loss = float("inf")
    ckpt_path = Path(cfg["output"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    max_epochs = cfg["train"]["max_epochs"]
    # Mean-only warmup (Nix & Weigend 1994's classic mitigation for the "inflate variance
    # instead of improving the mean" pathology): train in "mse" mode for the first
    # nll_warmup_epochs even when reconstruction="nll" is configured, so the mean head
    # gets a head start before the loss ever has a cheaper escape hatch via logvar.
    # Irrelevant (no-op) when reconstruction="mse", since effective_reconstruction is
    # already "mse" every epoch in that case regardless of this value.
    nll_warmup_epochs = cfg["loss"].get("nll_warmup_epochs", 0)

    for epoch in range(max_epochs):
        t0 = time.time()
        beta = kl_beta_schedule(epoch, max_epochs, cfg["loss"]["kl_cycles"], cfg["loss"]["kl_ramp_fraction"])
        effective_reconstruction = "mse" if (reconstruction == "nll" and epoch < nll_warmup_epochs) else reconstruction

        # NLL's recon_loss lives on a completely different scale than MSE's (a real
        # log-likelihood vs. a mean squared error -- tens vs. fractions in practice, see
        # the warmup-transition log lines) -- comparing across the mode switch would mean
        # "select on lowest val_recon_loss" always prefers whatever the last warmup (mse)
        # epoch happened to score, silently defeating the entire point of switching to nll.
        # Reset right at the transition so selection only ever compares within one mode.
        if reconstruction == "nll" and epoch == nll_warmup_epochs:
            logger.info("switching from mse warmup to nll -- resetting best_val_recon_loss (scales aren't comparable across modes)")
            best_val_recon_loss = float("inf")

        train_rng = np.random.default_rng(cfg["seed"] + epoch)
        train_pairs = train_sampler.draw(cfg["train"]["train_windows_per_epoch"], train_rng)
        train_ds = WindowDataset(feat, opens, closes, train_pairs)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate, **loader_kwargs
        )

        train_metrics = run_epoch(
            model, train_loader, optimizer, cfg["loss"], beta, price_scale, device, train=True,
            reconstruction=effective_reconstruction,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, cfg["loss"], beta, price_scale, device, train=False,
            reconstruction=effective_reconstruction,
        )

        # Pure forward-pass diagnostics (no gradient/optimizer impact) -- visibility into
        # whether sample() diversity/calibration degrades mid-training, not just after the
        # fact via an offline script run once training finishes. Cheap: k=8 samples over
        # at most 300 of the already-materialized val_pairs, negligible next to this
        # epoch's 20000-window backward passes.
        gen_diag = epoch_diagnostics(model, val_pairs, feat, opens, closes, device)

        logger.info(
            "epoch %d/%d  beta=%.2f  recon_mode=%s  train_recon=%.5f (kl=%.4f dir=%.4f price_std=%.5f vol_std=%.3f)  "
            "val_recon=%.5f (kl=%.4f dir=%.4f price_std=%.5f vol_std=%.3f)  "
            "gen_diversity_ratio=%.3f gen_crps=%.6f gen_var_ratio=%.3f  (%.1fs)",
            epoch + 1, max_epochs, beta, effective_reconstruction,
            train_metrics["recon_loss"], train_metrics["kl_loss"], train_metrics["direction_loss"],
            train_metrics["price_std"], train_metrics["vol_std"],
            val_metrics["recon_loss"], val_metrics["kl_loss"], val_metrics["direction_loss"],
            val_metrics["price_std"], val_metrics["vol_std"],
            gen_diag["gen_diversity_ratio"], gen_diag["gen_crps"], gen_diag["gen_var_ratio"],
            time.time() - t0,
        )

        # Select on reconstruction loss alone, NOT the raw total (recon + beta*kl_loss) --
        # beta cycles between kl_ramp_fraction's low point and 1.0 (see kl_beta_schedule),
        # so the raw total is only comparable *within* the same phase of a cycle. Selecting
        # on it would deterministically prefer whatever epoch has the lowest beta (i.e. the
        # first epoch of each cycle) regardless of how well-trained the model actually is --
        # exactly the failure mode this replaced.
        if val_metrics["recon_loss"] < best_val_recon_loss:
            best_val_recon_loss = val_metrics["recon_loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
            logger.info("  -> saved best checkpoint (val_recon=%.5f) to %s", best_val_recon_loss, ckpt_path)

    logger.info("done. best val_recon=%.5f, checkpoint=%s", best_val_recon_loss, ckpt_path)


if __name__ == "__main__":
    main()
