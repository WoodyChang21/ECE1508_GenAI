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

from src.data_pipeline import WindowDataset, WindowSampler, build_dataset, extract_arrays
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


def run_epoch(model, loader, optimizer, loss_cfg, beta, device, train: bool) -> dict:
    model.train(mode=train)
    totals = {"loss": 0.0, "price_loss": 0.0, "vol_loss": 0.0, "kl_loss": 0.0}
    n = 0
    for batch in loader:
        masked_tensor = batch["masked_tensor"].to(device)
        full_tensor = batch["full_tensor"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            price, volume, mu_p, logvar_p, mu_q, logvar_q = model(masked_tensor, full_tensor)
            loss, parts = cvae_loss(
                price, volume, y, mu_q, logvar_q, mu_p, logvar_p,
                loss_cfg["w_price"], loss_cfg["w_vol"], beta, loss_cfg["free_bits"],
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = y.shape[0]
        totals["loss"] += loss.item() * bs
        totals["price_loss"] += parts["price_loss"] * bs
        totals["vol_loss"] += parts["vol_loss"] * bs
        totals["kl_loss"] += parts["kl_loss"] * bs
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

    df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

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

    model = CVAEInpainting(**cfg["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_val_loss = float("inf")
    ckpt_path = Path(cfg["output"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    max_epochs = cfg["train"]["max_epochs"]

    for epoch in range(max_epochs):
        t0 = time.time()
        beta = kl_beta_schedule(epoch, max_epochs, cfg["loss"]["kl_cycles"], cfg["loss"]["kl_ramp_fraction"])

        train_rng = np.random.default_rng(cfg["seed"] + epoch)
        train_pairs = train_sampler.draw(cfg["train"]["train_windows_per_epoch"], train_rng)
        train_ds = WindowDataset(feat, opens, closes, train_pairs)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate, **loader_kwargs
        )

        train_metrics = run_epoch(model, train_loader, optimizer, cfg["loss"], beta, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, cfg["loss"], beta, device, train=False)

        logger.info(
            "epoch %d/%d  beta=%.2f  train_loss=%.5f (kl=%.4f)  val_loss=%.5f (kl=%.4f)  (%.1fs)",
            epoch + 1, max_epochs, beta,
            train_metrics["loss"], train_metrics["kl_loss"],
            val_metrics["loss"], val_metrics["kl_loss"],
            time.time() - t0,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
            logger.info("  -> saved best checkpoint (val_loss=%.5f) to %s", best_val_loss, ckpt_path)

    logger.info("done. best val_loss=%.5f, checkpoint=%s", best_val_loss, ckpt_path)


if __name__ == "__main__":
    main()
