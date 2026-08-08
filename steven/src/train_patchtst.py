"""Train the PatchTST benchmark.

Usage:
    python steven/src/train_patchtst.py --config steven/configs/patchtst.yaml
    python steven/src/train_patchtst.py --config steven/configs/patchtst.yaml \\
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
from src.losses import weighted_mse_loss
from src.models.patchtst import PatchTST
from src.data_pipeline import to_patchtst_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="steven/configs/patchtst.yaml")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--train-windows-per-epoch", type=int, default=None)
    p.add_argument("--windows-per-eval-set", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def collate(batch: list[dict]) -> dict:
    context = torch.stack(
        [torch.from_numpy(to_patchtst_input(b["masked_tensor"].numpy())[0]) for b in batch]
    )
    patch_pad = torch.stack(
        [torch.from_numpy(to_patchtst_input(b["masked_tensor"].numpy())[1]) for b in batch]
    )
    y = torch.stack([b["y"] for b in batch])
    return {"context": context, "patch_key_padding_mask": patch_pad, "y": y}


def run_epoch(model, loader, optimizer, loss_cfg, device, train: bool) -> dict:
    model.train(mode=train)
    total, price_total, vol_total, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        context = batch["context"].to(device)
        patch_pad = batch["patch_key_padding_mask"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            price, volume = model(context, patch_pad)
            loss, parts = weighted_mse_loss(price, volume, y, loss_cfg["w_price"], loss_cfg["w_vol"])

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = y.shape[0]
        total += loss.item() * bs
        price_total += parts["price_loss"] * bs
        vol_total += parts["vol_loss"] * bs
        n += bs

    return {"loss": total / n, "price_loss": price_total / n, "vol_loss": vol_total / n}


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

    # Opt-in, same pattern as train_cvae.py -- configs/patchtst.yaml has no
    # `momentum_features` key, so build_dataset's original 7-channel behavior is exactly
    # preserved for every existing config.
    momentum_cfg = cfg.get("momentum_features")
    if momentum_cfg and momentum_cfg.get("enabled"):
        df, bounds, stats, momentum_stats = mp.build_momentum_dataset(cfg["data_path"], momentum_cfg["vix_data_path"])
        logger.info(
            "momentum features enabled: ema_cross/trend_position/rsi/vix, N_FEATURE_CHANNELS=%d",
            dp.N_FEATURE_CHANNELS,
        )
    else:
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

    model = PatchTST(**cfg["model"], n_feature_channels=dp.N_FEATURE_CHANNELS).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_val_loss = float("inf")
    ckpt_path = Path(cfg["output"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["max_epochs"]):
        t0 = time.time()
        train_rng = np.random.default_rng(cfg["seed"] + epoch)
        train_pairs = train_sampler.draw(cfg["train"]["train_windows_per_epoch"], train_rng)
        train_ds = WindowDataset(feat, opens, closes, train_pairs)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate, **loader_kwargs
        )

        train_metrics = run_epoch(model, train_loader, optimizer, cfg["loss"], device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, cfg["loss"], device, train=False)

        logger.info(
            "epoch %d/%d  train_loss=%.5f  val_loss=%.5f  (%.1fs)",
            epoch + 1, cfg["train"]["max_epochs"],
            train_metrics["loss"], val_metrics["loss"], time.time() - t0,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
            logger.info("  -> saved best checkpoint (val_loss=%.5f) to %s", best_val_loss, ckpt_path)

    logger.info("done. best val_loss=%.5f, checkpoint=%s", best_val_loss, ckpt_path)


if __name__ == "__main__":
    main()
