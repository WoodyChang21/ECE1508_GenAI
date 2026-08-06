"""Train the HF PatchTSTModel-backed PatchTST (src/models/patchtst_hf.py).

Replaces the original hand-rolled patchtst.py with the real HF PatchTSTModel backbone
-- see docs/experiments.md for the comparison (the HF version substantially beats the
original on every metric). channel_attention is a CLI-overridable toggle;
channel_attention=False (configs/patchtst.yaml's default) is the main approach for now.

One deliberate simplification vs. the original: fixed context_length=MAX_CONTEXT (70
bars / 10 trading days) windows only, not the original's variable 2-10 day curriculum --
HF PatchTSTModel patchifies one fixed context_length per model. See docs/experiments.md.

Not yet updated for this model: evaluate.py and the long-only backtest still assume the
original patchtst.py's variable-context, padding-mask-aware interface -- that's deferred
until an architecture/channel_attention setting is picked from these training metrics.

Usage:
    python steven/src/train_patchtst.py --config steven/configs/patchtst.yaml
    python steven/src/train_patchtst.py --config steven/configs/patchtst.yaml \\
        --channel-attention --max-epochs 2 --train-windows-per-epoch 200
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import MAX_CONTEXT, N_FEATURE_CHANNELS, WindowSampler, build_dataset, build_window, extract_arrays
from src.losses import weighted_mse_loss
from src.models.patchtst_hf import PatchTSTHF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="steven/configs/patchtst.yaml")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--train-windows-per-epoch", type=int, default=None)
    p.add_argument("--windows-per-eval-set", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--channel-attention", action=argparse.BooleanOptionalAction, default=None,
        help="Overrides configs/patchtst.yaml's model.channel_attention. "
        "--channel-attention or --no-channel-attention.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class FixedContextDataset(Dataset):
    """Same window construction as data_pipeline.py's build_window, but every window
    uses the full MAX_CONTEXT (70 bars) -- see the fixed-context caveat in this module's
    docstring. Since ctx_bars == MAX_CONTEXT exactly, build_window's padding is always
    zero-length, so no padding_mask handling is needed here (unlike the original
    variable-curriculum WindowDataset + to_patchtst_input)."""

    def __init__(self, feat, opens, closes, start_indices):
        self.feat, self.opens, self.closes = feat, opens, closes
        self.start_indices = start_indices

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, i: int) -> dict:
        w = build_window(self.feat, self.opens, self.closes, int(self.start_indices[i]), MAX_CONTEXT)
        context = w["masked_tensor"][:MAX_CONTEXT, :N_FEATURE_CHANNELS]
        return {"past_values": torch.from_numpy(context), "y": torch.from_numpy(w["y"])}


def collate(batch: list[dict]) -> dict:
    return {
        "past_values": torch.stack([b["past_values"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def run_epoch(model, loader, optimizer, loss_cfg, device, train: bool) -> dict:
    model.train(mode=train)
    total, price_total, vol_total, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        past_values = batch["past_values"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train):
            price, volume = model(past_values)
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
    if args.channel_attention is not None:
        cfg["model"]["channel_attention"] = args.channel_attention

    device = resolve_device(cfg["device"])
    logger.info("device: %s, channel_attention: %s", device, cfg["model"]["channel_attention"])

    df, bounds, stats = build_dataset(cfg["data_path"])
    feat, opens, closes = extract_arrays(df)

    train_sampler = WindowSampler(*bounds["train"])
    val_sampler = WindowSampler(*bounds["val"])
    train_starts_full = train_sampler.valid_starts(MAX_CONTEXT)
    val_starts_full = val_sampler.valid_starts(MAX_CONTEXT)

    # train_ds is rebuilt fresh every epoch (new sampled windows), so no persistent_workers.
    loader_kwargs = {
        "num_workers": cfg["train"].get("num_workers", 0),
        "pin_memory": cfg["train"].get("pin_memory", False),
    }

    val_rng = np.random.default_rng(cfg["seed"])
    n_val = min(cfg["train"]["windows_per_eval_set"], len(val_starts_full))
    val_starts = val_rng.choice(val_starts_full, size=n_val, replace=False)
    val_ds = FixedContextDataset(feat, opens, closes, val_starts)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate, **loader_kwargs
    )

    model = PatchTSTHF(context_length=MAX_CONTEXT, **cfg["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_val_loss = float("inf")
    ckpt_path = Path(cfg["output"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["max_epochs"]):
        t0 = time.time()
        train_rng = np.random.default_rng(cfg["seed"] + epoch)
        n_train = min(cfg["train"]["train_windows_per_epoch"], len(train_starts_full))
        train_starts = train_rng.choice(train_starts_full, size=n_train, replace=False)
        train_ds = FixedContextDataset(feat, opens, closes, train_starts)
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
