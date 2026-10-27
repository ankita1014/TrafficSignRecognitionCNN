"""
train.py — Training pipeline for the Enhanced LeNet-5 CNN.

Usage:
  python -m src.train                      # train enhanced model
  python -m src.train --baseline            # train baseline LeNet-5
  python -m src.train --epochs 60           # override epochs
  python -m src.train --batch_size 32       # override batch size

Steps:
  1. Load / create the data split.
  2. Build the model.
  3. Train with augmentation + callbacks.
  4. Save best model, final model, and training history.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.utils import get_logger, set_seed, save_json, load_json  # noqa: E402
from src.dataset import load_split  # noqa: E402
from src.preprocessing import load_all, load_batch  # noqa: E402
from src.model import build_enhanced_lenet5, build_baseline_lenet5, get_callbacks  # noqa: E402

logger = get_logger("train")


def train_model(baseline: bool = False,
                epochs: int = config.EPOCHS,
                batch_size: int = config.BATCH_SIZE) -> dict:
    """
    Full training run.  Returns a dict with measured metrics and timing.
    """
    config.ensure_dirs()
    set_seed(config.RANDOM_SEED)

    # ── 1. Data split ────────────────────────────────────────────────────
    split = load_split()
    num_classes = split["num_classes"]
    train_paths  = np.array(split["train"]["paths"])
    train_labels = np.array(split["train"]["labels"])
    val_paths    = np.array(split["val"]["paths"])
    val_labels   = np.array(split["val"]["labels"])

    logger.info("Training samples   : %d", len(train_paths))
    logger.info("Validation samples  : %d", len(val_paths))
    logger.info("Number of classes   : %d", num_classes)

    # ── 2. Preload validation into RAM (no augmentation) ─────────────────
    logger.info("Preloading validation images...")
    X_val, y_val = load_all(val_paths, val_labels)
    logger.info("Validation data ready: X=%s y=%s", X_val.shape, y_val.shape)

    # ── 3. Build model ───────────────────────────────────────────────────
    if baseline:
        model = build_baseline_lenet5(num_classes)
        tag = "baseline"
    else:
        model = build_enhanced_lenet5(num_classes)
        tag = "enhanced"

    model.summary()

    # ── 4. Training generator (augmented) ────────────────────────────────
    train_gen = load_batch(train_paths, train_labels,
                            batch_size=batch_size,
                            augment=True,
                            shuffle=True)

    steps_per_epoch = max(1, len(train_paths) // batch_size)

    # ── 5. Callbacks ─────────────────────────────────────────────────────
    best_path  = config.best_model_path(baseline=baseline)
    final_path = config.final_model_path(baseline=baseline)
    cb_list = get_callbacks(best_path, baseline=baseline)

    # ── 6. Train ─────────────────────────────────────────────────────────
    logger.info("Starting training: epochs=%d batch_size=%d lr=%s",
                epochs, batch_size, config.LEARNING_RATE)
    t0 = time.time()

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=cb_list,
        verbose=1,
    )

    train_duration = time.time() - t0
    logger.info("Training completed in %.1f seconds (%.1f min)",
                train_duration, train_duration / 60)

    # ── 7. Save final model ──────────────────────────────────────────────
    model.save(final_path)
    logger.info("Final model saved → %s", final_path)

    # ── 8. Save history ──────────────────────────────────────────────────
    hist_dict = {
        "history": history.history,
        "train_duration_sec": round(train_duration, 2),
        "epochs_run": len(history.history["loss"]),
        "config": {
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "learning_rate": config.LEARNING_RATE,
            "val_split": config.VAL_SPLIT,
            "random_seed": config.RANDOM_SEED,
        },
    }
    hist_path = config.OUTPUTS_DIR / f"history_{tag}.json"
    save_json(hist_dict, hist_path)
    logger.info("History saved → %s", hist_path)

    return hist_dict


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced LeNet-5 for traffic sign recognition")
    parser.add_argument("--baseline", action="store_true",
                        help="Train the baseline LeNet-5 instead of enhanced")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Number of epochs (default: {config.EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    args = parser.parse_args()

    result = train_model(baseline=args.baseline,
                         epochs=args.epochs,
                         batch_size=args.batch_size)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs run         : {result['epochs_run']}")
    print(f"Training duration  : {result['train_duration_sec']:.1f} sec")
    final_acc = result["history"].get("val_accuracy", result["history"].get("val_acc", [0]))[-1]
    print(f"Final val accuracy : {final_acc:.4f}")


if __name__ == "__main__":
    main()
