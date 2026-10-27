"""
model.py — Enhanced LeNet-5 CNN and baseline LeNet-5.

Enhanced architecture (adapted from An, Yang & Zhang 2024):
  Input : 32×32×1 (grayscale, normalised)
  C1    : Conv2D(32, 3×3, padding='same') + PReLU
  Pool  : MaxPool2D(2×2)
  Drop  : Dropout(0.25)
  C3    : Conv2D(64, 3×3, padding='same') + PReLU
  Pool  : MaxPool2D(2×2)
  Drop  : Dropout(0.25)
  C5    : Conv2D(64, 3×3, padding='same') + PReLU
  Pool  : MaxPool2D(2×2)
  Drop  : Dropout(0.25)
  Flatten → Dense(512) + PReLU → Dropout(0.50) → Dense(num_classes, softmax)

Layer output shapes (with padding='same'):
  Input    : 32×32×1
  C1 conv  : 32×32×32   (padding=same, stride=1)
  Pool     : 16×16×32
  C3 conv  : 16×16×64
  Pool     : 8×8×64
  C5 conv  : 8×8×64
  Pool     : 4×4×64
  Flatten  : 4*4*64 = 1024   ← documented below

Note: The original paper reports a 576-element flattened vector with a
slight different kernel/padding scheme.  Using padding='same' with 3×3
kernels and 2×2 max-pooling on a 32×32 input gives 4×4×64=1024, which is
mathematically consistent and verified programmatically below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def build_enhanced_lenet5(num_classes: int) -> tf.keras.Model:
    """
    Build the enhanced LeNet-5 CNN.

    Parameters
    ----------
    num_classes : int — number of output neurons (matches dataset).

    Returns
    -------
    tf.keras.Model (compiled, ready for training)
    """
    model = models.Sequential(name="Enhanced_LeNet5")

    # Input
    model.add(layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1),
                           name="input_32x32_gray"))

    # ── C1 block ──────────────────────────────────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), padding="same", name="C1_conv"))
    model.add(layers.PReLU(name="C1_prelu"))
    model.add(layers.MaxPooling2D((2, 2), name="C1_pool"))
    model.add(layers.Dropout(config.DROPOUT_C1, name="C1_dropout"))

    # ── C3 block ──────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same", name="C3_conv"))
    model.add(layers.PReLU(name="C3_prelu"))
    model.add(layers.MaxPooling2D((2, 2), name="C3_pool"))
    model.add(layers.Dropout(config.DROPOUT_C3, name="C3_dropout"))

    # ── C5 block ──────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same", name="C5_conv"))
    model.add(layers.PReLU(name="C5_prelu"))
    model.add(layers.MaxPooling2D((2, 2), name="C5_pool"))
    model.add(layers.Dropout(config.DROPOUT_C5, name="C5_dropout"))

    # ── Flatten + Dense ──────────────────────────────────────────────────
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(512, name="dense_512"))
    model.add(layers.PReLU(name="dense_prelu"))
    model.add(layers.Dropout(config.DROPOUT_DENSE, name="dense_dropout"))

    # ── Output ───────────────────────────────────────────────────────────
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # ── Compile ──────────────────────────────────────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_lenet5(num_classes: int) -> tf.keras.Model:
    """
    Traditional LeNet-5 baseline for comparison.

    Original LeNet-5 (adapted for 32×32 grayscale):
      Conv2D(6, 5×5, padding='same') + ReLU → AvgPool(2×2)
      Conv2D(16, 5×5, padding='valid') + ReLU → AvgPool(2×2)
      Flatten → Dense(120) + ReLU → Dense(84) + ReLU → Dense(num_classes, softmax)

    Same preprocessing, same training protocol — only architecture differs.
    """
    model = models.Sequential(name="Baseline_LeNet5")

    model.add(layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1),
                           name="input_baseline"))
    model.add(layers.Conv2D(6, (5, 5), padding="same", activation="relu",
                            name="bl_c1"))
    model.add(layers.AveragePooling2D((2, 2), name="bl_p1"))
    model.add(layers.Conv2D(16, (5, 5), padding="valid", activation="relu",
                            name="bl_c3"))
    model.add(layers.AveragePooling2D((2, 2), name="bl_p2"))
    model.add(layers.Flatten(name="bl_flatten"))
    model.add(layers.Dense(120, activation="relu", name="bl_dense120"))
    model.add(layers.Dense(84, activation="relu", name="bl_dense84"))
    model.add(layers.Dense(num_classes, activation="softmax", name="bl_output"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def verify_shapes(num_classes: int) -> dict:
    """
    Build the enhanced model with a dummy forward pass and record every
    layer's output shape.  Returns a dict {layer_name: output_shape}.
    """
    model = build_enhanced_lenet5(num_classes)
    # Trigger a forward pass so all layer shapes are resolved
    dummy = tf.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 1))
    _ = model(dummy)
    shapes = {}
    for layer in model.layers:
        try:
            shapes[layer.name] = str(layer.output.shape)
        except Exception:
            shapes[layer.name] = "unknown"
    return shapes


def get_callbacks(model_path: Path, baseline: bool = False) -> list:
    """Standard callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
    ]


def print_summary(num_classes: int, baseline: bool = False) -> None:
    """Build model, print summary, and save architecture diagram."""
    model = build_baseline_lenet5(num_classes) if baseline else build_enhanced_lenet5(num_classes)
    model.summary()
    # Save text summary
    name = "baseline" if baseline else "enhanced"
    path = config.OUTPUTS_DIR / f"model_summary_{name}.txt"
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    # Save shapes
    shapes = verify_shapes(num_classes)
    from src.utils import save_json
    save_json(shapes, config.OUTPUTS_DIR / f"layer_shapes_{name}.json")
    print(f"\nLayer shapes saved → {path}")
    return model


if __name__ == "__main__":
    config.ensure_dirs()
    # Quick build test with a dummy class count
    n = 58
    print("=" * 60)
    print("ENHANCED LeNet-5")
    print("=" * 60)
    print_summary(n, baseline=False)
    print()
    print("=" * 60)
    print("BASELINE LeNet-5")
    print("=" * 60)
    print_summary(n, baseline=True)
