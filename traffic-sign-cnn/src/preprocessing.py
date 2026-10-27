"""
preprocessing.py — Reproducible image preprocessing pipeline.

Pipeline (matches the research paper's methodology):
  1. Resize to 32×32
  2. Convert RGB → grayscale
  3. Apply histogram equalization (CLAHE)
  4. Normalize to [-1, 1]

Augmentation (training only): rotation, width/height shift, zoom.

The same `preprocess_single()` function is used during training AND
single-image inference so preprocessing is always identical.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


# ── Core preprocessing for a single image ────────────────────────────────
def preprocess_single(image: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to one image array (uint8, H×W×C or H×W).

    Returns float32 array of shape (32, 32, 1) normalised to [-1, 1].
    """
    # 1. Resize
    if image.shape[:2] != (config.IMG_SIZE, config.IMG_SIZE):
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE),
                           interpolation=cv2.INTER_AREA)

    # 2. Grayscale
    if config.GRAYSCALE and image.ndim == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 3. Histogram equalization (CLAHE — adaptive, robust to lighting)
    if config.EQUALIZE_HIST:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        image = clahe.apply(image)

    # 4. Normalize  [0,255] → [-1, 1]
    image = image.astype(np.float32)
    image = (image / 255.0) * (config.NORM_HIGH - config.NORM_LOW) + config.NORM_LOW

    # Add channel dimension: (32, 32, 1)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    return image


def load_image(path: str) -> np.ndarray:
    """Load an image file as RGB uint8 numpy array."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def preprocess_file(path: str) -> np.ndarray:
    """Load + preprocess in one step.  Returns (32, 32, 1) float32."""
    img = load_image(path)
    return preprocess_single(img)


# ── Batch loader ──────────────────────────────────────────────────────────
def load_batch(paths, labels, batch_size, augment=False, shuffle=True):
    """
    Generator that yields (X_batch, y_batch).

    Parameters
    ----------
    paths   : list / array of file paths
    labels  : list / array of integer labels
    augment : if True, apply random augmentation (training only)
    shuffle : if True, reshuffle each epoch
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    paths  = np.array(paths)
    labels = np.array(labels)

    # We precompute all preprocessed images once (dataset fits in RAM).
    n = len(paths)
    X = np.zeros((n, config.IMG_SIZE, config.IMG_SIZE, 1), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    for i, p in enumerate(paths):
        X[i] = preprocess_file(p)

    if augment:
        datagen = ImageDataGenerator(
            rotation_range=config.AUG_ROTATION,
            width_shift_range=config.AUG_WIDTH_SHIFT,
            height_shift_range=config.AUG_HEIGHT_SHIFT,
            zoom_range=config.AUG_ZOOM,
            fill_mode=config.AUG_FILL_MODE,
        )
        # We iterate forever
        idx = 0
        while True:
            if shuffle:
                perm = np.random.permutation(n)
                X_shuf = X[perm]
                y_shuf = y[perm]
            else:
                X_shuf = X
                y_shuf = y
            # Generate augmented batches
            for batch_x_aug, batch_y in datagen.flow(X_shuf, y_shuf,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle):
                yield batch_x_aug, batch_y
    else:
        idx = 0
        while True:
            if shuffle:
                perm = np.random.permutation(n)
                X_shuf = X[perm]
                y_shuf = y[perm]
            else:
                X_shuf = X
                y_shuf = y
            # Yield raw (non-augmented) batches
            for start in range(0, n, batch_size):
                yield X_shuf[start:start + batch_size], y_shuf[start:start + batch_size]


def load_all(paths, labels) -> Tuple[np.ndarray, np.ndarray]:
    """Load all images into memory (used for val / test evaluation)."""
    n = len(paths)
    X = np.zeros((n, config.IMG_SIZE, config.IMG_SIZE, 1), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    for i, p in enumerate(paths):
        X[i] = preprocess_file(p)
    return X, y


# ── Augmentation config helper ───────────────────────────────────────────
def get_augmentation_config() -> dict:
    return {
        "rotation_range": config.AUG_ROTATION,
        "width_shift_range": config.AUG_WIDTH_SHIFT,
        "height_shift_range": config.AUG_HEIGHT_SHIFT,
        "zoom_range": config.AUG_ZOOM,
        "fill_mode": config.AUG_FILL_MODE,
    }


if __name__ == "__main__":
    # Quick self-test
    import glob
    sample = glob.glob(str(PROJECT_ROOT / config.TRAIN_DIR / "0" / "*.png"))[:3]
    for p in sample:
        out = preprocess_file(p)
        print(f"{Path(p).name}: shape={out.shape}, "
              f"min={out.min():.3f}, max={out.max():.3f}, dtype={out.dtype}")
