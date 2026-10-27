"""
dataset.py — Dataset inspection, loading, and stratified train/val/test split.

The Kaggle "Traffic Sign Dataset - Classification" contains:
  traffic_Data/
    DATA/   ← 58 class sub-folders (training images)
    TEST/   ← flat folder of test images (class ID encoded in filename)
  labels.csv  ← ClassId, Name

We treat DATA/ as the training+validation pool and TEST/ as the
held-out test set, preserving the original dataset split to prevent
data leakage.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.utils import (get_logger, save_json, load_json,  # noqa: E402
                       build_class_mapping, save_class_mapping, set_seed)

logger = get_logger("dataset")


# ── 1. Dataset inspection ──────────────────────────────────────────────────
def inspect_dataset() -> dict:
    """
    Walk the dataset directories, count images per class, detect issues,
    and return a summary dict that is saved to data/dataset_summary.json.
    """
    train_dir = PROJECT_ROOT / config.TRAIN_DIR
    test_dir  = PROJECT_ROOT / config.TEST_DIR
    csv_path  = PROJECT_ROOT / config.LABELS_CSV

    summary: dict = {}

    # labels.csv
    labels_df = pd.read_csv(csv_path)
    labels_df.columns = [c.strip() for c in labels_df.columns]
    summary["labels_csv_columns"] = list(labels_df.columns)
    summary["num_classes_in_csv"] = len(labels_df)
    summary["classes_in_csv"] = sorted(labels_df["ClassId"].astype(int).tolist())

    # DATA/ — training images
    train_counts: Dict[int, int] = {}
    corrupt_train: List[str] = []
    if train_dir.exists():
        for sub in sorted(train_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 9999):
            if not sub.is_dir():
                continue
            cid = int(sub.name)
            pngs = [f for f in sub.iterdir() if f.suffix.lower() == ".png"]
            train_counts[cid] = len(pngs)
            # spot-check first 3 images for corruption
            for img_path in pngs[:3]:
                try:
                    from PIL import Image
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception:
                    corrupt_train.append(str(img_path))
    summary["train_dir"] = str(config.TRAIN_DIR)
    summary["num_train_classes"] = len(train_counts)
    summary["train_total_images"] = sum(train_counts.values())
    summary["train_counts_per_class"] = train_counts
    summary["train_min"] = min(train_counts.values()) if train_counts else 0
    summary["train_max"] = max(train_counts.values()) if train_counts else 0
    summary["corrupt_train_samples"] = corrupt_train

    # TEST/ — test images (class ID from filename prefix)
    test_counts: Dict[int, int] = {}
    corrupt_test: List[str] = []
    test_files: List[str] = []
    if test_dir.exists():
        for f in test_dir.iterdir():
            if f.suffix.lower() != ".png":
                continue
            test_files.append(f.name)
            m = re.match(r"^0*(\d+)", f.name)   # strip leading zeros
            if m:
                cid = int(m.group(1))
                test_counts[cid] = test_counts.get(cid, 0) + 1
            try:
                from PIL import Image
                with Image.open(f) as im:
                    im.verify()
            except Exception:
                corrupt_test.append(str(f))
    summary["test_dir"] = str(config.TEST_DIR)
    summary["num_test_classes"] = len(test_counts)
    summary["test_total_images"] = sum(test_counts.values())
    summary["test_counts_per_class"] = test_counts
    summary["corrupt_test_samples"] = corrupt_test

    # Classes present in train but missing from test (and vice-versa)
    train_classes = set(train_counts.keys())
    test_classes  = set(test_counts.keys())
    summary["classes_missing_from_test"] = sorted(train_classes - test_classes)
    summary["classes_only_in_test"]      = sorted(test_classes - train_classes)

    # Duplicate / mirror pattern detection in TEST
    mirrored = [f for f in test_files if "_1_" in f]
    summary["test_mirrored_count"] = len(mirrored)

    # Class imbalance metrics
    counts = list(train_counts.values())
    if counts:
        summary["train_mean_per_class"] = round(float(np.mean(counts)), 1)
        summary["train_std_per_class"]  = round(float(np.std(counts)), 1)
        # Classes with very few samples (< 10)
        summary["rare_classes"] = {k: v for k, v in sorted(train_counts.items()) if v < 10}

    return summary


def save_dataset_summary() -> Path:
    """Inspect and save dataset summary JSON."""
    summary = inspect_dataset()
    path = config.summary_path()
    save_json(summary, path)
    logger.info("Dataset summary saved → %s", path)
    _print_summary(summary)
    return path


def _print_summary(s: dict) -> None:
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info("Classes in labels.csv : %d", s["num_classes_in_csv"])
    logger.info("Training classes       : %d", s["num_train_classes"])
    logger.info("Training images         : %d", s["train_total_images"])
    logger.info("  min / max per class   : %d / %d", s["train_min"], s["train_max"])
    logger.info("Test classes            : %d", s["num_test_classes"])
    logger.info("Test images             : %d", s["test_total_images"])
    logger.info("  mirrored (with _1_)   : %d", s["test_mirrored_count"])
    logger.info("Classes missing in test : %s", s["classes_missing_from_test"])
    if s.get("rare_classes"):
        logger.info("Rare classes (<10 imgs): %s", s["rare_classes"])
    logger.info("=" * 60)


# ── 2. File-list collection ───────────────────────────────────────────────
def collect_train_filepaths() -> Tuple[List[str], List[int]]:
    """Return (paths, labels) for every image in DATA/."""
    train_dir = PROJECT_ROOT / config.TRAIN_DIR
    paths: List[str] = []
    labels: List[int] = []
    for sub in sorted(train_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 9999):
        if not sub.is_dir():
            continue
        cid = int(sub.name)
        for f in sub.iterdir():
            if f.suffix.lower() == ".png":
                paths.append(str(f))
                labels.append(cid)
    return paths, labels


def collect_test_filepaths() -> Tuple[List[str], List[int]]:
    """Return (paths, labels) for every image in TEST/ (class from filename)."""
    test_dir = PROJECT_ROOT / config.TEST_DIR
    paths: List[str] = []
    labels: List[int] = []
    for f in sorted(test_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        m = re.match(r"^0*(\d+)", f.name)
        if m:
            paths.append(str(f))
            labels.append(int(m.group(1)))
    return paths, labels


# ── 3. Stratified train/validation split ──────────────────────────────────
def make_split() -> dict:
    """
    Stratified split of DATA/ into train / validation.
    TEST/ is kept as the held-out test set.

    For classes with < 4 samples we cannot do a standard stratified split
    (sklearn needs at least 2 per class per split).  For those rare classes
    we duplicate-assign: if a class has ≤ 2 images, all go to training
    (no validation samples for that class) to avoid empty strata.
    """
    set_seed(config.RANDOM_SEED)
    paths, labels = collect_train_filepaths()

    paths   = np.array(paths)
    labels  = np.array(labels)

    # Count per class
    unique, counts = np.unique(labels, return_counts=True)
    rare_threshold = 4  # classes with ≤ 4 images need special handling

    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []

    for cid in unique:
        mask = labels == cid
        c_paths  = paths[mask]
        c_labels = labels[mask]
        n = len(c_paths)

        if n <= rare_threshold:
            # Keep all in training; no validation for very rare classes
            train_paths.extend(c_paths.tolist())
            train_labels.extend(c_labels.tolist())
        else:
            p_val = config.VAL_SPLIT
            n_val = max(1, int(round(n * p_val)))
            # shuffle indices deterministically
            idx = np.random.RandomState(config.RANDOM_SEED).permutation(n)
            val_idx   = idx[:n_val]
            train_idx  = idx[n_val:]
            train_paths.extend(c_paths[train_idx].tolist())
            train_labels.extend(c_labels[train_idx].tolist())
            val_paths.extend(c_paths[val_idx].tolist())
            val_labels.extend(c_labels[val_idx].tolist())

    # Shuffle train / val
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    val_paths   = np.array(val_paths)
    val_labels  = np.array(val_labels)

    t_idx = np.random.RandomState(config.RANDOM_SEED).permutation(len(train_paths))
    train_paths  = train_paths[t_idx]
    train_labels = train_labels[t_idx]

    v_idx = np.random.RandomState(config.RANDOM_SEED + 1).permutation(len(val_paths))
    val_paths  = val_paths[v_idx]
    val_labels = val_labels[v_idx]

    # Test set
    test_paths, test_labels = collect_test_filepaths()

    split = {
        "random_seed": config.RANDOM_SEED,
        "val_split_fraction": config.VAL_SPLIT,
        "train": {
            "paths":  train_paths.tolist(),
            "labels": train_labels.tolist(),
            "count":  len(train_paths),
        },
        "val": {
            "paths":  val_paths.tolist(),
            "labels": val_labels.tolist(),
            "count":  len(val_paths),
        },
        "test": {
            "paths":  test_paths,
            "labels": test_labels,
            "count":  len(test_paths),
        },
        "num_classes": int(len(unique)),
    }

    save_json(split, config.split_path())
    logger.info("Split saved → %s", config.split_path())
    logger.info("  Train: %d | Val: %d | Test: %d",
                len(train_paths), len(val_paths), len(test_paths))
    return split


def load_split() -> dict:
    """Load existing split JSON or create a new one."""
    path = config.split_path()
    if path.exists():
        return load_json(path)
    return make_split()


# ── 4. Convenience loaders for generators ────────────────────────────────
def get_split_arrays() -> Tuple[Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray],
                                 Tuple[np.ndarray, np.ndarray]]:
    """
    Return (X_train, y_train), (X_val, y_val), (X_test, y_test) as
    *lists of file paths* (strings) and integer labels.
    Actual image loading happens in preprocessing.py via generators.

    Returns
    -------
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    split = load_split()
    tr = (np.array(split["train"]["paths"]), np.array(split["train"]["labels"]))
    va = (np.array(split["val"]["paths"]),   np.array(split["val"]["labels"]))
    te = (np.array(split["test"]["paths"]),  np.array(split["test"]["labels"]))
    return tr, va, te


if __name__ == "__main__":
    config.ensure_dirs()
    save_class_mapping()
    save_dataset_summary()
    make_split()
