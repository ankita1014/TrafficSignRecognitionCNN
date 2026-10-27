"""
config.py — Central configuration for the Traffic Sign Recognition project.

All paths are relative to the project root (traffic-sign-cnn/) so the
project is portable across machines.  Edit the values here to override
defaults; the training / evaluation / prediction scripts all import from
this file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset lives OUTSIDE the project folder (one level up, same as downloaded)
DATASET_ROOT   = Path("../traffic_Data")
TRAIN_DIR      = DATASET_ROOT / "DATA"     # 58 class sub-folders
TEST_DIR       = DATASET_ROOT / "TEST"     # flat test images, class in filename
LABELS_CSV     = Path("../labels.csv")     # ClassId,Name

# Output directories (created automatically)
MODELS_DIR     = PROJECT_ROOT / "models"
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"
GRAPHS_DIR     = PROJECT_ROOT / "outputs" / "graphs"
REPORTS_DIR    = PROJECT_ROOT / "outputs" / "reports"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
DATA_DIR       = PROJECT_ROOT / "data"    # dataset summary, split files

# ── Preprocessing ────────────────────────────────────────────────────────
IMG_SIZE       = 32          # paper: resize to 32×32
GRAYSCALE      = True         # paper: RGB → grayscale
EQUALIZE_HIST  = True         # paper: histogram equalization
# Normalisation: map [0,255] → [-1,1]  (paper-inspired)
NORM_LOW       = -1.0
NORM_HIGH      = 1.0

# ── Data augmentation (training only) ────────────────────────────────────
AUG_ROTATION     = 10      # ± degrees
AUG_WIDTH_SHIFT  = 0.10    # fraction of width
AUG_HEIGHT_SHIFT = 0.10   # fraction of height
AUG_ZOOM         = 0.10    # ± fraction
AUG_FILL_MODE    = "nearest"

# ── Training ─────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
VAL_SPLIT       = 0.15       # 15 % of TRAIN for validation (stratified)
BATCH_SIZE      = 64
EPOCHS          = 40         # EarlyStopping will usually halt sooner
LEARNING_RATE   = 1e-3
PATIENCE        = 8          # EarlyStopping patience
LR_PATIENCE     = 4          # ReduceLROnPlateau patience
LR_FACTOR       = 0.5        # new_lr = old_lr * factor
MIN_LR          = 1e-6
DROPOUT_C1      = 0.25       # after first conv-pool block
DROPOUT_C3      = 0.25       # after second conv-pool block
DROPOUT_C5      = 0.25       # after third conv-pool block
DROPOUT_DENSE   = 0.50       # before output layer

# ── Model / checkpoint names ─────────────────────────────────────────────
BEST_MODEL_NAME = "enhanced_lenet5_best.keras"
FINAL_MODEL_NAME = "enhanced_lenet5_final.keras"
BASELINE_BEST   = "baseline_lenet5_best.keras"
BASELINE_FINAL  = "baseline_lenet5_final.keras"
HISTORY_NAME    = "training_history.json"
SPLIT_NAME      = "data_split.json"
CLASS_MAP_NAME  = "class_mapping.json"
SUMMARY_NAME    = "dataset_summary.json"

# ── Convenience helpers ───────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create every output directory if it does not exist."""
    for d in (MODELS_DIR, OUTPUTS_DIR, GRAPHS_DIR, REPORTS_DIR,
              PREDICTIONS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def label_map_path() -> Path:
    return DATA_DIR / CLASS_MAP_NAME


def summary_path() -> Path:
    return DATA_DIR / SUMMARY_NAME


def split_path() -> Path:
    return DATA_DIR / SPLIT_NAME


def best_model_path(baseline: bool = False) -> Path:
    return MODELS_DIR / (BASELINE_BEST if baseline else BEST_MODEL_NAME)


def final_model_path(baseline: bool = False) -> Path:
    return MODELS_DIR / (BASELINE_FINAL if baseline else FINAL_MODEL_NAME)


def history_path(baseline: bool = False) -> Path:
    name = f"history_{'baseline' if baseline else 'enhanced'}.json"
    return OUTPUTS_DIR / name
