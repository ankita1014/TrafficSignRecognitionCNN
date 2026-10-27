"""
utils.py — shared helpers (seeds, logging, JSON I/O, class mapping).
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str = "tsr") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    return logger


# ── Reproducibility ────────────────────────────────────────────────────────
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass  # TF not yet available at import time


# ── JSON helpers ──────────────────────────────────────────────────────────
def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Class mapping ──────────────────────────────────────────────────────────
def load_labels_csv() -> pd.DataFrame:
    """Load labels.csv → DataFrame with columns ClassId, Name."""
    csv_path = PROJECT_ROOT / config.LABELS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {csv_path}.  "
            "Place it at the workspace root next to the traffic_Data folder."
        )
    df = pd.read_csv(csv_path)
    # Normalise column names just in case
    df.columns = [c.strip() for c in df.columns]
    if "ClassId" not in df.columns or "Name" not in df.columns:
        raise ValueError(
            f"labels.csv must contain 'ClassId' and 'Name' columns. "
            f"Found: {list(df.columns)}"
        )
    df["ClassId"] = df["ClassId"].astype(int)
    return df


def build_class_mapping() -> Dict[int, str]:
    """Return {class_id: class_name} from labels.csv."""
    df = load_labels_csv()
    return dict(zip(df["ClassId"].astype(int), df["Name"].astype(str)))


def save_class_mapping() -> Path:
    mapping = build_class_mapping()
    path = config.label_map_path()
    save_json(mapping, path)
    return path


def load_class_mapping() -> Dict[int, str]:
    path = config.label_map_path()
    if not path.exists():
        save_class_mapping()
    # JSON keys are strings → convert back to int
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def class_name(class_id: int, mapping: Optional[Dict[int, str]] = None) -> str:
    if mapping is None:
        mapping = load_class_mapping()
    return mapping.get(class_id, f"Class {class_id}")
