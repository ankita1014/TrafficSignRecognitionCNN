"""
gui_helpers.py — Shared backend logic for the Streamlit GUI.

Wraps model loading (cached), class mapping, preprocessing preview,
and discovery of saved reports / history so the GUI pages can stay
clean and focused on presentation.

Importing this module does NOT import TensorFlow until a model is
actually requested, which keeps the Streamlit nav fast.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root bootstrap ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


# ── 1. Class mapping ─────────────────────────────────────────────────────
def get_class_mapping() -> Dict[int, str]:
    """Return {class_id: class_name} from the saved JSON or labels.csv."""
    try:
        from src.utils import load_class_mapping
        return load_class_mapping()
    except Exception:
        return {}


def num_classes_from_mapping(mapping: Dict[int, str]) -> int:
    return max(mapping.keys()) + 1 if mapping else 0


# ── 2. Model loading (cached) ────────────────────────────────────────────
_model_cache: Dict[str, object] = {}


def load_trained_model(baseline: bool = False):
    """
    Load the trained Keras model once and cache it for the session.
    Returns (model, model_path_str) or (None, error_msg).
    """
    import tensorflow as tf  # deferred import

    tag = "baseline" if baseline else "enhanced"
    if tag in _model_cache:
        return _model_cache[tag], None

    path = config.best_model_path(baseline=baseline)
    if not path.exists():
        return None, f"Model file not found: {path.name}"

    try:
        model = tf.keras.models.load_model(str(path), compile=False)
        _model_cache[tag] = model
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def model_file_info(baseline: bool = False) -> Tuple[bool, str, float]:
    """Return (exists, filename, size_mb)."""
    path = config.best_model_path(baseline=baseline)
    if path.exists():
        return True, path.name, round(path.stat().st_size / (1024 * 1024), 2)
    return False, path.name, 0.0


def model_param_count(model) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    try:
        total = int(model.count_params())
    except Exception:
        total = 0
    try:
        trainable = int(sum(
            np.prod(w.shape) for w in model.trainable_weights
        ))
    except Exception:
        trainable = total
    return trainable, total


def model_input_shape(model) -> Optional[Tuple]:
    try:
        return tuple(model.inputs[0].shape[1:])
    except Exception:
        try:
            return tuple(model.input_shape[1:])
        except Exception:
            return None


def model_output_shape(model) -> Optional[Tuple]:
    try:
        return tuple(model.outputs[0].shape[1:])
    except Exception:
        try:
            return tuple(model.output_shape[1:])
        except Exception:
            return None


def model_layer_table(model) -> List[dict]:
    """Return list of {name, type, output_shape, params} for each layer."""
    rows = []
    # Trigger a forward pass so output shapes are materialised
    in_shape = model_input_shape(model)
    if in_shape is not None:
        import tensorflow as tf
        dummy = tf.zeros((1,) + in_shape)
        try:
            _ = model(dummy)
        except Exception:
            pass
    for layer in model.layers:
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = "—"
        rows.append({
            "Layer": layer.name,
            "Type": type(layer).__name__,
            "Output Shape": shape,
            "Params": layer.count_params(),
        })
    return rows


# ── 3. Preprocessing preview (step-by-step) ─────────────────────────────
def preprocessing_steps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Reproduce the training preprocessing pipeline step by step and
    return every intermediate image so the GUI can display them.

    Returns a dict with keys:
        original, resized, grayscale, equalized, normalized
    """
    import cv2

    steps = {}
    original = image.copy()

    # 1. Resize
    if image.shape[:2] != (config.IMG_SIZE, config.IMG_SIZE):
        resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    steps["original"] = original
    steps["resized"] = resized

    # 2. Grayscale
    if config.GRAYSCALE and resized.ndim == 3 and resized.shape[2] >= 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = resized if resized.ndim == 2 else resized[:, :, 0]
    steps["grayscale"] = gray

    # 3. Histogram equalization (CLAHE)
    if config.EQUALIZE_HIST:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        equalized = clahe.apply(gray)
    else:
        equalized = gray.copy()
    steps["equalized"] = equalized

    # 4. Normalize to [-1, 1]
    norm = equalized.astype(np.float32)
    norm = (norm / 255.0) * (config.NORM_HIGH - config.NORM_LOW) + config.NORM_LOW
    steps["normalized"] = norm  # float32 in [-1, 1]

    return steps


def preprocess_for_inference(image: np.ndarray) -> np.ndarray:
    """Run the full pipeline and return (1, 32, 32, 1) float32 batch."""
    from src.preprocessing import preprocess_single
    proc = preprocess_single(image)
    return np.expand_dims(proc, axis=0)


def run_inference(model, image: np.ndarray, mapping: Dict[int, str]) -> dict:
    """
    Preprocess + predict.  Returns:
        predicted_id, predicted_name, confidence, top3, all_probs
    """
    X = preprocess_for_inference(image)
    preds = model.predict(X, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = []
    for idx in top3_idx:
        top3.append({
            "rank": len(top3) + 1,
            "class_id": int(idx),
            "class_name": mapping.get(int(idx), f"Class {int(idx)}"),
            "score": float(preds[idx]),
        })
    # Re-rank properly
    for i, t in enumerate(top3, 1):
        t["rank"] = i

    return {
        "predicted_id": class_id,
        "predicted_name": mapping.get(class_id, f"Class {class_id}"),
        "confidence": confidence,
        "top3": top3,
        "all_probs": preds.tolist(),
    }


# ── 4. File discovery for reports / history ──────────────────────────────
def load_training_history(baseline: bool = False) -> Optional[dict]:
    path = config.history_path(baseline=baseline)
    if not path.exists():
        return None
    try:
        from src.utils import load_json
        return load_json(path)
    except Exception:
        return None


def load_dataset_summary() -> Optional[dict]:
    path = config.summary_path()
    if not path.exists():
        return None
    try:
        from src.utils import load_json
        return load_json(path)
    except Exception:
        return None


def load_layer_shapes(baseline: bool = False) -> Optional[dict]:
    name = f"layer_shapes_{'baseline' if baseline else 'enhanced'}.json"
    path = config.OUTPUTS_DIR / name
    if not path.exists():
        return None
    try:
        from src.utils import load_json
        return load_json(path)
    except Exception:
        return None


def find_evaluation_reports() -> List[Path]:
    """Return list of evaluation_results_*.json files in reports dir."""
    reports = []
    if config.REPORTS_DIR.exists():
        for f in config.REPORTS_DIR.glob("evaluation_results_*.json"):
            reports.append(f)
    return sorted(reports)


def load_evaluation_report(baseline: bool = False) -> Optional[dict]:
    name = f"evaluation_results_{'baseline' if baseline else 'enhanced'}.json"
    path = config.REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        from src.utils import load_json
        return load_json(path)
    except Exception:
        return None


def find_classification_report_csv(baseline: bool = False) -> Optional[Path]:
    name = f"classification_report_{'baseline' if baseline else 'enhanced'}.csv"
    path = config.REPORTS_DIR / name
    return path if path.exists() else None


def find_confusion_matrix_csv(baseline: bool = False) -> Optional[Path]:
    name = f"confusion_matrix_{'baseline' if baseline else 'enhanced'}.csv"
    path = config.REPORTS_DIR / name
    return path if path.exists() else None


def find_confusion_matrix_png(baseline: bool = False) -> Optional[Path]:
    name = f"confusion_matrix_{'baseline' if baseline else 'enhanced'}.png"
    path = config.GRAPHS_DIR / name
    return path if path.exists() else None


def find_training_curves_png(baseline: bool = False) -> Optional[Path]:
    name = f"training_curves_{'baseline' if baseline else 'enhanced'}.png"
    path = config.GRAPHS_DIR / name
    return path if path.exists() else None


# ── 5. Sample test images ────────────────────────────────────────────────
def sample_test_images(limit: int = 6) -> List[Path]:
    """Return diverse sample test image paths across different classes for quick testing."""
    test_dir = (config.PROJECT_ROOT / config.TEST_DIR).resolve()
    if not test_dir.exists():
        return []
    target_prefixes = ["002", "014", "021", "027", "035", "055"]
    samples: List[Path] = []
    for prefix in target_prefixes:
        matches = [p for p in test_dir.glob(f"{prefix}_*.png") if "_1_" not in p.name]
        if matches:
            samples.append(matches[0])
        if len(samples) >= limit:
            break
    # Fallback to general search if targets were not found
    if len(samples) < limit:
        seen = {s.stem.split("_")[0] for s in samples}
        for p in sorted(test_dir.glob("*.png")):
            cls = p.stem.split("_")[0]
            if cls not in seen and "_1_" not in p.name:
                seen.add(cls)
                samples.append(p)
                if len(samples) >= limit:
                    break
    return samples
