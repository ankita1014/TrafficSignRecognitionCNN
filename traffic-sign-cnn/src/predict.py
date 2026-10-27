"""
predict.py — Single-image prediction with the trained Enhanced LeNet-5.

Usage:
  python -m src.predict --image path/to/image.png
  python -m src.predict --image path/to/image.png --baseline
  python -m src.predict --gui          # open a Windows file dialog

For each prediction it displays:
  - input image (preprocessing preview)
  - predicted class ID and name
  - confidence score
  - top-3 predicted classes with scores
  - saves a result figure to outputs/predictions/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.utils import get_logger, load_class_mapping  # noqa: E402
from src.preprocessing import load_image, preprocess_single  # noqa: E402

logger = get_logger("predict")


def predict_image(image_path: str, baseline: bool = False, show_plot: bool = False) -> dict:
    """
    Run prediction on a single image file.

    Returns a dict with class_id, class_name, confidence, top3.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    img_path = Path(image_path)
    if not img_path.exists():
        msg = f"Image file not found: {image_path}"
        logger.error(msg)
        return {"error": msg}

    best_path = config.best_model_path(baseline=baseline)
    if not best_path.exists():
        msg = (f"Trained model not found: {best_path}\n"
               "Train the model first:  python -m src.train" +
               (" --baseline" if baseline else ""))
        logger.error(msg)
        return {"error": msg}

    # ── Load model ────────────────────────────────────────────────────────
    model = tf.keras.models.load_model(best_path)

    # ── Preprocess (identical to training) ────────────────────────────────
    raw_img = load_image(str(img_path))
    processed = preprocess_single(raw_img)
    X = np.expand_dims(processed, axis=0)  # (1, 32, 32, 1)

    # ── Predict ───────────────────────────────────────────────────────────
    preds = model.predict(X, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    # ── Top-3 ─────────────────────────────────────────────────────────────
    top3_idx = np.argsort(preds)[::-1][:3]
    mapping = load_class_mapping()

    top3 = []
    for idx in top3_idx:
        top3.append({
            "class_id": int(idx),
            "class_name": mapping.get(int(idx), f"Class {int(idx)}"),
            "score": round(float(preds[idx]), 6),
        })

    class_name = mapping.get(class_id, f"Class {class_id}")

    result = {
        "image_path": str(img_path),
        "predicted_class_id": class_id,
        "predicted_class_name": class_name,
        "confidence": round(confidence, 6),
        "top3": top3,
    }

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PREDICTION RESULT")
    print("=" * 60)
    print(f"Image         : {img_path.name}")
    print(f"Predicted ID  : {class_id}")
    print(f"Predicted name: {class_name}")
    print(f"Confidence    : {confidence:.4f}")
    print(f"Top-3:")
    for i, t in enumerate(top3, 1):
        print(f"  {i}. [{t['class_id']:>3}] {t['class_name']:<35} {t['score']:.4f}")
    print("=" * 60)

    # ── Save figure ───────────────────────────────────────────────────────
    config.ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input: {img_path.name}")
    axes[0].axis("off")
    # Preprocessed (denormalized for display)
    proc_display = (processed[:, :, 0] + 1) / 2 * 255
    axes[1].imshow(proc_display.astype(np.uint8), cmap="gray")
    axes[1].set_title(f"Preprocessed → {class_name}\n({confidence:.2%})")
    axes[1].axis("off")
    plt.tight_layout()
    out_name = f"pred_{img_path.stem}.png"
    out_path = config.PREDICTIONS_DIR / out_name
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info("Prediction figure saved → %s", out_path)

    if show_plot:
        plt.show()

    return result


def predict_with_dialog():
    """Open a Windows file dialog to choose an image, then predict."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter is not available. Use: python -m src.predict --image <path>")
        return

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a traffic sign image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.ppm"),
                   ("All files", "*.*")]
    )
    root.destroy()

    if not file_path:
        print("No file selected.")
        return

    predict_image(file_path)


def main():
    parser = argparse.ArgumentParser(description="Predict a single traffic sign image")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to the image file to classify")
    parser.add_argument("--baseline", action="store_true",
                        help="Use baseline model instead of enhanced")
    parser.add_argument("--gui", action="store_true",
                        help="Open a file dialog to choose the image")
    parser.add_argument("--show", action="store_true",
                        help="Display the prediction figure on screen")
    args = parser.parse_args()

    if args.gui:
        predict_with_dialog()
    elif args.image:
        predict_image(args.image, baseline=args.baseline, show_plot=args.show)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
