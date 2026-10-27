"""
evaluate.py — Evaluate the trained model on the held-out TEST set.

Generates and saves:
  - test_accuracy, test_loss
  - precision, recall, F1 (macro + weighted)
  - per-class classification report (CSV)
  - confusion matrix (PNG + CSV)
  - training vs validation accuracy/loss graphs (PNG)
  - per-class accuracy bar chart (PNG)
  - sample correctly & incorrectly classified images (PNG)
  - model summary, parameter count, model file size
  - training duration + inference latency

Usage:
  python -m src.evaluate                     # evaluate enhanced model
  python -m src.evaluate --baseline           # evaluate baseline model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                                precision_recall_fscore_support)

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.utils import get_logger, load_json, load_class_mapping  # noqa: E402
from src.dataset import load_split  # noqa: E402
from src.preprocessing import load_all  # noqa: E402

logger = get_logger("evaluate")


def evaluate_model(baseline: bool = False) -> dict:
    config.ensure_dirs()
    split = load_split()

    # ── Load model ───────────────────────────────────────────────────────
    best_path = config.best_model_path(baseline=baseline)
    if not best_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {best_path}\n"
            "Run training first:  python -m src.train" +
            (" --baseline" if baseline else "")
        )

    logger.info("Loading model: %s", best_path)
    model = tf.keras.models.load_model(best_path)

    # ── Load test data ────────────────────────────────────────────────────
    test_paths  = np.array(split["test"]["paths"])
    test_labels = np.array(split["test"]["labels"])
    logger.info("Test images: %d", len(test_paths))

    logger.info("Preprocessing test images...")
    X_test, y_test = load_all(test_paths, test_labels)

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info("Running inference...")
    t0 = time.time()
    preds = model.predict(X_test, verbose=0)
    infer_time = time.time() - t0
    y_pred = np.argmax(preds, axis=1)

    # ── Metrics ──────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    logger.info("Test accuracy : %.4f", test_acc)
    logger.info("Macro  P/R/F1 : %.4f / %.4f / %.4f", prec, rec, f1)
    logger.info("Weight P/R/F1: %.4f / %.4f / %.4f", prec_w, rec_w, f1_w)
    logger.info("Inference time: %.2f sec for %d images (%.2f ms/img)",
                infer_time, len(X_test), infer_time / len(X_test) * 1000)

    # ── Classification report ─────────────────────────────────────────────
    mapping = load_class_mapping()
    target_names = [mapping.get(i, f"Class {i}") for i in range(split["num_classes"])]
    report = classification_report(y_test, y_pred,
                                   labels=list(range(split["num_classes"])),
                                   target_names=target_names,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    tag = "baseline" if baseline else "enhanced"
    report_csv = config.REPORTS_DIR / f"classification_report_{tag}.csv"
    report_df.to_csv(report_csv)
    logger.info("Classification report -> %s", report_csv)

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=list(range(split["num_classes"])))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_csv = config.REPORTS_DIR / f"confusion_matrix_{tag}.csv"
    cm_df.to_csv(cm_csv)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {tag.capitalize()} LeNet-5")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    cm_png = config.GRAPHS_DIR / f"confusion_matrix_{tag}.png"
    plt.savefig(cm_png, dpi=150)
    plt.close()

    # ── Per-class accuracy ───────────────────────────────────────────────
    per_class_acc = {}
    for i in range(split["num_classes"]):
        mask = y_test == i
        if mask.sum() > 0:
            per_class_acc[i] = float((y_pred[mask] == i).sum() / mask.sum())
        else:
            per_class_acc[i] = None
    acc_df = pd.DataFrame([
        {"class_id": k, "class_name": mapping.get(k, ""), "accuracy": v}
        for k, v in per_class_acc.items()
    ])
    acc_csv = config.REPORTS_DIR / f"per_class_accuracy_{tag}.csv"
    acc_df.to_csv(acc_csv, index=False)

    fig, ax = plt.subplots(figsize=(16, 6))
    valid = [(k, v) for k, v in per_class_acc.items() if v is not None]
    keys, vals = zip(*valid)
    ax.bar(keys, vals, color="steelblue")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy — {tag.capitalize()} LeNet-5")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    acc_png = config.GRAPHS_DIR / f"per_class_accuracy_{tag}.png"
    plt.savefig(acc_png, dpi=150)
    plt.close()

    # ── Training history graphs ──────────────────────────────────────────
    hist_path = config.OUTPUTS_DIR / f"history_{tag}.json"
    if hist_path.exists():
        hist = load_json(hist_path)
        h = hist["history"]
        epochs_range = range(1, len(h["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs_range, h["accuracy"], label="Train")
        ax1.plot(epochs_range, h["val_accuracy"], label="Validation")
        ax1.set_title("Training vs Validation Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax2.plot(epochs_range, h["loss"], label="Train")
        ax2.plot(epochs_range, h["val_loss"], label="Validation")
        ax2.set_title("Training vs Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        plt.suptitle(f"{tag.capitalize()} LeNet-5 Training Curves")
        plt.tight_layout()
        curves_png = config.GRAPHS_DIR / f"training_curves_{tag}.png"
        plt.savefig(curves_png, dpi=150)
        plt.close()

    # ── Sample correct / incorrect images ───────────────────────────────
    correct_idx = np.where(y_pred == y_test)[0]
    incorrect_idx = np.where(y_pred != y_test)[0]
    np.random.seed(config.RANDOM_SEED)
    sample_correct = np.random.choice(correct_idx,
                                       min(12, len(correct_idx)), replace=False)
    sample_incorrect = np.random.choice(incorrect_idx,
                                         min(12, len(incorrect_idx)),
                                         replace=False) if len(incorrect_idx) > 0 else []

    def plot_samples(indices, title, filename):
        from src.preprocessing import load_image, preprocess_single
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        for i, idx in enumerate(indices):
            img = load_image(test_paths[idx])
            axes[i].imshow(img)
            axes[i].axis("off")
            true_name = mapping.get(int(y_test[idx]), f"Class {int(y_test[idx])}")[:25]
            pred_name = mapping.get(int(y_pred[idx]), f"Class {int(y_pred[idx])}")[:25]
            conf = float(np.max(preds[idx]))
            axes[i].set_title(f"T:{true_name}\nP:{pred_name} ({conf:.2f})",
                              fontsize=7)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=120)
        plt.close()

    if len(sample_correct) > 0:
        plot_samples(sample_correct, "Correctly Classified Samples",
                     config.GRAPHS_DIR / f"correct_samples_{tag}.png")
    if len(sample_incorrect) > 0:
        plot_samples(sample_incorrect, "Incorrectly Classified Samples",
                     config.GRAPHS_DIR / f"incorrect_samples_{tag}.png")

    # ── Model summary & stats ─────────────────────────────────────────────
    model_size = best_path.stat().st_size / (1024 * 1024)  # MB
    param_count = model.count_params()

    # ── Save consolidated results JSON ───────────────────────────────────
    results = {
        "model": str(best_path),
        "model_size_mb": round(model_size, 2),
        "param_count": param_count,
        "test_accuracy": round(float(test_acc), 6),
        "test_loss": round(float(test_loss), 6),
        "macro_precision": round(float(prec), 6),
        "macro_recall": round(float(rec), 6),
        "macro_f1": round(float(f1), 6),
        "weighted_precision": round(float(prec_w), 6),
        "weighted_recall": round(float(rec_w), 6),
        "weighted_f1": round(float(f1_w), 6),
        "inference_time_sec": round(float(infer_time), 4),
        "inference_per_image_ms": round(float(infer_time) / len(X_test) * 1000, 4),
        "num_test_images": len(X_test),
        "training_history": str(hist_path),
        "output_files": {
            "classification_report": str(report_csv),
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_png": str(cm_png),
            "per_class_accuracy_csv": str(acc_csv),
            "per_class_accuracy_png": str(acc_png),
        },
    }
    if hist_path.exists():
        results["output_files"]["training_curves_png"] = str(config.GRAPHS_DIR / f"training_curves_{tag}.png")
        results["training_duration_sec"] = load_json(hist_path).get("train_duration_sec")
        results["epochs_run"] = load_json(hist_path).get("epochs_run")

    results_path = config.REPORTS_DIR / f"evaluation_results_{tag}.json"
    from src.utils import save_json
    save_json(results, results_path)
    logger.info("Evaluation results -> %s", results_path)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — {tag.upper()} MODEL")
    print("=" * 60)
    print(f"Test accuracy      : {test_acc:.4f}")
    print(f"Test loss          : {test_loss:.4f}")
    print(f"Macro F1           : {f1:.4f}")
    print(f"Weighted F1        : {f1_w:.4f}")
    print(f"Parameters         : {param_count:,}")
    print(f"Model size         : {model_size:.2f} MB")
    print(f"Inference          : {infer_time:.2f}s ({infer_time/len(X_test)*1000:.2f} ms/img)")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate baseline model instead of enhanced")
    args = parser.parse_args()
    evaluate_model(baseline=args.baseline)


if __name__ == "__main__":
    main()
