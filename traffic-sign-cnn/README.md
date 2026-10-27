# Traffic Sign Recognition — Enhanced LeNet-5 CNN

A complete deep-learning project that implements, trains, evaluates, and tests a
CNN model for traffic sign recognition, based on the research paper:

> **A Lightweight Network Architecture for Traffic Sign Recognition Based on Enhanced LeNet-5 Network**
> Yuan An, Chunyu Yang, Shuo Zhang — *Frontiers in Neuroscience*, 2024
> DOI: [10.3389/fnins.2024.1431033](https://doi.org/10.3389/fnins.2024.1431033)

---

## 1. Problem Statement

Traffic sign recognition is a core component of Advanced Driver Assistance
Systems (ADAS).  The goal is to classify an image of a traffic sign into one
of N categories.  The referenced paper proposes an enhanced LeNet-5
architecture that is lightweight (suitable for edge deployment) yet achieves
high accuracy on the GTSRB benchmark.

This project adapts that architecture to a different dataset (58-class Kaggle
Traffic Sign Dataset) and provides a fully runnable training, evaluation, and
inference pipeline.

---

## 2. Project Objectives

- Implement an enhanced LeNet-5 CNN with PReLU activations, dropout, and three
  convolutional blocks.
- Implement a traditional LeNet-5 baseline for comparison.
- Preprocess images (resize → grayscale → histogram equalization → normalize
  to [-1, 1]) identically during training and inference.
- Train on CPU with reproducible settings, EarlyStopping, and checkpointing.
- Evaluate on a held-out test set and generate comprehensive metrics and graphs.
- Provide a single-image prediction script with a Windows file dialog option.

---

## 3. Dataset

**Source:** Kaggle — "Traffic Sign Dataset - Classification"

**Actual statistics (verified programmatically):**

| Metric | Value |
|---|---|
| Classes in `labels.csv` | 58 (IDs 0–57) |
| Training classes (DATA/) | 58 |
| Training images | 4,170 |
| Test classes (TEST/) | 54 |
| Test images | 1,994 |
| Min images / class (train) | 2 (class 25) |
| Max images / class (train) | 446 (class 28) |
| Classes missing from TEST | 9, 18, 19, 33 |
| Test mirror images (`_1_`) | 997 |

**Class encoding:** Zero-based (0–57).  `labels.csv` has columns `ClassId, Name`.

**Folder structure:**
```
traffic_Data/
  DATA/          ← 58 sub-folders (0/, 1/, ... 57/) with training PNGs
  TEST/          ← flat folder; class ID = leading digits of filename
labels.csv       ← ClassId, Name
```

**Limitations:**
- Severe class imbalance (2–446 images per class).
- 9 classes with fewer than 10 training images.
- 4 classes absent from the test set.
- Many "Unknown" placeholder classes (40–43, 45, 49, 52, 56, 57).
- TEST folder contains mirrored copies (filenames with `_1_`).

---

## 4. Preprocessing Methodology

Following the paper's pipeline:

1. **Resize** to 32 × 32 pixels (OpenCV `INTER_AREA`).
2. **Grayscale** conversion (RGB → single channel).
3. **Histogram equalization** using CLAHE (adaptive, clipLimit=2.0, tileGridSize=4×4).
   CLAHE is used instead of global equalization for better robustness to
   varying lighting — the principle is the same.
4. **Normalization:** `(pixel / 255.0) * (1 - (-1)) + (-1)` → range **[-1, 1]**.
   This maps 0 → -1 and 255 → 1, which is mathematically correct.
5. **Augmentation** (training only): rotation ±10°, width/height shift ±10%,
   zoom ±10%, fill mode `nearest`.  Never applied to validation or test images.

The same `preprocess_single()` function is used during training and
single-image inference, guaranteeing identical preprocessing.

**Limitation:** Grayscale conversion discards color information that is
meaningful for traffic signs (e.g., red prohibition signs vs. blue mandatory
signs).  This is a known trade-off of the paper's approach.

---

## 5. CNN Architecture

### 5.1 Enhanced LeNet-5 (main model)

| Layer | Operation | Output Shape |
|---|---|---|
| Input | — | 32 × 32 × 1 |
| C1 | Conv2D(32, 3×3, padding=same) + PReLU | 32 × 32 × 32 |
| Pool | MaxPool2D(2×2) | 16 × 16 × 32 |
| Drop | Dropout(0.25) | 16 × 16 × 32 |
| C3 | Conv2D(64, 3×3, padding=same) + PReLU | 16 × 16 × 64 |
| Pool | MaxPool2D(2×2) | 8 × 8 × 64 |
| Drop | Dropout(0.25) | 8 × 8 × 64 |
| C5 | Conv2D(64, 3×3, padding=same) + PReLU | 8 × 8 × 64 |
| Pool | MaxPool2D(2×2) | 4 × 4 × 64 |
| Drop | Dropout(0.25) | 4 × 4 × 64 |
| Flatten | — | 1024 |
| Dense | 512 + PReLU | 512 |
| Drop | Dropout(0.50) | 512 |
| Output | Dense(58, softmax) | 58 |

**Note on flattened dimension:** The paper reports a 576-element flatten
vector, but with `padding='same'` and 3×3 kernels on a 32×32 input through
three 2×2 pooling stages, the mathematically consistent result is
4×4×64 = **1024**.  This is verified programmatically by
`src/model.py:verify_shapes()`.

**Adaptation note:** The output layer has **58 neurons** (not 43 as in the
original GTSRB paper), because the Kaggle dataset has 58 classes.

### 5.2 Baseline LeNet-5 (comparison)

| Layer | Operation | Output Shape |
|---|---|---|
| Input | — | 32 × 32 × 1 |
| C1 | Conv2D(6, 5×5, same) + ReLU | 32 × 32 × 6 |
| Pool | AvgPool2D(2×2) | 16 × 16 × 6 |
| C3 | Conv2D(16, 5×5, valid) + ReLU | 12 × 12 × 16 |
| Pool | AvgPool2D(2×2) | 6 × 6 × 16 |
| Flatten | — | 576 |
| Dense | 120 + ReLU | 120 |
| Dense | 84 + ReLU | 84 |
| Output | Dense(58, softmax) | 58 |

---

## 6. Training Methodology

- **Loss:** Sparse categorical cross-entropy (integer labels).
- **Optimizer:** Adam, initial learning rate 1×10⁻³.
- **Metric:** Accuracy.
- **Batch size:** 64 (configurable).
- **Epochs:** 40 max (EarlyStopping usually halts sooner).
- **Callbacks:**
  - EarlyStopping (monitor `val_loss`, patience 8, restore best weights).
  - ModelCheckpoint (save best model on `val_loss`).
  - ReduceLROnPlateau (factor 0.5, patience 4, min_lr 1×10⁻⁶).
- **Random seed:** 42 (applied to Python, NumPy, and TensorFlow).
- **Data split:** 15% of DATA/ as validation (stratified, per-class).  TEST/
  preserved as held-out test set.  Rare classes (≤4 images) kept entirely in
  training to avoid empty strata.
- **Hardware:** CPU only (no GPU required).

The paper does not specify exact training hyperparameters (batch size,
learning rate, epochs), so the defaults above are reasonable choices for an
ordinary Windows laptop, not a reproduction of the paper's settings.

---

## 7. Evaluation Metrics

The evaluation script (`src/evaluate.py`) computes and saves:

- Test accuracy and test loss.
- Macro and weighted precision, recall, F1-score.
- Per-class classification report (CSV).
- Confusion matrix (PNG + CSV).
- Per-class accuracy bar chart (PNG).
- Training vs validation accuracy/loss curves (PNG).
- Sample correctly and incorrectly classified images (PNG).
- Model parameter count, model file size, inference latency.

Macro-averaged metrics are reported alongside overall accuracy to account for
class imbalance.  The test set is never used for model selection.

---

## 8. Installation and Execution

### 8.1 Prerequisites

- Windows 10/11
- Python 3.13 (already installed)
- Dataset downloaded and placed at workspace root:
  ```
  <workspace>/
    traffic_Data/
      DATA/
      TEST/
    labels.csv
    traffic-sign-cnn/     ← this project
  ```

### 8.2 Setup (run once)

Open **Command Prompt** and run:

```cmd
cd traffic-sign-cnn

:: Activate the virtual environment
venv\Scripts\activate

:: Install dependencies (already done, but for reference)
python -m pip install -r requirements.txt

:: Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### 8.3 Step-by-step commands

All commands below assume you are in `traffic-sign-cnn/` with the venv active.

```cmd
:: 1. Inspect the dataset and create the data split
python -m src.dataset

:: 2. Build and verify the model architecture
python -m src.model

:: 3. Train the enhanced model
python -m src.train

:: 4. (Optional) Train the baseline model
python -m src.train --baseline

:: 5. Evaluate the enhanced model on the test set
python -m src.evaluate

:: 6. (Optional) Evaluate the baseline model
python -m src.evaluate --baseline

:: 7. Predict a single image
python -m src.predict --image ..\traffic_Data\TEST\000_0001_j.png

:: 8. Predict with a Windows file dialog
python -m src.predict --gui
```

### 8.4 Override training parameters

```cmd
python -m src.train --epochs 60 --batch_size 32
```

---

## 9. Output Files

All outputs are generated under `traffic-sign-cnn/`:

```
models/
  enhanced_lenet5_best.keras       ← best checkpoint (lowest val_loss)
  enhanced_lenet5_final.keras      ← final model after training
  baseline_lenet5_best.keras       ← baseline best checkpoint
  baseline_lenet5_final.keras      ← baseline final model

outputs/
  history_enhanced.json            ← training history + duration
  history_baseline.json
  model_summary_enhanced.txt       ← model.summary() text
  model_summary_baseline.txt
  layer_shapes_enhanced.json       ← verified output shapes per layer
  layer_shapes_baseline.json
  graphs/
    confusion_matrix_enhanced.png
    confusion_matrix_baseline.png
    per_class_accuracy_enhanced.png
    per_class_accuracy_baseline.png
    training_curves_enhanced.png
    training_curves_baseline.png
    correct_samples_enhanced.png
    incorrect_samples_enhanced.png
    correct_samples_baseline.png
    incorrect_samples_baseline.png
  reports/
    classification_report_enhanced.csv
    classification_report_baseline.csv
    confusion_matrix_enhanced.csv
    confusion_matrix_baseline.csv
    per_class_accuracy_enhanced.csv
    per_class_accuracy_baseline.csv
    evaluation_results_enhanced.json
    evaluation_results_baseline.json
  predictions/
    pred_<image_name>.png          ← prediction result figures

data/
  dataset_summary.json             ← verified dataset statistics
  class_mapping.json               ← {ClassId: ClassName}
  data_split.json                  ← train/val/test file lists + labels
```

---

## 10. Results

### Reference (original paper, GTSRB)

The paper reports **97.53%** test accuracy for the enhanced LeNet-5 on the
43-class GTSRB benchmark.  This is a **reference point only** — our dataset is
different (58 classes, different image distribution, severe imbalance).

### Measured results (this project)

> Results are filled in after running `python -m src.evaluate`.
> See `outputs/reports/evaluation_results_enhanced.json` for actual numbers.
> No metrics are fabricated — all values are measured from a real training run.

---

## 11. Limitations and Future Improvements

**Limitations:**
- Grayscale preprocessing discards color, which is significant for traffic
  sign classification.
- Severe class imbalance (2–446 images/class) limits performance on rare
  classes.  Classes with ≤4 images have no validation representation.
- The TEST set contains mirrored copies (`_1_` files), which may inflate
  measured accuracy for those classes.
- 4 classes are entirely absent from the test set.
- Training on CPU is slower than GPU; large epoch counts are impractical.
- The flattened dimension (1024) differs from the paper's reported 576 due to
  padding choices.  Using `padding='valid'` would match 576 but changes the
  architecture.  The `same` padding choice is more standard and stable.

**Future improvements:**
- Train a 3-channel (RGB) variant and compare with the grayscale model.
- Apply class-weighted loss or oversampling for rare classes.
- Use `padding='valid'` in the convolutional layers to match the paper's
  576-element flatten dimension.
- Experiment with transfer learning from a lightweight pretrained backbone.
- Deploy as a TensorFlow Lite model for edge inference.

---

## 12. Project Structure

```
traffic-sign-cnn/
  README.md                    ← this file
  requirements.txt             ← pinned dependencies
  .gitignore
  config.py                    ← all configuration in one place
  data/
    README.md
  models/                      ← saved .keras models
  outputs/
    graphs/                    ← PNG figures
    reports/                   ← CSV + JSON metrics
    predictions/               ← prediction result figures
  src/
    __init__.py
    dataset.py                 ← inspection + stratified split
    preprocessing.py           ← resize, grayscale, CLAHE, normalize, augment
    model.py                   ← enhanced + baseline LeNet-5 builders
    train.py                   ← training pipeline with callbacks
    evaluate.py                ← full evaluation + graph generation
    predict.py                 ← single-image prediction + file dialog
    utils.py                   ← seeds, logging, JSON I/O, class mapping
```

---

## 13. Research Reference

Yuan An, Chunyu Yang, Shuo Zhang.
"A Lightweight Network Architecture for Traffic Sign Recognition Based on
Enhanced LeNet-5 Network."
*Frontiers in Neuroscience*, 2024. DOI: 10.3389/fnins.2024.1431033

This project is an **adaptation**, not an exact reproduction.  Key differences:
- Different dataset (58 classes vs. 43 in GTSRB).
- `padding='same'` (flattened dimension 1024 vs. paper's 576).
- CLAHE histogram equalization instead of global equalization.
- Hyperparameters are reasonable defaults, not paper-specified values.
