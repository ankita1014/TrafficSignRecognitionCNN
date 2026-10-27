"""
app.py  –  Streamlit GUI for Traffic Sign Recognition (Enhanced LeNet-5)

Run:
    venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── project root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.gui_helpers import (
    get_class_mapping, num_classes_from_mapping,
    load_trained_model, model_file_info, model_param_count,
    model_input_shape, model_output_shape, model_layer_table,
    preprocessing_steps, run_inference,
    load_training_history, load_dataset_summary, load_layer_shapes,
    load_evaluation_report, find_classification_report_csv,
    find_confusion_matrix_png,
    sample_test_images,
)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Traffic Sign Recognition — Enhanced LeNet-5",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  – safe overrides that work on light theme
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── main container ── */
.block-container { padding-top: 4rem; max-width: 1120px; }

/* ── sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #d6e4da;
}

/* ── metric card tweak ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d6e4da;
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── custom info card ── */
.ts-card {
    background: #ffffff;
    border: 1px solid #d6e4da;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.ts-card h3 { margin-top: 0; color: #1a3c34; font-size: 1.15rem; }
.ts-card p, .ts-card li { color: #2d4a40; font-size: 0.92rem; line-height: 1.5; }

/* ── workflow card ── */
.wf-card {
    background: #ffffff;
    border: 1px solid #d6e4da;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.wf-icon { font-size: 2.2rem; }
.wf-title { font-size: 1rem; font-weight: 700; color: #1a3c34; margin: 8px 0 4px; }
.wf-desc  { font-size: 0.82rem; color: #4a6b5a; }

/* ── result card ── */
.res-card {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f3 100%);
    border: 2px solid #81c784;
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
}
.res-id   { font-size: 0.95rem; color: #4a6b5a; margin-bottom: 4px; }
.res-name { font-size: 1.6rem; font-weight: 700; color: #1a3c34; }
.res-conf { font-size: 2.2rem; font-weight: 800; color: #2e7d32; margin-top: 6px; }
.res-label{ font-size: 0.85rem; color: #4a6b5a; }

/* ── status badge ── */
.badge-ok  { display:inline-block; background:#e8f5e9; color:#2e7d32;
             padding:3px 12px; border-radius:20px; font-size:.8rem; font-weight:600; }
.badge-err { display:inline-block; background:#ffebee; color:#c62828;
             padding:3px 12px; border-radius:20px; font-size:.8rem; font-weight:600; }

/* ── footer ── */
.ts-footer {
    margin-top: 2rem; padding-top: 1rem;
    border-top: 1px solid #d6e4da;
    font-size: 0.78rem; color: #7a9a8a; text-align: center;
}

/* ── tab bar ── */
[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px 8px !important;
    gap: 2px !important;
    border: 1px solid #c8ddd2 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    flex-wrap: wrap !important;
}

[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    color: #2d4a40 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    border: none !important;
    white-space: nowrap !important;
}

[data-baseweb="tab"] > div,
[data-baseweb="tab"] p,
[data-baseweb="tab"] span {
    color: #2d4a40 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

[data-baseweb="tab"]:hover {
    background: #e8f5e9 !important;
    color: #1a3c34 !important;
}
[data-baseweb="tab"]:hover > div,
[data-baseweb="tab"]:hover p,
[data-baseweb="tab"]:hover span {
    color: #1a3c34 !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: #2e7d32 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(46,125,50,0.35) !important;
}
[data-baseweb="tab"][aria-selected="true"] > div,
[data-baseweb="tab"][aria-selected="true"] p,
[data-baseweb="tab"][aria-selected="true"] span {
    color: #ffffff !important;
}

[data-baseweb="tab-highlight"] { background: transparent !important; }
[data-baseweb="tab-border"]    { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def _class_map():
    return get_class_mapping()

@st.cache_resource
def _model(baseline: bool):
    return load_trained_model(baseline=baseline)

@st.cache_data
def _summary():
    d = load_dataset_summary()
    return d if d else {}

@st.cache_data
def _history(baseline: bool):
    h = load_training_history(baseline=baseline)
    return h if h else {}

@st.cache_data
def _layer_shapes(baseline: bool):
    s = load_layer_shapes(baseline=baseline)
    return s if s else {}

@st.cache_data
def _eval_report(baseline: bool):
    r = load_evaluation_report(baseline=baseline)
    return r if r else {}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def sidebar() -> bool:
    st.sidebar.markdown("## 🚦 Traffic Sign CNN")
    st.sidebar.markdown("Enhanced LeNet-5 · 58 classes")
    st.sidebar.divider()

    choice = st.sidebar.radio(
        "Active model",
        ["Enhanced LeNet-5", "Baseline LeNet-5"],
        index=0,
        help="Switch active model across all tabs."
    )
    baseline = "Baseline" in choice

    st.sidebar.divider()
    exists, fname, size_mb = model_file_info(baseline=baseline)
    if exists:
        st.sidebar.markdown(
            '<span class="badge-ok">● Model loaded</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"{fname}  ·  {size_mb:.2f} MB")
    else:
        st.sidebar.markdown(
            '<span class="badge-err">● Not found</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Expected: {fname}")

    st.sidebar.divider()
    st.sidebar.markdown("**Quick tips**")
    st.sidebar.caption("• Switch model above to compare architectures.")
    st.sidebar.caption("• Use tabs above to test predictions & inspect metrics.")
    return baseline


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════
def page_home(baseline: bool):
    st.title("🚦 Traffic Sign Recognition — Enhanced LeNet-5 CNN")
    st.markdown(
        "A lightweight deep-learning system for classifying 58 traffic-sign classes, "
        "adapted from *An, Yang & Zhang (2024), Frontiers in Neuroscience*."
    )
    st.divider()

    # ── Workflow Cards ─────────────────────────────────────────────────────
    st.subheader("System Workflow")
    wf = st.columns(4)
    for col, (icon, title, desc) in zip(wf, [
        ("📤", "1. Input Image",        "Upload sign photo or select benchmark sample"),
        ("🔧", "2. Preprocessing",       "Resize (32×32) → Grayscale → CLAHE → [-1, 1]"),
        ("🧠", "3. CNN Inference",       "Enhanced LeNet-5 with PReLU & Dropout"),
        ("🚦", "4. Classification",     "Top prediction & confidence distribution"),
    ]):
        col.markdown(f"""
        <div class="wf-card">
            <div class="wf-icon">{icon}</div>
            <div class="wf-title">{title}</div>
            <div class="wf-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Essential Statistics ───────────────────────────────────────────────
    st.subheader("Project Overview")
    mapping  = _class_map()
    n_cls    = num_classes_from_mapping(mapping)
    exists, fname, size_mb = model_file_info(baseline=baseline)
    summary  = _summary()
    history  = _history(baseline=baseline)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Sign Classes",  n_cls or 58)
    c2.metric("Training Set Size",
              f"{summary['train_total_images']:,}" if summary.get("train_total_images") else "4,170")
    c3.metric("Active Model Size",     f"{size_mb:.2f} MB" if exists else "—")
    if history and "history" in history:
        va = history["history"].get("val_accuracy", [])
        c4.metric("Best Val Accuracy", f"{max(va):.2%}" if va else "—")
    else:
        c4.metric("Best Val Accuracy", "—")

    st.divider()

    # ── Core Project Purpose ───────────────────────────────────────────────
    st.markdown("""
    <div class="ts-card">
        <h3>Academic & Engineering Objective</h3>
        <p>Traffic Sign Recognition (TSR) is an essential component of Advanced Driver Assistance
        Systems (ADAS) and autonomous vehicles. The goal of this project is to build a lightweight,
        real-time convolutional neural network capable of robust classification across 58 traffic-sign
        categories without demanding heavy GPU hardware.</p>
        <p>By augmenting the classical LeNet-5 architecture with <b>three convolutional stages,
        Parametric ReLU (PReLU) activations</b>, and <b>multi-stage dropout</b>, the network delivers
        fast edge inference (~0.87 ms per image on CPU) with strong generalization.</p>
    </div>
    """, unsafe_allow_html=True)

    if exists:
        st.success("✅ Model weights loaded and ready. Switch to the **Prediction** tab to test images.")
    else:
        st.warning("⚠️ Model weights not found. Run `python -m src.train` to generate checkpoints.")

    st.markdown(
        '<div class="ts-footer">Final-Year Engineering Project · '
        'Based on An, Yang &amp; Zhang (2024), <i>Frontiers in Neuroscience</i></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def page_prediction(baseline: bool):
    label = "Baseline LeNet-5" if baseline else "Enhanced LeNet-5"
    st.title(f"🔍 Traffic Sign Prediction — {label}")
    st.markdown("Upload a traffic sign image or pick a test sample below to perform live inference.")
    st.divider()

    mapping = _class_map()

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload traffic sign image (JPG, PNG, BMP, PPM)",
        type=["png", "jpg", "jpeg", "bmp", "ppm"],
        help="Upload any clear crop of a traffic sign.",
    )

    # ── Diverse sample buttons ──────────────────────────────────────────────
    samples = sample_test_images(limit=6)
    selected_sample_path = None
    if samples:
        st.markdown("**Or test with a benchmark sign from the held-out test set:**")
        s_cols = st.columns(len(samples))
        for col, sp in zip(s_cols, samples):
            cls_id = int(sp.stem.split("_")[0])
            cls_name = mapping.get(cls_id, f"Class {cls_id}")
            # Format clean short label: e.g. "#2 Speed 30"
            btn_label = f"#{cls_id} {cls_name[:10]}"
            if col.button(btn_label, key=f"s_{sp.name}", use_container_width=True):
                selected_sample_path = sp
                if "pred_result" in st.session_state:
                    del st.session_state["pred_result"]

    # ── Image Loading ───────────────────────────────────────────────────────
    image_np, display_img = None, None

    if uploaded is not None:
        try:
            pil = Image.open(uploaded).convert("RGB")
            image_np = np.array(pil)
            display_img = pil
        except Exception as e:
            st.error(f"Cannot read image: {e}")

    elif selected_sample_path is not None:
        try:
            from src.preprocessing import load_image
            image_np = load_image(str(selected_sample_path))
            display_img = Image.fromarray(image_np)
        except Exception as e:
            st.error(f"Cannot load sample: {e}")

    # ── Interactive Layout ──────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Input Image")
        if display_img:
            st.image(display_img, use_container_width=True, caption=f"Resolution: {image_np.shape[1]}×{image_np.shape[0]} px")
        else:
            st.info("Upload an image or click a sample button above.")

    with right:
        st.markdown("#### Model Prediction")

        run = st.button(
            "🚦 Recognize Traffic Sign",
            type="primary",
            disabled=(image_np is None),
            use_container_width=True,
        )

        if run and image_np is not None:
            with st.spinner("Executing preprocessing & CNN inference…"):
                model, err = _model(baseline=baseline)
                if err:
                    st.error(f"Model error: {err}")
                else:
                    try:
                        result = run_inference(model, image_np, mapping)
                        st.session_state["pred_result"] = result
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        if "pred_result" in st.session_state and image_np is not None:
            r = st.session_state["pred_result"]
            st.markdown(f"""
            <div class="res-card">
                <div class="res-id">Identified Class ID: <b>#{r['predicted_id']}</b></div>
                <div class="res-name">{r['predicted_name']}</div>
                <div class="res-label">Softmax Confidence</div>
                <div class="res-conf">{r['confidence']:.1%}</div>
            </div>""", unsafe_allow_html=True)
            st.caption("ℹ️ *Confidence reflects the softmax output probability distribution across the 58 classes for the preprocessed input tensor.*")
        elif image_np is not None:
            st.info("Click **Recognize Traffic Sign** to run inference.")
        else:
            st.info("Awaiting input image.")

    # ── Top-3 Predictions ───────────────────────────────────────────────────
    if "pred_result" in st.session_state and image_np is not None:
        st.divider()
        st.subheader("Top-3 Class Candidates")
        r = st.session_state["pred_result"]
        top3 = r["top3"]

        df = pd.DataFrame([{
            "Rank": f"#{t['rank']}",
            "Class ID": t["class_id"],
            "Class Name": t["class_name"],
            "Softmax Confidence": f"{t['score']:.2%}",
        } for t in top3])
        st.dataframe(df, use_container_width=True, hide_index=True)

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[t["class_name"] for t in top3],
            y=[t["score"] for t in top3],
            marker_color=["#2e7d32", "#66bb6a", "#a5d6a7"],
            text=[f"{t['score']:.1%}" for t in top3],
            textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=20, b=40),
            xaxis_title="", yaxis_title="Probability",
            yaxis_tickformat=".0%", plot_bgcolor="#ffffff",
            paper_bgcolor="#f0f4f1",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def page_preprocessing(baseline: bool):
    st.title("🔧 Preprocessing Pipeline")
    st.markdown(
        "Both training and live inference use an **identical 4-step pipeline** to transform "
        "arbitrary RGB images into standardized `(32, 32, 1)` tensors."
    )
    st.divider()

    uploaded = st.file_uploader(
        "Upload an image to inspect each transformation",
        type=["png", "jpg", "jpeg", "bmp", "ppm"],
        key="pp_up",
    )
    image_np = None
    if uploaded:
        try:
            image_np = np.array(Image.open(uploaded).convert("RGB"))
        except Exception as e:
            st.error(f"Cannot read: {e}")

    if image_np is None:
        samps = sample_test_images(limit=4)
        if samps:
            st.markdown("**Or pick a sample image:**")
            sc = st.columns(len(samps))
            mapping = _class_map()
            for col, sp in zip(sc, samps):
                cls_id = int(sp.stem.split("_")[0])
                cls_name = mapping.get(cls_id, f"Class {cls_id}")
                if col.button(f"#{cls_id} {cls_name[:10]}", key=f"pp_{sp.name}", use_container_width=True):
                    from src.preprocessing import load_image
                    image_np = load_image(str(sp))

    if image_np is None:
        st.info("Select or upload an image to see the step-by-step pipeline output.")
        st.markdown("### Pipeline Specification")
        st.table(pd.DataFrame([
            ["1", "Resize", "32×32 px", "cv2.INTER_AREA (downsampling interpolation)"],
            ["2", "Grayscale", "1 Channel", "cv2.COLOR_RGB2GRAY (eliminates lighting color cast)"],
            ["3", "CLAHE", "Adaptive Contrast", "clipLimit=2.0, tileGridSize=(4,4) (handles shadows/glare)"],
            ["4", "Normalize", "[-1.0, 1.0]", "(px / 255.0) * 2.0 - 1.0 (float32 zero-centered)"],
        ], columns=["Step", "Operation", "Target Output", "Technical Rationale"]))
        return

    steps = preprocessing_steps(image_np)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Original Input")
        st.image(steps["original"], use_container_width=True)
        st.caption(f"Input Shape: {steps['original'].shape} · uint8")
    with col2:
        st.markdown("##### 2. Resized to 32×32")
        st.image(steps["resized"], use_container_width=True)
        st.caption(f"Spatial Dimensions: 32×32×3 · OpenCV INTER_AREA")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### 3. Grayscale Conversion")
        st.image(steps["grayscale"], use_container_width=True, channels="GRAY")
        st.caption(f"Single Channel: 32×32×1 · Standard RGB to Luminance")
    with col4:
        st.markdown("##### 4. CLAHE Equalization")
        st.image(steps["equalized"], use_container_width=True, channels="GRAY")
        st.caption("Contrast Limited Adaptive Histogram Equalization (local contrast enhancement)")

    st.markdown("##### 5. Normalized Float32 Tensor [-1.0, 1.0]")
    norm = steps["normalized"]
    disp = ((norm + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    st.image(disp, use_container_width=True, channels="GRAY")
    st.caption(f"dtype: float32  ·  Actual value range: [{norm.min():.3f}, {norm.max():.3f}]")

    st.divider()
    st.markdown("**CNN Input Tensor Specification:**")
    st.code(
        f"shape = (1, 32, 32, 1)   # [batch_size=1, height=32, width=32, channels=1]\n"
        f"dtype = float32\n"
        f"range = [{norm.min():.3f}, {norm.max():.3f}]   # Zero-centered [-1.0, 1.0]"
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════
def page_model_info(baseline: bool):
    label = "Baseline LeNet-5" if baseline else "Enhanced LeNet-5"
    st.title(f"🧠 Model Architecture — {label}")
    st.markdown("Inspect layer configurations, parameter budgets, and architectural decisions.")
    st.divider()

    model, err = _model(baseline=baseline)
    if err:
        st.warning(f"Model unavailable: {err}")
        return

    exists, fname, size_mb = model_file_info(baseline=baseline)
    in_shape  = model_input_shape(model)
    out_shape = model_output_shape(model)
    trainable, total = model_param_count(model)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Checkpoint", fname)
    c2.metric("Disk Footprint",   f"{size_mb:.2f} MB")
    c3.metric("Total Parameters", f"{total:,}")
    c4.metric("Trainable Params", f"{trainable:,}")

    st.divider()
    ia, ib = st.columns(2)
    ia.info(f"**Input Tensor Shape:** `{in_shape}` (32×32 Grayscale)")
    ib.info(f"**Output Layer Shape:** `{out_shape}` ({out_shape[-1] if out_shape else 58} Classes, Softmax)")

    # ── Architectural Rationale (Viva Ready) ──────────────────────────────
    if not baseline:
        st.markdown("""
        <div class="ts-card">
            <h3>Enhanced LeNet-5 Design Highlights</h3>
            <ul>
                <li><b>Three Convolutional Blocks:</b> C1 (32 filters), C3 (64 filters), and C5 (64 filters) with 3×3 kernels and <code>padding='same'</code> capture multi-scale spatial sign features.</li>
                <li><b>Parametric ReLU (PReLU):</b> Replaces static ReLUs with learned negative slopes, preserving subtle gradient flows and avoiding the "dying ReLU" failure mode.</li>
                <li><b>Staged Dropout (0.25 block, 0.50 dense):</b> Injected after each pooling operation to prevent co-adaptation, essential for generalization across heavily imbalanced classes.</li>
                <li><b>Lightweight Flatten Vector:</b> 4×4×64 = 1,024 nodes feeding a 512-dense layer, keeping parameters low (664K) for efficient CPU inference.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ts-card">
            <h3>Classical Baseline LeNet-5</h3>
            <ul>
                <li><b>Two Convolutional Layers:</b> 6 filters (5×5, same) and 16 filters (5×5, valid) with standard ReLU and Average Pooling.</li>
                <li><b>Traditional Dense Cascade:</b> Flatten (576) → Dense(120) → Dense(84) → Output(58).</li>
                <li><b>Ablation Benchmark:</b> Only 86,906 parameters; lacks modern activations (PReLU) and regularizers (Dropout), serving as a direct performance baseline.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Layer Table ────────────────────────────────────────────────────────
    st.subheader("Layer-by-Layer Configuration")
    rows = model_layer_table(model)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Collapsed Detailed Output Shapes ───────────────────────────────────
    shapes = _layer_shapes(baseline=baseline)
    if shapes:
        with st.expander("🔍 View Detailed Layer Output Shapes (JSON)", expanded=False):
            st.json(shapes)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
def page_performance(baseline: bool):
    label = "Baseline LeNet-5" if baseline else "Enhanced LeNet-5"
    st.title(f"📊 Model Performance — {label}")
    st.markdown("Quantitative evaluation metrics measured on the **1,994 held-out test images**.")
    st.divider()

    report = _eval_report(baseline=baseline)
    if report:
        st.subheader("Test-Set Evaluation Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test Accuracy",    f"{report.get('test_accuracy',0):.2%}")
        c2.metric("Macro F1-Score",   f"{report.get('macro_f1',0):.4f}")
        c3.metric("Weighted F1-Score",f"{report.get('weighted_f1',0):.4f}")
        c4.metric("Test Loss",        f"{report.get('test_loss',0):.4f}")
        c5.metric("Avg Latency",      f"{report.get('inference_per_image_ms',0):.2f} ms/img")

        # Academic Comparative Callout
        st.markdown(f"""
        <div class="ts-card">
            <h3>Evaluation Analysis</h3>
            <p>Tested on <b>{report.get('num_test_images', 1994):,} unaugmented test images</b>.
            {"<b>Enhanced LeNet-5 achieves 69.11% accuracy and 0.6317 Macro F1</b>, demonstrating a massive +18.86% accuracy improvement over the baseline." if not baseline else
             "<b>Baseline LeNet-5 achieves 50.25% accuracy and 0.4563 Macro F1</b>, suffering from underfitting on complex classes due to lack of regularization and smaller capacity."}
            Macro F1 treats every class equally, providing an unbiased metric despite extreme dataset imbalance.</p>
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix
        cm = find_confusion_matrix_png(baseline=baseline)
        if cm:
            st.subheader("Confusion Matrix (58 Classes)")
            st.image(str(cm), use_container_width=True)

        # Classification Report Table
        csv_p = find_classification_report_csv(baseline=baseline)
        if csv_p:
            st.subheader("Per-Class Classification Report")
            df_rep = pd.read_csv(csv_p)
            st.dataframe(df_rep, use_container_width=True, hide_index=True)

    else:
        st.warning("No evaluation report found for this model.")
        cmd = "python -m src.evaluate" + (" --baseline" if baseline else "")
        st.info(f"Generate results by running: `{cmd}`")

    # ── Genuine Training Curves ─────────────────────────────────────────────
    history = _history(baseline=baseline)
    if history and "history" in history:
        h = history["history"]
        epochs = list(range(1, len(h.get("loss", [])) + 1))

        st.divider()
        st.subheader("Training & Validation History")

        import plotly.graph_objects as go
        ca, cb = st.columns(2)

        if "accuracy" in h and "val_accuracy" in h:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=h["accuracy"],
                mode="lines+markers", name="Train Accuracy",
                line=dict(color="#2e7d32", width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=h["val_accuracy"],
                mode="lines+markers", name="Val Accuracy",
                line=dict(color="#1565c0", width=2, dash="dash")))
            fig.update_layout(title="Accuracy vs Epoch", height=360,
                xaxis_title="Epoch", yaxis_title="Accuracy",
                yaxis_tickformat=".0%", plot_bgcolor="#ffffff",
                paper_bgcolor="#f0f4f1", legend=dict(x=0.02, y=0.98))
            ca.plotly_chart(fig, use_container_width=True)

        if "loss" in h and "val_loss" in h:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=h["loss"],
                mode="lines+markers", name="Train Loss",
                line=dict(color="#2e7d32", width=2)))
            fig2.add_trace(go.Scatter(x=epochs, y=h["val_loss"],
                mode="lines+markers", name="Val Loss",
                line=dict(color="#1565c0", width=2, dash="dash")))
            fig2.update_layout(title="Loss vs Epoch", height=360,
                xaxis_title="Epoch", yaxis_title="Loss",
                plot_bgcolor="#ffffff", paper_bgcolor="#f0f4f1",
                legend=dict(x=0.98, y=0.98, xanchor="right"))
            cb.plotly_chart(fig2, use_container_width=True)

        ep  = history.get("epochs_run", len(h.get("loss", [])))
        dur = history.get("train_duration_sec", 0)
        st.caption(f"Trained for **{ep} epochs** in **{dur:.1f} s** on standard CPU with early stopping and LR scheduling.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — PROJECT INFO & VIVA DEFENSE
# ═══════════════════════════════════════════════════════════════════════════
def page_project(baseline: bool):
    st.title("📋 Project Information & Viva Defense")
    st.markdown("Academic attribution, technical adaptations, dataset constraints, and defense talking points.")
    st.divider()

    # ── Paper Citation ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="ts-card">
        <h3>Research Paper Reference</h3>
        <p><b>A Lightweight Network Architecture for Traffic Sign Recognition Based on Enhanced LeNet-5 Network</b><br>
        Authors: Yuan An, Chunyu Yang, Shuo Zhang<br>
        Journal: <i>Frontiers in Neuroscience</i>, 2024 · DOI: <a href="https://doi.org/10.3389/fnins.2024.1431033" target="_blank">10.3389/fnins.2024.1431033</a></p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Adaptations ────────────────────────────────────────────────────
    st.subheader("Key Adaptations from the Reference Paper")
    st.markdown("""
    1. **Dataset Extension (58 Classes vs. 43 GTSRB Classes):** The original paper evaluated on the German GTSRB benchmark (43 classes). We adapted the architecture to classify 58 distinct traffic-sign categories from the Kaggle Traffic Sign Dataset.
    2. **Mathematically Consistent Flattening (1,024 vs. 576):** With standard `padding='same'` on a 32×32 input through three 2×2 max-pooling layers, the spatial dimension is 4×4 with 64 feature maps (4×4×64 = 1,024). The paper's 576 resulted from valid padding.
    3. **CLAHE Equalization:** We implemented Contrast Limited Adaptive Histogram Equalization rather than global histogram equalization to prevent over-amplifying noise in homogeneous sky/road regions while improving sign legibility under strong glare.
    """)

    # ── Dataset Realities & Viva Challenges ────────────────────────────────
    st.subheader("Dataset Realities & Known Limitations (Viva Discussion)")
    st.markdown("""
    - **Severe Class Imbalance:** Training samples per class range from a minimum of **2 images** (Class 25: *Keep Left*) to **446 images** (Class 28: *Watch out for cars*). 9 classes have fewer than 10 training samples.
    - **Absent Test Classes:** 4 classes (IDs 9, 18, 19, 33) have zero representation in the official held-out `TEST/` folder.
    - **Color Information Trade-Off:** The grayscale pipeline discards color cues (e.g., distinguishing red regulatory circles from blue mandatory signs), but offers faster CPU execution and lower edge parameter count.
    - **Mirrored Test Samples:** The test folder contains mirrored variations (identifiable by `_1_`), which can cause correlation between train/test distributions.
    """)

    # ── Viva Defense Q&A ───────────────────────────────────────────────────
    st.subheader("Core Viva Defense Q&A")
    with st.expander("Q1: Why choose an Enhanced LeNet-5 rather than modern architectures like ResNet or Vision Transformers?"):
        st.write(
            "For automotive edge deployment (such as in-car microcontrollers or embedded ADAS units), "
            "ResNet-50 (>25M parameters) and ViTs (>80M parameters) require dedicated GPU accelerators and introduce "
            "higher latency and thermal dissipation. Enhanced LeNet-5 operates with only **~664K parameters** and achieves "
            "**< 1 ms per image** CPU latency, making it practically deployable on budget hardware."
        )

    with st.expander("Q2: Why use PReLU activations instead of standard ReLU?"):
        st.write(
            "Standard ReLU maps all negative inputs to zero ($f(x) = 0$ for $x < 0$), which can lead to 'dying neurons' "
            "where gradients permanently vanish during backpropagation. PReLU introduces a learnable parameter $\\alpha$ "
            "($f(x) = \\alpha x$ for $x < 0$). Backpropagating $\\alpha$ allows the network to adaptively retain weak "
            "negative signals without manual hyperparameter tuning."
        )

    with st.expander("Q3: Why report Macro F1-score alongside Test Accuracy?"):
        st.write(
            "In heavily imbalanced datasets (e.g., Class 28 has 446 images while Class 25 has only 2), overall accuracy is "
            "heavily biased towards majority classes. Macro F1 calculates the metric independently for each class and "
            "averages them with equal weight, providing a rigorous, honest indicator of minority class performance."
        )

    # ── Future Work ────────────────────────────────────────────────────────
    st.subheader("Future Scope")
    st.markdown("""
    - **RGB 3-Channel Extension:** Retain chromatic channels to separate chromatic sign semantics.
    - **Focal Loss / Class Weighting:** Address extreme class imbalance by penalizing minority misclassifications more heavily.
    - **TensorFlow Lite INT8 Quantization:** Export model weights to quantized 8-bit integers for ARM/microcontroller deployment.
    """)

    st.markdown(
        '<div class="ts-footer">Traffic Sign Recognition · Clean Academic Refactoring</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    baseline = sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home",
        "🔍 Prediction",
        "🔧 Preprocessing",
        "🧠 Model Info",
        "📊 Performance",
        "📋 Project",
    ])

    with tab1: page_home(baseline)
    with tab2: page_prediction(baseline)
    with tab3: page_preprocessing(baseline)
    with tab4: page_model_info(baseline)
    with tab5: page_performance(baseline)
    with tab6: page_project(baseline)


if __name__ == "__main__":
    main()
