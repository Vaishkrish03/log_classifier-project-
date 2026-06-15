"""
app.py
------
Streamlit UI for the Universal Log Classifier.
Dark theme with color-coded severity levels.
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Log Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Critical": "#FF4B4B",
    "Warning":  "#FFA500",
    "Info":     "#4B8BFF",
    "Normal":   "#00C48C",
}
CLASS_EMOJIS = {
    "Critical": "🔴",
    "Warning":  "🟠",
    "Info":     "🔵",
    "Normal":   "🟢",
}
MODEL_NAMES = ["NaiveBayes", "SVM", "RandomForest", "XGBoost"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base dark background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stSidebar"] {
    background-color: #161B27;
    border-right: 1px solid #2D2D44;
}
/* Cards */
.result-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
    border-left: 5px solid;
    background: #1A1D2E;
}
.label-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 8px;
}
.confidence-bar-bg {
    background: #2D2D44;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}
.confidence-bar-fill {
    border-radius: 8px;
    height: 10px;
}
.model-section {
    background: #1A1D2E;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    border: 1px solid #2D2D44;
}
.metric-box {
    background: #1A1D2E;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    border: 1px solid #2D2D44;
}
h1, h2, h3 { color: #FAFAFA !important; }
.stTextArea textarea {
    background-color: #1A1D2E !important;
    color: #FAFAFA !important;
    border: 1px solid #3D3D5C !important;
    border-radius: 8px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 13px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-size: 15px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
div[data-testid="stSelectbox"] > div {
    background-color: #1A1D2E !important;
    border: 1px solid #3D3D5C !important;
    color: #FAFAFA !important;
}
.stDataFrame { background: #1A1D2E !important; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #161B27;
    border-radius: 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1A1D2E;
    color: #AAAACC;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def models_exist() -> bool:
    return all(os.path.exists(os.path.join(MODELS_DIR, f"{m}.pkl")) for m in MODEL_NAMES)


@st.cache_resource(show_spinner=False)
def load_predictor():
    from src.predictor import predict_single, predict_all_models, predict_batch
    return predict_single, predict_all_models, predict_batch


@st.cache_data(show_spinner=False)
def load_eval():
    from src.evaluator import load_eval_results
    return load_eval_results()


def render_result_card(result: dict, show_model: bool = False):
    label = result.get("label", "Unknown")
    color = CLASS_COLORS.get(label, "#888")
    emoji = CLASS_EMOJIS.get(label, "⚪")
    conf = result.get("confidence", 0)
    model = result.get("model", "")

    header = f"{emoji} {label}" + (f" <span style='color:#888;font-size:12px;margin-left:8px'>via {model}</span>" if show_model else "")

    st.markdown(f"""
    <div class="result-card" style="border-left-color: {color};">
        <div class="label-badge" style="background:{color};">{header}</div>
        <div style="font-size:13px;color:#AAA;margin-bottom:6px;">Confidence: <b style="color:{color}">{conf:.1f}%</b></div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{conf}%;background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    if "probabilities" in result:
        with st.expander("Class probabilities"):
            for cls, prob in result["probabilities"].items():
                c = CLASS_COLORS.get(cls, "#888")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
                    <span style="width:80px;color:{c};font-weight:600">{cls}</span>
                    <div style="flex:1;background:#2D2D44;border-radius:6px;height:8px;">
                        <div style="width:{prob}%;background:{c};height:8px;border-radius:6px;"></div>
                    </div>
                    <span style="color:#AAA;font-size:12px;width:45px;text-align:right">{prob:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)


def train_models_ui():
    st.warning("⚠️ Models not trained yet. Click below to train.")
    if st.button("🚀 Train Models Now"):
        with st.spinner("Training all 4 models... this takes ~1-2 minutes"):
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "src.trainer"],
                    capture_output=True, text=True, cwd=os.path.dirname(__file__)
                )
                if result.returncode == 0:
                    st.success("✅ Models trained successfully! Refreshing...")
                    st.rerun()
                else:
                    st.error(f"Training failed:\n{result.stderr}")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-size:36px">🔍</div>
        <div style="font-size:20px;font-weight:700;color:#FAFAFA">Log Classifier</div>
        <div style="font-size:12px;color:#888;margin-top:4px">Universal · Multi-class · OS-agnostic</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤖 Model")
    selected_model = st.selectbox(
        "Choose classifier",
        ["All Models (Compare)"] + MODEL_NAMES,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    for cls, color in CLASS_COLORS.items():
        emoji = CLASS_EMOJIS[cls]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
            <div style="width:12px;height:12px;border-radius:50%;background:{color};"></div>
            <span style="color:#DDD;font-size:14px">{emoji} {cls}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Sources")
    st.markdown("""
    <div style="font-size:12px;color:#888;line-height:1.8;">
    • HDFS logs (Hadoop)<br>
    • Linux system logs<br>
    • Windows event logs
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Retrain Models"):
        for m in MODEL_NAMES:
            path = os.path.join(MODELS_DIR, f"{m}.pkl")
            if os.path.exists(path):
                os.remove(path)
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            os.remove(vec_path)
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="text-align:center;background:linear-gradient(135deg,#667eea,#f093fb);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:2.4rem;margin-bottom:4px;">
    Universal Log Classifier
</h1>
<p style="text-align:center;color:#888;font-size:14px;margin-bottom:24px;">
    OS-agnostic · NLP-powered · 4-class severity detection
</p>
""", unsafe_allow_html=True)

# Check if models exist
if not models_exist():
    train_models_ui()
    st.stop()

predict_single, predict_all_models, predict_batch = load_predictor()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔎 Single Log", "📂 Bulk Upload", "📊 Model Evaluation"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Paste a log line")

    SAMPLE_LOGS = {
        "Pick a sample...": "",
        "🔴 Critical — Exception": "ERROR: java.lang.NullPointerException thrown at DataNode.java:342 — failed to read block",
        "🟠 Warning — High Load": "WARNING: CPU usage exceeded threshold 90% — consider scaling resources",
        "🔵 Info — Service Start": "INFO: Namenode started successfully and is listening on port 9000",
        "🟢 Normal — Heartbeat": "DEBUG: Heartbeat sent to master node — status OK",
        "Windows Event": "The service 'wuauserv' failed to start due to the following error: Access is denied.",
        "Linux Syslog": "kernel: oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0",
    }

    sample = st.selectbox("Or pick a sample log:", list(SAMPLE_LOGS.keys()))
    default_text = SAMPLE_LOGS[sample]

    log_input = st.text_area(
        "Log line",
        value=default_text,
        height=100,
        placeholder="Paste any log line here — Windows, Linux, HDFS...",
        label_visibility="collapsed",
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        classify_btn = st.button("🔍 Classify", use_container_width=True)

    if classify_btn and log_input.strip():
        st.markdown("---")
        if selected_model == "All Models (Compare)":
            st.markdown("#### Results across all models")
            results = predict_all_models(log_input.strip())
            cols = st.columns(2)
            for i, (model, result) in enumerate(results.items()):
                with cols[i % 2]:
                    st.markdown(f"**{model}**")
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        render_result_card(result)
        else:
            result = predict_single(log_input.strip(), selected_model)
            st.markdown(f"#### Result — {selected_model}")
            render_result_card(result)

    elif classify_btn:
        st.warning("Please enter a log line first.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload a log file")
    st.markdown("<span style='color:#888;font-size:13px'>Supports .txt, .log, .csv (one log per line)</span>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload log file", type=["txt", "log", "csv"], label_visibility="collapsed")

    bulk_model = selected_model if selected_model != "All Models (Compare)" else "XGBoost"
    if selected_model == "All Models (Compare)":
        st.info("ℹ️ Bulk mode uses XGBoost by default. Select a specific model in the sidebar for bulk classification.")

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in content.splitlines() if len(l.strip()) > 5]

        st.markdown(f"<span style='color:#888'>Loaded **{len(lines)}** log lines</span>", unsafe_allow_html=True)

        if st.button("⚡ Classify All"):
            with st.spinner(f"Classifying {len(lines)} lines with {bulk_model}..."):
                results = predict_batch(lines, bulk_model)

            df = pd.DataFrame(results)
            df.columns = ["Log Line", "Class", "Class Index", "Confidence (%)"]

            # Color map for display
            def color_class(val):
                color = CLASS_COLORS.get(val, "#888")
                return f"color: {color}; font-weight: bold"

            st.markdown("#### Classification Results")
            st.dataframe(
                df[["Log Line", "Class", "Confidence (%)"]].style.applymap(color_class, subset=["Class"]),
                use_container_width=True,
                height=350,
            )

            # Distribution pie chart
            st.markdown("#### Distribution")
            counts = df["Class"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0E1117")
            colors_pie = [CLASS_COLORS.get(c, "#888") for c in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=counts.index,
                colors=colors_pie,
                autopct="%1.1f%%",
                startangle=140,
                textprops={"color": "white", "fontsize": 11},
            )
            for at in autotexts:
                at.set_color("white")
            ax.set_title("Log Severity Distribution", color="white", fontsize=13, fontweight="bold")
            fig.patch.set_facecolor("#0E1117")
            st.pyplot(fig)

            # Download results
            csv = df[["Log Line", "Class", "Confidence (%)"]].to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv, "classified_logs.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Evaluation")

    try:
        from src.evaluator import load_eval_results, plot_accuracy_comparison, plot_confusion_matrix, plot_per_class_f1

        results = load_eval_results()

        # Accuracy cards
        st.markdown("#### Accuracy")
        acc_cols = st.columns(4)
        model_colors = {
            "NaiveBayes":   "#A855F7",
            "SVM":          "#F97316",
            "RandomForest": "#22D3EE",
            "XGBoost":      "#FACC15",
        }
        for i, (model, data) in enumerate(results.items()):
            with acc_cols[i]:
                color = model_colors.get(model, "#888")
                st.markdown(f"""
                <div class="metric-box" style="border-top:3px solid {color};">
                    <div style="font-size:12px;color:#888;margin-bottom:4px">{model}</div>
                    <div style="font-size:26px;font-weight:700;color:{color}">{data['accuracy']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Accuracy Comparison")
            fig = plot_accuracy_comparison(results)
            st.pyplot(fig)

        with col_b:
            st.markdown("#### Per-Class F1 Score")
            fig2 = plot_per_class_f1(results)
            st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### Confusion Matrix")
        cm_model = st.selectbox("Select model:", MODEL_NAMES, key="cm_select")
        fig3 = plot_confusion_matrix(results, cm_model)
        st.pyplot(fig3)

    except FileNotFoundError:
        st.warning("No evaluation data found. Train the models first.")
