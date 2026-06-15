"""
evaluator.py
------------
Loads saved eval_results.json and generates plots for the Streamlit UI.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["Critical", "Warning", "Info", "Normal"]

MODEL_COLORS = {
    "NaiveBayes":    "#A855F7",
    "SVM":           "#F97316",
    "RandomForest":  "#22D3EE",
    "XGBoost":       "#FACC15",
}


def load_eval_results() -> dict:
    path = os.path.join(MODELS_DIR, "eval_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError("eval_results.json not found. Train models first.")
    with open(path, "r") as f:
        return json.load(f)


def plot_accuracy_comparison(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    names = list(results.keys())
    accs = [results[n]["accuracy"] * 100 for n in names]
    colors = [MODEL_COLORS.get(n, "#888") for n in names]

    bars = ax.bar(names, accs, color=colors, width=0.5, zorder=3)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Model Accuracy Comparison", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#333", zorder=0)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", color="white", fontsize=10)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(results: dict, model_name: str) -> plt.Figure:
    cm = np.array(results[model_name]["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=False, linewidths=0.5, linecolor="#333",
        annot_kws={"color": "white", "fontsize": 11}
    )
    ax.set_title(f"{model_name} — Confusion Matrix", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")

    fig.tight_layout()
    return fig


def plot_per_class_f1(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    model_names = list(results.keys())

    for i, model in enumerate(model_names):
        f1_scores = [
            results[model]["report"].get(cls, {}).get("f1-score", 0) * 100
            for cls in CLASS_NAMES
        ]
        ax.bar(x + i * width, f1_scores, width, label=model,
               color=MODEL_COLORS.get(model, "#888"), zorder=3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(CLASS_NAMES, color="white")
    ax.set_ylabel("F1 Score (%)", color="white")
    ax.set_title("Per-Class F1 Score by Model", color="white", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#333", zorder=0)
    ax.legend(facecolor="#1E1E2E", labelcolor="white", framealpha=0.8)

    fig.tight_layout()
    return fig