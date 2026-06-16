"""
predictor.py
------------
Loads trained models and vectorizer.
Hybrid approach: labeler override for high-confidence cases, ML for everything else.
"""

import os
import re
import pickle
import numpy as np
from src.preprocessor import load_vectorizer
from src.labeler import label_log_multi, CLASS_NAMES as LABEL_CLASS_NAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CLASS_NAMES = ["Critical", "Warning", "Info", "Normal"]

OVERRIDE_THRESHOLD = 4

_MODEL_CACHE = {}
_VEC_CACHE = [None]


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model '{name}' not found. Run trainer.py first.")
        with open(path, "rb") as f:
            _MODEL_CACHE[name] = pickle.load(f)
    return _MODEL_CACHE[name]


def _get_vectorizer():
    if _VEC_CACHE[0] is None:
        _VEC_CACHE[0] = load_vectorizer()
    return _VEC_CACHE[0]


def _get_proba(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)[0]
        e = np.exp(scores - scores.max())
        return e / e.sum()
    else:
        pred = clf.predict(X)[0]
        proba = np.zeros(len(CLASS_NAMES))
        proba[pred] = 1.0
        return proba


def _labeler_score(line: str) -> tuple:
    """Returns (class_idx, score) from the labeler."""
    from src.labeler import _score_patterns, CRITICAL_PATTERNS, WARNING_PATTERNS, INFO_PATTERNS, NORMAL_PATTERNS
    line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
    scores = {
        0: _score_patterns(line, CRITICAL_PATTERNS),
        1: _score_patterns(line, WARNING_PATTERNS),
        2: _score_patterns(line, INFO_PATTERNS),
        3: _score_patterns(line, NORMAL_PATTERNS),
    }
    best = max(scores, key=scores.get)
    return best, scores[best]


def predict_single(log_line: str, model_name: str) -> dict:
    vec = _get_vectorizer()
    clf = _load_model(model_name)

    labeler_class, labeler_score = _labeler_score(log_line)
    if labeler_score >= OVERRIDE_THRESHOLD:
        proba = np.zeros(len(CLASS_NAMES))
        proba[labeler_class] = 1.0
        return {
            "label": CLASS_NAMES[labeler_class],
            "label_idx": int(labeler_class),
            "confidence": 100.0,
            "probabilities": {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)},
            "model": model_name,
            "source": "rule-based override",
        }

    X = vec.transform([log_line])
    pred_idx = clf.predict(X)[0]
    proba = _get_proba(clf, X)

    return {
        "label": CLASS_NAMES[pred_idx],
        "label_idx": int(pred_idx),
        "confidence": round(float(proba[pred_idx]) * 100, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)},
        "model": model_name,
        "source": "ml model",
    }


def predict_all_models(log_line: str) -> dict:
    model_names = ["NaiveBayes", "SVM", "RandomForest", "XGBoost"]
    results = {}
    for name in model_names:
        try:
            results[name] = predict_single(log_line, name)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def predict_batch(log_lines: list[str], model_name: str) -> list[dict]:
    vec = _get_vectorizer()
    clf = _load_model(model_name)

    results = []
    for line in log_lines:
        labeler_class, labeler_score = _labeler_score(line)
        if labeler_score >= OVERRIDE_THRESHOLD:
            proba = np.zeros(len(CLASS_NAMES))
            proba[labeler_class] = 1.0
            results.append({
                "line": line,
                "label": CLASS_NAMES[labeler_class],
                "label_idx": int(labeler_class),
                "confidence": 100.0,
            })
        else:
            X = vec.transform([line])
            pred_idx = clf.predict(X)[0]
            proba = _get_proba(clf, X)
            results.append({
                "line": line,
                "label": CLASS_NAMES[pred_idx],
                "label_idx": int(pred_idx),
                "confidence": round(float(proba[pred_idx]) * 100, 2),
            })

    return results
