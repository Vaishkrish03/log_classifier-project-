"""
preprocessor.py
---------------
TF-IDF vectorization for log lines.
Handles fit (training) and transform (inference) separately.
"""

import os
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


def _tokenize_log(text: str) -> str:
    """
    Custom tokenizer that:
    - Lowercases
    - Splits camelCase (e.g. 'IOException' → 'IO Exception')
    - Removes numbers-only tokens
    """
    # Split camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    # Remove purely numeric tokens
    text = re.sub(r"\b\d+\b", "", text)
    return text


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=_tokenize_log,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    os.makedirs(MODELS_DIR, exist_ok=True)
    vec = build_vectorizer()
    vec.fit(texts)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vec, f)
    print(f"[preprocessor] Vectorizer fitted and saved → {VECTORIZER_PATH}")
    return vec


def load_vectorizer() -> TfidfVectorizer:
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer not found. Run trainer.py first.")
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def get_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Fit vectorizer on training split only (no data leakage).
    Returns X_train, X_test, y_train, y_test, vectorizer
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].tolist(),
    )

    vec = fit_vectorizer(X_train_raw)
    X_train = vec.transform(X_train_raw)
    X_test = vec.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, vec
