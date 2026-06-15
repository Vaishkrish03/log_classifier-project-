"""
trainer.py
----------
Trains Naive Bayes, SVM, Random Forest, XGBoost on the combined log dataset.
Saves each model as a .pkl file under models/.
"""

import os
import pickle
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from src.data_loader import load_dataset
from src.preprocessor import get_splits

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CLASSIFIERS = {
    "NaiveBayes": MultinomialNB(alpha=0.1),
    "SVM": LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=42, verbosity=0),
}

CLASS_NAMES = ["Critical", "Warning", "Info", "Normal"]


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("LOG CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    df = load_dataset()

    # get_splits fits vectorizer on train only — no data leakage
    X_train, X_test, y_train, y_test, vec = get_splits(df)
    print(f"\n[trainer] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[trainer] Vocab size: {X_train.shape[1]}\n")

    # Apply SMOTE on vectorized training data only
    print("[trainer] Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"[trainer] After SMOTE — train size: {X_train.shape[0]}\n")

    results = {}

    for name, clf in CLASSIFIERS.items():
        print(f"── Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"   Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"   Saved → {model_path}\n")

        results[name] = {
            "accuracy": round(acc, 4),
            "report": report,
            "confusion_matrix": cm,
        }

    results_path = os.path.join(MODELS_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[trainer] Evaluation results saved → {results_path}")
    print("\n✅ All models trained successfully.")
    return results


if __name__ == "__main__":
    train_all()