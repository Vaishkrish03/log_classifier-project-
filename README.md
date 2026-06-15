# Universal Log Classifier

An OS-agnostic, NLP-powered log classification system that categorizes system logs into **Critical**, **Warning**, **Info**, and **Normal** severity levels.

Trained on HDFS, Linux, and Windows log data from [loghub](https://github.com/logpai/loghub).

---

## Project Structure

```
log_classifier/
├── src/
│   ├── labeler.py        # Keyword-based multi-class label scoring
│   ├── data_loader.py    # Downloads & combines HDFS/Linux/Windows logs
│   ├── preprocessor.py   # TF-IDF vectorization
│   ├── trainer.py        # Trains NB, SVM, RF, XGBoost — saves .pkl files
│   ├── predictor.py      # Inference: single line & batch
│   └── evaluator.py      # Metrics & plots for UI
├── models/               # Saved model .pkl files (auto-created on training)
├── data/                 # Cached log files (auto-downloaded)
├── app.py                # Streamlit UI
├── train.py              # Training entrypoint
└── requirements.txt
```

---

## Setup & Run Locally

```bash
# 1. Clone / download the project
cd log_classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (downloads data automatically)
python train.py

# 4. Launch the app
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. Deploy — Streamlit Cloud will install `requirements.txt` automatically

> **Note:** On first load, the app will prompt you to train models. Click "Train Models Now" in the UI.

---

## Models

| Model | Description |
|-------|-------------|
| Naive Bayes | Fast probabilistic baseline |
| SVM (LinearSVC) | Strong linear classifier for text |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting — typically best accuracy |

---

## Classes

| Class | Color | Description |
|-------|-------|-------------|
| 🔴 Critical | Red | Errors, exceptions, crashes, failures |
| 🟠 Warning | Orange | Degraded performance, retries, thresholds |
| 🔵 Info | Blue | Service starts, connections, completions |
| 🟢 Normal | Green | Heartbeats, debug traces, routine ops |
