"""
data_loader.py
--------------
Downloads sample log files from loghub GitHub and combines them
into a single labeled DataFrame for training.
"""

import os
import re
import pandas as pd
import requests
from src.labeler import label_log_multi

# Raw URLs for log samples from loghub
LOG_SOURCES = {
    "HDFS": "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "Linux": "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log",
    "Windows": "https://raw.githubusercontent.com/logpai/loghub/master/Windows/Windows_2k.log",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _download_log(name: str, url: str) -> list[str]:
    """Download a log file and return lines."""
    local_path = os.path.join(DATA_DIR, f"{name}_2k.log")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(local_path):
        print(f"[data_loader] Using cached {name} logs.")
        with open(local_path, "r", errors="ignore") as f:
            return f.readlines()

    print(f"[data_loader] Downloading {name} logs...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.splitlines(keepends=True)
        with open(local_path, "w", errors="ignore") as f:
            f.writelines(lines)
        print(f"[data_loader] Downloaded {len(lines)} lines for {name}.")
        return lines
    except Exception as e:
        print(f"[data_loader] Failed to download {name}: {e}")
        return []


def _clean_line(line: str) -> str:
    """Strip timestamps, IPs, hex addresses — keep the semantic content."""
    line = line.strip()
    # Remove common timestamp patterns
    line = re.sub(r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}[.,\d]*", "", line)
    line = re.sub(r"\d{2}:\d{2}:\d{2}[.,\d]*", "", line)
    # Remove IP addresses
    line = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b", "", line)
    # Remove hex addresses / block IDs
    line = re.sub(r"\b[0-9a-fA-F]{6,}\b", "", line)
    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line


def load_dataset(sources: dict = None) -> pd.DataFrame:
    """
    Download logs from loghub, clean and label them.
    Returns a DataFrame with columns: [text, label]
    """
    if sources is None:
        sources = LOG_SOURCES

    all_lines = []
    for name, url in sources.items():
        raw_lines = _download_log(name, url)
        for line in raw_lines:
            cleaned = _clean_line(line)
            if len(cleaned) > 10:  # skip near-empty lines
                all_lines.append({"text": cleaned, "source": name})

    df = pd.DataFrame(all_lines)
    df["label"] = df["text"].apply(label_log_multi)

    print(f"\n[data_loader] Total samples: {len(df)}")
    print(df["label"].value_counts().rename({0: "Critical", 1: "Warning", 2: "Info", 3: "Normal"}))
    return df
if __name__ == "__main__":
    # 1. Run the pipeline to build the structured dataset
    df = load_dataset()
    
    # 2. Print the first 10 rows to the terminal so the guide can see it
    print("\n--- FIRST 10 STRUCTURED LOGS ---")
    print(df.head(10))
    
    # 3. Export it as a CSV file he can scroll through
    df.to_csv("final_structured_dataset.csv", index=False)
    print("\n✅ Dataset exported to 'final_structured_dataset.csv'")
