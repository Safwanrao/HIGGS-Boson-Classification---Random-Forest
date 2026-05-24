# =============================================================================
# HIGGS Boson Classification using Random Forest
# Dataset: HIGGS Dataset (UCI ML Repository) — ~7.5 GB
# Author: [Your Name] | Roll: [Your Roll Number]
# =============================================================================

import os
import gzip
import shutil
import urllib.request
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
GZ_PATH     = "HIGGS.csv.gz"
CSV_PATH    = "HIGGS.csv"
SAMPLE_ROWS = 500_000      # Use 500k rows for speed; remove cap for full 11M
RANDOM_STATE = 42
N_ESTIMATORS = 100
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "label",
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "jet1_pT", "jet1_eta", "jet1_phi", "jet1_b-tag",
    "jet2_pT", "jet2_eta", "jet2_phi", "jet2_b-tag",
    "jet3_pT", "jet3_eta", "jet3_phi", "jet3_b-tag",
    "jet4_pT", "jet4_eta", "jet4_phi", "jet4_b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
]

# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD DATASET
# ─────────────────────────────────────────────
def download_dataset():
    print("\n" + "="*60)
    print("STEP 1: Downloading HIGGS Dataset (~2.6 GB compressed / ~7.5 GB uncompressed)")
    print("="*60)

    if os.path.exists(CSV_PATH):
        size_gb = os.path.getsize(CSV_PATH) / (1024**3)
        print(f"  ✔  Dataset already exists on disk ({size_gb:.2f} GB). Skipping download.")
        return

    print(f"  ↓  Downloading from: {DATASET_URL}")
    start = time.time()
    urllib.request.urlretrieve(DATASET_URL, GZ_PATH)
    elapsed = time.time() - start
    print(f"  ✔  Download complete in {elapsed:.1f}s")

    print("  ↓  Extracting .gz archive …")
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(GZ_PATH)

    size_gb = os.path.getsize(CSV_PATH) / (1024**3)
    print(f"  ✔  Extraction complete. File size: {size_gb:.2f} GB")


# ─────────────────────────────────────────────
# STEP 2 — READ DATASET
# ─────────────────────────────────────────────
def read_dataset():
    print("\n" + "="*60)
    print("STEP 2: Reading Dataset")
    print("="*60)

    print(f"  ↓  Reading {SAMPLE_ROWS:,} rows from {CSV_PATH} …")
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        names=FEATURE_NAMES,
        nrows=SAMPLE_ROWS,
    )

    print(f"  ✔  Shape          : {df.shape}")
    print(f"  ✔  Memory usage   : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  ✔  Class balance  :\n{df['label'].value_counts().to_string()}")
    print(f"\n  First 5 rows:\n{df.head()}")
    return df


# ─────────────────────────────────────────────
# STEP 3 — PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    print("\n" + "="*60)
    print("STEP 3: Preprocessing")
    print("="*60)

    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"  ✔  Missing values : {missing}")

    # Separate features and label
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✔  Train samples  : {X_train.shape[0]:,}")
    print(f"  ✔  Test  samples  : {X_test.shape[0]:,}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("  ✔  StandardScaler applied")

    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
# STEP 4 — TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 4: Training Random Forest Classifier")
    print("="*60)
    print(f"  n_estimators = {N_ESTIMATORS}  |  random_state = {RANDOM_STATE}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\n  ✔  Training complete in {elapsed:.1f}s")
    return clf


# ─────────────────────────────────────────────
# STEP 5 — EVALUATE & PLOT RESULTS
# ─────────────────────────────────────────────
def evaluate_and_plot(clf, X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("STEP 5: Evaluation & Results")
    print("="*60)

    y_pred      = clf.predict(X_test)
    y_proba     = clf.predict_proba(X_test)[:, 1]

    acc         = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)
    cm          = confusion_matrix(y_test, y_pred)
    report      = classification_report(y_test, y_pred, target_names=["Background (0)", "Signal (1)"])
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  ROC AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # ── Plot 1: Confusion Matrix ──────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Background", "Signal"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/1_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  ✔  Saved: {RESULTS_DIR}/1_confusion_matrix.png")

    # ── Plot 2: ROC Curve ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e63946", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Random Forest", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/2_roc_curve.png", dpi=150)
    plt.close()
    print(f"  ✔  Saved: {RESULTS_DIR}/2_roc_curve.png")

    # ── Plot 3: Feature Importances (Top 15) ──────
    top_n = 15
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.bar(range(top_n),
           importances[indices[:top_n]],
           color=colors, edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] for i in indices[:top_n]],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_title("Top 15 Feature Importances — Random Forest",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/3_feature_importances.png", dpi=150)
    plt.close()
    print(f"  ✔  Saved: {RESULTS_DIR}/3_feature_importances.png")

    # ── Plot 4: Class Distribution ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels_map = {0: "Background", 1: "Signal"}
    counts = pd.Series(y_test).value_counts().sort_index()

    axes[0].bar([labels_map[i] for i in counts.index],
                counts.values,
                color=["#457b9d", "#e63946"], edgecolor="white", width=0.5)
    axes[0].set_title("Test Set Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    axes[1].bar([labels_map[i] for i in pred_counts.index],
                pred_counts.values,
                color=["#2a9d8f", "#f4a261"], edgecolor="white", width=0.5)
    axes[1].set_title("Predicted Class Distribution", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Actual vs Predicted Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/4_class_distribution.png", dpi=150)
    plt.close()
    print(f"  ✔  Saved: {RESULTS_DIR}/4_class_distribution.png")

    # ── Plot 5: Prediction Probability Distribution ─
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_proba[y_test == 0], bins=60, alpha=0.6,
            color="#457b9d", label="Background (True Label 0)")
    ax.hist(y_proba[y_test == 1], bins=60, alpha=0.6,
            color="#e63946", label="Signal (True Label 1)")
    ax.set_xlabel("Predicted Probability (Signal)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/5_probability_distribution.png", dpi=150)
    plt.close()
    print(f"  ✔  Saved: {RESULTS_DIR}/5_probability_distribution.png")

    # ── Summary metrics txt ───────────────────────
    with open(f"{RESULTS_DIR}/metrics_summary.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write("RANDOM FOREST — HIGGS CLASSIFICATION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"ROC AUC  : {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"  ✔  Saved: {RESULTS_DIR}/metrics_summary.txt")
    print("\n  All results saved to the 'results/' folder.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HIGGS Boson Detection — Random Forest Pipeline")
    print("="*60)

    download_dataset()
    df                                      = read_dataset()
    X_train, X_test, y_train, y_test, cols = preprocess(df)
    clf                                     = train_model(X_train, y_train)
    evaluate_and_plot(clf, X_test, y_test, cols)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE ✔")
    print("="*60 + "\n")
