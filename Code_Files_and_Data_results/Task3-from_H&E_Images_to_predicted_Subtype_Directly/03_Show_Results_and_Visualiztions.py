#!/usr/bin/env python3
# visualize_he_to_subtype_results.py
# Reads the saved artifacts and produces clear, publication-ready plots per model.

import os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

OUT_ROOT = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_Task3_H&E_to_Subtype"
MODELS = ["retccl_resnet50", "resnet50_imagenet", "vit_b16_imagenet"]

def load_metrics(model_dir: Path):
    with open(model_dir / "metrics.json") as f:
        return json.load(f)

def plot_confusion(model_dir: Path, save_png=True):
    cm_df = pd.read_csv(model_dir / "confusion_matrix_normalized.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm_df.values, vmin=0, vmax=1)
    ax.set_xticks(range(len(cm_df.columns))); ax.set_xticklabels(cm_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(cm_df.index)));   ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized Confusion Matrix")
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, f"{cm_df.values[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm_df.values[i,j]>0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_png:
        fig.savefig(model_dir / "confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)

def plot_roc(model_dir: Path, class_names=None, save_png=True):
    data = np.load(model_dir / "roc_curves.npz", allow_pickle=True)
    if class_names is None:
        # infer from keys
        class_names = sorted(set(k.split("_", 1)[1] for k in data.files if k.startswith("fpr_")))
    fig, ax = plt.subplots(figsize=(6,5))
    for cls in class_names:
        fpr = data[f"fpr_{cls}"]; tpr = data[f"tpr_{cls}"]; auc_v = data[f"auc_{cls}"].item() if hasattr(data[f"auc_{cls}"], "item") else float(data[f"auc_{cls}"])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_v:.2f})")
    ax.plot([0,1],[0,1],'--',lw=1,color='gray')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("One-vs-Rest ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if save_png: fig.savefig(model_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

def plot_pr(model_dir: Path, class_names=None, save_png=True):
    data = np.load(model_dir / "pr_curves.npz", allow_pickle=True)
    if class_names is None:
        class_names = sorted(set(k.split("_", 1)[1] for k in data.files if k.startswith("prec_")))
    fig, ax = plt.subplots(figsize=(6,5))
    for cls in class_names:
        prec = data[f"prec_{cls}"]; rec = data[f"rec_{cls}"]; auc_v = data[f"aucPR_{cls}"].item() if hasattr(data[f"aucPR_{cls}"], "item") else float(data[f"aucPR_{cls}"])
        ax.plot(rec, prec, label=f"{cls} (AUC={auc_v:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("One-vs-Rest Precision–Recall")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    if save_png: fig.savefig(model_dir / "pr_curves.png", dpi=160)
    plt.close(fig)

def plot_calibration(model_dir: Path, save_png=True):
    df = pd.read_csv(model_dir / "calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1],'--', color='gray', lw=1)
    ax.plot(df["bin_mean_confidence"], df["bin_fraction_positive"], marker='o')
    ax.set_xlabel("Mean predicted confidence (top class)")
    ax.set_ylabel("Fraction of correct predictions")
    ax.set_title("Calibration (reliability curve)")
    fig.tight_layout()
    if save_png: fig.savefig(model_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)

def plot_confidence_hist(model_dir: Path, save_png=True):
    df = pd.read_csv(model_dir / "confidence_hist.csv")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(0.5*(df["bin_left"]+df["bin_right"]), df["count"], width=(df["bin_right"]-df["bin_left"]))
    ax.set_xlabel("Top-class probability"); ax.set_ylabel("Count"); ax.set_title("Confidence histogram (test)")
    fig.tight_layout()
    if save_png: fig.savefig(model_dir / "confidence_hist.png", dpi=160)
    plt.close(fig)

def comparison_table(out_root: Path, models: list):
    rows = []
    for m in models:
        p = out_root / m / "metrics.json"
        if p.exists():
            with open(p) as f: met = json.load(f)
            met["model"] = m
            rows.append(met)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_root / "models_comparison.csv", index=False)
        print("Saved models_comparison.csv with metrics across models.")

def main():
    ap = argparse.ArgumentParser(description="Make plots for HE→Subtype classification (reads saved artifacts).")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    for m in args.models:
        model_dir = out_root / m
        if not model_dir.exists():
            print(f"Skip {m}: folder missing.")
            continue
        print(f"Plotting for {m} ...")
        # infer class names from confusion CSV
        cm_path = model_dir / "confusion_matrix_normalized.csv"
        class_names = list(pd.read_csv(cm_path, index_col=0, nrows=0).columns) if cm_path.exists() else None

        plot_confusion(model_dir, save_png=True)
        plot_roc(model_dir, class_names, save_png=True)
        plot_pr(model_dir, class_names, save_png=True)
        plot_calibration(model_dir, save_png=True)
        plot_confidence_hist(model_dir, save_png=True)

    comparison_table(out_root, args.models)

if __name__ == "__main__":
    main()
