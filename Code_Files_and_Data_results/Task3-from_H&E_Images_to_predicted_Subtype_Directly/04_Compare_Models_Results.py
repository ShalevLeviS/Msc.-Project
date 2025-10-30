#!/usr/bin/env python3
# comprehensive_model_comparison.py
# Creates ONE comprehensive figure with all model comparisons

import os, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ===================== CONFIG =====================
OUT_ROOT = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_Task3_H&E_to_Subtype"
MODELS = ["retccl_resnet50", "resnet50_imagenet", "vit_b16_imagenet"]

MODEL_DISPLAY_NAMES = {
    "retccl_resnet50": "RetCCL-ResNet50",
    "resnet50_imagenet": "ResNet50-ImageNet",
    "vit_b16_imagenet": "ViT-B/16-ImageNet"
}

# ===================== IO HELPERS =====================
def load_all_data(out_root: Path, models):
    """Load all metrics and data needed for comparison from per-model folders."""
    data = {}
    for model in models:
        model_dir = out_root / model
        model_data = {}

        # metrics
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                model_data["metrics"] = json.load(f)

        # per-class report
        per_class_path = model_dir / "per_class_report.csv"
        if per_class_path.exists():
            # Expecting columns: class, precision, recall, f1, support (or similar)
            df = pd.read_csv(per_class_path)
            # normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            if "class" not in df.columns:
                # try to infer class column
                for cand in ["label", "name", "classes"]:
                    if cand in df.columns:
                        df = df.rename(columns={cand: "class"})
                        break
            model_data["per_class"] = df

        # confusion matrix (normalized)
        cm_path = model_dir / "confusion_matrix_normalized.csv"
        if cm_path.exists():
            model_data["confusion"] = pd.read_csv(cm_path, index_col=0)

        # ROC curves
        roc_path = model_dir / "roc_curves.npz"
        if roc_path.exists():
            model_data["roc"] = np.load(roc_path, allow_pickle=True)

        data[model] = model_data
    return data

# ===================== PLOTTING =====================
def _annotate_bars(ax, bars, values, fontsize=8, fmt="{:.3f}"):
    """Annotate bars with their values."""
    for bar, val in zip(bars, values):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        h = bar.get_height()
        if h > 0:  # Only annotate if bar has height
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, fmt.format(val),
                    ha="center", va="bottom", fontsize=fontsize, fontweight='bold')

def create_comprehensive_comparison(out_root: Path, models, save_png=True):
    """Create ONE big figure:
       Row 1: Overall metrics (left) + Per-class F1 (right) - SIDE BY SIDE
       Row 2: Confusion matrices for all 3 models (+ shared colorbar)
       Row 3: ROC-AUC per model (macro/micro if available, else OvR)
    """
    data = load_all_data(out_root, models)
    if not data or all(len(data[m]) == 0 for m in models):
        print("No data found!")
        return

    # --------- Figure layout ---------
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6C5B7B', '#17A398']

    # ================== ROW 1: Overall Metrics (left) + Per-class F1 (right) - EQUAL WIDTH ==================
    # Create a nested GridSpec for row 1 to get equal widths
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], wspace=0.3)
    
    # --- Overall metrics (left half) ---
    ax_metrics = fig.add_subplot(gs_row1[0])
    metrics_to_plot = ["accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc"]
    metric_labels = ["Accuracy", "Balanced Acc", "Macro F1", "ROC-AUC"]
    x = np.arange(len(metric_labels))
    width = 0.25  # Width for each bar

    for idx, model in enumerate(models):
        if "metrics" not in data[model]:
            continue
        values = [data[model]["metrics"].get(metric, 0.0) for metric in metrics_to_plot]
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        offset = width * (idx - 1)  # Center the bars (-1, 0, 1 for 3 models)
        bars = ax_metrics.bar(x + offset, values, width, label=display_name, color=colors[idx % len(colors)])
        # Annotate each bar with its value
        _annotate_bars(ax_metrics, bars, values, fontsize=9, fmt="{:.3f}")

    ax_metrics.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax_metrics.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax_metrics.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_labels, fontsize=10)
    ax_metrics.legend(fontsize=9, loc='lower right')
    ax_metrics.grid(axis='y', alpha=0.3)
    ax_metrics.set_ylim(0, 1.1)  # Extra space for annotations

    # --- Per-class F1 (right half) ---
    ax_f1 = fig.add_subplot(gs_row1[1])
    if all("per_class" in data[m] for m in models):
        # Use classes from the first model for consistent ordering
        classes = data[models[0]]["per_class"]["class"].values
        x_cls = np.arange(len(classes))
        width_f1 = 0.25

        for idx, model in enumerate(models):
            # align by class name
            df = data[model]["per_class"]
            # Make sure we have 'class' and 'f1'
            if "class" not in df.columns or "f1" not in df.columns:
                continue
            # map by class
            cls2f1 = dict(zip(df["class"].values, df["f1"].values))
            f1_scores = [float(cls2f1.get(c, np.nan)) for c in classes]

            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            offset = width_f1 * (idx - 1)  # Center the bars
            bars = ax_f1.bar(x_cls + offset, f1_scores, width_f1, label=display_name, color=colors[idx % len(colors)])
            
            # Annotate each bar with its value
            _annotate_bars(ax_f1, bars, f1_scores, fontsize=8, fmt="{:.3f}")

        ax_f1.set_xlabel('Subtype', fontsize=11, fontweight='bold')
        ax_f1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax_f1.set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
        ax_f1.set_xticks(x_cls)
        ax_f1.set_xticklabels(classes, fontsize=9, rotation=45, ha='right')
        ax_f1.legend(fontsize=9, loc='lower right')
        ax_f1.grid(axis='y', alpha=0.3)
        ax_f1.set_ylim(0, 1.1)  # Extra space for annotations
    else:
        ax_f1.text(0.5, 0.5, "Per-class reports not found for all models",
                   ha='center', va='center', fontsize=10)
        ax_f1.axis('off')

    # ================== ROW 2: Confusion Matrices (3 models side by side) ==================
    for idx, model in enumerate(models[:3]):  # up to 3 columns
        ax_cm = fig.add_subplot(gs[1, idx])
        if "confusion" in data[model]:
            cm = data[model]["confusion"]
            im = ax_cm.imshow(cm.values, vmin=0, vmax=1, cmap='Blues')

            # Add colorbar for each confusion matrix
            cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            cbar.set_label('Fraction', fontsize=8)

            ax_cm.set_xticks(range(len(cm.columns)))
            ax_cm.set_xticklabels(cm.columns, rotation=45, ha='right', fontsize=8)
            ax_cm.set_yticks(range(len(cm.index)))
            ax_cm.set_yticklabels(cm.index, fontsize=8)
            ax_cm.set_xlabel('Predicted', fontsize=9, fontweight='bold')
            ax_cm.set_ylabel('True Label', fontsize=9, fontweight='bold')
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            ax_cm.set_title(f'Confusion Matrix: {display_name}', fontsize=10, fontweight='bold')

            # overlay values
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm.values[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax_cm.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color, fontweight='bold')
        else:
            ax_cm.text(0.5, 0.5, "Confusion matrix not found",
                       ha='center', va='center', fontsize=9)
            ax_cm.set_axis_off()

    # ================== ROW 3: ROC-AUC (one panel per model) ==================
    for m_idx, model in enumerate(models[:3]):
        ax_roc = fig.add_subplot(gs[2, m_idx])
        if "roc" not in data[model]:
            ax_roc.text(0.5, 0.5, "ROC data not found", ha='center', va='center', fontsize=9)
            ax_roc.set_axis_off()
            continue

        roc_data = data[model]["roc"]
        display_name = MODEL_DISPLAY_NAMES.get(model, model)

        plotted = False
        files = getattr(roc_data, "files", list(roc_data.keys()))
        # Prefer macro → micro → OvR
        if all(k in files for k in ["fpr_macro", "tpr_macro", "auc_macro"]):
            fpr = roc_data["fpr_macro"]; tpr = roc_data["tpr_macro"]; auc_val = float(roc_data["auc_macro"])
            ax_roc.plot(fpr, tpr, linewidth=2.5, label=f'Macro (AUC={auc_val:.3f})', color=colors[m_idx % len(colors)])
            plotted = True
        elif all(k in files for k in ["fpr_micro", "tpr_micro", "auc_micro"]):
            fpr = roc_data["fpr_micro"]; tpr = roc_data["tpr_micro"]; auc_val = float(roc_data["auc_micro"])
            ax_roc.plot(fpr, tpr, linewidth=2.5, label=f'Micro (AUC={auc_val:.3f})', color=colors[m_idx % len(colors)])
            plotted = True
        else:
            cls_names = sorted({k.split("_", 1)[1] for k in files if k.startswith("fpr_") and "_" in k})
            aucs = []
            for ci, cls in enumerate(cls_names):
                fpr_k, tpr_k, auc_k = f"fpr_{cls}", f"tpr_{cls}", f"auc_{cls}"
                if fpr_k in files and tpr_k in files:
                    fpr = roc_data[fpr_k]; tpr = roc_data[tpr_k]
                    auc_val = float(roc_data[auc_k]) if auc_k in files else np.nan
                    aucs.append(auc_val)
                    ax_roc.plot(fpr, tpr, linewidth=2, label=f'{cls} (AUC={auc_val:.3f})')
                    plotted = True
            if aucs and np.isfinite(np.nanmean(aucs)):
                ax_roc.text(0.98, 0.02, f"Mean AUC: {np.nanmean(aucs):.3f}",
                            ha='right', va='bottom', fontsize=9, transform=ax_roc.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_roc.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5, alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate', fontsize=9, fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontsize=9, fontweight='bold')
        ax_roc.set_title(f'ROC Curve: {display_name}', fontsize=10, fontweight='bold')
        if plotted:
            ax_roc.legend(fontsize=8, loc='lower right')
        ax_roc.grid(alpha=0.3)
        ax_roc.tick_params(labelsize=8)

    # Main title
    fig.suptitle('Comprehensive Model Comparison: H&E → PAM50 Subtype Classification',
                 fontsize=15, fontweight='bold', y=0.98)

    # Save
    if save_png:
        output_path = out_root / "comprehensive_model_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comprehensive comparison to: {output_path}")

    plt.close(fig)

    # Also create a simple summary table
    create_summary_table(data, models, out_root)

# ===================== SUMMARY TABLE =====================
def create_summary_table(data, models, out_root: Path):
    rows = []
    for model in models:
        metrics = data.get(model, {}).get("metrics", {})
        if not metrics:
            continue
        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(model, model),
            "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Balanced_Acc": f"{metrics.get('balanced_accuracy', 0):.4f}",
            "Macro_F1": f"{metrics.get('macro_f1', 0):.4f}",
            "ROC_AUC": f"{metrics.get('macro_roc_auc', 0):.4f}",
            "PR_AUC": f"{metrics.get('macro_pr_auc', 0):.4f}",
            "Log_Loss": f"{metrics.get('log_loss', 0):.4f}",
            "N_Test": metrics.get("n_test", 0),
        })
    df = pd.DataFrame(rows)
    out_path = out_root / "model_comparison_summary.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No metrics found.")
    print("="*70)

# ===================== MAIN =====================
def main():
    out_root = Path(OUT_ROOT)
    print("Creating comprehensive model comparison figure...")
    create_comprehensive_comparison(out_root, MODELS)
    print("\nGenerated files:")
    print("  - comprehensive_model_comparison.png (ONE figure with everything)")
    print("  - model_comparison_summary.csv")

if __name__ == "__main__":
    main()