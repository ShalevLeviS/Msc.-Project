#!/usr/bin/env python3
# 01_Run_Models_for_prediction.py
# Trains an XGBoost PAM50 subtype classifier separately for each features column (.npz paths),
# evaluates on a held-out test set, and saves all artifacts needed for later visualization.
# Runs WITHOUT CLI args by using defaults below; CLI is optional.

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             log_loss, classification_report)

import xgboost as xgb

# ----------------- DEFAULTS (edit here if you like) -----------------
DEFAULT_CSV = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_all_tasks/ST_Image_CSV_paths.csv"
DEFAULT_OUT = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_Task3_H&E_to_Subtype"
DEFAULT_SEED = 1337
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1

# features columns per model (exact CSV column names expected)
FEATURE_COLS = {
    "retccl_resnet50":   "feat_retccl_resnet50_npz",
    "resnet50_imagenet": "feat_resnet50_imagenet_npz",
    "vit_b16_imagenet":  "feat_vit_b16_imagenet_npz",
}

# class set
VALID_CLASSES = ["LumA", "LumB", "Her2", "Basal", "Normal"]

# try these keys inside npz in order
NPZ_KEYS_TRY = ("features", "feat", "emb", "X", "arr_0")

# be quiet
warnings.filterwarnings("ignore")

NPZ_KEYS_TRY = ("feats", "features", "feat", "emb", "X", "arr_0")

# ----------------- utils -----------------
def load_npz_vector(npz_path: str, pool: str = "mean") -> np.ndarray:
    """Load tile embeddings from .npz and pool to a single slide vector."""
    if not npz_path or not Path(npz_path).exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    arr = None
    for k in NPZ_KEYS_TRY:
        if k in data:
            arr = data[k]
            break

    if arr is None:
        raise ValueError(
            f"No embedding array found in {npz_path}. "
            f"Tried keys: {NPZ_KEYS_TRY}. Available: {list(data.files)}"
        )

    # expected shape: [n_tiles, d] or [d]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        vec = arr.astype(np.float32)
    elif arr.ndim == 2:
        if pool == "mean":
            vec = np.nanmean(arr, axis=0).astype(np.float32)
        elif pool == "max":
            vec = np.nanmax(arr, axis=0).astype(np.float32)
        elif pool == "median":
            vec = np.nanmedian(arr, axis=0).astype(np.float32)
        else:
            raise ValueError("Unsupported pooling method")
    elif arr.ndim == 3:
        # ViT case: [n_tiles, n_patches, d]
        # Extract CLS token (index 0) and pool over tiles
        cls_tokens = arr[:, 0, :]  # [n_tiles, d]
        
        if pool == "mean":
            vec = np.nanmean(cls_tokens, axis=0).astype(np.float32)
        elif pool == "max":
            vec = np.nanmax(cls_tokens, axis=0).astype(np.float32)
        elif pool == "median":
            vec = np.nanmedian(cls_tokens, axis=0).astype(np.float32)
        else:
            raise ValueError("Unsupported pooling method")
    else:
        raise ValueError(f"Unexpected embedding shape {arr.shape} in {npz_path}")

    if not np.all(np.isfinite(vec)):
        raise ValueError(f"Non-finite values after pooling in {npz_path}")

    return vec

def build_dataset(master_csv: str, feature_col: str):
    """
    Return pooled features (X), encoded labels (y as strings), and sample_ids
    for rows that have valid labels + existing npz files.
    """
    df = pd.read_csv(master_csv)
    df = df[df["pam50_subtype"].isin(VALID_CLASSES)].copy()
    df = df[df[feature_col].notna() & (df[feature_col].astype(str) != "")]

    print(f"  📊 Found {len(df)} rows with valid labels and non-empty {feature_col}")

    X_list, y_list, ids = [], [], []
    error_counts = {}
    dim_counts = {}  # Track dimensions
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Load & pool ({feature_col})"):
        try:
            vec = load_npz_vector(str(row[feature_col]), pool="mean")
            if not np.all(np.isfinite(vec)):
                error_counts["non_finite"] = error_counts.get("non_finite", 0) + 1
                continue
            
            # Track dimension
            dim = len(vec)
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
            
            X_list.append(vec)
            y_list.append(row["pam50_subtype"])
            ids.append(row["sample_id"])
        except FileNotFoundError as e:
            error_counts["file_not_found"] = error_counts.get("file_not_found", 0) + 1
        except ValueError as e:
            error_type = str(e)[:50]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        except Exception as e:
            error_type = f"{type(e).__name__}: {str(e)[:30]}"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

    print(f"  ✅ Successfully loaded: {len(X_list)} samples")
    
    # Show dimension distribution
    if dim_counts:
        print(f"  📏 Feature dimensions found:")
        for dim, count in sorted(dim_counts.items()):
            print(f"     - {dim}-d: {count} samples")
    
    if error_counts:
        print(f"  ❌ Errors encountered:")
        for err, count in error_counts.items():
            print(f"     - {err}: {count} files")

    if len(X_list) == 0:
        return np.zeros((0,)), np.array([]), np.array([])
    
    # Find most common dimension
    most_common_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
    
    # Filter to only keep samples with the most common dimension
    X_filtered, y_filtered, ids_filtered = [], [], []
    for vec, label, sid in zip(X_list, y_list, ids):
        if len(vec) == most_common_dim:
            X_filtered.append(vec)
            y_filtered.append(label)
            ids_filtered.append(sid)
    
    if len(X_filtered) < len(X_list):
        print(f"  ⚠️  Filtered to {len(X_filtered)} samples with consistent {most_common_dim}-d features")
    
    X = np.vstack(X_filtered).astype(np.float32)
    y = np.array(y_filtered)
    sample_ids = np.array(ids_filtered)
    return X, y, sample_ids

def compute_sample_weights(y_enc: np.ndarray, n_classes: int):
    """
    Balanced sample weights (inverse frequency) for multiclass to mitigate imbalance.
    """
    counts = np.bincount(y_enc, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv[y_enc]
    # normalize weights to mean 1
    return weights * (len(y_enc) / weights.sum())

def save_json(path: Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def train_eval_xgb(X, y, sample_ids, out_dir, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE):
    """
    Train XGBoost multiclass on pooled features, evaluate on held-out test split,
    and save artifacts for later plotting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
    num_classes = len(le.classes_)
    if num_classes < 2:
        raise RuntimeError("Not enough classes to train.")

    # stratified splits: hold out TEST, then split TRAIN/VAL
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (trainval_idx, test_idx) = next(sss1.split(X, y_enc))

    X_trainval, y_trainval, sid_trainval = X[trainval_idx], y_enc[trainval_idx], sample_ids[trainval_idx]
    X_test, y_test, sid_test = X[test_idx], y_enc[test_idx], sample_ids[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    (train_idx, val_idx) = next(sss2.split(X_trainval, y_trainval))

    X_train, y_train, sid_train = X_trainval[train_idx], y_trainval[train_idx], sid_trainval[train_idx]
    X_val,   y_val,   sid_val   = X_trainval[val_idx],   y_trainval[val_idx],   sid_trainval[val_idx]

    # save splits
    save_json(out_dir / "splits.json", {
        "train": [str(x) for x in sid_train],
        "val":   [str(x) for x in sid_val],
        "test":  [str(x) for x in sid_test],
        "seed":  int(seed)
    })
    save_json(out_dir / "class_mapping.json", class_mapping)

    # Balanced sample weights
    w_train = compute_sample_weights(y_train, num_classes)
    w_val   = compute_sample_weights(y_val,   num_classes)

    # XGBoost params
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": seed,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval   = xgb.DMatrix(X_val,   label=y_val,   weight=w_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    evals = [(dtrain, "train"), (dval, "val")]
    bst = xgb.train(
        params, dtrain,
        num_boost_round=600,
        evals=evals,
        early_stopping_rounds=40,
        verbose_eval=False
    )

    # predict on test
    y_proba = bst.predict(dtest)  # [n_test, num_classes]
    y_pred = y_proba.argmax(axis=1)

    # ---------------- save core results ----------------
    # per-slide test results csv (your naming)
    test_results = pd.DataFrame({
        "sample_id": sid_test,
        "true_subtype": le.inverse_transform(y_test),
        "pred_subtype": le.inverse_transform(y_pred),
        "pred_prob": y_proba.max(axis=1),
    })
    test_results.to_csv(out_dir / "test_result_to_classification.csv", index=False)

    # metrics
    metrics = {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "log_loss": float(log_loss(y_test, y_proba)),
    }

    # per-class report
    clf_rep = classification_report(
        y_test, y_pred, target_names=list(le.classes_), output_dict=True, zero_division=0
    )
    per_class_rows = []
    for cls in le.classes_:
        row = clf_rep.get(cls, {"precision":0,"recall":0,"f1-score":0,"support":0})
        per_class_rows.append({
            "class": cls,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        })
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_report.csv", index=False)

    # confusion matrix (row-normalized)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    cm_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")

    # ROC & PR curves (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(num_classes)))
    roc_pack = {}
    pr_pack = {}
    macro_roc_aucs, macro_pr_aucs = [], []

    for i, cls in enumerate(le.classes_):
        fpr, tpr, thr = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(rec, prec)

        roc_pack[f"fpr_{cls}"] = fpr
        roc_pack[f"tpr_{cls}"] = tpr
        roc_pack[f"thr_{cls}"] = thr
        roc_pack[f"auc_{cls}"] = roc_auc

        pr_pack[f"prec_{cls}"] = prec
        pr_pack[f"rec_{cls}"]  = rec
        pr_pack[f"aucPR_{cls}"] = pr_auc

        macro_roc_aucs.append(roc_auc)
        macro_pr_aucs.append(pr_auc)

    metrics["macro_roc_auc"] = float(np.mean(macro_roc_aucs))
    metrics["macro_pr_auc"]  = float(np.mean(macro_pr_aucs))

    # save curve packs
    np.savez_compressed(out_dir / "roc_curves.npz", **roc_pack)
    np.savez_compressed(out_dir / "pr_curves.npz",  **pr_pack)

    # calibration & confidence histogram (top-class prob)
    top_conf = y_proba.max(axis=1)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins+1)
    bin_ids = np.digitize(top_conf, bins) - 1
    bin_means, bin_fracs, bin_counts = [], [], []
    correct = (y_pred == y_test).astype(int)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_means.append(float(top_conf[mask].mean()))
            bin_fracs.append(float(correct[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means.append(float((bins[b]+bins[b+1])/2))
            bin_fracs.append(np.nan)
            bin_counts.append(0)
    calib_df = pd.DataFrame({
        "bin_mean_confidence": bin_means,
        "bin_fraction_positive": bin_fracs,
        "count": bin_counts
    })
    calib_df.to_csv(out_dir / "calibration_curve.csv", index=False)

    hist_df = pd.DataFrame({
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "count": np.histogram(top_conf, bins=bins)[0]
    })
    hist_df.to_csv(out_dir / "confidence_hist.csv", index=False)

    # save metrics summary
    save_json(out_dir / "metrics.json", metrics)

    # also return to aggregate in a top-level comparison table
    return metrics, test_results

# ----------------- orchestration -----------------
def run_training_pipeline(csv_paths=DEFAULT_CSV,
                          out_root=DEFAULT_OUT,
                          seed=DEFAULT_SEED,
                          test_size=DEFAULT_TEST_SIZE,
                          val_size=DEFAULT_VAL_SIZE):
    os.makedirs(out_root, exist_ok=True)
    
    comparison_rows = []
    for model_name, feat_col in FEATURE_COLS.items():
        print(f"\n=== Processing model: {model_name} (column: {feat_col}) ===")
        out_dir = Path(out_root) / model_name

        X, y, sample_ids = build_dataset(csv_paths, feat_col)
        if X.ndim != 2 or len(y) == 0 or len(np.unique(y)) < 2:
            print("⚠️  Not enough data/classes to train. Skipping.")
            continue

        metrics, _ = train_eval_xgb(
            X, y, sample_ids, out_dir,
            seed=seed, test_size=test_size, val_size=val_size
        )
        metrics["model"] = model_name
        comparison_rows.append(metrics)

    # save model comparison table
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(Path(out_root) / "models_comparison.csv", index=False)
        print("\nSaved models_comparison.csv (high-level metrics per model).")
    else:
        print("\nNo models were trained (insufficient data?).")

# ----------------- CLI (optional) -----------------
def main_cli():
    ap = argparse.ArgumentParser(description="Train XGBoost subtype classifier per features column and save artifacts.")
    ap.add_argument("--csv-paths", required=True, help="Path to master CSV (pam50_subtype + npz paths).")
    ap.add_argument("--out-root", default=DEFAULT_OUT, help="Output root dir (artifacts saved per model).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    args = ap.parse_args()

    run_training_pipeline(
        csv_paths=args.csv_paths,
        out_root=args.out_root,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size
    )

# ----------------- entry -----------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # no CLI args -> run with defaults declared at top
        run_training_pipeline(
            csv_paths=DEFAULT_CSV,
            out_root=DEFAULT_OUT,
            seed=DEFAULT_SEED,
            test_size=DEFAULT_TEST_SIZE,
            val_size=DEFAULT_VAL_SIZE
        )
    else:
        main_cli()
