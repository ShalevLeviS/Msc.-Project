#!/usr/bin/env python3
# 02_train_gene_regressors_xgb_optimized.py
# Train XGBoost multi-target regressors with auto-detection of CPU cores (32-256)
# Optimized for gene expression prediction from H&E image features

import os
import json
import warnings
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from scipy.stats import pearsonr, spearmanr
import joblib

warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise RuntimeError("xgboost is required. Install with: pip install xgboost") from e

# ----------------------- AUTO CPU DETECTION -----------------------
def get_optimal_cpu_config():
    """
    Auto-detect available CPU cores and configure optimal parallelization.
    Returns (n_parallel_genes, cores_per_gene) based on available cores.
    """
    total_cores = os.cpu_count() or 32  # Fallback to 32 if detection fails
    
    # Optimal configurations based on total cores
    if total_cores >= 256:
        # 256 cores: train 32 genes at once, 8 cores each
        return 32, 8
    elif total_cores >= 128:
        # 128 cores: train 16 genes at once, 8 cores each
        return 16, 8
    elif total_cores >= 64:
        # 64 cores: train 8 genes at once, 8 cores each
        return 8, 8
    elif total_cores >= 32:
        # 32 cores: train 4 genes at once, 8 cores each
        return 4, 8
    else:
        # < 32 cores: train 2 genes at once, remaining cores each
        cores_per_gene = max(1, total_cores // 2)
        return 2, cores_per_gene

N_PARALLEL_GENES, CORES_PER_GENE = get_optimal_cpu_config()
TOTAL_CORES = os.cpu_count() or 32

print(f"\n{'='*70}")
print(f"🖥️  CPU CONFIGURATION AUTO-DETECTED")
print(f"{'='*70}")
print(f"Total CPU cores available: {TOTAL_CORES}")
print(f"Parallel genes (workers): {N_PARALLEL_GENES}")
print(f"Cores per gene: {CORES_PER_GENE}")
print(f"Total cores utilized: {N_PARALLEL_GENES * CORES_PER_GENE}")
print(f"{'='*70}\n")

# ----------------------- PAM50 GENES -----------------------
PAM50_GENES = [
    'ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
    'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
    'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
    'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
    'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'NDC80', 'NUF2', 'ORC6',
    'PGR', 'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B',
    'TYMS', 'UBE2C', 'UBE2T'
]

# ----------------------- DEFAULT PATHS -----------------------
DEFAULT_PATHS_CSV = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/ST_Image_CSV_paths.csv"
DEFAULT_FEATS_ROOT = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/Task1-from_H&E_Images_to_predicted_Gene_Expression"
DEFAULT_RESULTS_ROOT = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/Task1-from_H&E_Images_to_predicted_Gene_Expression"
DEFAULT_FE_MODELS = ["uni_v2", "gigapath_tile", "retccl_resnet50"]
SEED = 42

# ----------------------- ID NORMALIZATION -----------------------
_BARCODE_RE = re.compile(r"([ACGT]+-\d+)$", re.IGNORECASE)

def _canon_id(x: str) -> str:
    """Canonicalize an ID string for matching."""
    s = str(x).strip().strip('"').strip("'").lower()
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try: 
            return str(int(float(s)))
        except: 
            return s
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s

def _extract_barcode_like(s: str) -> str:
    """Extract barcode-like suffix (e.g., ACGT-1234) from string."""
    m = _BARCODE_RE.search(str(s))
    return m.group(1).lower() if m else str(s).lower()

def _best_orientation_and_ids(gexp: pd.DataFrame, coords_idx: pd.Index) -> Tuple[pd.DataFrame, List[str], str]:
    """Determine if gene expression matrix needs transposing to have tiles as rows."""
    coords_ids = [_canon_id(i) for i in coords_idx.astype(str)]
    coords_set = set(coords_ids)

    idx_ids = [_canon_id(i) for i in gexp.index.astype(str)]
    col_ids = [_canon_id(c) for c in gexp.columns.astype(str)]
    inter_i = len(set(idx_ids) & coords_set)
    inter_c = len(set(col_ids) & coords_set)

    if inter_c > inter_i:
        gexp = gexp.T
        return gexp, [_canon_id(c) for c in gexp.index.astype(str)], "columns_as_tiles_transposed"
    return gexp, idx_ids, "index_as_tiles"

def _resolve_tile_ids_robust(gexp_tiles: pd.DataFrame, coords_idx: pd.Index) -> Tuple[List[str], Dict]:
    """Match gene expression tile IDs to coordinate tile IDs using multiple strategies."""
    coords_raw = coords_idx.astype(str).tolist()
    coords_canon = [_canon_id(x) for x in coords_raw]
    coords_bar = [_extract_barcode_like(x) for x in coords_raw]
    map_canon_to_orig: Dict[str, str] = {}
    map_bar_to_orig: Dict[str, str] = {}
    for orig, cn, br in zip(coords_raw, coords_canon, coords_bar):
        map_canon_to_orig.setdefault(cn, orig)
        map_bar_to_orig.setdefault(br, orig)

    idx_raw = gexp_tiles.index.astype(str).tolist()
    idx_canon = [_canon_id(x) for x in idx_raw]
    idx_bar = [_extract_barcode_like(x) for x in idx_raw]

    A = [map_canon_to_orig[i] for i in idx_canon if i in map_canon_to_orig]
    B = [map_bar_to_orig[i] for i in idx_bar if i in map_bar_to_orig]
    chosen = "A_canonical" if len(A) >= len(B) else "B_barcode_suffix"
    tile_ids = A if len(A) >= len(B) else B
    return tile_ids, {"A_matches": len(A), "B_matches": len(B), "chosen": chosen}

def load_gene_as_tiles_index(gene_path: str, coords_index: pd.Index) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Load gene expression CSV and orient it so tiles are rows."""
    g = pd.read_csv(gene_path, index_col=0)
    g.index = g.index.astype(str).str.strip()
    g.columns = g.columns.astype(str).str.strip()
    g_tiles, _, orient = _best_orientation_and_ids(g, coords_index)
    tile_ids, dbg = _resolve_tile_ids_robust(g_tiles, coords_index)
    dbg["orient"] = orient
    return g_tiles, tile_ids, dbg

# ----------------------- DATA ASSEMBLY -----------------------
def assemble_dataset(paths_csv: str, feats_root: str, fe_model: str, 
                     filter_pam50: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Assemble features and gene expression data.
    
    Args:
        paths_csv: Path to CSV with sample_id and gene_exp_path columns
        feats_root: Root directory containing feature NPZ files
        fe_model: Feature extraction model name (subdirectory)
        filter_pam50: If True, only keep PAM50 genes in the output
    
    Returns:
        X (n, d): Feature matrix
        Y (n, G): Target gene expression (log1p transformed)
        groups (n,): Sample ID for each tile
        genes (G,): Gene names
        rows_meta: DataFrame with sample_id and tile_id for each row
    """
    df = pd.read_csv(paths_csv)
    need = {"sample_id", "gene_exp_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"{paths_csv} must contain columns: {need}")

    feats_dir = Path(feats_root) / fe_model
    if not feats_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {feats_dir}")

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    groups: List[str] = []
    metas: List[Dict[str, str]] = []
    genes_ref: List[str] = []

    print(f"\n{'='*60}")
    print(f"Assembling dataset for: {fe_model}")
    if filter_pam50:
        print(f"Filtering to PAM50 genes only ({len(PAM50_GENES)} genes)")
    print(f"{'='*60}\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{fe_model}] Loading samples"):
        sample_id = str(row["sample_id"])
        gene_path = str(row["gene_exp_path"])
        npz_path = feats_dir / f"{sample_id}.npz"
        if not npz_path.exists():
            continue

        # Load features
        z = np.load(npz_path, allow_pickle=True)
        feats = z["feats"]  # (N_tiles, D_features)
        tile_ids_npz = [str(x) for x in z["tile_ids"].tolist()]
        if feats.size == 0 or len(tile_ids_npz) == 0:
            continue

        # Use NPZ tile IDs as coordinate reference
        coords_index = pd.Index(tile_ids_npz, name="tile_id")

        # Load and orient gene expression matrix
        g_tiles, tile_ids, dbg = load_gene_as_tiles_index(gene_path, coords_index)
        if len(tile_ids) == 0:
            continue

        # Ensure numeric values and handle duplicate gene columns
        g_tiles = g_tiles.apply(pd.to_numeric, errors="coerce")
        if not g_tiles.columns.is_unique:
            # Average duplicate gene columns
            g_tiles = g_tiles.T.groupby(level=0).mean().T

        # Filter to PAM50 genes if requested
        if filter_pam50:
            available_pam50 = [g for g in PAM50_GENES if g in g_tiles.columns]
            if len(available_pam50) == 0:
                continue  # Skip samples with no PAM50 genes
            g_tiles = g_tiles[available_pam50]

        # Align tile ordering between features and gene expression
        keep_set = set(tile_ids)
        valid_tids = [t for t in tile_ids_npz if t in keep_set]
        if not valid_tids:
            continue

        order = [i for i, t in enumerate(tile_ids_npz) if t in keep_set]
        Xs = feats[order, :]  # Select matching tiles from features

        # Build target frame in same row order as valid_tids
        Y_frame = g_tiles.loc[valid_tids]  # tiles × genes
        cur_genes = list(Y_frame.columns)

        # Establish global gene order across all samples
        if len(genes_ref) == 0:
            genes_ref = cur_genes
        elif cur_genes != genes_ref:
            # Align to reference gene list, fill missing with 0
            Y_frame = Y_frame.reindex(columns=genes_ref).fillna(0.0)

        # Log1p transform
        Ys = np.log1p(Y_frame.values)

        # Accumulate data
        X_list.append(Xs)
        Y_list.append(Ys)
        groups.extend([sample_id] * len(valid_tids))
        metas.extend([{"sample_id": sample_id, "tile_id": t} for t in valid_tids])

    if len(X_list) == 0:
        raise RuntimeError(f"No data assembled for feature model '{fe_model}'. Check NPZs and CSVs.")

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    groups_arr = np.array(groups)
    rows_meta = pd.DataFrame(metas)

    print(f"\n✓ Dataset assembled:")
    print(f"  - {len(X)} tiles")
    print(f"  - {X.shape[1]} features")
    print(f"  - {len(genes_ref)} genes")
    if filter_pam50:
        print(f"  - Filtered to PAM50 genes")
    print(f"  - {len(np.unique(groups_arr))} unique samples\n")

    return X, Y, groups_arr, genes_ref, rows_meta

# ----------------------- PARALLEL TRAINING -----------------------
def train_single_gene(args: Tuple[int, str, np.ndarray, np.ndarray, Dict]) -> Tuple[int, str, XGBRegressor]:
    """
    Train one XGBoost model for a single gene.
    This function is called in parallel by ProcessPoolExecutor.
    """
    gene_idx, gene_name, X_train, y_train, base_params = args
    mdl = XGBRegressor(**base_params)
    mdl.fit(X_train, y_train, verbose=False)
    return gene_idx, gene_name, mdl

class MultiXGBParallel:
    """
    Train multiple XGBoost models in parallel.
    
    Optimized for multi-core systems (32-256 cores).
    """
    def __init__(self, base_params: Dict, n_targets: int, n_workers: int = N_PARALLEL_GENES):
        self.base_params = dict(base_params)
        self.n_targets = int(n_targets)
        self.n_workers = n_workers
        self.models: List[Optional[XGBRegressor]] = [None] * n_targets

    def fit(self, X: np.ndarray, Y: np.ndarray, desc: str = "Training models") -> "MultiXGBParallel":
        """Fit all models in parallel."""
        # Prepare tasks for parallel execution
        tasks: List[Tuple[int, str, np.ndarray, np.ndarray, Dict]] = []
        for j in range(self.n_targets):
            tasks.append((j, f"gene_{j}", X, Y[:, j], self.base_params))
        
        print(f"\n{'='*60}")
        print(f"PARALLEL TRAINING CONFIGURATION:")
        print(f"  - Total genes: {self.n_targets}")
        print(f"  - Parallel workers: {self.n_workers} genes at once")
        print(f"  - Cores per gene: {self.base_params.get('n_jobs', 1)}")
        print(f"  - Total cores used: {self.n_workers * self.base_params.get('n_jobs', 1)}")
        print(f"{'='*60}\n")
        
        # Train models in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(train_single_gene, task): task for task in tasks}
            
            with tqdm(total=self.n_targets, desc=desc, ncols=100) as pbar:
                for future in as_completed(futures):
                    gene_idx, gene_name, model = future.result()
                    self.models[gene_idx] = model
                    pbar.update(1)
        
        print(f"\n✓ All {self.n_targets} models trained successfully!\n")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using all trained models."""
        preds: List[np.ndarray] = []
        for mdl in tqdm(self.models, desc="Predicting", total=len(self.models), ncols=100):
            if mdl is None:
                raise ValueError("Model not trained")
            pred = mdl.predict(X)
            pred_array = np.asarray(pred, dtype=np.float64)
            pred_reshaped = np.reshape(pred_array, (-1, 1))
            preds.append(pred_reshaped)
        return np.hstack(preds)

# ----------------------- METRICS -----------------------
def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, kind: str = "pearson") -> float:
    """Safely compute correlation, return NaN on error."""
    try:
        if kind == "pearson":
            result = pearsonr(y_true, y_pred)
            # Handle both old scipy (returns tuple) and new scipy (returns object with attributes)
            if hasattr(result, 'statistic'):
                return float(result.statistic)  # type: ignore
            elif hasattr(result, 'correlation'):
                return float(result.correlation)  # type: ignore
            else:
                # Old scipy: returns (r, p) tuple
                return float(result[0])  # type: ignore
        else:
            result = spearmanr(y_true, y_pred)
            # Handle both old scipy (returns tuple) and new scipy (returns object with attributes)
            if hasattr(result, 'statistic'):
                return float(result.statistic)  # type: ignore
            elif hasattr(result, 'correlation'):
                return float(result.correlation)  # type: ignore
            else:
                # Old scipy: returns (r, p) tuple
                return float(result[0])  # type: ignore
    except Exception:
        return float('nan')

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error in original scale."""
    yt = np.expm1(y_true)
    yp = np.expm1(y_pred)
    denom = np.clip(np.abs(yt), 1e-8, None)
    return float(np.mean(np.abs(yt - yp) / denom))

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, genes: List[str], 
                       level: str, split: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute comprehensive metrics for all genes.
    
    Returns:
        main_metrics: DataFrame with per-gene metrics
        residuals_df: DataFrame with per-prediction residuals
    """
    main_rows = []
    residual_rows = []
    
    for j, gene_name in enumerate(tqdm(genes, desc=f"Computing {level}-level metrics", ncols=100)):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        
        # Calculate residuals
        residuals = yt - yp
        abs_errors = np.abs(residuals)
        
        # Relative error (in original scale)
        yt_orig = np.expm1(yt)
        yp_orig = np.expm1(yp)
        rel_errors = np.abs(yt_orig - yp_orig) / np.clip(np.abs(yt_orig), 1e-8, None) * 100
        
        # Main metrics
        main_rows.append({
            "gene_name": gene_name,
            "split": split,
            "level": level,
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if np.std(yt) > 0 else np.nan,
            "pearson_r": _safe_corr(yt, yp, "pearson"),
            "spearman_r": _safe_corr(yt, yp, "spearman"),
            "mape": _safe_mape(yt, yp),
            "evs": float(explained_variance_score(yt, yp)),
            "n_samples": len(yt),
            "mean_abs_error": float(np.mean(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "std_error": float(np.std(abs_errors)),
            "max_error": float(np.max(abs_errors)),
            "q95_error": float(np.percentile(abs_errors, 95))
        })
        
        # Residuals (save for each prediction)
        for i in range(len(yt)):
            residual_rows.append({
                "gene_name": gene_name,
                "split": split,
                "level": level,
                "sample_id": meta.iloc[i]["sample_id"] if "sample_id" in meta.columns else "",
                "tile_id": meta.iloc[i]["tile_id"] if "tile_id" in meta.columns else "",
                "y_true": float(yt[i]),
                "y_pred": float(yp[i]),
                "residual": float(residuals[i]),
                "abs_error": float(abs_errors[i]),
                "rel_error_pct": float(rel_errors[i])
            })
    
    return pd.DataFrame(main_rows), pd.DataFrame(residual_rows)

def compute_r2_distribution(main_metrics: pd.DataFrame, split: str, level: str) -> pd.DataFrame:
    """Compute R² distribution bins for each gene."""
    sub = main_metrics[(main_metrics["split"] == split) & (main_metrics["level"] == level)]
    
    rows = []
    for _, row in sub.iterrows():
        r2 = row["r2"]
        if pd.isna(r2):
            r2_bin = "missing"
        elif r2 < 0:
            r2_bin = "lt_0"
        elif r2 < 0.2:
            r2_bin = "0_02"
        elif r2 < 0.4:
            r2_bin = "02_04"
        elif r2 < 0.6:
            r2_bin = "04_06"
        elif r2 < 0.8:
            r2_bin = "06_08"
        else:
            r2_bin = "08_10"
        
        rows.append({
            "gene_name": row["gene_name"],
            "split": split,
            "level": level,
            "r2": r2,
            "r2_bin": r2_bin
        })
    
    return pd.DataFrame(rows)

# ----------------------- MAIN TRAINING FUNCTION -----------------------
def train_xgb_for_feature_model(paths_csv: str, feats_root: str, results_root: str,
                                fe_model: str, val_frac: float = 0.2, seed: int = SEED,
                                use_parallel: bool = True, filter_pam50: bool = False,
                                skip_if_exists: bool = True):
    """
    Train XGBoost models for gene expression prediction.
    
    Args:
        paths_csv: CSV with sample_id and gene_exp_path columns
        feats_root: Root directory with feature NPZ files
        results_root: Where to save results
        fe_model: Feature extraction model name
        val_frac: Validation split fraction
        seed: Random seed
        use_parallel: Use parallel training
        filter_pam50: Only predict PAM50 genes
        skip_if_exists: Skip if model already trained (default: True)
    """
    # Check if model already exists
    pam50_suffix = "_pam50" if filter_pam50 else ""
    model_dir = Path(results_root) / f"{fe_model}{pam50_suffix}"
    
    if skip_if_exists and model_dir.exists():
        # Check for required output files
        existing_runs = sorted(model_dir.glob("*/"), key=lambda x: x.name, reverse=True)
        
        if existing_runs:
            latest_run = existing_runs[0]
            required_files = [
                latest_run / "trained_models.pkl",
                latest_run / "config.json",
                latest_run / "metrics_csv" / "main_metrics.csv",
                latest_run / "metrics_csv" / "residuals.csv"
            ]
            
            if all(f.exists() for f in required_files):
                print(f"\n{'='*70}")
                print(f"✓ SKIPPING {fe_model} - Already trained!")
                print(f"  Found existing results at: {latest_run}")
                print(f"  Use skip_if_exists=False to retrain")
                print(f"{'='*70}\n")
                return
    
    print(f"\n{'#'*70}")
    print(f"# TRAINING: {fe_model}")
    print(f"# PAM50 only: {filter_pam50}")
    print(f"# Parallel: {use_parallel}")
    print(f"{'#'*70}\n")

    # Assemble dataset
    X, Y, groups, genes, rows_meta = assemble_dataset(
        paths_csv, feats_root, fe_model, filter_pam50=filter_pam50
    )

    # Train/validation split (random at tile level)
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True)
    Xtr, Xva = X[tr_idx], X[va_idx]
    Ytr, Yva = Y[tr_idx], Y[va_idx]
    meta_tr = rows_meta.iloc[tr_idx].reset_index(drop=True)
    meta_va = rows_meta.iloc[va_idx].reset_index(drop=True)

    print(f"Split: {len(tr_idx)} train tiles, {len(va_idx)} validation tiles\n")

    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pam50_suffix = "_pam50" if filter_pam50 else ""
    run_dir = Path(results_root) / f"{fe_model}{pam50_suffix}" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create flat metrics directory
    metrics_dir = run_dir / "metrics_csv"
    metrics_dir.mkdir(exist_ok=True)

    # XGBoost parameters
    base_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": CORES_PER_GENE,
        "tree_method": "hist",
        "verbosity": 0
    }
    
    # Save configuration
    config = {
        "timestamp": ts,
        "seed": seed,
        "fe_model": fe_model,
        "regressor": "xgb",
        "filter_pam50": filter_pam50,
        "reg_params": base_params,
        "cpu_config": {
            "total_cores_available": TOTAL_CORES,
            "use_parallel": use_parallel,
            "n_parallel_genes": N_PARALLEL_GENES if use_parallel else 1,
            "cores_per_gene": CORES_PER_GENE,
            "total_cores_used": N_PARALLEL_GENES * CORES_PER_GENE if use_parallel else CORES_PER_GENE
        },
        "target_transform": "log1p",
        "genes": genes,
        "n_genes": len(genes),
        "split": {"type": "random", "val_frac": val_frac},
        "data_stats": {
            "n_train_tiles": len(tr_idx),
            "n_val_tiles": len(va_idx),
            "n_features": X.shape[1],
            "n_samples": len(np.unique(groups))
        }
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Train models
    if use_parallel:
        mxgb = MultiXGBParallel(base_params, n_targets=len(genes), n_workers=N_PARALLEL_GENES)
    else:
        print("Warning: Sequential mode - will be slower than parallel.")
        # Simple sequential wrapper
        class MultiXGBSequential:
            def __init__(self, base_params: Dict, n_targets: int):
                self.base_params = base_params
                self.n_targets = n_targets
                self.models: List[XGBRegressor] = []
            
            def fit(self, X: np.ndarray, Y: np.ndarray, desc: str = "Training") -> "MultiXGBSequential":
                self.models = []
                for j in tqdm(range(self.n_targets), desc=desc, ncols=100):
                    mdl = XGBRegressor(**self.base_params)
                    mdl.fit(X, Y[:, j], verbose=False)
                    self.models.append(mdl)
                return self
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                preds: List[np.ndarray] = []
                for mdl in tqdm(self.models, desc="Predicting", ncols=100):
                    pred = mdl.predict(X)
                    pred_array = np.asarray(pred, dtype=np.float64)
                    pred_reshaped = np.reshape(pred_array, (-1, 1))
                    preds.append(pred_reshaped)
                return np.hstack(preds)
        
        mxgb = MultiXGBSequential(base_params, n_targets=len(genes))
    
    mxgb.fit(Xtr, Ytr, desc=f"[{fe_model}] Training XGBoost")

    # Save trained models (for later feature importance extraction)
    print("\nSaving models...")
    joblib.dump({
        "models": mxgb.models,
        "params": base_params,
        "genes": genes,
        "filter_pam50": filter_pam50,
        "feature_names": [f"feat_{i}" for i in range(X.shape[1])]
    }, run_dir / "trained_models.pkl", compress=3)
    print(f"✓ Saved: trained_models.pkl")

    # Make predictions
    print("\nMaking predictions...")
    Yhat_va = mxgb.predict(Xva)

    # Compute tile-level metrics
    print("\n{'='*60}")
    print("COMPUTING TILE-LEVEL METRICS")
    print(f"{'='*60}")
    main_metrics_tile, residuals_tile = compute_all_metrics(
        Yva, Yhat_va, genes, level="tile", split="val", meta=meta_va
    )
    
    # Aggregate to slide-level
    print(f"\n{'='*60}")
    print("AGGREGATING TO SLIDE-LEVEL")
    print(f"{'='*60}")
    
    slide_data = []
    slide_meta = []
    
    for gene_idx, gene_name in enumerate(tqdm(genes, desc="Aggregating genes", ncols=100)):
        tmp = pd.DataFrame({
            "sample_id": meta_va["sample_id"].values,
            "y": Yva[:, gene_idx],
            "p": Yhat_va[:, gene_idx]
        })
        grp = tmp.groupby("sample_id", sort=False).mean(numeric_only=True)
        
        if gene_idx == 0:
            slide_meta = [{"sample_id": sid} for sid in grp.index]
        
        slide_data.append({
            "y": grp["y"].values,
            "p": grp["p"].values
        })
    
    # Build slide-level arrays
    n_samples_slide = len(slide_meta)
    Yva_slide = np.zeros((n_samples_slide, len(genes)))
    Yhat_slide = np.zeros((n_samples_slide, len(genes)))
    
    for gene_idx in range(len(genes)):
        Yva_slide[:, gene_idx] = slide_data[gene_idx]["y"]
        Yhat_slide[:, gene_idx] = slide_data[gene_idx]["p"]
    
    meta_slide = pd.DataFrame(slide_meta)
    
    # Compute slide-level metrics
    print(f"\n{'='*60}")
    print("COMPUTING SLIDE-LEVEL METRICS")
    print(f"{'='*60}")
    main_metrics_slide, residuals_slide = compute_all_metrics(
        Yva_slide, Yhat_slide, genes, level="slide", split="val", meta=meta_slide
    )
    
    # Combine all metrics
    main_metrics_all = pd.concat([main_metrics_tile, main_metrics_slide], ignore_index=True)
    residuals_all = pd.concat([residuals_tile, residuals_slide], ignore_index=True)
    
    # Compute R² distribution
    r2_dist_tile = compute_r2_distribution(main_metrics_all, "val", "tile")
    r2_dist_slide = compute_r2_distribution(main_metrics_all, "val", "slide")
    r2_dist_all = pd.concat([r2_dist_tile, r2_dist_slide], ignore_index=True)
    
    # Save all metrics to flat CSV files
    print(f"\n{'='*60}")
    print("SAVING METRICS TO CSV")
    print(f"{'='*60}")
    
    main_metrics_all.to_csv(metrics_dir / "main_metrics.csv", index=False)
    print(f"✓ Saved: main_metrics.csv ({len(main_metrics_all)} rows)")
    
    residuals_all.to_csv(metrics_dir / "residuals.csv", index=False)
    print(f"✓ Saved: residuals.csv ({len(residuals_all)} rows)")
    
    r2_dist_all.to_csv(metrics_dir / "r2_distribution.csv", index=False)
    print(f"✓ Saved: r2_distribution.csv ({len(r2_dist_all)} rows)")
    
    # Save split assignment
    split_df = rows_meta.copy()
    split_df["split"] = "train"
    split_df.loc[va_idx, "split"] = "val"
    split_df.to_csv(metrics_dir / "split_assignment.csv", index=False)
    print(f"✓ Saved: split_assignment.csv")
    
    # Compute overall summary statistics
    tile_summary = main_metrics_tile.describe().T
    slide_summary = main_metrics_slide.describe().T
    
    tile_summary.to_csv(metrics_dir / "summary_tile.csv")
    slide_summary.to_csv(metrics_dir / "summary_slide.csv")
    print(f"✓ Saved: summary_tile.csv, summary_slide.csv")
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY FOR {fe_model}")
    print(f"{'='*70}")
    print(f"Tile-level:")
    print(f"  R² (mean):      {main_metrics_tile['r2'].mean():.4f}")
    print(f"  R² (median):    {main_metrics_tile['r2'].median():.4f}")
    print(f"  Pearson (mean): {main_metrics_tile['pearson_r'].mean():.4f}")
    print(f"  RMSE (mean):    {main_metrics_tile['rmse'].mean():.4f}")
    print(f"\nSlide-level:")
    print(f"  R² (mean):      {main_metrics_slide['r2'].mean():.4f}")
    print(f"  R² (median):    {main_metrics_slide['r2'].median():.4f}")
    print(f"  Pearson (mean): {main_metrics_slide['pearson_r'].mean():.4f}")
    print(f"  RMSE (mean):    {main_metrics_slide['rmse'].mean():.4f}")
    print(f"{'='*70}\n")

    # Save to comparison directory
    compare_dir = Path(results_root) / "compare_results"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # Append to comparison summary
    summ_row = {
        "timestamp": ts,
        "fe_model": fe_model,
        "regressor": "xgb",
        "filter_pam50": filter_pam50,
        "n_genes": len(genes),
        "n_train_tiles": len(tr_idx),
        "n_val_tiles": len(va_idx),
        "tile_r2_mean": float(main_metrics_tile['r2'].mean()),
        "tile_r2_median": float(main_metrics_tile['r2'].median()),
        "tile_pearson_mean": float(main_metrics_tile['pearson_r'].mean()),
        "tile_rmse_mean": float(main_metrics_tile['rmse'].mean()),
        "slide_r2_mean": float(main_metrics_slide['r2'].mean()),
        "slide_r2_median": float(main_metrics_slide['r2'].median()),
        "slide_pearson_mean": float(main_metrics_slide['pearson_r'].mean()),
        "slide_rmse_mean": float(main_metrics_slide['rmse'].mean()),
    }
    
    summ_path = compare_dir / "comparison_summary.csv"
    if summ_path.exists():
        old = pd.read_csv(summ_path)
        pd.concat([old, pd.DataFrame([summ_row])], ignore_index=True).to_csv(summ_path, index=False)
    else:
        pd.DataFrame([summ_row]).to_csv(summ_path, index=False)
    
    print(f"✓ Appended to: {summ_path}")
    
    print(f"\n{'#'*70}")
    print(f"# COMPLETED: {fe_model}")
    print(f"# Metrics saved to: {metrics_dir}")
    print(f"# Models saved to: {run_dir / 'trained_models.pkl'}")
    print(f"{'#'*70}\n")

# ----------------------- SCRIPT ENTRY POINT -----------------------
if __name__ == "__main__":
    # Run directly from code (no CLI)
    RUN_FROM_CODE = True
    
    if RUN_FROM_CODE:
        print(f"\n{'#'*70}")
        print(f"# STARTING TRAINING FOR ALL MODELS")
        print(f"# Skip if exists: TRUE")
        print(f"{'#'*70}\n")
        
        # Train on PAM50 genes only
        for fe_model in DEFAULT_FE_MODELS:
            try:
                train_xgb_for_feature_model(
                    paths_csv=DEFAULT_PATHS_CSV,
                    feats_root=DEFAULT_FEATS_ROOT,
                    results_root=DEFAULT_RESULTS_ROOT,
                    fe_model=fe_model,
                    val_frac=0.2,
                    seed=SEED,
                    use_parallel=True,
                    filter_pam50=True,  # PAM50 GENES ONLY
                    skip_if_exists=True  # ← SKIP IF ALREADY TRAINED
                )
            except Exception as e:
                print(f"\n{'!'*70}")
                print(f"! ERROR in {fe_model}: {e}")
                print(f"! Continuing with next model...")
                print(f"{'!'*70}\n")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'#'*70}")
        print(f"# ALL MODELS COMPLETED (or skipped)")
        print(f"{'#'*70}\n")
    else:
        print("Set RUN_FROM_CODE=True or implement CLI with argparse")