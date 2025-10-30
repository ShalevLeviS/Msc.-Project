#!/usr/bin/env python3
# pam50_inplace_pipeline.py
# Pools tile-level expression -> slide-level, runs PAM50 using genefu centroids (Python-side),
# writes results IN-PLACE to your CSV: pam50_subtype, pam50_prob.
# No extra files created.

import argparse
import os
import sys
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# rpy2 (only to load pam50 centroids once)
from rpy2.robjects import r
from rpy2.robjects.packages import importr

warnings.filterwarnings("ignore")

# ======== DEFAULTS ========
DEFAULT_CSV = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_all_tasks/ST_Image_CSV_paths.csv"

# ======== quiet Rscript runner to avoid HPC shell noise ========
def _run_rscript_quiet(cmd_list):
    env = os.environ.copy()
    # Strip exported bash functions that confuse /bin/sh in batch envs
    for k in list(env.keys()):
        if k.startswith("BASH_FUNC_") or k in ("ENV", "BASH_ENV"):
            env.pop(k, None)
    env.setdefault("LC_ALL", "C")
    return subprocess.check_output(cmd_list, stderr=subprocess.STDOUT, text=True, env=env)

# ======== ensure genefu available (no molecular.subtyping used) ========
def ensure_genefu(verbose=True) -> bool:
    try:
        import rpy2  # noqa: F401
    except Exception:
        if verbose: print("⏳ Installing rpy2 ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "rpy2>=3.5"])
        except subprocess.CalledProcessError:
            if verbose:
                print("❌ Failed to install rpy2. Please ensure R and Python dev headers are available.")
            return False

    if shutil.which("Rscript") is None:
        if verbose:
            print("❌ Rscript not found on PATH. Install R / Rscript (e.g., apt-get install r-base).")
        return False

    try:
        r('suppressPackageStartupMessages(library(genefu))')
        if verbose: print("✅ genefu already available.")
        return True
    except Exception:
        pass

    if verbose: print("⏳ Installing genefu via BiocManager ...")
    try:
        out = _run_rscript_quiet([
            "Rscript", "-e",
            'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", repos="https://cloud.r-project.org"); '
            'BiocManager::install("genefu", ask=FALSE, update=FALSE, quiet=TRUE); '
            'suppressMessages(library(genefu)); cat("OK\\n")'
        ])
        if "OK" in out:
            if verbose: print("✅ genefu installed.")
            return True
    except subprocess.CalledProcessError as e:
        if verbose:
            print("❌ genefu install failed.")
            print(e.output[-800:])
    return False

# ======== Load pam50 centroids (once) and legal probes from genefu ========
_PAM50_CENTROIDS = None  # pandas DataFrame: rows=probes, cols=subtypes
_PAM50_LEGAL = None      # set of probe IDs present in centroids

def pam50_setup():
    """Load genefu::pam50 and cache pam50$centroids + legal probe IDs."""
    global _PAM50_CENTROIDS, _PAM50_LEGAL
    if _PAM50_CENTROIDS is not None and _PAM50_LEGAL is not None:
        return

    if not ensure_genefu(verbose=False):
        raise RuntimeError("genefu not available")

    r('suppressPackageStartupMessages(library(genefu))')
    r('suppressMessages(data(pam50))')
    # Some installations miss pam50.robust; not needed here, but align with your working env:
    r('if (!exists("pam50.robust")) pam50.robust <- pam50')

    legal = list(r('rownames(pam50$centroids)'))
    centroid_names = list(r('colnames(pam50$centroids)'))
    vals = []
    for i in range(len(centroid_names)):
        vals.append(list(r(f'pam50$centroids[, {i+1}]')))

    _PAM50_LEGAL = set(legal)
    _PAM50_CENTROIDS = pd.DataFrame(
        {name: col for name, col in zip(centroid_names, vals)},
        index=legal
    )

# ======== Robust pooling from tiles -> slide ========
def _winsorize(x, p=0.05):
    if len(x) == 0:
        return x
    lo = np.nanpercentile(x, 100 * p, method="linear")
    hi = np.nanpercentile(x, 100 * (1 - p), method="linear")
    return np.clip(x, lo, hi)

def robust_pool_tile_to_slide(tile_by_gene: pd.DataFrame, method="trimmed_mean", trim=0.1, winsor=0.05):
    """
    tile_by_gene: rows=tiles, cols=genes. Returns pd.Series of pooled values per gene.
    """
    df = tile_by_gene.apply(pd.to_numeric, errors="coerce")
    pooled = {}
    for g in df.columns:
        col = df[g].values.astype(float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            pooled[g] = np.nan
            continue
        if method == "trimmed_mean":
            col_w = _winsorize(col, winsor)
            if col_w.size < 5:
                pooled[g] = float(np.nanmedian(col_w))
            else:
                k = int(np.floor(trim * col_w.size))
                col_s = np.sort(col_w)
                col_t = col_s[k: col_s.size - k] if col_s.size - 2 * k > 0 else col_s
                pooled[g] = float(np.nanmean(col_t))
        elif method == "median":
            pooled[g] = float(np.nanmedian(col))
        elif method == "mean":
            pooled[g] = float(np.nanmean(col))
        else:
            raise ValueError("Unsupported pooling method")
    return pd.Series(pooled)

# ======== Robust loader for per-tile gene tables (tiles x genes) ========
def load_tile_gene_table(path: str) -> pd.DataFrame:
    """
    Robust loader for per-tile gene tables:
      - auto-detect delimiter (CSV/TSV), gz supported
      - tries to use gene symbol/name columns if present (gene_name, symbol, hgnc_symbol, ...)
      - decides whether genes are rows or columns by matching PAM50 probes
      - returns a DataFrame shaped: tiles x genes
    """
    # 1) read with sniffed separator
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep="\t")

    df_orig = df.copy()

    # 2) Identify potential gene ID columns (prefer symbol-like)
    gene_cols_pref = [
        "gene_symbol","hgnc_symbol","symbol","gene_name","Gene","GENE","gene",
        "feature_name","feature","gene_id","ensembl","ensembl_id","ENSG","id","ID"
    ]
    present_gene_cols = [c for c in df.columns if c in gene_cols_pref or c.lower() in gene_cols_pref]
    chosen_gene_col = None
    if present_gene_cols:
        for pref in ["gene_symbol","symbol","hgnc_symbol","gene_name","GENE","Gene","gene"]:
            if any(c.lower() == pref for c in present_gene_cols):
                chosen_gene_col = next(c for c in present_gene_cols if c.lower() == pref)
                break
        if chosen_gene_col is None:
            chosen_gene_col = present_gene_cols[0]

    # 3) If we found a gene column, assume genes are rows → set index to that col
    if chosen_gene_col is not None:
        df = df.set_index(chosen_gene_col)
        # keep only numeric columns (tiles)
        keep_numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if keep_numeric:
            df = df[keep_numeric]

    # 4) Decide orientation by overlap with pam50 legal probes (case-insensitive)
    def _ci_set(lst): return {str(x).strip().upper() for x in lst}
    # For orientation we don't strictly need pam50 yet, but using it helps
    pam = set(x.upper() for x in (_PAM50_LEGAL or []))
    overlap_rows = len(pam & _ci_set(df.index))
    overlap_cols = len(pam & _ci_set(df.columns))

    if overlap_rows == 0 and overlap_cols == 0:
        # try original frame / transposed with first column as gene
        ov_rows0 = len(pam & _ci_set(df_orig.index))
        ov_cols0 = len(pam & _ci_set(df_orig.columns))
        if ov_cols0 > ov_rows0:
            df = df_orig  # genes as columns
        else:
            df = df_orig.set_index(df_orig.columns[0]).T  # first col is gene -> transpose
        overlap_rows = len(pam & _ci_set(df.index))
        overlap_cols = len(pam & _ci_set(df.columns))

    # 5) Normalize to tiles x genes
    if overlap_rows > overlap_cols:
        df = df.T  # rows were genes -> transpose to tiles x genes

    # 6) Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ======== Normalization (your CPM→log2→median center→robust scale) ========
def enhanced_pam50_normalization(patient_expr_series: pd.Series, legal_probes_set: set):
    """Normalize a slide-level counts vector on PAM50 probes only."""
    pam50_expr = patient_expr_series[patient_expr_series.index.isin(legal_probes_set)]
    if pam50_expr.empty:
        return pd.Series(dtype=float), {}

    total_counts = pam50_expr.sum()
    if total_counts <= 0:
        return pd.Series(dtype=float), {}

    cpm = (pam50_expr / total_counts) * 1e6
    log_cpm = np.log2(cpm + 1.0)
    median_expr = log_cpm.median()
    centered = log_cpm - median_expr
    mad = np.median(np.abs(centered - centered.median()))
    scaled = centered / (1.4826 * mad) if mad > 0 else centered

    stats = {
        "total_counts": float(total_counts),
        "median_log_cpm": float(median_expr),
        "mad": float(mad),
        "n_genes_normalized": int(len(scaled))
    }
    return scaled, stats

# ======== Python-side PAM50 classifier (Spearman vs centroids) ========
def classify_pam50_python(expr_norm: pd.Series, centroids_df: pd.DataFrame):
    """
    expr_norm: normalized PAM50 vector (index = probe IDs)
    centroids_df: rows=probes, cols=subtypes
    Returns: (subtype:str|None, prob:float|nan, corr_map:dict)
    """
    # align genes
    common = expr_norm.index.intersection(centroids_df.index)
    if len(common) < 35:
        return None, np.nan, {}

    x = expr_norm.loc[common]
    cors = {}
    for st in centroids_df.columns:
        y = centroids_df.loc[common, st]
        corr = x.corr(y, method='spearman')
        cors[st] = 0.0 if pd.isna(corr) else float(corr)

    # choose best subtype
    subtype = max(cors, key=cors.get)
    # convert correlations into a probability-like score via Fisher-z + softmax
    arr = np.array([cors[s] for s in centroids_df.columns])
    arr = np.clip(arr, -0.999, 0.999)
    z = np.arctanh(arr)
    ez = np.exp(z - z.max())
    probs = ez / ez.sum()
    prob_map = {st: float(p) for st, p in zip(centroids_df.columns, probs)}
    return subtype, prob_map.get(subtype, np.nan), cors

# ======== Main in-place pipeline ========
def run_pipeline_inplace(csv_paths: str,
                         pooling_method: str = "trimmed_mean",
                         trim: float = 0.1,
                         winsor: float = 0.05,
                         min_genes_required: int = 35,
                         sample_id: str = None,
                         n_samples: int = None) -> pd.DataFrame:
    """
    Update ST_Image_CSV_paths.csv IN-PLACE with PAM50 results.
      - Adds/overwrites: pam50_subtype (str), pam50_prob (float)
    Optionally restrict processing to a single sample_id OR the first N rows (for testing).
    """
    # Ensure centroids ready
    pam50_setup()

    meta_full = pd.read_csv(csv_paths)
    if sample_id is not None:
        meta = meta_full[meta_full["sample_id"].astype(str) == str(sample_id)].copy()
    elif n_samples is not None:
        meta = meta_full.head(int(n_samples)).copy()
    else:
        meta = meta_full.copy()

    if "pam50_subtype" not in meta_full.columns:
        meta_full["pam50_subtype"] = np.nan
    if "pam50_prob" not in meta_full.columns:
        meta_full["pam50_prob"] = np.nan

    results = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="PAM50 subtyping"):
        sid = row.get('sample_id', Path(str(row.get('image_path', ''))).stem)
        gpath = str(row.get('gene_exp_path', '') or '')

        if not gpath or not Path(gpath).exists():
            results.append({'sample_id': sid, 'pam50_subtype': np.nan, 'pam50_prob': np.nan, 'status': 'no_gene_table'})
            continue

        try:
            # 1) tiles x genes
            tiles_by_gene = load_tile_gene_table(gpath)

            # 2) pool tiles -> slide counts
            pooled = robust_pool_tile_to_slide(tiles_by_gene, method=pooling_method, trim=trim, winsor=winsor)

            # 3) normalize on PAM50 probes only
            expr_norm, norm_stats = enhanced_pam50_normalization(pooled, _PAM50_LEGAL)
            if len(expr_norm) < min_genes_required:
                results.append({'sample_id': sid, 'pam50_subtype': np.nan, 'pam50_prob': np.nan,
                                'status': f'few_pam50_genes ({len(expr_norm)})'})
                continue

            # 4) classify in Python against genefu centroids
            subtype, pam_prob, corr_map = classify_pam50_python(expr_norm, _PAM50_CENTROIDS)
            if subtype is None:
                results.append({'sample_id': sid, 'pam50_subtype': np.nan, 'pam50_prob': np.nan,
                                'status': f'few_pam50_genes ({len(expr_norm)})'})
                continue

            # record + write back to the master CSV
            results.append({'sample_id': sid, 'pam50_subtype': subtype, 'pam50_prob': pam_prob, 'status': 'ok'})

            mask = meta_full["sample_id"].astype(str) == str(sid)
            meta_full.loc[mask, "pam50_subtype"] = subtype
            meta_full.loc[mask, "pam50_prob"] = pam_prob

        except Exception as e:
            results.append({'sample_id': sid, 'pam50_subtype': np.nan, 'pam50_prob': np.nan, 'status': f'error: {e}'})
            # do not raise; continue to next sample

    # Save IN PLACE (only this CSV updated; no extra files)
    meta_full.to_csv(csv_paths, index=False)
    return pd.DataFrame(results)

# ======== CLI ========
def main_cli():
    ap = argparse.ArgumentParser(
        description="Pool tiles -> slide, classify PAM50 using genefu centroids (Python), update CSV in place."
    )
    ap.add_argument("--csv-paths", default=None,
                    help=f"Path to ST_Image_CSV_paths.csv. If omitted, uses DEFAULT_CSV: {DEFAULT_CSV}")
    ap.add_argument("--pool", default="trimmed_mean", choices=["trimmed_mean", "median", "mean"])
    ap.add_argument("--trim", type=float, default=0.1)
    ap.add_argument("--winsor", type=float, default=0.05)
    ap.add_argument("--min-genes", type=int, default=35,
                    help="Minimum PAM50 genes present to attempt subtyping (default=35).")
    ap.add_argument("--sample-id", default=None, help="Process only this sample_id (testing).")
    ap.add_argument("--n-samples", type=int, default=None, help="Process only the first N samples (testing).")
    args = ap.parse_args()

    csv_path = args.csv_paths or DEFAULT_CSV
    # Ensure centroids load once (also checks genefu presence)
    pam50_setup()

    summary = run_pipeline_inplace(
        csv_paths=csv_path,
        pooling_method=args.pool,
        trim=args.trim,
        winsor=args.winsor,
        min_genes_required=args.min_genes,
        sample_id=args.sample_id,
        n_samples=args.n_samples
    )
    print("✅ Updated CSV in place:", csv_path)
    try:
        print(summary.head())
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pam50_setup()
        _summary = run_pipeline_inplace(
            csv_paths=DEFAULT_CSV,
            pooling_method="trimmed_mean",
            trim=0.1,
            winsor=0.05,
            min_genes_required=35,
            # n_samples=1,   # <— add this to limit default run
        )
        print("✅ Updated CSV in place:", DEFAULT_CSV)
        try:
            print(_summary.head())
        except Exception:
            pass
    else:
        main_cli()
