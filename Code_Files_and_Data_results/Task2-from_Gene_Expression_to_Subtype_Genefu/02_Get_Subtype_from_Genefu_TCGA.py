#!/usr/bin/env python3
# pam50_from_star_pipeline.py
# Read TCGA STAR-Counts TSVs (paths listed in image_to_star_counts.csv),
# run PAM50 (genefu centroids via rpy2) per sample, and SAVE a NEW CSV
# with added columns: pam50_subtype, pam50_prob, pam50_status.
#
# Default input:  /local1/ofir/shalevle/TCGA/TCGA-Data/image_to_star_counts.csv
# Default output: /local1/ofir/shalevle/TCGA/Data_for_Task3_H&E_to_Subtype/<base>_with_pam50.csv

import argparse, os, sys, shutil, subprocess, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======== DEFAULTS ========
DEFAULT_IN_CSV  = "/local1/ofir/shalevle/TCGA/TCGA-Data/image_to_star_counts.csv"
DEFAULT_OUT_DIR = "/local1/ofir/shalevle/TCGA/Data_for_Task3_H&E_to_Subtype"
STAR_PATH_COL   = "star_count_path"
MIN_PAM50_GENES = 35

# ======== quiet Rscript runner (HPC-safe) ========
def _run_rscript_quiet(cmd_list):
    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith("BASH_FUNC_") or k in ("ENV", "BASH_ENV"):
            env.pop(k, None)
    env.setdefault("LC_ALL", "C")
    return subprocess.check_output(cmd_list, stderr=subprocess.STDOUT, text=True, env=env)

# ======== Ensure R deps (rpy2, genefu) ========
def ensure_r_pkg(package, bioc=False, verbose=True):
    if shutil.which("Rscript") is None:
        if verbose:
            print("❌ Rscript not found on PATH. Install R (e.g., apt-get install r-base).")
        return False
    try:
        _run_rscript_quiet(["Rscript", "-e", f"suppressPackageStartupMessages(library({package}))"])
        return True
    except subprocess.CalledProcessError:
        pass

    if verbose:
        print(f"⏳ Installing R package: {package} {'(Bioconductor)' if bioc else ''} ...")

    try:
        if bioc:
            out = _run_rscript_quiet([
                "Rscript","-e",
                'if (!requireNamespace("BiocManager", quietly=TRUE)) '
                'install.packages("BiocManager", repos="https://cloud.r-project.org"); '
                f'BiocManager::install("{package}", ask=FALSE, update=FALSE, quiet=TRUE); '
                f'suppressMessages(library({package})); cat("OK\\n")'
            ])
        else:
            out = _run_rscript_quiet([
                "Rscript","-e",
                'options(repos="https://cloud.r-project.org"); '
                f'install.packages("{package}", quiet=TRUE); '
                f'suppressMessages(library({package})); cat("OK\\n")'
            ])
        return "OK" in out
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"❌ Failed to install {package}.")
            print(e.output[-800:])
        return False

def ensure_genefu(verbose=True):
    # rpy2 python-side
    try:
        import rpy2  # noqa: F401
    except Exception:
        if verbose: print("⏳ Installing rpy2 ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "rpy2>=3.5"])
        except subprocess.CalledProcessError:
            if verbose:
                print("❌ Failed to install rpy2. Ensure R + dev headers are available.")
            return False
    # R genefu
    return ensure_r_pkg("genefu", bioc=True, verbose=verbose)

# ======== Load PAM50 centroids via rpy2 (cached) ========
_PAM50_CENTROIDS = None  # DataFrame rows=genes, cols=subtypes
_PAM50_LEGAL = None      # set of gene symbols present in centroids

def pam50_setup():
    global _PAM50_CENTROIDS, _PAM50_LEGAL
    if _PAM50_CENTROIDS is not None and _PAM50_LEGAL is not None:
        return
    if not ensure_genefu(verbose=False):
        raise RuntimeError("genefu not available")
    from rpy2.robjects import r
    r('suppressPackageStartupMessages(library(genefu))')
    r('suppressMessages(data(pam50))')
    r('if (!exists("pam50.robust")) pam50.robust <- pam50')
    legal = list(r('rownames(pam50$centroids)'))
    centroid_names = list(r('colnames(pam50$centroids)'))
    vals = []
    for i in range(len(centroid_names)):
        vals.append(list(r(f'pam50$centroids[, {i+1}]')))
    _PAM50_LEGAL = set(legal)
    _PAM50_CENTROIDS = pd.DataFrame({name: col for name, col in zip(centroid_names, vals)}, index=legal)

# ======== Optional Ensembl→Symbol mapping (fallback only) ========
def map_ensembl_to_symbol(ensembl_ids):
    """Best-effort map Ensembl IDs (no version) to HGNC symbols using org.Hs.eg.db."""
    if not ensembl_ids:
        return {}
    ok = ensure_r_pkg("org.Hs.eg.db", bioc=True, verbose=False) and ensure_r_pkg("AnnotationDbi", bioc=True, verbose=False)
    if not ok:
        return {}
    from rpy2.robjects import r, StrVector
    ids = list(ensembl_ids)
    rids = StrVector(ids)
    r('suppressPackageStartupMessages(library(org.Hs.eg.db)); suppressPackageStartupMessages(library(AnnotationDbi))')
    r.assign("ids_in", rids)
    res = r('as.character(AnnotationDbi::mapIds(org.Hs.eg.db, keys=ids_in, keytype="ENSEMBL", column="SYMBOL", multiVals="first"))')
    mapping = {}
    for i, k in enumerate(ids):
        val = str(res[i]) if i < len(res) else "NA"
        if val != "NA" and val != "None":
            mapping[k] = val
    return mapping

# ======== Normalization & Classifier ========
def enhanced_pam50_normalization(expr_series: pd.Series, legal_probes_set: set):
    x = expr_series[expr_series.index.isin(legal_probes_set)]
    if x.empty: return pd.Series(dtype=float), {}
    total = x.sum()
    if total <= 0: return pd.Series(dtype=float), {}
    cpm = (x / total) * 1e6
    log_cpm = np.log2(cpm + 1.0)
    med = log_cpm.median()
    centered = log_cpm - med
    mad = np.median(np.abs(centered - centered.median()))
    scaled = centered / (1.4826 * mad) if mad > 0 else centered
    return scaled, {
        "total_counts": float(total),
        "median_log_cpm": float(med),
        "mad": float(mad),
        "n_genes_normalized": int(len(scaled))
    }

def classify_pam50_python(expr_norm: pd.Series, centroids_df: pd.DataFrame):
    common = expr_norm.index.intersection(centroids_df.index)
    if len(common) < MIN_PAM50_GENES: return None, np.nan, {}
    x = expr_norm.loc[common]
    cors = {st: float(x.corr(centroids_df.loc[common, st], method='spearman')) for st in centroids_df.columns}
    subtype = max(cors, key=cors.get)
    arr = np.clip(np.array([cors[s] for s in centroids_df.columns]), -0.999, 0.999)
    z = np.arctanh(arr)
    ez = np.exp(z - z.max())
    den = ez.sum()
    prob = float(ez[list(centroids_df.columns).index(subtype)] / den) if den > 0 else np.nan
    return subtype, prob, cors

# ======== STAR-Counts TSV loader → Series(index=SYMBOL, value=counts) ========
# --- REPLACE ONLY THIS FUNCTION ---
def load_star_counts_series(tsv_path: str) -> pd.Series:
    """
    Load a STAR-Counts file where the real header might appear on line 6,
    and where gene_name and unstranded are the 3rd and 4th columns.
    Returns Series indexed by SYMBOL (upper) with raw counts.
    """
    if not tsv_path or not Path(tsv_path).exists():
        raise FileNotFoundError(f"STAR TSV missing: {tsv_path}")

    # 1) Find the header row (the one containing both 'gene_name' and 'unstranded')
    header_idx = None
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = [p.strip().lower() for p in line.rstrip("\n").split("\t")]
            if "gene_name" in parts and "unstranded" in parts:
                header_idx = i
                break
    # fallback to line 5 (0-based) if not found (your file has header at 6th line)
    if header_idx is None:
        header_idx = 5

    # 2) Read the table using the detected header row
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=header_idx,
        dtype=str,
        engine="python"
    )

    # 3) Identify symbol and count columns
    cols_lower = {c.lower(): c for c in df.columns}
    sym_col = cols_lower.get("gene_name")
    cnt_col = cols_lower.get("unstranded")

    # If names are missing, fall back to positions: 3rd (index 2) & 4th (index 3)
    if sym_col is None or cnt_col is None:
        if len(df.columns) < 4:
            raise ValueError(f"Unexpected STAR file format. Columns: {list(df.columns)}")
        sym_col = sym_col or df.columns[2]  # 3rd column
        cnt_col = cnt_col or df.columns[3]  # 4th column

    # 4) Drop banner / summary / comment rows
    def _bad_symbol(s: str) -> bool:
        s = str(s)
        return (
            s.startswith("N_") or     # N_unmapped, N_multimapping, etc.
            s.startswith("__") or     # __no_feature (other tools)
            s.startswith("#") or      # '# gene-model: ...' if slipped into data
            s.strip().lower() in {"(comment)", "comment"}
        )

    # Prefer to filter by gene_name; if it's NaN, also try gene_id if present
    df = df[~df[sym_col].fillna("").map(_bad_symbol)]
    if "gene_id" in df.columns:
        df = df[~df["gene_id"].fillna("").map(_bad_symbol)]

    # 5) Build symbols (uppercase) and numeric counts
    symbols = df[sym_col].astype(str).str.strip().str.upper()
    counts = pd.to_numeric(df[cnt_col], errors="coerce")

    # 6) Aggregate duplicates and clean NaNs
    ser = pd.Series(counts.values, index=symbols.values)
    ser = ser[ser.index.notna()]
    ser = ser.groupby(ser.index).sum(min_count=1)
    ser = ser[np.isfinite(ser)]
    return ser
# --- END REPLACEMENT ---


# ======== Main runner ========
def run_pam50_for_star_counts(pairs_csv: str,
                              out_dir: str,
                              n_samples: int = None,
                              overwrite_inplace: bool = False) -> str:
    pam50_setup()  # ensures centroids + legal

    df = pd.read_csv(pairs_csv)
    if STAR_PATH_COL not in df.columns:
        raise KeyError(f"Column '{STAR_PATH_COL}' not found in {pairs_csv}")

    work = df.copy()
    if n_samples is not None:
        work = work.head(int(n_samples)).copy()

    # Ensure output columns exist
    for col in ["pam50_subtype","pam50_prob","pam50_status"]:
        if col not in df.columns:
            df[col] = np.nan

    results = []
    for idx, row in tqdm(work.iterrows(), total=len(work), desc="PAM50 (STAR-Counts)"):
        sid = str(row.get("sample_id", row.get("case_id", row.get("submitter_id", idx))))
        tsv = str(row[STAR_PATH_COL])

        if not isinstance(tsv, str) or not tsv or not Path(tsv).exists():
            results.append((idx, sid, np.nan, np.nan, "no_star_tsv"))
            continue
        try:
            ser = load_star_counts_series(tsv)
            expr_norm, _stats = enhanced_pam50_normalization(ser, _PAM50_LEGAL)
            if len(expr_norm) < MIN_PAM50_GENES:
                results.append((idx, sid, np.nan, np.nan, f"few_pam50_genes ({len(expr_norm)})"))
                continue
            subtype, prob, _cors = classify_pam50_python(expr_norm, _PAM50_CENTROIDS)
            if subtype is None:
                results.append((idx, sid, np.nan, np.nan, f"few_pam50_genes ({len(expr_norm)})"))
                continue
            results.append((idx, sid, subtype, prob, "ok"))
        except Exception as e:
            results.append((idx, sid, np.nan, np.nan, f"error: {e}"))

    # Apply results to full df (keep row order)
    for idx, sid, subtype, prob, status in results:
        df.at[idx, "pam50_subtype"] = subtype
        df.at[idx, "pam50_prob"]    = prob
        df.at[idx, "pam50_status"]  = status

    # Save
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if overwrite_inplace:
        out_path = pairs_csv
    else:
        base = Path(pairs_csv).stem
        out_path = str(Path(out_dir) / f"{base}_with_pam50.csv")
    df.to_csv(out_path, index=False)
    return out_path

# ======== CLI ========
def main():
    ap = argparse.ArgumentParser(description="PAM50 subtyping from TCGA STAR-Counts TSVs listed in a pairs CSV.")
    ap.add_argument("--in-csv", default=DEFAULT_IN_CSV, help="Path to image_to_star_counts.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where to save the NEW CSV")
    ap.add_argument("--n-samples", type=int, default=None, help="Limit number of rows processed (for smoke tests)")
    ap.add_argument("--overwrite-inplace", action="store_true", help="Overwrite the input CSV instead of writing a new one")
    args = ap.parse_args()

    try:
        out_path = run_pam50_for_star_counts(
            pairs_csv=args.in_csv,
            out_dir=args.out_dir,
            n_samples=args.n_samples,
            overwrite_inplace=args.overwrite_inplace
        )
        print(f"✅ Saved: {out_path}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
