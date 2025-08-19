#!/usr/bin/env python3
"""
Auto-detect and sanity-check pipeline outputs (tagged or untagged).
Looks in OUTDIR for the most recently modified matching file per artifact.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---------- Config ----------
OUTDIR = Path("wsi_geneexp_pipeline")  # change if needed


# ---------- Helpers ----------
def find_latest(*patterns):
    """
    Return the most-recently modified file in OUTDIR matching any of the glob patterns.
    Example: find_latest("tile_features__*.npz", "tile_features.npz")
    """
    cands = []
    for pat in patterns:
        cands.extend(OUTDIR.glob(pat))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def show_npz_tile_features(p):
    npz = np.load(p, allow_pickle=True)
    ids, feats = npz["ids"], npz["feats"]
    print(f"Tile features [{p.name}]: feats {feats.shape} | ids {ids.shape}")
    if feats.size:
        print("  preview tile id:", ids[0])


def show_tile_to_slide(p):
    df = pd.read_csv(p)
    print(f"Tile→Slide map [{p.name}]: {df.shape}")
    print(df.head(3))


def show_missing_json(p):
    missing = json.load(open(p))
    print(f"Missing PAM50 [{p.name}]: {missing}")


def show_cv_csv(p, label):
    df = pd.read_csv(p)
    print(f"{label} [{p.name}]: {df.shape}  (columns={len(df.columns)})")
    print(df.head(2))


def show_attn_reps(p):
    npz = np.load(p, allow_pickle=True)
    slides, reps = npz["slides"], npz["reps"]
    print(f"Slide reps [{p.name}]: reps {reps.shape} | slides {slides.shape}")
    if len(slides):
        print("  first slide:", slides[0])


def show_pred_table(p):
    df = pd.read_csv(p, index_col=0)
    print(f"Predicted PAM50 table [{p.name}]: {df.shape}")
    print("  first 5 columns:", df.columns[:5].tolist())
    print(df.head(2))


def show_pickle_df(p, label):
    df = pd.read_pickle(p)
    print(f"{label} [{p.name}]: {df.shape}")
    print(df.head(2))


# ---------- Checks ----------
print("\n=== CHUNK 1 & 2: Inputs and Features ===")

gexp_pkl   = find_latest("gexp_all__*.pkl", "gexp_all.pkl", "gexp_pam50__*.pkl", "gexp_pam50.pkl")
coords_pkl = find_latest("coords_all__*.pkl", "coords_all.pkl")
missing_js = find_latest("missing_pam50__*.json", "missing_pam50.json")
t2s_csv    = find_latest("tile_to_slide__*.csv", "tile_to_slide.csv")
feat_npz   = find_latest("tile_features__*.npz", "tile_features.npz")
tile_ids   = find_latest("tile_ids__*.csv", "tile_ids.csv")

if gexp_pkl:   show_pickle_df(gexp_pkl, "Gene expression (tile-level, PAM50 cols)")
else:          print("Missing gene expression pickle.")
if coords_pkl: show_pickle_df(coords_pkl, "Coordinates (tile-level)")
else:          print("Missing coords pickle.")
if missing_js: show_missing_json(missing_js)
else:          print("Missing missing_pam50.json.")
if t2s_csv:    show_tile_to_slide(t2s_csv)
else:          print("Missing tile_to_slide.csv.")
if feat_npz:   show_npz_tile_features(feat_npz)
else:          print("Missing tile_features npz.")
if tile_ids:
    df = pd.read_csv(tile_ids)
    print(f"Tile IDs [{tile_ids.name}]: {df.shape}")
    print(df.head(3))
else:
    print("Missing tile_ids.csv.")


print("\n=== CHUNK 3: Mean-pool XGBoost CV ===")
cv_meanpool = find_latest("cv_gene_r_meanpool_xgb__*.csv", "cv_gene_r_meanpool_xgb.csv")
if cv_meanpool: show_cv_csv(cv_meanpool, "Mean-pool CV")
else:           print("Missing mean-pool CV results CSV.")


print("\n=== CHUNK 4: Attention MIL Reps ===")
attn_npz = find_latest("slide_reps_attention__*.npz", "slide_reps_attention.npz")
if attn_npz: show_attn_reps(attn_npz)
else:        print("Missing attention reps npz.")


print("\n=== CHUNK 5: XGBoost on Attention Reps CV ===")
cv_attn = find_latest("cv_gene_r_attnrep_xgb__*.csv", "cv_gene_r_attnrep_xgb.csv")
if cv_attn: show_cv_csv(cv_attn, "Attention reps CV")
else:       print("Missing attention reps CV results CSV.")


print("\n=== CHUNK 6: Final Predictions ===")
pred_csv = find_latest("predicted_pam50_table__*.csv", "predicted_pam50_table.csv")
if pred_csv: show_pred_table(pred_csv)
else:        print("Missing predicted PAM50 table CSV.")

print("\n=== DONE ===")
