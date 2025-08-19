# ====================== CHUNK 1 (subset + caching) ======================
# Purpose: Load paths + data, optional subsampling, focus PAM50, build tile→slide map
# -----------------------------------------------------------------------
import os, re, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------- CONFIG ----------
PATH_CSV = Path("/local1/ofir/shalevle/STImage_1K4M/ST_Image_CSV_paths.csv")
OUTDIR   = Path("./wsi_geneexp_pipeline"); OUTDIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 224
BATCH    = 64
SEED     = 42

# Subset controls (set ONE of the two below; both None = use ALL samples)
SUBSET_SAMPLE_IDS = None                 # e.g., ["TCGA-XX-0001", "TCGA-YY-0002", ...]
SUBSET_N_SAMPLES  = None                   # e.g., 10 (ignored if SUBSET_SAMPLE_IDS is not None)
SUBSET_SEED       = 123                  # reproducible sampling

# Limit tiles per slide (None = keep all tiles). If set, selects a random subset per slide.
TILE_CAP_PER_SLIDE = None               # e.g., 500 or 1000

# PAM50 genes
PAM50 = [
 "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6",
 "CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1",
 "FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT","MDM2","MELK",
 "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6L",
 "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T"
]

# ---------- HELPERS ----------
tile_re = re.compile(r"(.+?)_\d+x\d+$")  # "…_10x13" → slide base
def slide_from_tile(tile_id: str) -> str:
    m = tile_re.match(tile_id)
    return m.group(1) if m else tile_id

def _subset_tag(sample_ids, tile_cap):
    """Stable tag for cache filenames."""
    if sample_ids is None:
        base = "all"
    else:
        h = hashlib.md5(json.dumps(sorted(sample_ids)).encode()).hexdigest()[:8]
        base = f"ids{len(sample_ids)}_{h}"
    if tile_cap is not None:
        base += f"_tilecap{int(tile_cap)}"
    return base

# ---------- LOAD PATHS CSV ----------
paths_df = pd.read_csv(PATH_CSV)
paths_df.columns = [c.strip() for c in paths_df.columns]
assert {"sample_id","image_path","coord_path","gene_exp_path"}.issubset(paths_df.columns), \
    "CSV must have sample_id,image_path,coord_path,gene_exp_path"

# ---------- APPLY SUBSET (by sample_id) ----------
if SUBSET_SAMPLE_IDS is not None:
    keep_ids = [sid for sid in SUBSET_SAMPLE_IDS if sid in set(paths_df["sample_id"])]
else:
    if SUBSET_N_SAMPLES is not None:
        n = min(int(SUBSET_N_SAMPLES), len(paths_df))
        keep_ids = paths_df["sample_id"].sample(n=n, random_state=SUBSET_SEED).tolist()
    else:
        keep_ids = None

if keep_ids is not None:
    paths_df = paths_df[paths_df["sample_id"].isin(keep_ids)].copy()
else:
    keep_ids = paths_df["sample_id"].tolist()  # all samples

print(f"[Subset] Using {len(keep_ids)} sample(s).")

sample2img   = dict(zip(paths_df["sample_id"], paths_df["image_path"]))
sample2coord = dict(zip(paths_df["sample_id"], paths_df["coord_path"]))
sample2gexp  = dict(zip(paths_df["sample_id"], paths_df["gene_exp_path"]))

# ---------- CACHE PATHS (tagged by subset/tile cap) ----------
TAG           = _subset_tag(keep_ids, TILE_CAP_PER_SLIDE)
GEXP_CACHE    = OUTDIR / f"gexp_pam50__{TAG}.pkl"
COORDS_CACHE  = OUTDIR / f"coords_all__{TAG}.pkl"
MISSING_JSON  = OUTDIR / f"missing_pam50__{TAG}.json"
TILE2SLIDE_CS = OUTDIR / f"tile_to_slide__{TAG}.csv"

# ---------- LOAD OR BUILD DATA ----------
if GEXP_CACHE.exists() and COORDS_CACHE.exists() and MISSING_JSON.exists() and TILE2SLIDE_CS.exists():
    gexp_all   = pd.read_pickle(GEXP_CACHE)
    coords_all = pd.read_pickle(COORDS_CACHE)
    with open(MISSING_JSON) as f: missing = json.load(f)
    pam50_cols = [g for g in PAM50 if g in gexp_all.columns]
    print(f"[Chunk 1] Using cached subset='{TAG}'. PAM50 present: {len(pam50_cols)}/50 | missing: {len(missing)}")
else:
    def read_all_gene_exp() -> pd.DataFrame:
        dfs = []
        ALIAS = {"ORC6L": "ORC6"}
        for sid, gpath in tqdm(sample2gexp.items(),
                               total=len(sample2gexp),
                               desc="Reading gene expression (PAM50 only)"):
            hdr = pd.read_csv(gpath, nrows=0)
            cols = set(hdr.columns)

            to_read = ['Unnamed: 0']
            rename_map = {}
            for g in PAM50:
                if g in cols:
                    to_read.append(g)
                alias = ALIAS.get(g)
                if alias and alias in cols:
                    to_read.append(alias)
                    rename_map[alias] = g

            if len(to_read) == 1:
                continue

            df = pd.read_csv(
                gpath,
                usecols=to_read,
                index_col=0,
                dtype={c: np.float32 for c in to_read if c != 'Unnamed: 0'}
            )
            if rename_map:
                df = df.rename(columns=rename_map)
            dfs.append(df)

        out = pd.concat(dfs, axis=0, copy=False)
        present = [g for g in PAM50 if g in out.columns]
        return out[present]

    def read_all_coords() -> pd.DataFrame:
        dfs = []
        for sid, cpath in tqdm(sample2coord.items(),
                               total=len(sample2coord),
                               desc="Reading coords"):
            df = pd.read_csv(cpath, index_col=0)
            df.columns = [c.strip().lower() for c in df.columns]
            df = df[['yaxis', 'xaxis', 'r']].astype(np.float32)
            dfs.append(df)
        return pd.concat(dfs, axis=0, copy=False)

    gexp_all   = read_all_gene_exp()
    coords_all = read_all_coords()

    # sync tiles
    common_tiles = gexp_all.index.intersection(coords_all.index)
    gexp_all   = gexp_all.loc[common_tiles]
    coords_all = coords_all.loc[common_tiles]

    # alias safety
    ALIAS = {"ORC6L": "ORC6"}
    for wanted, actual in ALIAS.items():
        if wanted not in gexp_all.columns and actual in gexp_all.columns:
            gexp_all = gexp_all.rename(columns={actual: wanted})

    # focus PAM50 + report missing
    pam50_cols = [g for g in PAM50 if g in gexp_all.columns]
    missing = sorted(set(PAM50) - set(pam50_cols))
    print(f"PAM50 present: {len(pam50_cols)}/50 | missing: {len(missing)}")

    # optional: cap tiles per slide
    if TILE_CAP_PER_SLIDE is not None:
        rng = np.random.default_rng(SUBSET_SEED)
        tile_to_slide_ser = gexp_all.index.to_series().str.replace(r'_\d+x\d+$', '', regex=True)
        groups = {}
        for tile, slide in zip(tile_to_slide_ser.index, tile_to_slide_ser.values):
            groups.setdefault(slide, []).append(tile)

        kept_tiles = []
        for slide, tiles in groups.items():
            k = min(int(TILE_CAP_PER_SLIDE), len(tiles))
            if k < len(tiles):
                sel = rng.choice(tiles, size=k, replace=False)
                kept_tiles.extend(sel.tolist())
            else:
                kept_tiles.extend(tiles)
        kept_tiles = list(dict.fromkeys(kept_tiles))  # stable unique
        gexp_all   = gexp_all.loc[kept_tiles]
        coords_all = coords_all.loc[kept_tiles]
        print(f"[Subset] Tile cap per slide = {TILE_CAP_PER_SLIDE} → kept {len(kept_tiles)} tiles total.")

    # tile→slide mapping
    tile_to_slide = gexp_all.index.to_series().str.replace(r'_\d+x\d+$', '', regex=True).rename('slide')
    tile_to_slide.to_csv(TILE2SLIDE_CS)

    # cache dataframes for future fast path
    gexp_all.to_pickle(GEXP_CACHE)
    coords_all.to_pickle(COORDS_CACHE)
    with open(MISSING_JSON, "w") as f: json.dump(missing, f, indent=2)

print("############ Chunk 1 Completed ############")


# ====================== CHUNK 2 (subset-aware + caching) ======================
# Purpose: Create tile dataset → extract ViT/DINO features → save .npz
# -------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# feature caches tagged by subset
FEAT_NPZ     = OUTDIR / f"tile_features__{TAG}.npz"
TILE_IDS_CSV = OUTDIR / f"tile_ids__{TAG}.csv"

if FEAT_NPZ.exists() and TILE_IDS_CSV.exists():
    print(f"[Chunk 2] Cached features found for subset='{TAG}' → skipping extraction.\n"
          f" - {FEAT_NPZ}\n - {TILE_IDS_CSV}")
else:
    tx = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

    class TileDataset(Dataset):
        def __init__(self, gexp_df, coords_df, slide2img_map):
            self.tiles  = gexp_df.index.tolist()
            self.coords = coords_df
            self.slide2img = {sid: Path(p) for sid, p in slide2img_map.items()}
            keep = []
            for t in self.tiles:
                s = slide_from_tile(t)
                p = self.slide2img.get(s, None)
                if p is not None and p.exists():
                    keep.append(t)
            self.tiles = keep

        def __len__(self): return len(self.tiles)

        def __getitem__(self, i):
            tid = self.tiles[i]
            s   = slide_from_tile(tid)
            img_path = self.slide2img[s]

            # coords
            row = self.coords.loc[tid]
            y = float(row['yaxis']); x = float(row['xaxis']); r = float(row['r'])
            side = int(max(2 * r, 2 * round(r)))
            if side < 8: side = 8

            # crop box (top-left); clamp later against image bounds
            x0 = max(0, int(round(x - side / 2)))
            y0 = max(0, int(round(y - side / 2)))

            use_vips = globals().get("USE_VIPS", False)
            if use_vips:
                import pyvips
                W = pyvips.Image.new_from_file(str(img_path), access="sequential")
                w = min(side, max(0, W.width  - x0))
                h = min(side, max(0, W.height - y0))
                if w <= 0 or h <= 0:
                    x0 = max(0, min(W.width  - side, x0))
                    y0 = max(0, min(W.height - side, y0))
                    w = min(side, W.width  - x0)
                    h = min(side, W.height - y0)
                patch_vips = W.crop(x0, y0, w, h)
                arr = np.ndarray(
                    buffer=patch_vips.write_to_memory(),
                    dtype=np.uint8,
                    shape=[h, w, patch_vips.bands],
                )
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                patch = Image.fromarray(arr, mode="RGB")
            else:
                with Image.open(img_path) as W:
                    W = W.convert("RGB")
                    x1 = min(W.width,  x0 + side)
                    y1 = min(W.height, y0 + side)
                    if x0 >= x1 or y0 >= y1:
                        x0 = max(0, min(W.width  - side, x0))
                        y0 = max(0, min(W.height - side, y0))
                        x1 = min(W.width,  x0 + side)
                        y1 = min(W.height, y0 + side)
                    patch = W.crop((x0, y0, x1, y1))

            return tid, tx(patch)

    # encoder: ViT-B/16 DINO headless
    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0).to(DEVICE).eval()
    feat_dim = model.num_features
    print("Encoder:", "vit_base_patch16_224.dino", "| feat_dim:", feat_dim)

    def _silence_pil_in_workers(_):
        import warnings
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    pam50_cols = [g for g in PAM50 if g in gexp_all.columns]
    ds = TileDataset(gexp_all[pam50_cols], coords_all, sample2img)
    loader = DataLoader(
        ds, batch_size=BATCH, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=_silence_pil_in_workers
    )

    from tqdm.auto import tqdm

    all_ids, all_feats = [], []
    with torch.no_grad():
        for tids, xb in tqdm(
            loader,
            total=len(loader),
            desc=f"Extracting {len(ds):,} tiles (bs={BATCH}) [subset '{TAG}']"
        ):
            xb = xb.to(DEVICE, non_blocking=True)
            z  = model(xb)             # [B, D]
            all_ids.extend(list(tids))
            all_feats.append(z.cpu().numpy())

    feats = np.vstack(all_feats) if all_feats else np.empty((0, feat_dim), dtype=np.float32)
    np.savez_compressed(FEAT_NPZ, ids=np.array(all_ids), feats=feats)
    pd.DataFrame({"tile_id": all_ids}).to_csv(TILE_IDS_CSV, index=False)
    print("Saved features:", feats.shape, "→", FEAT_NPZ)

print("############ Chunk 2 Completed ############")



# ------------- CHUNKS 3–6 (cache/tag aware + NaN-robust) -------------
import re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr
import xgboost as xgb

# Reuse OUTDIR if already defined; otherwise default
OUTDIR = Path(globals().get("OUTDIR", "./wsi_geneexp_pipeline"))

# ---------- Helpers to resolve caches ----------
def _resolve_feat_file(outdir: Path) -> Path:
    cands = []
    # Prefer explicit TAG if set earlier in your script
    if "TAG" in globals():
        cands.append(outdir / f"tile_features__{TAG}.npz")
    # Then common names
    cands.append(outdir / "tile_features__all.npz")
    cands.append(outdir / "tile_features.npz")
    # Finally any tagged features
    cands.extend(sorted(outdir.glob("tile_features__*.npz")))
    for p in cands:
        if p.exists(): return p
    raise FileNotFoundError("No tile_features*.npz found in OUTDIR")

def _resolve_gexp_file(outdir: Path, tag_used: str) -> Path:
    cands = []
    if tag_used not in ("untagged", None):
        cands.append(outdir / f"gexp_pam50__{tag_used}.pkl")
    cands.append(outdir / "gexp_pam50__all.pkl")
    cands.append(outdir / "gexp_pam50.pkl")
    cands.extend(sorted(outdir.glob("gexp_pam50__*.pkl")))
    for p in cands:
        if p.exists(): return p
    raise FileNotFoundError("No gexp_pam50 cache found in OUTDIR")

def slide_from_tile(tile_id: str) -> str:
    m = re.match(r"(.+?)_\d+x\d+$", tile_id)
    return m.group(1) if m else tile_id

# ---------- Load features (auto-detect tag) ----------
FEAT_NPZ = _resolve_feat_file(OUTDIR)
npz = np.load(FEAT_NPZ, allow_pickle=True)
tile_ids = npz["ids"].astype(str)
feats = npz["feats"]

m = re.match(r"tile_features__(.+)\.npz", FEAT_NPZ.name)
TAG_USED = m.group(1) if m else ("all" if FEAT_NPZ.name == "tile_features__all.npz" else "untagged")
print(f"[Chunk 3] Using features: {FEAT_NPZ.name} (tag={TAG_USED})  | feats shape: {feats.shape}")

# ---------- Load gene expression for labels ----------
GEXP_PKL = _resolve_gexp_file(OUTDIR, TAG_USED)
gexp_all = pd.read_pickle(GEXP_PKL)
# Try to reuse pam50_cols from earlier; otherwise infer from gexp_all
if "pam50_cols" not in globals():
    if "PAM50" in globals():
        pam50_cols = [g for g in PAM50 if g in gexp_all.columns]
    else:
        pam50_cols = list(gexp_all.columns)  # fallback if PAM50 constant not in scope
print(f"[Chunk 3] Label genes: {len(pam50_cols)}")

# --- map tile -> slide
tile_to_slide = pd.Series({tid: slide_from_tile(tid) for tid in tile_ids}, name="slide")

# --- Slide-level labels: mean of tile-level PAM50
gexp_pam50 = gexp_all.loc[tile_ids, pam50_cols].copy()
slide_labels = gexp_pam50.groupby(tile_to_slide).mean()

# --- Slide-level features: mean of tile features per slide
tile_df = pd.DataFrame({"tile_id": tile_ids, "slide": tile_to_slide.loc[tile_ids].values})
feat_df = pd.DataFrame(feats).assign(slide=tile_df["slide"].values)
slide_feats_mean = feat_df.groupby("slide").mean()

# --- CLEANING: drop slides with any NaN/Inf in Y or X
def clean_align(slide_feats_df: pd.DataFrame, slide_labels_df: pd.DataFrame):
    slide_labels_df = slide_labels_df.replace([np.inf, -np.inf], np.nan)
    slide_feats_df  = slide_feats_df.replace([np.inf, -np.inf], np.nan)

    # Align indices
    common = slide_labels_df.index.intersection(slide_feats_df.index)
    slide_labels_df = slide_labels_df.loc[common]
    slide_feats_df  = slide_feats_df.loc[common]

    # Drop label-NaN slides first
    bad_y = slide_labels_df.isna().any(axis=1)
    if bad_y.any():
        print(f"[Clean] Dropping {bad_y.sum()} slides with NaN labels.")
        slide_labels_df = slide_labels_df.loc[~bad_y]
        slide_feats_df  = slide_feats_df.loc[slide_labels_df.index]

    # Then drop feature-NaN slides
    bad_x = slide_feats_df.isna().any(axis=1)
    if bad_x.any():
        print(f"[Clean] Dropping {bad_x.sum()} slides with NaN features.")
        slide_feats_df  = slide_feats_df.loc[~bad_x]
        slide_labels_df = slide_labels_df.loc[slide_feats_df.index]

    X = slide_feats_df.values
    Y = slide_labels_df.values
    slides = slide_labels_df.index.values
    print(f"[Clean] Remaining slides: {len(slides)}")

    assert np.isfinite(X).all(), "Non-finite values in X after cleaning."
    assert np.isfinite(Y).all(), "Non-finite values in Y after cleaning."
    return X, Y, slides, slide_feats_df, slide_labels_df

X, Y, slides, slide_feats_mean, slide_labels = clean_align(slide_feats_mean, slide_labels)
print("Slides:", len(slides))

from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

def pearsonr_safe(y_true_1d, y_pred_1d):
    if y_true_1d.ndim != 1: y_true_1d = y_true_1d.ravel()
    if y_pred_1d.ndim != 1: y_pred_1d = y_pred_1d.ravel()
    if y_true_1d.shape[0] < 2: return np.nan
    if np.allclose(y_true_1d, y_true_1d[0]):  # zero variance
        return np.nan
    try:
        return pearsonr(y_true_1d, y_pred_1d)[0]
    except Exception:
        return np.nan

def spearmanr_safe(y_true_1d, y_pred_1d):
    if y_true_1d.ndim != 1: y_true_1d = y_true_1d.ravel()
    if y_pred_1d.ndim != 1: y_pred_1d = y_pred_1d.ravel()
    if y_true_1d.shape[0] < 2: return np.nan
    if np.allclose(y_true_1d, y_true_1d[0]):
        return np.nan
    try:
        return spearmanr(y_true_1d, y_pred_1d, nan_policy="omit").correlation
    except Exception:
        return np.nan

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def per_gene_metrics(Yte, Yhat, gene_names):
    """Return dict of lists keyed by metric name (len = n_genes)."""
    nG = Yte.shape[1]
    out = { "pearson": [], "spearman": [], "mae": [], "rmse": [], "r2": [] }
    for j in range(nG):
        yt, yp = Yte[:, j], Yhat[:, j]
        out["pearson"].append(pearsonr_safe(yt, yp))
        out["spearman"].append(spearmanr_safe(yt, yp))
        out["mae"].append(mean_absolute_error(yt, yp))
        out["rmse"].append(rmse(yt, yp))
        # R² can be negative; defined if len>=2
        out["r2"].append(r2_score(yt, yp) if yt.shape[0] >= 2 else np.nan)
    # Map to arrays with gene names later
    return out

def fold_summary_from_gene_metrics(m, r_threshold=0.3):
    """m: dict from per_gene_metrics; returns dict of fold-level summaries."""
    pear = np.array(m["pearson"], dtype=float)
    spear = np.array(m["spearman"], dtype=float)
    mae   = np.array(m["mae"], dtype=float)
    rr    = np.array(m["rmse"], dtype=float)
    r2s   = np.array(m["r2"], dtype=float)
    return {
        "median_pearson": np.nanmedian(pear),
        "pct_genes_r_ge_0.3": np.nanmean(pear >= r_threshold) * 100.0,
        "median_spearman": np.nanmedian(spear),
        "median_mae": np.nanmedian(mae),
        "median_rmse": np.nanmedian(rr),
        "mean_r2": np.nanmean(r2s),
    }

def per_slide_profile_metrics(Yte, Yhat, slide_ids):
    """Profile-level metrics (across 50 genes) per slide in test fold."""
    rows = []
    for i in range(Yte.shape[0]):
        yt, yp = Yte[i, :], Yhat[i, :]
        rows.append({
            "slide": slide_ids[i],
            "pearson_profile": pearsonr_safe(yt, yp),
            "spearman_profile": spearmanr_safe(yt, yp),
            "mae_profile": mean_absolute_error(yt, yp),
            "rmse_profile": rmse(yt, yp),
            "r2_profile": r2_score(yt, yp) if yt.shape[0] >= 2 else np.nan,
            "ev_profile": explained_variance_score(yt, yp) if yt.shape[0] >= 2 else np.nan,
        })
    return pd.DataFrame(rows)

def baseline_predictor(Ytr, Yte):
    """Predict per-gene training mean; return Yhat_base (same shape as Yte)."""
    mu = np.nanmean(Ytr, axis=0)
    return np.tile(mu, (Yte.shape[0], 1))



# --- 5-fold (or less) GroupKFold CV on slides with rich metrics ---
n_splits = min(5, len(slides))
if n_splits < 2:
    raise RuntimeError("Not enough slides for CV after cleaning.")
gkf = GroupKFold(n_splits=n_splits)

# Storage: per-gene metrics per fold
per_gene_frames = []
fold_summ_rows  = []
per_slide_frames = []

for fold, (tr, te) in enumerate(gkf.split(X, Y, groups=slides), 1):
    Xtr, Xte = X[tr], X[te]
    Ytr, Yte = Y[tr], Y[te]
    test_slides = slides[te]

    base = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=42+fold
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(Xtr, Ytr)

    Yhat = model.predict(Xte)

    # --- Per-gene metrics ---
    gm = per_gene_metrics(Yte, Yhat, pam50_cols)
    df_gene = pd.DataFrame({
        **{f"pearson_{g}":[gm['pearson'][j]] for j,g in enumerate(pam50_cols)},
        **{f"spearman_{g}":[gm['spearman'][j]] for j,g in enumerate(pam50_cols)},
        **{f"mae_{g}":[gm['mae'][j]] for j,g in enumerate(pam50_cols)},
        **{f"rmse_{g}":[gm['rmse'][j]] for j,g in enumerate(pam50_cols)},
        **{f"r2_{g}":[gm['r2'][j]] for j,g in enumerate(pam50_cols)},
    })
    df_gene.insert(0, "fold", fold)
    per_gene_frames.append(df_gene)

    # --- Fold summary (also baseline comparison) ---
    summ = fold_summary_from_gene_metrics(gm)
    Yhat_base = baseline_predictor(Ytr, Yte)
    gm_base = per_gene_metrics(Yte, Yhat_base, pam50_cols)
    summ.update({
        "fold": fold,
        "baseline_median_mae": np.nanmedian(np.array(gm_base["mae"])),
        "baseline_median_rmse": np.nanmedian(np.array(gm_base["rmse"])),
        "impr_median_mae_vs_base": summ["median_mae"] - np.nanmedian(np.array(gm_base["mae"])),
        "impr_median_rmse_vs_base": summ["median_rmse"] - np.nanmedian(np.array(gm_base["rmse"])),
    })
    fold_summ_rows.append(summ)

    # --- Per-slide profile metrics ---
    df_ps = per_slide_profile_metrics(Yte, Yhat, test_slides)
    df_ps.insert(0, "fold", fold)
    per_slide_frames.append(df_ps)

    print(f"Fold {fold}: median r={summ['median_pearson']:.3f}, %r>=0.3={summ['pct_genes_r_ge_0.3']:.1f}%, "
          f"median ρ={summ['median_spearman']:.3f}, median MAE={summ['median_mae']:.4f}, "
          f"baseline MAE={summ['baseline_median_mae']:.4f}")

# Save enriched outputs
cv_meanpool_gene_csv   = OUTDIR / f"cv_meanpool_metrics_per_gene__{TAG_USED}.csv"
cv_meanpool_summary    = OUTDIR / f"cv_meanpool_fold_summary__{TAG_USED}.csv"
cv_meanpool_per_slide  = OUTDIR / f"cv_meanpool_per_slide_profile__{TAG_USED}.csv"

pd.concat(per_gene_frames, ignore_index=True).to_csv(cv_meanpool_gene_csv, index=False)
pd.DataFrame(fold_summ_rows).to_csv(cv_meanpool_summary, index=False)
pd.concat(per_slide_frames, ignore_index=True).to_csv(cv_meanpool_per_slide, index=False)

# (Optional) keep your original simple CSV for backward compat:
cv_meanpool_simple = OUTDIR / f"cv_gene_r_meanpool_xgb__{TAG_USED}.csv"
pd.concat(
    [df.filter(regex=r"^pearson_") for df in per_gene_frames],
    ignore_index=True
).to_csv(cv_meanpool_simple, index=False)

print("Saved:",
      cv_meanpool_gene_csv.name, ",",
      cv_meanpool_summary.name, ",",
      cv_meanpool_per_slide.name)
print("############ Chunk 3 Completed (enriched metrics) ############")

# ------------- CHUNK 4: attention-MIL pooling -> export pooled slide embeddings -------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Device (reuse from Chunk 2 if present, else detect)
DEVICE = globals().get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Build bags ONLY for cleaned slides
valid_slides_set = set(slides)
tileid_to_row = {tid: i for i, tid in enumerate(tile_ids)}
slide_to_tiles = {}
for tid in tile_ids:
    s = slide_from_tile(tid)
    if s in valid_slides_set:
        slide_to_tiles.setdefault(s, []).append(tid)

bags = {s: feats[[tileid_to_row[t] for t in ts]] for s, ts in slide_to_tiles.items()}
slides_order = list(slides)
n_genes = slide_labels.shape[1]
feat_dim = feats.shape[1]

class BagDataset(Dataset):
    def __init__(self, slides_order, bags_dict, y_df):
        self.slides = slides_order
        self.bags = bags_dict
        self.y = y_df.loc[self.slides].values.astype(np.float32)
    def __len__(self): return len(self.slides)
    def __getitem__(self, i):
        s = self.slides[i]
        B = torch.from_numpy(self.bags[s]).float()
        y = torch.from_numpy(self.y[i])
        return s, B, y

class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=256, out_dim=256):
        super().__init__()
        self.attn_V = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, X):
        V = self.attn_V(X); U = self.attn_U(X)
        A = self.attn_w(V * U).squeeze(-1)
        w = torch.softmax(A, dim=0)
        pooled = torch.sum(w.unsqueeze(-1) * X, dim=0)
        return self.proj(pooled), w

class MILRegressor(nn.Module):
    def __init__(self, in_dim, attn_dim=256, rep_dim=256, n_genes=50):
        super().__init__()
        self.mil = GatedAttentionMIL(in_dim, attn_dim, rep_dim)
        self.head = nn.Sequential(nn.Linear(rep_dim, 256), nn.ReLU(), nn.Linear(256, n_genes))
    def forward(self, X):
        rep, w = self.mil(X)
        yhat = self.head(rep)
        return yhat, rep, w

mil = MILRegressor(in_dim=feat_dim, attn_dim=256, rep_dim=256, n_genes=n_genes).to(DEVICE)
opt = optim.AdamW(mil.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# simple train/val split
np.random.seed(42)
perm = np.random.permutation(len(slides_order))
split = int(0.8 * len(perm)) if len(perm) >= 5 else max(1, int(0.7 * len(perm)))
train_idx, val_idx = perm[:split], perm[split:]
train_slides = [slides_order[i] for i in train_idx]
val_slides   = [slides_order[i] for i in val_idx]

train_loader = DataLoader(BagDataset(train_slides, bags, slide_labels), batch_size=1, shuffle=True)
val_loader   = DataLoader(BagDataset(val_slides,   bags, slide_labels), batch_size=1, shuffle=False)

best_val = float("inf"); best_state = None
for epoch in range(20):
    mil.train(); tr_loss = 0.0
    for _, B, y in train_loader:
        B = B[0].to(DEVICE); y = y.to(DEVICE)
        yhat, _, _ = mil(B)
        loss = loss_fn(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item()
    mil.eval(); vs = []
    with torch.no_grad():
        for _, B, y in val_loader:
            B = B[0].to(DEVICE); y = y.to(DEVICE)
            yhat, _, _ = mil(B)
            vs.append(loss_fn(yhat, y).item())
    vloss = float(np.mean(vs)) if len(vs) else float("inf")
    print(f"Epoch {epoch+1:02d} | train {tr_loss/max(1,len(train_loader)):.4f} | val {vloss:.4f}")
    if vloss < best_val:
        best_val, best_state = vloss, {k: v.cpu() for k, v in mil.state_dict().items()}

if best_state is not None:
    mil.load_state_dict(best_state)

# Export pooled slide embeddings
mil.eval()
slide_reps = {}
with torch.no_grad():
    for s in slides_order:
        B = torch.from_numpy(bags[s]).float().to(DEVICE)
        _, rep, _ = mil(B)
        slide_reps[s] = rep.cpu().numpy()

S = np.vstack([slide_reps[s] for s in slides_order])
REPS_NPZ = OUTDIR / f"slide_reps_attention__{TAG_USED}.npz"
np.savez_compressed(REPS_NPZ, slides=np.array(slides_order), reps=S)
print("Exported attention-MIL reps:", S.shape, "->", REPS_NPZ.name)
print("############ Chunk 4 Completed ############")

# ------------- CHUNK 5: XGBoost on attention reps (with label cleaning) -------------
npz = np.load(REPS_NPZ, allow_pickle=True)
slides_att = npz["slides"].astype(str)
X_att = npz["reps"]

# Align Y to slides_att, drop slides with any NaN in labels
dfY = slide_labels.reindex(slides_att).replace([np.inf, -np.inf], np.nan)
good_mask = ~dfY.isna().any(axis=1)
if (~good_mask).any():
    print(f"[Chunk 5] Dropping {(~good_mask).sum()} slides with NaN labels before XGB.")
X_att = X_att[good_mask.values]
slides_used = dfY.index[good_mask].values
Y = dfY.loc[good_mask, :].values

# --- CV on attention reps with rich metrics ---
n_splits = min(5, len(slides_used))
if n_splits < 2:
    raise RuntimeError("Not enough slides for CV after cleaning (attn reps).")
gkf = GroupKFold(n_splits=n_splits)

per_gene_frames = []
fold_summ_rows  = []
per_slide_frames = []

for fold, (tr, te) in enumerate(gkf.split(X_att, Y, groups=slides_used), 1):
    base = xgb.XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", random_state=1337+fold
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_att[tr], Y[tr])

    Yhat = model.predict(X_att[te])
    test_slides = slides_used[te]

    gm = per_gene_metrics(Y[te], Yhat, list(slide_labels.columns))
    df_gene = pd.DataFrame({
        **{f"pearson_{g}":[gm['pearson'][j]] for j,g in enumerate(slide_labels.columns)},
        **{f"spearman_{g}":[gm['spearman'][j]] for j,g in enumerate(slide_labels.columns)},
        **{f"mae_{g}":[gm['mae'][j]] for j,g in enumerate(slide_labels.columns)},
        **{f"rmse_{g}":[gm['rmse'][j]] for j,g in enumerate(slide_labels.columns)},
        **{f"r2_{g}":[gm['r2'][j]] for j,g in enumerate(slide_labels.columns)},
    })
    df_gene.insert(0, "fold", fold)
    per_gene_frames.append(df_gene)

    summ = fold_summary_from_gene_metrics(gm)
    Yhat_base = baseline_predictor(Y[tr], Y[te])
    gm_base = per_gene_metrics(Y[te], Yhat_base, list(slide_labels.columns))
    summ.update({
        "fold": fold,
        "baseline_median_mae": np.nanmedian(np.array(gm_base["mae"])),
        "baseline_median_rmse": np.nanmedian(np.array(gm_base["rmse"])),
        "impr_median_mae_vs_base": summ["median_mae"] - np.nanmedian(np.array(gm_base["mae"])),
        "impr_median_rmse_vs_base": summ["median_rmse"] - np.nanmedian(np.array(gm_base["rmse"])),
    })
    fold_summ_rows.append(summ)

    df_ps = per_slide_profile_metrics(Y[te], Yhat, test_slides)
    df_ps.insert(0, "fold", fold)
    per_slide_frames.append(df_ps)

    print(f"[attn+XGB] Fold {fold}: median r={summ['median_pearson']:.3f}, %r>=0.3={summ['pct_genes_r_ge_0.3']:.1f}%, "
          f"median ρ={summ['median_spearman']:.3f}, median MAE={summ['median_mae']:.4f}, "
          f"baseline MAE={summ['baseline_median_mae']:.4f}")

# Save enriched outputs
attn_gene_csv   = OUTDIR / f"cv_attnrep_metrics_per_gene__{TAG_USED}.csv"
attn_summary    = OUTDIR / f"cv_attnrep_fold_summary__{TAG_USED}.csv"
attn_per_slide  = OUTDIR / f"cv_attnrep_per_slide_profile__{TAG_USED}.csv"

pd.concat(per_gene_frames, ignore_index=True).to_csv(attn_gene_csv, index=False)
pd.DataFrame(fold_summ_rows).to_csv(attn_summary, index=False)
pd.concat(per_slide_frames, ignore_index=True).to_csv(attn_per_slide, index=False)

# (Optional) keep the simple Pearson-only CSV for compatibility:
attn_simple = OUTDIR / f"cv_gene_r_attnrep_xgb__{TAG_USED}.csv"
pd.concat(
    [df.filter(regex=r"^pearson_") for df in per_gene_frames],
    ignore_index=True
).to_csv(attn_simple, index=False)

print("Saved:",
      attn_gene_csv.name, ",",
      attn_summary.name, ",",
      attn_per_slide.name)
print("############ Chunk 5 Completed (enriched metrics) ############")

# ------------- CHUNK 6: final training on all (cleaned) & inference helper -------------
# Final fit on all cleaned attention reps
dfY_all = slide_labels.reindex(slides_att).replace([np.inf, -np.inf], np.nan)
good_mask = ~dfY_all.isna().any(axis=1)
if (~good_mask).any():
    print(f"[Chunk 6] Dropping {(~good_mask).sum()} slides with NaN labels before final fit.")
X_final = X_att[good_mask.values]
slides_final = slides_att[good_mask.values]
Y_final = dfY_all.loc[good_mask].values

base = xgb.XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
    tree_method="hist", random_state=1
)
final_model = MultiOutputRegressor(base, n_jobs=-1)
final_model.fit(X_final, Y_final)

Yhat_all = final_model.predict(X_final)
pred_df = pd.DataFrame(Yhat_all, index=slides_final, columns=list(slide_labels.columns))
pred_csv = OUTDIR / f"predicted_pam50_table__{TAG_USED}.csv"
pred_df.to_csv(pred_csv)
print("Wrote:", pred_csv)
print("############ Chunk 6 Completed ############")
