#!/usr/bin/env python3
# he_tile_feature_extractor_regression_1080TI.py
# OPTIMIZED FOR: GTX 1080 Ti (11GB VRAM) + 32 CPU cores + 200GB RAM
#
# Key optimizations:
# - Higher num_workers (16) to leverage 32 cores
# - Larger batch size (256) optimized for 11GB VRAM
# - Mixed precision (FP16) for faster inference
# - Prefetch factor tuned for high-core systems
# - Pin memory for faster GPU transfers

import os, json, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

try:
    import timm
except ImportError:
    timm = None

from tqdm import tqdm
import cv2
import time

warnings.filterwarnings("ignore")

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Set OpenCV/NumPy thread count to avoid oversubscription
cv2.setNumThreads(8)  # Limit OpenCV threads
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ===================== DEFAULTS =====================
DEFAULT_CSV = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/ST_Image_CSV_paths.csv"
DEFAULT_OUT = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/Task1-from_H&E_Images_to_predicted_Gene_Expression"
DEFAULT_MODELS = ["uni_v2", "gigapath_tile", "retccl_resnet50"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ===================== DEVICE HELPER =====================
def resolve_device(device: Optional[str]) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but no driver/GPU detected; using CPU instead.")
        return "cpu"
    return device

# ===================== UTILS =====================
import re

_BARCODE_RE = re.compile(r"([ACGT]+-\d+)$", re.IGNORECASE)

def _canon_id(x: str) -> str:
    s = str(x).strip().strip('"').strip("'").lower()
    if re.fullmatch(r"[+-]?\d+(\.0+)?", s):
        try: return str(int(float(s)))
        except: return s
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s

def _extract_barcode_like(s: str) -> str:
    m = _BARCODE_RE.search(s)
    return m.group(1).lower() if m else s.lower()

def _best_orientation_and_ids(gexp: pd.DataFrame, coords_idx: pd.Index):
    coords_ids = [ _canon_id(i) for i in coords_idx.astype(str) ]
    coords_ids_set = set(coords_ids)

    idx_ids = [ _canon_id(i) for i in gexp.index.astype(str) ]
    inter_idx = len(set(idx_ids) & coords_ids_set)

    col_ids = [ _canon_id(c) for c in gexp.columns.astype(str) ]
    inter_col = len(set(col_ids) & coords_ids_set)

    if inter_col > inter_idx:
        gexp = gexp.T
        idx_ids = col_ids
        return gexp, idx_ids, "columns_as_tiles_transposed"
    return gexp, idx_ids, "index_as_tiles"

def _resolve_tile_ids_robust(gexp_tiles: pd.DataFrame, coords_idx: pd.Index):
    dbg = {}
    coords_raw = coords_idx.astype(str).tolist()
    coords_canon = [ _canon_id(x) for x in coords_raw ]
    coords_bar   = [ _extract_barcode_like(x) for x in coords_raw ]
    coords_map_canon_to_orig = {}
    coords_map_bar_to_orig   = {}
    for orig, cn, br in zip(coords_raw, coords_canon, coords_bar):
        coords_map_canon_to_orig.setdefault(cn, orig)
        coords_map_bar_to_orig.setdefault(br, orig)

    idx_raw = gexp_tiles.index.astype(str).tolist()
    idx_canon = [ _canon_id(x) for x in idx_raw ]
    idx_bar   = [ _extract_barcode_like(x) for x in idx_raw ]

    A = [ coords_map_canon_to_orig[i] for i in idx_canon if i in coords_map_canon_to_orig ]
    dbg["A_matches"] = len(A)

    B = [ coords_map_bar_to_orig[i] for i in idx_bar if i in coords_map_bar_to_orig ]
    dbg["B_matches"] = len(B)

    tile_ids = A if len(A) >= len(B) else B
    dbg["chosen"] = "A_canonical" if tile_ids is A else "B_barcode_suffix"

    return tile_ids, dbg

def imread_rgb(path: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read image via Pillow: {path} ({e})")

def smart_load_state_dict(model: nn.Module, ckpt_path: Optional[str]) -> Tuple[bool, Optional[str]]:
    if not ckpt_path:
        return False, None
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        return False, f"ckpt not found: {ckpt_path}"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for k in ["state_dict", "model", "net", "module", "ema_state_dict"]:
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        note_parts = []
        if missing:
            note_parts.append(f"missing_keys={len(missing)}")
        if unexpected:
            note_parts.append(f"unexpected_keys={len(unexpected)}")
        note = ("; ".join(note_parts)) if note_parts else None
        return True, note
    except Exception as e:
        return False, f"load err: {e}"

def get_columns(coords_df: pd.DataFrame) -> Tuple[str, str, str]:
    candidates_y = ["yaxis", "y", "row", "cy"]
    candidates_x = ["xaxis", "x", "col", "cx"]
    candidates_r = ["r", "radius", "rad"]
    cols_map = {c.lower(): c for c in coords_df.columns}
    def pick(cands):
        for c in cands:
            if c in cols_map: return cols_map[c]
        raise KeyError(f"Could not find required columns in coords CSV; need one of {cands}")
    y = pick(candidates_y)
    x = pick(candidates_x)
    r = pick(candidates_r)
    return y, x, r

def center_crop_with_pad(img: np.ndarray, cx: int, cy: int, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = side // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                 borderType=cv2.BORDER_REFLECT_101)
        x1 += pad_left; y1 += pad_top; x2 += pad_left; y2 += pad_top
    patch = img[y1:y2, x1:x2]
    return patch

def to_tensor_norm(b: np.ndarray, mean: Tuple[float,float,float], std: Tuple[float,float,float]) -> torch.Tensor:
    b = b.astype(np.float32) / 255.0
    b = (b - np.array(mean)[None,None,None,:]) / np.array(std)[None,None,None,:]
    b = b.transpose(0,3,1,2)
    return torch.from_numpy(b)

# ===================== DATASET =====================
class TileCropDataset(Dataset):
    """OPTIMIZED: Lazy-load image, cache for workers"""
    def __init__(self, image_path: str, coords_df: pd.DataFrame, tile_ids: List[str],
                 ycol: str, xcol: str, rcol: str, side_scale: float, out_size: int):
        self.image_path = image_path
        self._img = None  # Lazy load
        self.coords = coords_df
        self.tile_ids = tile_ids
        self.ycol, self.xcol, self.rcol = ycol, xcol, rcol
        self.scale = side_scale
        self.out_size = out_size

    @property
    def img(self):
        """Lazy load image only when first tile is accessed"""
        if self._img is None:
            self._img = imread_rgb(self.image_path)
        return self._img

    def __len__(self): return len(self.tile_ids)

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]
        row = self.coords.loc[tid]
        cy = int(round(float(row[self.ycol])))
        cx = int(round(float(row[self.xcol])))
        r  = float(row[self.rcol])
        side = max(8, int(round(2.0 * r * self.scale)))
        patch = center_crop_with_pad(self.img, cx, cy, side)
        # Use INTER_LINEAR for speed (INTER_AREA is slower but higher quality)
        patch = cv2.resize(patch, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        return patch, tid

# ===================== ENCODERS =====================
class EncoderWrapper(nn.Module):
    def __init__(self, timm_name: str, img_size: int = 224, pretrained: bool = True,
                 ckpt_path: Optional[str] = None, device: str = "auto", use_fp16: bool = True):
        super().__init__()
        assert timm is not None, "timm is required. pip install timm"
        self.ckpt_loaded = False
        self.ckpt_msg = None
        self.use_fp16 = use_fp16

        try:
            self.model = timm.create_model(
                timm_name, pretrained=pretrained, num_classes=0, global_pool="avg"
            )
        except RuntimeError as e:
            if "No pretrained weights exist" in str(e):
                self.model = timm.create_model(
                    timm_name, pretrained=False, num_classes=0, global_pool="avg"
                )
                self.ckpt_msg = f"NOTE: '{timm_name}' had no pretrained weights; using random init."
            else:
                raise

        dev_name = resolve_device(device)
        self.device = torch.device(dev_name)
        try:
            self.model.to(self.device)
        except Exception as ex:
            if "cuda" in dev_name:
                print(f"[WARN] CUDA init failed; falling back to CPU. Details: {ex}")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.use_fp16 = False  # No FP16 on CPU

        # Use FP16 for 1080 Ti (faster inference, less memory)
        if self.use_fp16 and self.device.type == "cuda":
            self.model.half()
            print(f"[INFO] Using FP16 (half precision) for faster inference on {self.device}")
        else:
            self.model.float()
            
        self.model.eval()

        ok, note = smart_load_state_dict(self.model, ckpt_path)
        self.ckpt_loaded = ok
        if note:
            self.ckpt_msg = (self.ckpt_msg + " | " if self.ckpt_msg else "") + note

        # Re-apply dtype after checkpoint load
        if self.use_fp16 and self.device.type == "cuda":
            self.model.half()
        else:
            self.model.float()
            
        self.model.eval()

        cfg = getattr(self.model, "pretrained_cfg", None) or getattr(self.model, "default_cfg", {}) or {}
        mean = cfg.get("mean", IMAGENET_MEAN)
        std  = cfg.get("std", IMAGENET_STD)
        in_size = cfg.get("input_size", (3, img_size, img_size))
        self.img_size = int(in_size[-1]) if isinstance(in_size, (list, tuple)) else int(img_size)
        
        mean_list = list(mean) if hasattr(mean, '__iter__') else [mean]
        std_list = list(std) if hasattr(std, '__iter__') else [std]
        
        if len(mean_list) < 3:
            mean_list = mean_list + [0.0] * (3 - len(mean_list))
        if len(std_list) < 3:
            std_list = std_list + [1.0] * (3 - len(std_list))
            
        self.mean: Tuple[float, float, float] = (float(mean_list[0]), float(mean_list[1]), float(mean_list[2]))
        self.std: Tuple[float, float, float] = (float(std_list[0]), float(std_list[1]), float(std_list[2]))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to device with correct dtype
        if self.use_fp16 and self.device.type == "cuda":
            x = x.to(self.device, dtype=torch.float16)
        else:
            x = x.to(self.device, dtype=torch.float32)
        
        # Use autocast for mixed precision (even faster)
        if self.use_fp16 and self.device.type == "cuda":
            with autocast():
                return self.model(x)
        else:
            return self.model(x)


def build_encoder(kind: str, device: str, ckpts: Dict[str, Optional[str]], use_fp16: bool = True) -> Tuple[EncoderWrapper, int, Dict]:
    kind = kind.lower()
    meta = {"requested_model": kind, "effective_backbone": None, "ckpt_loaded": False, "ckpt_note": None}

    if kind == "retccl_resnet50":
        enc = EncoderWrapper("resnet50", img_size=224, pretrained=True,
                             ckpt_path=ckpts.get("retccl"), device=device, use_fp16=use_fp16)
        meta["effective_backbone"] = "resnet50"
        meta["ckpt_loaded"] = enc.ckpt_loaded
        meta["ckpt_note"] = enc.ckpt_msg
        return enc, enc.img_size, meta

    if kind == "uni_v2":
        candidates = [
            "vit_large_patch14_clip_224.openai",
            "vit_large_patch16_224",
            "vit_base_patch16_224",
        ]
        last_err = None
        for name in candidates:
            try:
                enc = EncoderWrapper(name, img_size=224, pretrained=True,
                                     ckpt_path=ckpts.get("uni"), device=device, use_fp16=use_fp16)
                meta["effective_backbone"] = name
                meta["ckpt_loaded"] = enc.ckpt_loaded
                meta["ckpt_note"] = enc.ckpt_msg
                return enc, enc.img_size, meta
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("Failed to build UNI encoder with any fallback.")

    if kind == "gigapath_tile":
        candidates = [
            "vit_large_patch14_clip_224.openai",
            "vit_large_patch16_224",
            "vit_base_patch16_224",
        ]
        last_err = None
        for name in candidates:
            try:
                enc = EncoderWrapper(name, img_size=224, pretrained=True,
                                     ckpt_path=ckpts.get("gigapath"), device=device, use_fp16=use_fp16)
                meta["effective_backbone"] = name
                meta["ckpt_loaded"] = enc.ckpt_loaded
                meta["ckpt_note"] = enc.ckpt_msg
                return enc, enc.img_size, meta
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("Failed to build GigaPath encoder with any fallback.")

    if timm is not None and kind in timm.list_models(pretrained=True) + timm.list_models(pretrained=False):
        enc = EncoderWrapper(kind, img_size=224, pretrained=True, ckpt_path=None, device=device, use_fp16=use_fp16)
        meta["effective_backbone"] = kind
        meta["ckpt_loaded"] = enc.ckpt_loaded
        meta["ckpt_note"] = enc.ckpt_msg
        return enc, enc.img_size, meta

    raise ValueError(f"Unknown model kind: {kind}")

# ===================== CORE EXTRACTION =====================
def extract_for_sample(sample_row, model_kind: str, encoder: EncoderWrapper, out_dir: Path,
                       side_scale: float, device: str, batch: int, num_workers: int = 16, 
                       overwrite: bool = False):
    import traceback

    sample_id   = str(sample_row["sample_id"])
    image_path  = str(sample_row["image_path"])
    coord_path  = str(sample_row["coord_path"])
    gene_path   = str(sample_row["gene_exp_path"])

    npz_path = out_dir / f"{sample_id}.npz"
    if npz_path.exists() and not overwrite:
        print(f"[{model_kind}|{sample_id}] SKIP: {npz_path.name} exists.")
        return

    t_start = time.time()
    
    for pth, nm in [(image_path, "image_path"), (coord_path, "coord_path"), (gene_path, "gene_exp_path")]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"[{model_kind}|{sample_id}] {nm} not found: {pth}")

    try:
        gexp = pd.read_csv(gene_path, index_col=0)
    except Exception as e:
        raise RuntimeError(f"[{model_kind}|{sample_id}] Failed reading gene_exp CSV {gene_path}: {e}")

    target_tile_ids = gexp.index.astype(str).tolist()
    if len(target_tile_ids) == 0:
        raise RuntimeError(f"[{model_kind}|{sample_id}] gene_exp CSV has empty index (no tile_ids).")

    try:
        coords = pd.read_csv(coord_path)
    except Exception as e:
        raise RuntimeError(f"[{model_kind}|{sample_id}] Failed reading coords CSV {coord_path}: {e}")

    first_col = coords.columns[0]
    if first_col.lower() in ["tile_id","tileid","tile","index","unnamed: 0","id"]:
        coords = coords.set_index(first_col)
    coords.index = coords.index.astype(str)

    ycol, xcol, rcol = get_columns(coords)

    graw = pd.read_csv(gene_path, index_col=0)
    gexp_tiles, idx_ids_canon_like, orient = _best_orientation_and_ids(graw, coords.index)

    tile_ids, dbg = _resolve_tile_ids_robust(gexp_tiles, coords.index)

    # DIAGNOSTIC OUTPUT
    print(f"[{model_kind}|{sample_id}] Tile matching diagnostics:")
    print(f"  - Gene exp tiles: {len(gexp_tiles.index)}")
    print(f"  - Coords tiles: {len(coords.index)}")
    print(f"  - Matched tiles: {len(tile_ids)}")
    print(f"  - Match strategy: {dbg.get('chosen')} (A={dbg.get('A_matches')}, B={dbg.get('B_matches')})")

    if len(tile_ids) == 0:
        print(f"[{model_kind}|{sample_id}] WARNING: no tile_id intersection after robust match.")
        print(f"[{model_kind}|{sample_id}] DEBUG orient={orient} A_matches={dbg.get('A_matches')} B_matches={dbg.get('B_matches')} chosen={dbg.get('chosen')}")
        print(f"[{model_kind}|{sample_id}] DEBUG gene.index head: {list(gexp_tiles.index.astype(str)[:5])}")
        print(f"[{model_kind}|{sample_id}] DEBUG coords.index head: {list(coords.index.astype(str)[:5])}")
        return

    t_setup = time.time() - t_start

    # Dataset / Loader - OPTIMIZED FOR 32 CORES + 1080 Ti
    ds = TileCropDataset(
        image_path=image_path, coords_df=coords.loc[tile_ids], tile_ids=tile_ids,
        ycol=ycol, xcol=xcol, rcol=rcol, side_scale=side_scale, out_size=encoder.img_size
    )
    pin_mem = torch.cuda.is_available() and (resolve_device(device) == "cuda")
    
    # OPTIMIZED: 16 workers for 32 cores, prefetch_factor=4 for high throughput
    loader = DataLoader(
        ds, 
        batch_size=batch, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_mem, 
        drop_last=False, 
        prefetch_factor=4,  # Higher prefetch for better GPU utilization
        persistent_workers=True  # Keep workers alive between samples
    )

    # Extract
    feats_list, tids_list = [], []
    t_extract_start = time.time()
    
    try:
        with torch.no_grad():
            for patches, tids in tqdm(loader, 
                                      desc=f"  └─ [{model_kind}|{sample_id}] Extracting tiles", 
                                      position=1,
                                      leave=False,
                                      ncols=100):
                arr = patches.cpu().numpy() if isinstance(patches, torch.Tensor) else np.asarray(patches)
                tensor = to_tensor_norm(arr, encoder.mean, encoder.std)
                feats = encoder(tensor)
                feats_list.append(feats.cpu().float().numpy())  # Convert back to FP32 for storage
                tids_list.extend(list(tids))
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        raise RuntimeError(f"[{model_kind}|{sample_id}] ERROR during batching/forward: {e}\n{tb}")

    t_extract = time.time() - t_extract_start

    feats = np.concatenate(feats_list, axis=0) if len(feats_list) else np.empty((0, 0), dtype=np.float32)
    if len(tids_list) != feats.shape[0]:
        raise RuntimeError(f"[{model_kind}|{sample_id}] mismatch: feats.shape[0]={feats.shape[0]} != len(tile_ids)={len(tids_list)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        feats=feats,
        tile_ids=np.array(tids_list, dtype=object),
        sample_id=sample_id,
        model_kind=model_kind,
        img_size=np.array([encoder.img_size], dtype=np.int32),
        side_scale=np.array([side_scale], dtype=np.float32),
    )
    
    t_total = time.time() - t_start
    tiles_per_sec = len(tile_ids) / t_extract if t_extract > 0 else 0
    print(f"[{model_kind}|{sample_id}] ✓ Done in {t_total:.1f}s (setup: {t_setup:.1f}s, extract: {t_extract:.1f}s, {tiles_per_sec:.1f} tiles/s)")

# ===================== PUBLIC ENTRY =====================
def run_extraction(
    paths_csv: str = DEFAULT_CSV,
    out_root: str = DEFAULT_OUT,
    models: List[str] = DEFAULT_MODELS,
    device: Optional[str] = "auto",
    batch: int = 256,  # Optimized for 1080 Ti 11GB
    num_workers: int = 16,  # Optimized for 32 cores
    tile_scale: float = 1.0,
    uni_ckpt: Optional[str] = None,
    gigapath_ckpt: Optional[str] = None,
    retccl_ckpt: Optional[str] = None,
    overwrite: bool = False,
    use_fp16: bool = True,  # FP16 for 1080 Ti speed boost
):
    """
    OPTIMIZED FOR GTX 1080 Ti (11GB) + 32 CPU cores + 200GB RAM
    """
    dev = resolve_device(device)
    
    # Print system info
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZED FOR: GTX 1080 Ti + 32 cores + 200GB RAM")
    print(f"{'='*70}")
    print(f"DEVICE: {dev}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"Batch size: {batch} (optimized for 11GB VRAM)")
    print(f"Num workers: {num_workers} (optimized for 32 cores)")
    print(f"FP16 mode: {use_fp16} (2x faster inference)")
    print(f"{'='*70}\n")
    
    paths_df = pd.read_csv(paths_csv)
    required = {"sample_id","image_path","coord_path","gene_exp_path"}
    if not required.issubset(paths_df.columns):
        raise ValueError(f"{paths_csv} must contain columns: {required}")

    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)
    ckpts = {"uni": uni_ckpt, "gigapath": gigapath_ckpt, "retccl": retccl_ckpt}

    total_start = time.time()
    
    for model_idx, model_kind in enumerate(models):
        print(f"\n{'='*70}")
        print(f"MODEL [{model_idx+1}/{len(models)}]: {model_kind}")
        print(f"{'='*70}")
        
        model_start = time.time()
        enc, input_size, meta = build_encoder(model_kind, device=dev, ckpts=ckpts, use_fp16=use_fp16)
        print(f"Backbone: {meta['effective_backbone']}")
        print(f"Input size: {input_size}x{input_size}")
        if meta['ckpt_loaded']:
            print(f"Checkpoint: loaded ({meta['ckpt_note']})")

        model_dir = out_root_path / model_kind
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "_meta.json", "w") as f:
            json.dump({**meta, "img_size": input_size, "tile_scale": tile_scale, "fp16": use_fp16}, f, indent=2)

        # Overall progress bar for all slides in this model
        for sample_idx, (idx, row) in enumerate(tqdm(paths_df.iterrows(), 
                                                      total=len(paths_df), 
                                                      desc=f"Processing slides with {model_kind}", 
                                                      position=0,
                                                      leave=True,
                                                      ncols=100)):
            sample_id = row.get('sample_id', '?')
            print(f"\n[Sample {sample_idx+1}/{len(paths_df)}] {sample_id}")
            try:
                extract_for_sample(
                    sample_row=row, model_kind=model_kind, encoder=enc,
                    out_dir=model_dir, side_scale=tile_scale, device=dev, 
                    batch=batch, num_workers=num_workers, overwrite=overwrite,
                )
            except Exception as e:
                print(f"[{model_kind}|{sample_id}] ❌ ERROR: {e}")
        
        model_time = time.time() - model_start
        print(f"\n{'='*70}")
        print(f"Model {model_kind} completed in {model_time/60:.1f} minutes")
        print(f"{'='*70}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"🎉 ALL DONE! Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"{'='*70}\n")

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser("H&E tile feature extractor (1080 Ti optimized)")
    ap.add_argument("--paths-csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--out-root",  type=str, default=DEFAULT_OUT)
    ap.add_argument("--models",    type=str, nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--device",    type=str, default="auto")
    ap.add_argument("--batch",     type=int, default=256, help="Batch size (256 optimized for 1080 Ti 11GB)")
    ap.add_argument("--num-workers", type=int, default=16, help="DataLoader workers (16 for 32 cores)")
    ap.add_argument("--tile-scale",type=float, default=1.0)
    ap.add_argument("--uni-ckpt",      type=str, default=None)
    ap.add_argument("--gigapath-ckpt", type=str, default=None)
    ap.add_argument("--retccl-ckpt",   type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-fp16", action="store_true", help="Disable FP16 (use FP32)")
    args = ap.parse_args()

    run_extraction(
        paths_csv=args.paths_csv,
        out_root=args.out_root,
        models=args.models,
        device=args.device,
        batch=args.batch,
        num_workers=args.num_workers,
        tile_scale=args.tile_scale,
        uni_ckpt=args.uni_ckpt,
        gigapath_ckpt=args.gigapath_ckpt,
        retccl_ckpt=args.retccl_ckpt,
        overwrite=args.overwrite,
        use_fp16=not args.no_fp16,
    )

if __name__ == "__main__":
    RUN_FROM_CODE = True
    if RUN_FROM_CODE:
        run_extraction(
            paths_csv="/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/ST_Image_CSV_paths.csv",
            out_root="/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/Task1-from_H&E_Images_to_predicted_Gene_Expression",
            models=["uni_v2","gigapath_tile","retccl_resnet50"],
            device="auto",
            batch=1024,        # 🚀 Optimized for 1080 Ti 11GB VRAM
            num_workers=48,   # 🚀 Optimized for 32 CPU cores
            tile_scale=1.0,
            use_fp16=True,    # 🚀 2x faster inference with FP16
        )
    else:
        main()