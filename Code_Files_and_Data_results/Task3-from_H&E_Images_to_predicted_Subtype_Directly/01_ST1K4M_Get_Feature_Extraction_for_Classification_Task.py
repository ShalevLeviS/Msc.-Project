#!/usr/bin/env python3
# he_tile_feature_extractor_hybrid.py

import os, argparse, shutil, warnings
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import timm
except ImportError:
    timm = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import openslide
    if TYPE_CHECKING:
        from openslide import OpenSlide
except ImportError:
    openslide = None

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
DEFAULT_CSV = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_all_tasks/ST_Image_CSV_paths.csv"
DEFAULT_OUT_ROOT = "/local1/ofir/shalevle/STImage_1K4M/Outputs_From_tasks/Data_for_all_tasks/Feature_Extraction_Files_Per_Model"
DEFAULT_MODELS = ["retccl_resnet50", "resnet50_imagenet", "vit_b16_imagenet"]  # virchow removed (gated)
DEFAULT_BATCH = 128
DEFAULT_WORKERS = 0
DEFAULT_TILE_SCALE = 1.2
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_REGISTRY = {
    "resnet50_imagenet": {
        "arch": "resnet50", "source": "timm",
        "pretrained": True, "weight_path": None,
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "vit_b16_imagenet": {
        "arch": "vit_base_patch16_224", "source": "timm",
        "pretrained": True, "weight_path": None,
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "retccl_resnet50": {
        "arch": "resnet50", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/retccl_resnet50.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    # New models from paper
    "phikon_v2": {
        "arch": "vit_base_patch16_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/phikon_v2.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "conch_v15": {
        "arch": "vit_base_patch16_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/conch_v15.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "uni_v2": {
        "arch": "vit_large_patch16_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/uni_v2.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "h_optimus_1": {
        "arch": "vit_base_patch16_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/h_optimus_1.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "titan": {
        "arch": "vit_base_patch16_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/titan.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "virchow_v2": {
        "arch": "vit_huge_patch14_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/virchow_v2.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
    "gigapath": {
        "arch": "vit_giant_patch14_224", "source": "timm",
        "pretrained": False, "weight_path": "/path/to/gigapath.pth",
        "img_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
    },
}

# ===================== HELPERS =====================
def _find_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower: return lower[cand.lower()]
    return None

def crop_circle(tile_rgb, side):
    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (side//2, side//2), side//2, 255, thickness=-1)
    return cv2.bitwise_and(tile_rgb, tile_rgb, mask=mask)

# ===================== DATASET =====================
class TileDataset(Dataset):
    """Read tiles from coord CSV; uses OpenSlide if WSI, else cv2/PIL."""
    def __init__(self, image_path, coord_csv, img_size=224, tile_scale=1.0,
                 side_override=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.image_path = image_path
        self.coords = pd.read_csv(coord_csv)

        self.xcol = _find_column(self.coords, ["xaxis","x","X"])
        self.ycol = _find_column(self.coords, ["yaxis","y","Y"])
        self.rcol = _find_column(self.coords, ["r","radius","R"])
        if not (self.xcol and self.ycol and self.rcol):
            raise ValueError(f"coord CSV missing x/y/r columns in {coord_csv}")

        if "tile_id" not in self.coords.columns:
            self.coords = self.coords.reset_index().rename(columns={"index": "tile_id"})
        self.id_col = "tile_id"

        ext = Path(self.image_path).suffix.lower()
        self.use_openslide = openslide is not None and ext in [".svs",".tif",".tiff",".ndpi",".scn"]
        
        # Properly initialize slide and img_bgr based on use_openslide
        if self.use_openslide and openslide is not None:
            self.slide = openslide.OpenSlide(self.image_path)
            self.img_bgr: Optional[np.ndarray] = None
        else:
            self.slide = None
            self.img_bgr = self._load_image()

        self.img_size = int(img_size)
        self.tile_scale = float(tile_scale)
        self.side_override = side_override
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_size = (self.img_size, self.img_size)

    def _load_image(self) -> np.ndarray:
        """Load image with fallback methods to handle large images."""
        # Try OpenCV with increased pixel limit
        try:
            # Set OpenCV environment variable to allow larger images
            os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
            im = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if im is not None:
                return im
        except Exception as e:
            print(f"OpenCV failed for {self.image_path}: {e}")
        
        # Fallback to PIL
        try:
            # Increase PIL decompression bomb limit
            Image.MAX_IMAGE_PIXELS = None
            pil_img = Image.open(self.image_path).convert("RGB")
            im = np.array(pil_img)[:, :, ::-1]  # RGB to BGR
            return im
        except Exception as e:
            raise RuntimeError(f"Failed to load image {self.image_path}: {e}")

    def __len__(self): return len(self.coords)

    def _to_tensor_norm(self, tile_rgb):
        tile = cv2.resize(tile_rgb, self.resize_size, interpolation=cv2.INTER_LINEAR)
        t = tile.astype(np.float32)/255.0
        t = t.transpose(2,0,1)
        for c in range(3): t[c] = (t[c]-self.mean[c])/self.std[c]
        return torch.from_numpy(t)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        row = self.coords.iloc[idx]
        try:
            x = int(round(float(row[self.xcol])))
            y = int(round(float(row[self.ycol])))
            r = float(row[self.rcol])
        except Exception:
            side = self.side_override if self.side_override else 64
            blank = np.zeros((side,side,3),dtype=np.uint8)
            return self._to_tensor_norm(blank), f"badrow_{idx}"

        side = self.side_override if self.side_override else int(round(2*r*self.tile_scale))
        side = max(side,16)

        if self.use_openslide and self.slide is not None:
            region = self.slide.read_region((x - side//2, y - side//2), 0, (side, side))
            tile_rgb = np.array(region.convert("RGB"))
        elif self.img_bgr is not None:
            h,w = self.img_bgr.shape[:2]
            x1, y1 = max(0, x-side//2), max(0, y-side//2)
            x2, y2 = min(w, x1+side), min(h, y1+side)
            tile_bgr = self.img_bgr[y1:y2, x1:x2]
            if tile_bgr.size==0:
                tile_bgr = np.zeros((side,side,3),dtype=np.uint8)
            else:
                tile_bgr = cv2.copyMakeBorder(tile_bgr,0,side-tile_bgr.shape[0],
                                              0,side-tile_bgr.shape[1],
                                              cv2.BORDER_CONSTANT,value=0)
            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: create blank tile if neither method is available
            tile_rgb = np.zeros((side,side,3),dtype=np.uint8)

        tile_rgb = crop_circle(tile_rgb, side)
        return self._to_tensor_norm(tile_rgb), str(row[self.id_col])

# ===================== ENCODER =====================
class FeatureEncoder(nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self,x):
        if hasattr(self.model,"forward_features"): t=self.model.forward_features(x)
        else: t=self.model(x)
        if isinstance(t,dict): t=list(t.values())[-1]
        if isinstance(t,(list,tuple)): t=t[-1]
        if t.ndim==4: t=t.mean(dim=[2,3])
        return t

def _strip_module_prefix(state_dict):
    return { (k[len("module."):] if k.startswith("module.") else k): v for k,v in state_dict.items() }

def build_encoder(spec):
    if timm is None: raise ImportError("timm required")
    model = timm.create_model(spec["arch"], pretrained=spec.get("pretrained", False), num_classes=0)
    enc = FeatureEncoder(model)
    if spec.get("weight_path"):
        wp=Path(spec["weight_path"])
        if wp.exists():
            ckpt=torch.load(wp,map_location="cpu")
            state=ckpt.get("state_dict",ckpt)
            enc.model.load_state_dict(_strip_module_prefix(state),strict=False)
    enc.eval()
    with torch.no_grad():
        dummy=torch.zeros(1,3,spec["img_size"],spec["img_size"])
        feat_dim=enc(dummy).shape[1]
    return enc,spec["img_size"],feat_dim

# ===================== EXTRACTION =====================
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def extract_for_model(model_id, df, out_root, batch_size, num_workers, tile_scale, side_override, device):
    if model_id not in MODEL_REGISTRY: return df
    spec=MODEL_REGISTRY[model_id]
    try: encoder,img_size,feat_dim=build_encoder(spec)
    except Exception as e: print(f"❌ Failed {model_id}: {e}"); return df
    encoder.to(device).eval()

    model_out=Path(out_root)/model_id; ensure_dir(model_out)
    feat_col=f"feat_{model_id}_npz"
    if feat_col not in df.columns: df[feat_col]=""

    rows_iter=range(len(df))
    pbar=tqdm(rows_iter,desc=f"[{model_id}] samples") if tqdm else rows_iter

    for idx in pbar:
        row=df.iloc[idx]
        img_path,coord_path=str(row["image_path"]),str(row["coord_path"])
        if not img_path or not coord_path or not Path(img_path).exists() or not Path(coord_path).exists():
            continue
        out_npz=model_out/f"{Path(img_path).stem}.npz"
        if out_npz.exists():
            try: np.load(out_npz); df.at[idx,feat_col]=str(out_npz); continue
            except Exception: pass

        try:
            ds=TileDataset(img_path,coord_path,img_size,tile_scale,side_override,
                           mean=spec["mean"],std=spec["std"])
        except Exception as e:
            print(f"⚠️ Skip {img_path}: {e}"); continue
        if len(ds)==0:
            print(f"⚠️ No tiles for {img_path}"); continue

        loader=DataLoader(ds,batch_size=batch_size,num_workers=num_workers,
                          shuffle=False,pin_memory=(device=="cuda"))
        feats_list,ids_list=[],[]
        with torch.no_grad():
            for xb,tile_ids in loader:
                xb=xb.to(device).float()
                fb=encoder(xb).cpu().numpy()
                feats_list.append(fb); ids_list.extend(tile_ids)
        feats=np.concatenate(feats_list,axis=0)
        np.savez_compressed(out_npz,feats=feats,tile_ids=np.array(ids_list))
        df.at[idx,feat_col]=str(out_npz)
    return df

def create_ensemble_features(df, out_root, model_ids):
    """Create ensemble features by averaging features from multiple models."""
    if not model_ids: return df
    
    ensemble_name = "ensemble_" + "_".join(model_ids)
    ensemble_out = Path(out_root) / ensemble_name
    ensure_dir(ensemble_out)
    feat_col = f"feat_{ensemble_name}_npz"
    
    if feat_col not in df.columns: 
        df[feat_col] = ""
    
    print(f"\n🔗 Creating ensemble from: {', '.join(model_ids)}")
    
    rows_iter = range(len(df))
    pbar = tqdm(rows_iter, desc=f"[{ensemble_name}] samples") if tqdm else rows_iter
    
    for idx in pbar:
        row = df.iloc[idx]
        img_path = str(row["image_path"])
        stem = Path(img_path).stem
        out_npz = ensemble_out / f"{stem}.npz"
        
        if out_npz.exists():
            try: 
                np.load(out_npz)
                df.at[idx, feat_col] = str(out_npz)
                continue
            except Exception: 
                pass
        
        # Load features from all models
        feats_to_avg = []
        tile_ids: Optional[np.ndarray] = None
        
        for mid in model_ids:
            feat_path_col = f"feat_{mid}_npz"
            if feat_path_col not in df.columns or not df.at[idx, feat_path_col]:
                continue
            
            feat_path = Path(df.at[idx, feat_path_col])
            if not feat_path.exists():
                continue
            
            try:
                data = np.load(feat_path)
                feats = data['feats']
                if tile_ids is None:
                    tile_ids = data['tile_ids']
                feats_to_avg.append(feats)
            except Exception as e:
                print(f"⚠️ Failed to load {feat_path}: {e}")
                continue
        
        if not feats_to_avg:
            print(f"⚠️ No features found for ensemble at idx {idx}")
            continue
        
        # Check that tile_ids is not None before saving
        if tile_ids is None:
            print(f"⚠️ No tile_ids found for ensemble at idx {idx}")
            continue
        
        # Average features
        ensemble_feats = np.mean(feats_to_avg, axis=0)
        np.savez_compressed(out_npz, feats=ensemble_feats, tile_ids=tile_ids)
        df.at[idx, feat_col] = str(out_npz)
    
    return df

# ===================== MAIN =====================
def backup_csv(path):
    p=Path(path); bak=p.with_suffix(p.suffix+".backup")
    if not bak.exists(): shutil.copy2(p,bak)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv-paths",default=DEFAULT_CSV)
    parser.add_argument("--models",nargs="+",default=DEFAULT_MODELS)
    parser.add_argument("--out-root",default=DEFAULT_OUT_ROOT)
    parser.add_argument("--batch-size",type=int,default=DEFAULT_BATCH)
    parser.add_argument("--num-workers",type=int,default=DEFAULT_WORKERS)
    parser.add_argument("--tile-scale",type=float,default=DEFAULT_TILE_SCALE)
    parser.add_argument("--side-override",type=int,default=None)
    parser.add_argument("--device",default=DEFAULT_DEVICE)
    parser.add_argument("--ensemble",nargs="+",default=None,
                       help="List of models to ensemble (e.g., --ensemble gigapath virchow_v2)")
    parser.add_argument("--skip-individual",action="store_true",
                       help="Skip individual model extraction, only create ensemble")
    args=parser.parse_args()

    if args.device=="cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU"); args.device="cpu"

    df=pd.read_csv(args.csv_paths)
    backup_csv(args.csv_paths)
    
    # Extract features for individual models (unless skipped)
    if not args.skip_individual:
        for mid in args.models:
            df=extract_for_model(mid,df,args.out_root,args.batch_size,
                                args.num_workers,args.tile_scale,
                                args.side_override,args.device)
    
    # Create ensemble if requested
    if args.ensemble:
        df = create_ensemble_features(df, args.out_root, args.ensemble)
    
    df.to_csv(args.csv_paths,index=False)
    print("✅ Done. CSV updated:",args.csv_paths)

if __name__=="__main__": main()