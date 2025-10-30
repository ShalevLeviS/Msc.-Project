#!/usr/bin/env python3
"""
Multi-Model Histopathology Feature Extraction Pipeline
Extracts tile-level embeddings from WSI slides using multiple foundation models.
Optimized for XGBoost downstream classification tasks.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2

try:
    import openslide
except ImportError:
    print("ERROR: openslide-python not installed. Install with: pip install openslide-python")
    sys.exit(1)

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    print("ERROR: safetensors not installed. Install with: pip install safetensors")
    sys.exit(1)


# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Global configuration"""
    # Paths
    csv_path: str = "/local1/ofir/shalevle/TCGA/Data_for_Task3_H%26E_to_Subtype/image_to_star_counts_with_pam50.csv"
    output_root: str = "/local1/ofir/shalevle/TCGA/Data_for_Task3_H&E_to_Subtype/Feature_Extraction_Data"
    model_weights_dir: str = "/local1/ofir/shalevle/Models_FE"
    config_dir: str = "/local1/ofir/shalevle/TCGA/Data_for_Task3_H&E_to_Subtype/Feature_Extraction_Data"
    
    # Tiling parameters
    target_magnification: float = 20.0  # Target ~20x
    tile_size: int = 256  # Base tile size
    stride: int = 256  # No overlap
    top_k_tiles: int = 1000  # Select top K tiles per slide
    tissue_threshold: float = 0.2  # Minimum tissue ratio to keep tile
    
    # Model parameters
    batch_size: int = 128  # Will auto-reduce on OOM
    normalize_embeddings: bool = True  # L2 normalize
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4 if torch.cuda.is_available() else 2


# ==================== MODEL REGISTRY ====================

MODEL_REGISTRY = {
    "GigaPath": {
        "weight_file": "Giga_Path_pytorch_model.bin",
        "config_file": "GIGA_PATH_config.json",
        "input_size": 224,  # Resize 256→224
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "UNI": {
        "weight_file": "UNI_pytorch_model.bin",
        "config_file": "UNI_config.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "Conch": {
        "weight_file": "Conch_pytorch_model.bin",
        "config_file": "conch_ViT-B-16.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "PHIKON": {
        "weight_file": "PHIKON_model.safetensors",
        "config_file": "PHIKON_config.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "H_Optimus1": {
        "weight_file": "H-optimus1_model.safetensors",
        "config_file": "H_OPTIMUS_config.json",
        "input_size": 224,
        # Special normalization for H-Optimus
        "normalize_mean": [0.707223, 0.578729, 0.703617],
        "normalize_std": [0.211883, 0.230117, 0.177517],
    },
    # Skip Titan as it's slide-level encoder
}


# ==================== VISION TRANSFORMER IMPLEMENTATION ====================

class VisionTransformer(nn.Module):
    """Generic Vision Transformer for loading various ViT models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract config parameters with safe defaults
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.num_channels = config.get('num_channels', 3)
        self.hidden_size = config.get('hidden_size', 768)
        self.num_hidden_layers = config.get('num_hidden_layers', 12)
        self.num_attention_heads = config.get('num_attention_heads', 12)
        self.intermediate_size = config.get('intermediate_size', 3072)
        self.hidden_dropout_prob = config.get('hidden_dropout_prob', 0.0)
        self.attention_probs_dropout_prob = config.get('attention_probs_dropout_prob', 0.0)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Positional embeddings
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                self.hidden_dropout_prob,
                self.attention_probs_dropout_prob
            ) for _ in range(self.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_size, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return cls token embedding
        return x[:, 0]
    
    def forward_features(self, x):
        """Alias for compatibility"""
        return self.forward(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Attention block
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== MODEL LOADER ====================

class ModelLoader:
    """Loads models from config + weights"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_name: str) -> Tuple[nn.Module, int, transforms.Compose]:
        """Load model, return (model, embedding_dim, preprocessing)"""
        
        model_info = MODEL_REGISTRY[model_name]
        
        # Load config
        config_path = Path(self.config.config_dir) / model_info["config_file"]
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            model_config = json.load(f)
        
        self.logger.info(f"Loading {model_name}...")
        self.logger.info(f"Config: {json.dumps(model_config, indent=2)[:200]}...")
        
        # Create model
        model = VisionTransformer(model_config)
        
        # Load weights
        weight_path = Path(self.config.model_weights_dir) / model_info["weight_file"]
        if not weight_path.exists():
            raise FileNotFoundError(f"Weights not found: {weight_path}")
        
        if weight_path.suffix == ".safetensors":
            state_dict = load_safetensors(weight_path)
        else:
            state_dict = torch.load(weight_path, map_location="cpu")
            # Handle nested state_dict
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Load with strict=False to handle missing/extra keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            self.logger.warning(f"Missing keys: {missing[:5]}...")
        if unexpected:
            self.logger.warning(f"Unexpected keys: {unexpected[:5]}...")
        
        model = model.to(self.config.device)
        model.eval()
        
        # Detect embedding dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, model_info["input_size"], model_info["input_size"]).to(self.config.device)
            embedding = model.forward_features(dummy)
            embedding_dim = embedding.shape[-1]
        
        self.logger.info(f"✓ {model_name} loaded: embedding_dim={embedding_dim}")
        
        # Create preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(model_info["input_size"]),
            transforms.CenterCrop(model_info["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=model_info["normalize_mean"],
                std=model_info["normalize_std"]
            )
        ])
        
        return model, embedding_dim, preprocess


# ==================== TILE EXTRACTION ====================

class TileExtractor:
    """Extracts tiles from WSI at specified magnification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_tiles(self, slide_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Dict]:
        """Extract tiles from slide, return (tiles, coords, metadata)"""
        
        # Open slide
        slide = openslide.OpenSlide(slide_path)
        
        # Get best level for target magnification
        level, actual_mag = self._get_best_level(slide)
        
        # Get level dimensions
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        # Create tissue mask
        tissue_mask = self._create_tissue_mask(slide)
        
        # Generate tile coordinates
        all_coords = []
        all_scores = []
        
        tile_size = self.config.tile_size
        stride = self.config.stride
        
        for y in range(0, level_dims[1] - tile_size, stride):
            for x in range(0, level_dims[0] - tile_size, stride):
                # Check tissue mask
                mask_x = int(x * tissue_mask.shape[1] / level_dims[0])
                mask_y = int(y * tissue_mask.shape[0] / level_dims[1])
                mask_region = tissue_mask[
                    mask_y:mask_y + int(tile_size * tissue_mask.shape[0] / level_dims[1]),
                    mask_x:mask_x + int(tile_size * tissue_mask.shape[1] / level_dims[0])
                ]
                
                tissue_ratio = mask_region.mean()
                if tissue_ratio < self.config.tissue_threshold:
                    continue
                
                # Convert to level 0 coordinates for reading
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                # Read tile
                tile = slide.read_region((x0, y0), level, (tile_size, tile_size))
                tile = np.array(tile.convert('RGB'))
                
                # Score tile (variance-based quality)
                score = self._score_tile(tile)
                
                all_coords.append((x, y))
                all_scores.append(score)
        
        # Select top-K tiles
        if len(all_scores) > self.config.top_k_tiles:
            top_k_indices = np.argsort(all_scores)[-self.config.top_k_tiles:]
            selected_coords = [all_coords[i] for i in top_k_indices]
            selected_scores = [all_scores[i] for i in top_k_indices]
        else:
            selected_coords = all_coords
            selected_scores = all_scores
        
        # Extract selected tiles
        tiles = []
        final_coords = []
        final_scores = []
        
        for (x, y), score in zip(selected_coords, selected_scores):
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            tile = slide.read_region((x0, y0), level, (tile_size, tile_size))
            tile = np.array(tile.convert('RGB'))
            tiles.append(tile)
            final_coords.append((x, y))
            final_scores.append(score)
        
        metadata = {
            'level': level,
            'magnification': actual_mag,
            'tile_size': tile_size,
            'stride': stride,
            'n_tiles_total': len(all_coords),
            'n_tiles_kept': len(tiles),
            'level_dimensions': level_dims,
            'downsample': downsample
        }
        
        slide.close()
        
        return tiles, final_coords, final_scores, metadata
    
    def _get_best_level(self, slide: openslide.OpenSlide) -> Tuple[int, float]:
        """Get best pyramid level for target magnification"""
        
        # Try to get objective power
        try:
            objective_power = float(slide.properties.get(
                openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40
            ))
        except:
            objective_power = 40.0  # Default assumption
        
        # Calculate target downsample
        target_downsample = objective_power / self.config.target_magnification
        
        # Find closest level
        downsamples = slide.level_downsamples
        best_level = min(range(len(downsamples)), 
                        key=lambda i: abs(downsamples[i] - target_downsample))
        
        actual_mag = objective_power / downsamples[best_level]
        
        return best_level, actual_mag
    
    def _create_tissue_mask(self, slide: openslide.OpenSlide) -> np.ndarray:
        """Create binary tissue mask from thumbnail"""
        
        # Get thumbnail
        thumb_size = (2000, 2000)
        thumb = slide.get_thumbnail(thumb_size)
        thumb = np.array(thumb.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        
        # Otsu threshold
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert (tissue = 1, background = 0)
        mask = 255 - mask
        mask = mask.astype(float) / 255.0
        
        return mask
    
    def _score_tile(self, tile: np.ndarray) -> float:
        """Score tile quality (higher = better)"""
        
        # Use RGB variance as quality metric
        return float(np.var(tile))


# ==================== TILE DATASET ====================

class TileDataset(Dataset):
    """Dataset for tile batch encoding"""
    
    def __init__(self, tiles: List[np.ndarray], transform: transforms.Compose):
        self.tiles = tiles
        self.transform = transform
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]
        tile_pil = Image.fromarray(tile)
        return self.transform(tile_pil)


# ==================== FEATURE EXTRACTOR ====================

class FeatureExtractor:
    """Extracts features for a single model across all slides"""
    
    def __init__(self, config: Config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.model_dir = Path(config.output_root) / model_name
        self.embeddings_dir = self.model_dir / "embeddings"
        self.manifests_dir = self.model_dir / "manifests"
        self.done_dir = self.model_dir / "done"
        
        for d in [self.embeddings_dir, self.manifests_dir, self.done_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load model
        loader = ModelLoader(config)
        self.model, self.embedding_dim, self.preprocess = loader.load_model(model_name)
        
        # Tile extractor
        self.tile_extractor = TileExtractor(config)
        
        # Batch size (will auto-reduce on OOM)
        self.batch_size = config.batch_size
    
    def process_slide(self, slide_path: str, slide_id: str) -> Optional[str]:
        """Process single slide, return path to saved NPZ or None if failed"""
        
        # Check if already done
        done_marker = self.done_dir / f"{slide_id}.ok"
        npz_path = self.embeddings_dir / f"{slide_id}.npz"
        
        if done_marker.exists() and npz_path.exists():
            self.logger.info(f"Skipping {slide_id} (already processed)")
            return str(npz_path)
        
        try:
            start_time = time.time()
            
            # Extract tiles
            tiles, coords, scores, metadata = self.tile_extractor.extract_tiles(slide_path)
            
            if len(tiles) == 0:
                self.logger.warning(f"No tiles extracted for {slide_id}")
                return None
            
            # Encode tiles
            embeddings = self._encode_tiles(tiles)
            
            # Normalize
            if self.config.normalize_embeddings:
                embeddings = F.normalize(torch.from_numpy(embeddings), dim=1).numpy()
            
            # Compute aggregations for XGBoost
            slide_embedding_mean = np.mean(embeddings, axis=0).astype(np.float32)
            slide_embedding_max = np.max(embeddings, axis=0).astype(np.float32)
            slide_embedding_std = np.std(embeddings, axis=0).astype(np.float32)
            
            # Weighted mean using tile scores
            scores_normalized = np.array(scores, dtype=np.float32)
            scores_normalized = scores_normalized / scores_normalized.sum()
            slide_embedding_weighted = (embeddings.T @ scores_normalized).astype(np.float32)
            
            # Save NPZ
            metadata['slide_id'] = slide_id
            metadata['model_name'] = self.model_name
            metadata['embedding_dim'] = self.embedding_dim
            metadata['extraction_time_sec'] = time.time() - start_time
            
            np.savez_compressed(
                npz_path,
                embeddings=embeddings.astype(np.float32),
                tile_coords=np.array(coords, dtype=np.int32),
                tile_scores=np.array(scores, dtype=np.float32),
                slide_embedding_mean=slide_embedding_mean,
                slide_embedding_max=slide_embedding_max,
                slide_embedding_std=slide_embedding_std,
                slide_embedding_weighted=slide_embedding_weighted,
                metadata=np.array([metadata], dtype=object)
            )
            
            # Mark as done
            done_marker.touch()
            
            return str(npz_path)
            
        except Exception as e:
            self.logger.error(f"Failed to process {slide_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _encode_tiles(self, tiles: List[np.ndarray]) -> np.ndarray:
        """Encode tiles in batches"""
        
        dataset = TileDataset(tiles, self.preprocess)
        
        # Try with current batch size, reduce on OOM
        while self.batch_size >= 1:
            try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.device == "cuda"
                )
                
                embeddings = []
                with torch.no_grad():
                    for batch in dataloader:
                        batch = batch.to(self.config.device)
                        emb = self.model.forward_features(batch)
                        embeddings.append(emb.cpu().numpy())
                
                return np.vstack(embeddings)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.batch_size = max(1, self.batch_size // 2)
                    self.logger.warning(f"OOM! Reducing batch size to {self.batch_size}")
                    torch.cuda.empty_cache()
                else:
                    raise


# ==================== MAIN PIPELINE ====================

def setup_logging(output_root: str, model_name: str):
    """Setup logging"""
    log_dir = Path(output_root) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'run.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution pipeline"""
    
    config = Config()
    
    print("=" * 80)
    print("HISTOPATHOLOGY MULTI-MODEL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"\nDevice: {config.device}")
    print(f"Models to process: {list(MODEL_REGISTRY.keys())}")
    print(f"Output root: {config.output_root}")
    print(f"Top-K tiles per slide: {config.top_k_tiles}")
    print()
    
    # Load CSV
    print(f"Loading CSV: {config.csv_path}")
    df = pd.read_csv(config.csv_path)
    print(f"Found {len(df)} slides")
    
    # Verify image_path column exists
    if 'image_path' not in df.columns:
        raise ValueError("CSV must have 'image_path' column")
    
    # Process each model
    for model_name in MODEL_REGISTRY.keys():
        print("\n" + "=" * 80)
        print(f"PROCESSING MODEL: {model_name}")
        print("=" * 80 + "\n")
        
        setup_logging(config.output_root, model_name)
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize extractor
            extractor = FeatureExtractor(config, model_name)
            
            # Process slides
            embedding_paths = []
            manifest_rows = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[{model_name}] Slides"):
                slide_path = row['image_path']
                slide_id = Path(slide_path).stem
                
                if not os.path.exists(slide_path):
                    logger.warning(f"Slide not found: {slide_path}")
                    embedding_paths.append(None)
                    continue
                
                # Process slide
                npz_path = extractor.process_slide(slide_path, slide_id)
                embedding_paths.append(npz_path)
                
                if npz_path:
                    manifest_rows.append({
                        'slide_id': slide_id,
                        'svs_path': slide_path,
                        'embedding_path': npz_path
                    })
            
            # Update CSV with embedding paths
            df[f'{model_name}_embedding_path'] = embedding_paths
            
            # Save manifest
            manifest_df = pd.DataFrame(manifest_rows)
            manifest_path = Path(config.output_root) / model_name / "manifests" / "embeddings_index.csv"
            manifest_df.to_csv(manifest_path, index=False)
            logger.info(f"Saved manifest: {manifest_path}")
            
            print(f"\n✓ {model_name} completed: {len(manifest_rows)}/{len(df)} slides processed\n")
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save updated CSV
    output_csv = config.csv_path.replace('.csv', '_with_embeddings.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Updated CSV saved: {output_csv}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()