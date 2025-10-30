#!/usr/bin/env python3
"""
Optimized Multi-Model Histopathology Feature Extraction Pipeline
Uses 1 GPU + 256 CPU cores efficiently with HuggingFace integration
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
from PIL import Image
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

try:
    import openslide
except ImportError:
    print("ERROR: openslide-python not installed. Install with: pip install openslide-python")
    sys.exit(1)

try:
    from transformers import AutoModel, AutoImageProcessor
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers")


# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Global configuration optimized for 1 GPU + 256 CPU cores"""
    # Paths
    csv_path: str = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/TCGA_PATHS_with_start_count.csv"
    output_root: str = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/Feature_Extraction_Files"
    model_weights_dir: str = "/sise/ofircohen-group/datasets/TCGA/TCGA-BRCA-FULL/Models_FE"
    config_dir: str = "/sise/ofircohen-group/datasets/TCGA/TCGA-BRCA-FULL/Models_FE"
    
    # Tiling parameters
    target_magnification: float = 20.0
    tile_size: int = 256
    stride: int = 256  # CHANGED: 512 stride = 2x spacing, extract 4x fewer tiles
    top_k_tiles: int = 1000
    tissue_threshold: float = 0.2  # CHANGED: 0.5 = need 50% tissue (was 0.2)
    
    # Performance optimization
    batch_size: int = 256  # Optimal for most GPUs
    num_cpu_workers: int = 256  # Use all 256 CPU cores
    prefetch_factor: int = 2  # Less prefetching overhead
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision (FP16)
    
    # Parallel processing
    tile_extraction_workers: int = 8  # REDUCED: Even less I/O contention
    preprocessing_workers: int = 8  # Parallel preprocessing
    
    # Model parameters
    normalize_embeddings: bool = True
    
    # Device
    device: str = "cuda:0"  # Explicit single GPU
    
    def __post_init__(self):
        """Validate and adjust settings"""
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_amp = False
            self.batch_size = 32
        
        # Adjust workers based on available CPUs
        available_cpus = mp.cpu_count()
        if self.num_cpu_workers > available_cpus:
            print(f"WARNING: Requested {self.num_cpu_workers} workers but only {available_cpus} CPUs available")
            self.num_cpu_workers = min(self.num_cpu_workers, available_cpus)


# ==================== MODEL REGISTRY WITH HUGGINGFACE ====================

MODEL_REGISTRY = {
    "UNI": {
        "huggingface_id": "MahmoodLab/UNI",  # Official HF model if available
        "weight_file": "UNI_pytorch_model.bin",
        "config_file": "UNI_config.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "use_huggingface": False,  # Set to True if model is on HF
    },
    "Conch": {
        "weight_file": "Conch_pytorch_model.bin",
        "config_file": "conch_ViT-B-16.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "use_huggingface": False,
    },
    "PHIKON": {
        "huggingface_id": "owkin/phikon",
        "weight_file": "PHIKON_model.safetensors",
        "config_file": "PHIKON_config.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "use_huggingface": True,  # Available on HuggingFace
    },
    "H_Optimus1": {
        "weight_file": "H-optimus1_model.safetensors",
        "config_file": "H_OPTIMUS_config.json",
        "input_size": 224,
        "normalize_mean": [0.707223, 0.578729, 0.703617],
        "normalize_std": [0.211883, 0.230117, 0.177517],
        "use_huggingface": False,
    },
    "GigaPath": {
        "weight_file": "Giga_Path_pytorch_model.bin",
        "config_file": "GIGA_PATH_config.json",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "use_huggingface": False,
    },
}


# ==================== PARALLEL TILE EXTRACTION ====================

def extract_tile_worker(args: Tuple) -> Optional[Tuple[np.ndarray, int, int, float]]:
    """Worker function for parallel tile extraction with tissue scoring"""
    slide_path, x, y, tile_size, level, tissue_threshold, downsample = args
    
    try:
        # Open slide (this is slow but necessary per worker)
        slide = openslide.OpenSlide(slide_path)
        
        # Read tile at original coordinates
        tile = slide.read_region((int(x * downsample), int(y * downsample)), level, (tile_size, tile_size))
        tile = tile.convert('RGB')
        tile_np = np.array(tile)
        
        slide.close()
        
        # Fast tissue detection with multiple filters
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        
        # Quick rejection filters
        mean_val = gray.mean()
        if mean_val > 220 or mean_val < 50:
            return None
        
        # Calculate tissue ratio
        tissue_mask = (gray > 50) & (gray < 220)
        tissue_ratio = tissue_mask.sum() / tissue_mask.size
        
        if tissue_ratio >= tissue_threshold:
            return (tile_np, int(x * downsample), int(y * downsample), tissue_ratio)
        return None
        
    except Exception:
        return None


class ParallelTileExtractor:
    """Parallel tile extraction using multiprocessing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_tiles(self, slide_path: str) -> List[Tuple[np.ndarray, int, int, float]]:
        """Extract tiles in parallel using process pool with early stopping"""
        
        try:
            slide = openslide.OpenSlide(slide_path)
            
            # Calculate optimal level
            level = self._get_optimal_level(slide)
            level_dims = slide.level_dimensions[level]
            downsample = slide.level_downsamples[level]
            
            # Generate tile coordinates with spacing
            tile_coords = []
            for y in range(0, level_dims[1], self.config.stride):
                for x in range(0, level_dims[0], self.config.stride):
                    tile_coords.append((
                        slide_path,
                        x,
                        y,
                        self.config.tile_size,
                        level,
                        self.config.tissue_threshold,
                        downsample
                    ))
            
            slide.close()
            
            # Only check 1.5x tiles needed (more aggressive)
            max_tiles_to_check = min(len(tile_coords), int(self.config.top_k_tiles * 1.5))
            self.logger.info(f"Checking {max_tiles_to_check} tile positions (stride={self.config.stride})")
            
            # Parallel extraction with early stopping
            tiles = []
            
            with ProcessPoolExecutor(max_workers=self.config.tile_extraction_workers) as executor:
                # Submit only what we need to check
                coords_to_process = tile_coords[:max_tiles_to_check]
                futures = [executor.submit(extract_tile_worker, args) for args in coords_to_process]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc="Extracting tiles", leave=False):
                    result = future.result()
                    if result is not None:
                        tiles.append(result)
                    
                    # Early stopping if we have enough good tiles
                    if len(tiles) >= self.config.top_k_tiles:
                        self.logger.info(f"Found {len(tiles)} tiles, stopping early")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
            
            self.logger.info(f"Extracted {len(tiles)} valid tiles")
            return tiles
            
        except Exception as e:
            self.logger.error(f"Tile extraction failed: {e}")
            return []
    
    def _get_optimal_level(self, slide: openslide.OpenSlide) -> int:
        """Find optimal magnification level"""
        try:
            objective = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
            target_downsample = objective / self.config.target_magnification
            
            best_level = 0
            min_diff = float('inf')
            
            for level in range(slide.level_count):
                downsample = slide.level_downsamples[level]
                diff = abs(downsample - target_downsample)
                if diff < min_diff:
                    min_diff = diff
                    best_level = level
            
            return best_level
        except:
            return 0


# ==================== OPTIMIZED DATASET ====================

class FastTileDataset(Dataset):
    """Memory-efficient dataset with preprocessing"""
    
    def __init__(self, tiles: List[Tuple[np.ndarray, int, int, float]], 
                 transform: transforms.Compose):
        self.tiles = tiles
        self.transform = transform
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile_np, x, y, score = self.tiles[idx]
        
        # Convert to PIL and apply transforms
        tile_pil = Image.fromarray(tile_np)
        tile_tensor = self.transform(tile_pil)
        
        return tile_tensor, (x, y)


# ==================== MODEL LOADER WITH HUGGINGFACE ====================

class OptimizedModelLoader:
    """Load models efficiently with HuggingFace integration"""
    
    def __init__(self, config: Config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model_config = MODEL_REGISTRY[model_name]
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> Union[nn.Module, Any]:
        """Load model with optimal settings"""
        
        # Try HuggingFace first if available
        if self.model_config.get("use_huggingface", False) and HUGGINGFACE_AVAILABLE:
            return self._load_huggingface_model()
        else:
            return self._load_local_model()
    
    def _load_huggingface_model(self) -> Union[nn.Module, Any]:
        """Load from HuggingFace Hub"""
        self.logger.info(f"Loading {self.model_name} from HuggingFace Hub")
        
        try:
            hf_id = self.model_config["huggingface_id"]
            model = AutoModel.from_pretrained(
                hf_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.use_amp else torch.float32
            )
            model = model.to(self.config.device)
            model.eval()
            
            self.logger.info(f"Successfully loaded {self.model_name} from HuggingFace")
            return model
            
        except Exception as e:
            self.logger.warning(f"HuggingFace load failed: {e}. Falling back to local.")
            return self._load_local_model()
    
    def _load_local_model(self) -> Union[nn.Module, Any]:
        """Load from local weights"""
        self.logger.info(f"Loading {self.model_name} from local weights")
        
        try:
            # Load config
            config_path = Path(self.config.config_dir) / self.model_config["config_file"]
            with open(config_path) as f:
                model_config = json.load(f)
            
            # Create model
            model: nn.Module = VisionTransformer(model_config)
            
            # Load weights
            weight_path = Path(self.config.model_weights_dir) / self.model_config["weight_file"]
            
            if weight_path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(str(weight_path))
            else:
                checkpoint = torch.load(weight_path, map_location='cpu')
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    # Checkpoint is already a state dict
                    state_dict = checkpoint
            
            # Ensure state_dict is a dictionary
            if not isinstance(state_dict, dict):
                raise ValueError(f"Expected state_dict to be a dictionary, got {type(state_dict)}")
            
            model.load_state_dict(state_dict, strict=False)
            
            # Optimize model
            model = model.to(self.config.device)
            model.eval()
            
            # Enable optimizations
            if self.config.use_amp:
                model = model.half()  # Convert to FP16
            
            # Compile model for faster inference (PyTorch 2.0+)
            # Note: torch.compile returns a CompiledModel which is callable but not strictly nn.Module
            compiled_model: Union[nn.Module, Any] = model
            if hasattr(torch, 'compile') and self.config.device != "cpu":
                try:
                    compiled_model = torch.compile(model, mode='max-autotune')
                    self.logger.info(f"Successfully compiled {self.model_name} with torch.compile")
                except Exception as e:
                    self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
            
            self.logger.info(f"Successfully loaded {self.model_name} from local weights")
            return compiled_model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_transform(self) -> transforms.Compose:
        """Get preprocessing transform"""
        
        input_size = self.model_config["input_size"]
        mean = self.model_config["normalize_mean"]
        std = self.model_config["normalize_std"]
        
        return transforms.Compose([
            transforms.Resize((input_size, input_size), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


# ==================== VISION TRANSFORMER (FALLBACK) ====================

class VisionTransformer(nn.Module):
    """Optimized Vision Transformer implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.image_size = config.get('img_size', config.get('image_size', 224))
        self.patch_size = config.get('patch_size', 16)
        self.num_channels = config.get('in_chans', config.get('num_channels', 3))
        self.hidden_size = config.get('num_features', 
                                     config.get('hidden_size', 
                                               config.get('embed_dim', 768)))
        self.num_hidden_layers = config.get('depth', 
                                           config.get('num_hidden_layers', 12))
        
        # Fix num_heads to ensure it divides hidden_size
        requested_heads = config.get('num_heads', config.get('num_attention_heads', 12))
        if self.hidden_size % requested_heads != 0:
            # Find a valid number of heads
            valid_heads = [h for h in [12, 16, 8, 6, 4] if self.hidden_size % h == 0]
            if valid_heads:
                self.num_attention_heads = valid_heads[0]
                print(f"WARNING: Adjusted num_heads from {requested_heads} to {self.num_attention_heads} "
                      f"(hidden_size={self.hidden_size} must be divisible by num_heads)")
            else:
                # Last resort: use hidden_size // 64 (typical head dimension)
                self.num_attention_heads = max(1, self.hidden_size // 64)
                print(f"WARNING: Using calculated num_heads={self.num_attention_heads} for hidden_size={self.hidden_size}")
        else:
            self.num_attention_heads = requested_heads
        
        mlp_ratio = config.get('mlp_ratio', 4)
        self.intermediate_size = int(mlp_ratio * self.hidden_size)
        
        self.hidden_dropout_prob = config.get('drop_rate', 0.0)
        self.attention_probs_dropout_prob = config.get('attn_drop_rate', 0.0)
        
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
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]
    
    def forward_features(self, x):
        return self.forward(x)


class TransformerBlock(nn.Module):
    """Optimized transformer block"""
    
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=attention_dropout, 
            batch_first=True, bias=True
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== OPTIMIZED FEATURE EXTRACTOR ====================

class OptimizedFeatureExtractor:
    """High-performance feature extraction with GPU + CPU parallelization"""
    
    def __init__(self, config: Config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tile_extractor = ParallelTileExtractor(config)
        self.model_loader = OptimizedModelLoader(config, model_name)
        
        # Load model (can be nn.Module or compiled version)
        self.logger.info(f"Loading model: {model_name}")
        self.model: Union[nn.Module, Any] = self.model_loader.load_model()
        self.transform = self.model_loader.get_transform()
        
        # Setup output directory
        self.output_dir = Path(config.output_root) / model_name / "embeddings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def process_slide(self, slide_path: str, slide_id: str) -> Optional[str]:
        """Process single slide with optimizations"""
        
        try:
            # Check if embeddings already exist
            npz_path = self.output_dir / f"{slide_id}.npz"
            if npz_path.exists():
                self.logger.info(f"Skipping {slide_id} - embeddings already exist")
                return str(npz_path)
            
            start_time = time.time()
            
            # Step 1: Parallel tile extraction
            self.logger.info(f"Extracting tiles from {slide_id}")
            tiles = self.tile_extractor.extract_tiles(slide_path)
            
            if len(tiles) == 0:
                self.logger.warning(f"No tiles extracted from {slide_id}")
                return None
            
            self.logger.info(f"Extracted {len(tiles)} tiles in {time.time()-start_time:.1f}s")
            
            # Step 2: Tile selection (top-k by tissue content)
            if len(tiles) > self.config.top_k_tiles:
                tiles = self._select_top_tiles(tiles)
            
            # Step 3: Batch inference with AMP
            embeddings = self._extract_embeddings(tiles)
            
            # Step 4: Save results
            npz_path = self._save_embeddings(
                embeddings, tiles, slide_id, slide_path, start_time
            )
            
            return npz_path
            
        except Exception as e:
            self.logger.error(f"Failed to process {slide_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_top_tiles(self, tiles: List[Tuple[np.ndarray, int, int, float]]) -> List:
        """Select top-k tiles by tissue content (scores already computed)"""
        
        # Extract scores (already computed during extraction)
        scores = [score for _, _, _, score in tiles]
        
        # Select top-k by score
        top_indices = np.argsort(scores)[-self.config.top_k_tiles:]
        selected = [tiles[i] for i in top_indices]
        
        self.logger.info(f"Selected top {len(selected)} tiles from {len(tiles)} candidates")
        return selected
    
    def _extract_embeddings(self, tiles: List[Tuple[np.ndarray, int, int, float]]) -> np.ndarray:
        """Extract embeddings with optimized batching"""
        
        dataset = FastTileDataset(tiles, self.transform)
        
        # Use minimal workers - more workers = more overhead for small datasets
        num_workers = 2
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=2,
            persistent_workers=False
        )
        
        embeddings = []
        
        with torch.no_grad():
            for batch, coords in tqdm(dataloader, desc="Extracting features", leave=False):
                batch = batch.to(self.config.device, non_blocking=True)
                
                # Mixed precision inference
                if self.config.use_amp:
                    with autocast():
                        # Handle both forward_features method and direct forward
                        if hasattr(self.model, 'forward_features') and callable(self.model.forward_features):
                            emb = self.model.forward_features(batch)
                        else:
                            emb = self.model(batch)
                else:
                    # Handle both forward_features method and direct forward
                    if hasattr(self.model, 'forward_features') and callable(self.model.forward_features):
                        emb = self.model.forward_features(batch)
                    else:
                        emb = self.model(batch)
                
                # Ensure emb is a tensor
                if not isinstance(emb, torch.Tensor):
                    raise TypeError(f"Expected model output to be a Tensor, got {type(emb)}")
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                
                embeddings.append(emb.cpu().float().numpy())
        
        return np.vstack(embeddings)
    
    def _save_embeddings(self, embeddings: np.ndarray, 
                        tiles: List[Tuple[np.ndarray, int, int, float]],
                        slide_id: str, slide_path: str, start_time: float) -> str:
        """Save embeddings and metadata"""
        
        coords = np.array([(x, y) for _, x, y, _ in tiles])
        
        # Compute slide-level aggregated features
        slide_features_mean = embeddings.mean(axis=0)  # Shape: (D,)
        slide_features_max = embeddings.max(axis=0)    # Shape: (D,)
        slide_features_std = embeddings.std(axis=0)    # Shape: (D,)
        
        # Concatenate all aggregations for richer representation
        slide_features_concat = np.concatenate([
            slide_features_mean,
            slide_features_max,
            slide_features_std
        ])  # Shape: (3*D,) = (2304,) for D=768
        
        metadata = {
            'slide_id': slide_id,
            'slide_path': slide_path,
            'model': self.model_name,
            'n_tiles_total': len(tiles),
            'n_tiles_kept': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'processing_time': time.time() - start_time,
            'config': asdict(self.config)
        }
        
        npz_path = self.output_dir / f"{slide_id}.npz"
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,  # Tile-level: (N, D)
            coordinates=coords,  # Tile positions: (N, 2)
            slide_features_mean=slide_features_mean,  # (D,) - mean pooling
            slide_features_max=slide_features_max,    # (D,) - max pooling
            slide_features_std=slide_features_std,    # (D,) - std pooling
            slide_features_concat=slide_features_concat,  # (3*D,) - all combined
            metadata=np.array([metadata], dtype=object)
        )
        
        self.logger.info(f"Saved tile embeddings: {embeddings.shape}, "
                        f"slide features - mean: {slide_features_mean.shape}, "
                        f"concat: {slide_features_concat.shape}")
        
        return str(npz_path)


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
    
    # Initialize config
    config = Config()
    
    print("=" * 80)
    print("OPTIMIZED HISTOPATHOLOGY FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"\nDevice: {config.device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Mixed Precision (AMP): {config.use_amp}")
    
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Tile Extraction Workers: {config.tile_extraction_workers}")
    print(f"DataLoader Workers: {min(config.num_cpu_workers, 32)}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Models: {list(MODEL_REGISTRY.keys())}")
    print(f"Output: {config.output_root}")
    print()
    
    # Load CSV
    print(f"Loading CSV: {config.csv_path}")
    df = pd.read_csv(config.csv_path)
    print(f"Found {len(df)} slides")
    
    if 'image_path' not in df.columns:
        raise ValueError("CSV must have 'image_path' column")
    
    # Process each model
    pipeline_start = time.time()
    
    for model_idx, model_name in enumerate(MODEL_REGISTRY.keys(), 1):
        print(f"\n{'=' * 80}")
        print(f"MODEL {model_idx}/{len(MODEL_REGISTRY)}: {model_name}")
        print(f"{'=' * 80}\n")
        
        setup_logging(config.output_root, model_name)
        logger = logging.getLogger(__name__)
        
        model_start = time.time()
        
        try:
            # Initialize extractor
            extractor = OptimizedFeatureExtractor(config, model_name)
            
            # Process slides
            results: List[Optional[str]] = []
            successful = 0
            failed = 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), 
                               desc=f"[{model_name}] Processing slides"):
                slide_path = row['image_path']
                slide_id = Path(slide_path).stem
                
                if not os.path.exists(slide_path):
                    logger.warning(f"Slide not found: {slide_path}")
                    results.append(None)
                    failed += 1
                    continue
                
                npz_path = extractor.process_slide(slide_path, slide_id)
                results.append(npz_path)
                
                if npz_path:
                    successful += 1
                else:
                    failed += 1
            
            # Update CSV
            df[f'{model_name}_embedding_path'] = results
            
            # Save manifest
            manifest_rows = []
            for idx, (_, row) in enumerate(df.iterrows()):
                npz_path = results[idx]
                if npz_path is not None:
                    manifest_rows.append({
                        'slide_id': Path(row['image_path']).stem,
                        'svs_path': row['image_path'],
                        'embedding_path': npz_path
                    })
            
            manifest_df = pd.DataFrame(manifest_rows)
            manifest_path = Path(config.output_root) / model_name / "embeddings_manifest.csv"
            manifest_df.to_csv(manifest_path, index=False)
            
            # Summary
            model_elapsed = time.time() - model_start
            print(f"\n{'-' * 80}")
            print(f"✓ {model_name} COMPLETED")
            print(f"{'-' * 80}")
            print(f"  Successful: {successful}/{len(df)} slides")
            print(f"  Failed: {failed}/{len(df)} slides")
            print(f"  Time: {model_elapsed/3600:.2f} hours")
            print(f"  Avg time/slide: {model_elapsed/len(df):.1f} sec")
            print(f"{'-' * 80}\n")
            
            # Cleanup
            del extractor
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final CSV
    output_csv = config.csv_path.replace('.csv', '_with_embeddings_optimized.csv')
    df.to_csv(output_csv, index=False)
    
    # Final summary
    pipeline_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total time: {pipeline_elapsed/3600:.2f} hours")
    print(f"Models processed: {len(MODEL_REGISTRY)}")
    print(f"Updated CSV: {output_csv}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Set optimal multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()