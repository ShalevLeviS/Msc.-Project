#!/usr/bin/env python3
# fix_tcga_paths.py

import os
from pathlib import Path
import pandas as pd

# ================== CONFIG ==================
INPUT_CSV  = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/image_to_star_counts_with_pam50.csv"
IMAGE_ROOT = "/sise/ofircohen-group/datasets/TCGA/TCGA-BRCA-FULL/slides"
STAR_ROOT  = "/sise/ofircohen-group/datasets/TCGA/TCGA-BRCA-FULL/STAR-Counts"
OUT_DIR    = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks"
OUT_NAME   = "TCGA_PATHS_with start_count.csv"

IMAGE_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
STAR_EXTS  = {".tsv", ".txt", ".tsv.gz", ".txt.gz", ".counts", ".counts.gz"}

# ============================================

def build_index(root: Path, allowed_exts: set[str]) -> dict[str, list[str]]:
    idx = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in {".gz", ".bz2", ".xz"}:
                ext = "".join(Path(fn).suffixes[-2:]).lower()
            if allowed_exts and ext not in allowed_exts:
                continue
            full = str(Path(dirpath) / fn)
            idx.setdefault(fn, []).append(full)
    return idx

from typing import Optional, List
def choose_path(candidates: List[str]) -> Optional[str]:

    if not candidates:
        return None
    return candidates[0] if len(candidates) == 1 else None

def main():
    df = pd.read_csv(INPUT_CSV)
    if "image_path" not in df.columns or "star_count_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' and 'star_count_path' columns.")

    print("Indexing image files...")
    img_idx = build_index(Path(IMAGE_ROOT), IMAGE_EXTS)
    print(f"  {sum(len(v) for v in img_idx.values())} image files indexed")

    print("Indexing STAR count files...")
    star_idx = build_index(Path(STAR_ROOT), STAR_EXTS)
    print(f"  {sum(len(v) for v in star_idx.values())} STAR count files indexed")

    new_img_paths, new_star_paths = [], []
    img_statuses, star_statuses = [], []

    for _, row in df.iterrows():
        # fix image
        img_name = Path(str(row["image_path"])).name
        img_candidates = img_idx.get(img_name, [])
        chosen_img = choose_path(img_candidates)
        if chosen_img:
            new_img_paths.append(chosen_img)
            img_statuses.append("fixed")
        elif not img_candidates:
            new_img_paths.append(row["image_path"])
            img_statuses.append("missing")
        else:
            new_img_paths.append(row["image_path"])
            img_statuses.append("ambiguous")

        # fix star count
        star_name = Path(str(row["star_count_path"])).name
        star_candidates = star_idx.get(star_name, [])
        chosen_star = choose_path(star_candidates)
        if chosen_star:
            new_star_paths.append(chosen_star)
            star_statuses.append("fixed")
        elif not star_candidates:
            new_star_paths.append(row["star_count_path"])
            star_statuses.append("missing")
        else:
            new_star_paths.append(row["star_count_path"])
            star_statuses.append("ambiguous")

    df["image_path"] = new_img_paths
    df["star_count_path"] = new_star_paths
    df["image_path_status"] = img_statuses
    df["star_count_path_status"] = star_statuses

    out_csv = Path(OUT_DIR) / OUT_NAME
    df.to_csv(out_csv, index=False)
    print(f"\n✅ New CSV saved to:\n{out_csv}")

    print("\nStatus summary:")
    print(df[["image_path_status", "star_count_path_status"]]
          .apply(lambda s: s.value_counts())
          .fillna(0)
          .astype(int))

if __name__ == "__main__":
    main()
