import os
import csv
from pathlib import Path

"""
File Path Collector for Spatial Transcriptomics Data

Purpose: Maps sample IDs to their corresponding data files and generates CSV lookup table

Inputs:
- Sample list file: Text file with sample IDs (one per line)
- Base directory: Contains data files in subdirectories
- Expected files per sample: {sample_id}.png, {sample_id}_coord.csv, {sample_id}_count.csv

Process:
1. Build file index from directory tree
2. Match each sample ID to its 3 expected files
3. Track missing files
4. Generate CSV mapping table

Output: CSV with columns [sample_id, image_path, coord_path, gene_exp_path]
"""


# Inputs
sample_list_file = "/gpfs0/bgu-ofircohen/users/shalevle/tcga-proj/Gather_Data/all_data_from_st_1k4m.txt"
base_dir = "/local1/ofir/shalevle/STImage_1K4M"
output_csv = "/local1/ofir/shalevle/STImage_1K4M/ST_Image_CSV_paths.csv"

# Build file index once for better performance
print("Building file index...")
file_index = {}
for root, dirs, files in os.walk(base_dir):
    for file in files:
        file_index[file] = os.path.join(root, file)

print(f"Found {len(file_index)} files in directory tree")

# Read sample IDs and collect all at once for better error reporting
with open(sample_list_file, "r") as infile:
    sample_ids = [line.strip() for line in infile if line.strip()]

print(f"Processing {len(sample_ids)} samples...")

# Write CSV
with open(output_csv, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["sample_id", "image_path", "coord_path", "gene_exp_path"])  # Added sample_id column
    
    missing_files = []
    
    for i, sample_id in enumerate(sample_ids, 1):
        if i % 100 == 0:  # Progress every 100 samples
            print(f"Processing {i}/{len(sample_ids)}: {sample_id}")
        
        # Look up files in index
        img_file = f"{sample_id}.png"
        coord_file = f"{sample_id}_coord.csv"
        gene_file = f"{sample_id}_count.csv"
        
        img_path = file_index.get(img_file)
        coord_path = file_index.get(coord_file)
        gene_path = file_index.get(gene_file)
        
        # Track missing files
        missing = []
        if not img_path:
            missing.append(f"image: {img_file}")
        if not coord_path:
            missing.append(f"coord: {coord_file}")
        if not gene_path:
            missing.append(f"gene: {gene_file}")
            
        if missing:
            missing_files.append(f"{sample_id}: {', '.join(missing)}")
        
        # Write row (use empty string for missing paths, or you could skip the row entirely)
        writer.writerow([
            sample_id,
            img_path or "",
            coord_path or "",
            gene_path or ""
        ])

print(f"✅ CSV saved to: {output_csv}")

if missing_files:
    print(f"\n⚠️  Warning: {len(missing_files)} samples had missing files:")
    for missing in missing_files[:10]:  # Show first 10
        print(f"  {missing}")
    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")
else:
    print("✅ All files found successfully!")