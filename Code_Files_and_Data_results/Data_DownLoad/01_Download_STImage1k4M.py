from huggingface_hub import hf_hub_download
import os

"""
HuggingFace Dataset Downloader for Spatial Transcriptomics Data

Purpose: Downloads STimage-1K4M dataset files from HuggingFace Hub for given sample IDs

Inputs:
- Sample list file: Text file with sample IDs (one per line)
- HuggingFace repo: "jiawennnn/STimage-1K4M" dataset
- Expected file types: .png (images), _coord.csv (coordinates), _count.csv (gene expression)

Process:
1. Read sample IDs from input file
2. For each sample, try downloading 3 file types from 3 possible subfolders
3. Search order: ST/ → Visium/ → VisiumHD/ for each file type
4. Stop at first successful download per file

Output: Downloaded files organized in local directory structure
"""


input_file = "/gpfs0/bgu-ofircohen/users/shalevle/tcga-proj/Gather_Data/all_data_from_st_1k4m.txt"
save_dir = "/local1/ofir/shalevle/STImage_1K4M"
os.makedirs(save_dir, exist_ok=True)

subfolders = ["ST", "Visium", "VisiumHD"]
file_specs = [
    ("image", ".png"),
    ("coord", "_coord.csv"),
    ("gene_exp", "_count.csv")
]

with open(input_file, "r") as file:
    for i, line in enumerate(file, 1):
        sample_id = line.strip()
        if not sample_id:
            continue

        for file_type, suffix in file_specs:
            filename = f"{sample_id}{suffix}"
            success = False

            for folder in subfolders:
                path = f"{folder}/{file_type}/{filename}"
                print(f"{i:03d} ⏳ Trying: {path}")
                try:
                    file_path = hf_hub_download(
                        repo_id="jiawennnn/STimage-1K4M",
                        filename=filename,
                        subfolder=f"{folder}/{file_type}",
                        local_dir=save_dir,
                        local_dir_use_symlinks=False,
                        repo_type="dataset"
                    )
                    print(f"{i:03d} ✅ Downloaded: {filename}")
                    success = True
                    break
                except Exception:
                    continue

            if not success:
                print(f"{i:03d} ❌ Failed: {filename}")
