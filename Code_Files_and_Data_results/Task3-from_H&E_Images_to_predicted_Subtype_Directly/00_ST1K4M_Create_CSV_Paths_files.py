import os
import csv
from pathlib import Path

"""
Auto-discover sample files under:
base_dir/
  ST/
    coord/
    gene_exp/
    image/
  Visium/
    coord/
    gene_exp/
    image/

Expected file patterns:
  image: {sample_id}.png
  coord: {sample_id}_coord.csv
  gene:  {sample_id}_count.csv

Output CSV columns:
  [sample_id, image_path, coord_path, gene_exp_path]
"""

# ===== Config =====
base_dir = "/sise/ofircohen-group/datasets/ST1K4M"
output_csv = "/sise/ofircohen-group/Shalev/tcga-proj/Code_Files_and_Data_results/Data_From_Tasks/ST_Image_CSV_paths.csv"

# ===== Parsers (no 3.10 union types) =====
def parse_sample_from_image(name):
    n = name.lower()
    if n.endswith(".png"):
        return Path(name).stem
    return None

def parse_sample_from_coord(name):
    n = name.lower()
    if n.endswith("_coord.csv"):
        # remove the trailing "_coord"
        return Path(name).stem[:-6]
    return None

def parse_sample_from_gene(name):
    n = name.lower()
    if n.endswith("_count.csv"):
        # remove the trailing "_count"
        return Path(name).stem[:-6]
    return None

# ===== Collect paths =====
images = {}   # sample_id -> image_path
coords = {}   # sample_id -> coord_path
genes  = {}   # sample_id -> gene_exp_path
dup_warn = {"image": set(), "coord": set(), "gene": set()}

print(f"Scanning directory tree under: {base_dir}")
for root, dirs, files in os.walk(base_dir):
    rel = Path(root).relative_to(base_dir)
    parts = [p.lower() for p in rel.parts]

    # Only consider expected leaves
    in_image   = "image" in parts
    in_coord   = "coord" in parts
    in_geneexp = "gene_exp" in parts

    for fname in files:
        fpath = os.path.join(root, fname)

        if in_image:
            sid = parse_sample_from_image(fname)
            if sid:
                if sid in images:
                    dup_warn["image"].add(sid)
                images[sid] = fpath

        elif in_coord:
            sid = parse_sample_from_coord(fname)
            if sid:
                if sid in coords:
                    dup_warn["coord"].add(sid)
                coords[sid] = fpath

        elif in_geneexp:
            sid = parse_sample_from_gene(fname)
            if sid:
                if sid in genes:
                    dup_warn["gene"].add(sid)
                genes[sid] = fpath

# ===== Write CSV =====
all_sample_ids = sorted(set(images) | set(coords) | set(genes))
print(f"Discovered {len(all_sample_ids)} unique sample IDs")

Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

missing_rows = 0
with open(output_csv, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["sample_id", "image_path", "coord_path", "gene_exp_path"])

    for i, sid in enumerate(all_sample_ids, 1):
        if i % 200 == 0:
            print(f"Writing {i}/{len(all_sample_ids)}: {sid}")
        img = images.get(sid, "")
        crd = coords.get(sid, "")
        gen = genes.get(sid, "")
        if not img or not crd or not gen:
            missing_rows += 1
        writer.writerow([sid, img, crd, gen])

print(f"✅ CSV saved to: {output_csv}")

# ===== Reports =====
def _report_dups(label, s):
    if s:
        print(f"⚠️  Duplicate {label} entries for {len(s)} sample IDs (kept the last seen path). Examples:")
        for k in list(s)[:10]:
            print(f"  - {k}")

_report_dups("image", dup_warn["image"])
_report_dups("coord", dup_warn["coord"])
_report_dups("gene",  dup_warn["gene"])

if missing_rows:
    print(f"⚠️  {missing_rows} rows have at least one missing path (image/coord/gene).")
else:
    print("✅ All discovered sample IDs have all three files.")
