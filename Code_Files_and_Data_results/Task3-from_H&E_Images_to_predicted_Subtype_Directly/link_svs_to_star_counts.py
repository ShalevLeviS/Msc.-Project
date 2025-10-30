#!/usr/bin/env python3
# link_svs_to_star_counts.py
# Pairs TCGA SVS slides with matching RNA-Seq "STAR - Counts" files, writes a CSV, and (optionally) downloads via gdc-client.

import os, re, sys, json, time, argparse, shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import pandas as pd

# ====== DEFAULTS (your paths) ======
DEFAULT_SLIDES_ROOT = "/local1/ofir/TCGA-BRCA-HaE/slides"
DEFAULT_SLIDES_MANIFEST_TSV = "/local1/ofir/TCGA-BRCA-HaE/diagnostic_slides_manifest.tsv"
DEFAULT_OUT_ROOT = "/local1/ofir/shalevle/TCGA/TCGA-Data"
DEFAULT_PROJECT_ID = "TCGA-BRCA"
DEFAULT_GDC_CLIENT = "/local1/ofir/shalevle/TCGA/gdc-client"  # <- your path

# ====== CONSTANTS ======
GDC_BASE = "https://api.gdc.cancer.gov"
SAMPLE_CODE_TO_LABEL = {
    "01": "Primary Tumor",
    "11": "Solid Tissue Normal",
}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

# ====== HTTP HELPERS ======
def gdc_post(endpoint: str, payload: Dict[str, Any], size: int = 2000) -> Dict[str, Any]:
    url = f"{GDC_BASE}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    data = dict(payload)
    data.setdefault("size", size)
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
    r.raise_for_status()
    return r.json()

def gdc_files_query(filters: Dict[str, Any], fields: str, size: int = 2000) -> Dict[str, Any]:
    payload = {
        "filters": filters,
        "fields": fields,
        "format": "JSON",
        "pretty": "false",
        "size": size,
    }
    return gdc_post("files", payload, size=size)

# ====== UTIL ======
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_uuid(s: Optional[str]) -> bool:
    return bool(s) and bool(UUID_RE.match(str(s)))

def parse_sample_code_from_name(filename: str) -> Optional[str]:
    """
    Extract the 2-digit TCGA sample code from a slide filename if present.
    Example: TCGA-BH-A18V-11A-01-TSA....svs -> '11'
    """
    stem = Path(filename).name
    # Normalize dots to dashes for easier matching
    m = re.search(r"TCGA-[^-]+-[^-]+-(\d{2})", stem.replace(".", "-"))
    return m.group(1) if m else None

def find_gdc_client(preferred: str = "") -> str:
    """
    Resolve path to gdc-client. Prefer explicit path, else use PATH.
    """
    if preferred:
        p = Path(preferred).expanduser()
        if not p.exists():
            raise SystemExit(f"gdc-client not found at '{p}'")
        if not os.access(p, os.X_OK):
            raise SystemExit(f"gdc-client at '{p}' is not executable. Run: chmod +x '{p}'")
        return str(p)
    found = shutil.which("gdc-client")
    if found:
        return found
    raise SystemExit(
        "gdc-client not found in PATH. Provide --gdc-client /path/to/gdc-client "
        "or export GDC_CLIENT=/path/to/gdc-client"
    )

# ====== STEP 1: Scan SVS tree ======
def collect_svs_paths(slides_root: str) -> pd.DataFrame:
    svs_rows = []
    for root, _, files in os.walk(slides_root):
        for f in files:
            if f.lower().endswith(".svs"):
                full = str(Path(root) / f)
                parent_uuid = Path(root).name
                svs_rows.append({
                    "image_path": full,
                    "filename": f,
                    "parent_uuid": parent_uuid if is_uuid(parent_uuid) else None,
                    "sample_code_from_name": parse_sample_code_from_name(f),
                })
    return pd.DataFrame(svs_rows)

# ====== STEP 2: Read slides manifest (filename <-> file_id) ======
def read_slides_manifest(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    # Be permissive with column names
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("file_id") or "id"
    filename_col = cols.get("filename") or "filename"
    df = df.rename(columns={id_col: "id", filename_col: "filename"})
    if not {"id", "filename"}.issubset(df.columns):
        raise SystemExit("Slides manifest must contain 'id' (or 'file_id') and 'filename' columns.")
    return df[["id", "filename"]].copy()

# ====== STEP 3: Slide file_id -> case/sample info ======
def fetch_slide_case_sample_info(svs_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    fields = ",".join([
        "file_id","file_name",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
    ])
    out: Dict[str, Dict[str, Any]] = {}
    batch = 500
    for i in range(0, len(svs_ids), batch):
        subset = svs_ids[i:i+batch]
        filters = {"op": "in", "content": {"field": "files.file_id", "value": subset}}
        js = gdc_files_query(filters, fields, size=5000)
        hits = js.get("data", {}).get("hits", [])
        for hit in hits:
            fid = hit["file_id"]
            case_ids = []
            sample_types = []
            sample_submitters = []
            for c in hit.get("cases", []):
                if "submitter_id" in c:
                    case_ids.append(c["submitter_id"])
                for s in c.get("samples", []):
                    if "sample_type" in s:
                        sample_types.append(s["sample_type"])
                    if "submitter_id" in s:
                        sample_submitters.append(s["submitter_id"])
            out[fid] = {
                "slide_file_name": hit.get("file_name"),
                "case_submitter_ids": sorted(set(case_ids)),
                "slide_sample_types": sorted(set(sample_types)),
                "slide_sample_submitter_ids": sorted(set(sample_submitters)),
            }
        time.sleep(0.15)
    return out

def desired_sample_label(sample_code_from_name: Optional[str], slide_sample_types: List[str]) -> str:
    if sample_code_from_name in SAMPLE_CODE_TO_LABEL:
        return SAMPLE_CODE_TO_LABEL[sample_code_from_name]
    if "Primary Tumor" in slide_sample_types:
        return "Primary Tumor"
    if "Solid Tissue Normal" in slide_sample_types:
        return "Solid Tissue Normal"
    return slide_sample_types[0] if slide_sample_types else "Primary Tumor"

# ====== STEP 4: Query RNA STAR - Counts per case + sample_type ======
def fetch_rna_star_for_case(case_submitter_id: str, sample_type_label: str, project: str) -> List[Dict[str, Any]]:
    fields = ",".join([
        "file_id","file_name","md5sum","file_size","updated_datetime",
        "cases.submitter_id","cases.samples.submitter_id","cases.samples.sample_type"
    ])
    filters = {
        "op":"and",
        "content":[
            {"op":"in","content":{"field":"cases.project.project_id","value":[project]}},
            {"op":"in","content":{"field":"data_category","value":["Transcriptome Profiling"]}},
            {"op":"in","content":{"field":"data_type","value":["Gene Expression Quantification"]}},
            {"op":"in","content":{"field":"analysis.workflow_type","value":["STAR - Counts"]}},
            {"op":"in","content":{"field":"cases.submitter_id","value":[case_submitter_id]}},
            {"op":"in","content":{"field":"cases.samples.sample_type","value":[sample_type_label]}},
        ]
    }
    js = gdc_files_query(filters, fields, size=2000)
    return js.get("data", {}).get("hits", [])

def pick_best_rna_file(hits: List[Dict[str, Any]], preferred_sample_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    def sample_code(sid: Optional[str]) -> Optional[str]:
        m = re.search(r"TCGA-[^-]+-[^-]+-(\d{2})", sid or "")
        return m.group(1) if m else None

    # Prefer same sample code (e.g., 01 vs 11) when possible
    if preferred_sample_code:
        preferred = []
        for h in hits:
            for c in h.get("cases", []):
                for s in c.get("samples", []):
                    if sample_code(s.get("submitter_id")) == preferred_sample_code:
                        preferred.append(h)
                        break
        if preferred:
            hits = preferred

    # Break ties by larger size, then latest updated
    def key(h):
        size = h.get("file_size") or 0
        upd = h.get("updated_datetime") or ""
        return (int(size), upd)

    return sorted(hits, key=key, reverse=True)[0] if hits else None

# ====== STEP 5: Write manifest and download ======
def write_rna_manifest(rows: List[tuple], path_tsv: str) -> None:
    df = pd.DataFrame(rows, columns=["id","filename","md5","size","state"])
    df.to_csv(path_tsv, sep="\t", index=False)

def gdc_download_with_manifest(manifest_tsv: str, out_dir: str, gdc_client_path: str) -> None:
    import subprocess
    ensure_dir(Path(out_dir))
    cmd = [gdc_client_path, "download", "-m", manifest_tsv, "-d", out_dir]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ====== MAIN ======
def main():
    ap = argparse.ArgumentParser(description="Link TCGA SVS to RNA-Seq STAR Counts and download them.")
    ap.add_argument("--slides-root", default=DEFAULT_SLIDES_ROOT)
    ap.add_argument("--slides-manifest", default=DEFAULT_SLIDES_MANIFEST_TSV)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--project", default=DEFAULT_PROJECT_ID)
    ap.add_argument("--gdc-client", default=os.environ.get("GDC_CLIENT", DEFAULT_GDC_CLIENT))
    ap.add_argument("--download", dest="download", action="store_true")
    ap.add_argument("--no-download", dest="download", action="store_false")
    ap.set_defaults(download=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    pairs_csv = out_root / "image_to_star_counts.csv"
    rna_manifest_tsv = out_root / "rna_manifest_star.tsv"
    download_dir = out_root / "STAR-Counts"

    ensure_dir(out_root)
    ensure_dir(download_dir)

    gdc_client_path = find_gdc_client(args.gdc_client)

    print("[1/6] Scanning SVS...")
    svs_df = collect_svs_paths(args.slides_root)
    if svs_df.empty:
        print("No .svs files found under:", args.slides_root)
        sys.exit(1)

    print("[2/6] Reading slides manifest...")
    man = read_slides_manifest(args.slides_manifest)

    # Merge by filename to get slide file_id; fallback to parent UUID folder name
    merged = svs_df.merge(man, how="left", left_on="filename", right_on="filename")
    merged["slide_file_id"] = merged.apply(
        lambda r: r["id"] if pd.notna(r["id"]) else (r["parent_uuid"] if is_uuid(r["parent_uuid"]) else None),
        axis=1
    )
    missing_ids = int(merged["slide_file_id"].isna().sum())
    if missing_ids:
        print(f"WARNING: {missing_ids} SVS without file_id (not in manifest and parent dir not UUID). They will be skipped.")
        merged = merged[merged["slide_file_id"].notna()].copy()

    if merged.empty:
        print("No SVS with resolvable file_id. Exiting.")
        sys.exit(1)

    print("[3/6] Querying GDC for slide→case/sample info...")
    slide_meta = fetch_slide_case_sample_info(merged["slide_file_id"].tolist())
    merged["case_submitter_id"] = merged["slide_file_id"].map(lambda fid: (slide_meta.get(fid, {}).get("case_submitter_ids") or [None])[0])
    merged["slide_sample_types"] = merged["slide_file_id"].map(lambda fid: slide_meta.get(fid, {}).get("slide_sample_types") or [])
    merged["desired_sample_type"] = merged.apply(
        lambda r: desired_sample_label(r["sample_code_from_name"], r["slide_sample_types"]),
        axis=1
    )
    merged["preferred_sample_code"] = merged["sample_code_from_name"].fillna("")
    merged = merged[merged["case_submitter_id"].notna()].copy()

    if merged.empty:
        print("No SVS with case_submitter_id. Exiting.")
        sys.exit(1)

    print("[4/6] Finding RNA STAR-Counts per SVS...")
    cache: Dict[tuple, Optional[Dict[str, Any]]] = {}
    rna_rows_for_manifest: Dict[str, tuple] = {}
    star_target_path: List[Optional[str]] = []

    for _, row in merged.iterrows():
        key = (row["case_submitter_id"], row["desired_sample_type"])
        if key not in cache:
            hits = fetch_rna_star_for_case(row["case_submitter_id"], row["desired_sample_type"], args.project)
            best = pick_best_rna_file(hits, preferred_sample_code=row["preferred_sample_code"])
            cache[key] = best
            time.sleep(0.15)
        best = cache[key]
        if not best:
            star_target_path.append(None)
            continue

        star_id = best["file_id"]
        star_name = best["file_name"]
        md5 = best.get("md5sum") or ""
        size = best.get("file_size") or ""
        state = "released"

        # Add single row per unique RNA file_id to manifest rows
        if star_id not in rna_rows_for_manifest:
            rna_rows_for_manifest[star_id] = (star_id, star_name, md5, size, state)

        expected_path = str(download_dir / star_id / star_name)
        star_target_path.append(expected_path)

    merged["star_count_path"] = star_target_path

    result_df = merged[merged["star_count_path"].notna()][
        ["image_path","star_count_path","slide_file_id","case_submitter_id","desired_sample_type"]
    ].copy()

    if result_df.empty:
        print("No matching STAR - Counts found. Exiting.")
        sys.exit(2)

    print("[5/6] Writing pairs CSV and manifest...")
    ensure_dir(out_root)
    result_df.to_csv(pairs_csv, index=False)
    write_rna_manifest(list(rna_rows_for_manifest.values()), str(rna_manifest_tsv))

    if args.download:
        print("[6/6] Downloading STAR counts with gdc-client...")
        gdc_download_with_manifest(str(rna_manifest_tsv), str(download_dir), gdc_client_path)
        print("Download complete.")

    print(f"\nPairs CSV : {pairs_csv}")
    print(f"Manifest  : {rna_manifest_tsv}")
    print(f"Downloads : {download_dir}")

if __name__ == "__main__":
    main()
