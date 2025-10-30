#!/usr/bin/env python3
import json, io, re
from pathlib import Path
import pandas as pd
import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"

SLIDES_TSV   = "/local1/ofir/TCGA-BRCA-HaE/diagnostic_slides_sample_sheet.tsv"
OUT_MANIFEST = "/local1/ofir/shalevle/TCGA/tcga_data_rnaseq/rnaseq_manifest_star.tsv"

# Accept only modern STAR gene expression TSVs (DR32+)
STAR_TSV_PATTERNS = [
    r"rna_seq\.augmented_star_gene_counts\.tsv$",
    r"star_gene_counts\.tsv$",
]

def read_case_ids(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "Case ID" in df.columns:
        cases = df["Case ID"].str[:12]
    else:
        m = df.apply(lambda r: next((re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", v).group(1)
                                     for v in r.astype(str)
                                     if re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", v)), None), axis=1)
        cases = m
    cases = [c for c in cases if isinstance(c, str) and len(c) == 12]
    return sorted(set(cases))

def chunk(it, n):
    for i in range(0, len(it), n):
        yield it[i:i+n]

def query_star_expr(cases):
    fields = ",".join([
        "file_id","file_name","md5sum","file_size","state","data_format",
        "project.project_id","data_category","experimental_strategy","data_type",
        "analysis.workflow_type","cases.samples.sample_type_code","access"
    ])
    rows = []
    for batch in chunk(cases, 400):
        filters = {
            "op":"and","content":[
                {"op":"in","content":{"field":"project.project_id","value":["TCGA-BRCA"]}},
                {"op":"in","content":{"field":"cases.submitter_id","value":batch}},
                {"op":"in","content":{"field":"data_category","value":["Transcriptome Profiling"]}},
                {"op":"in","content":{"field":"experimental_strategy","value":["RNA-Seq"]}},
                {"op":"in","content":{"field":"data_type","value":["Gene Expression Quantification"]}},
                {"op":"in","content":{"field":"analysis.workflow_type","value":["STAR - Counts"]}},
                {"op":"in","content":{"field":"cases.samples.sample_type_code","value":["01"]}},  # Primary Tumor
                {"op":"in","content":{"field":"access","value":["open"]}},
            ]
        }
        r = requests.post(
            f"{GDC_FILES}?size=20000&format=TSV&fields={fields}",
            headers={"Content-Type":"application/json"},
            data=json.dumps(filters), timeout=180
        )
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), sep="\t", dtype=str)
        if not df.empty:
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=fields.split(","))
    df = pd.concat(rows, ignore_index=True).drop_duplicates("file_id")

    # Final safety net: only the STAR gene-count TSV filenames
    pat = re.compile("|".join(STAR_TSV_PATTERNS), re.IGNORECASE)
    df = df[df["file_name"].str.contains(pat, na=False)]

    # Write gdc-client manifest columns only
    man = df.rename(columns={
        "file_id":"id","file_name":"filename","md5sum":"md5","file_size":"size"
    })[["id","filename","md5","size","state"]]
    return man

def main():
    cases = read_case_ids(SLIDES_TSV)
    print(f"Cases: {len(cases)}")
    man = query_star_expr(cases)
    print(f"RNA-seq STAR files: {len(man)}")
    Path(OUT_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT_MANIFEST, sep="\t", index=False)
    print(f"Manifest written: {OUT_MANIFEST}")

if __name__ == "__main__":
    main()
