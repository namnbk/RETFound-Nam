"""
Extract OLIVES B-scans annotated with the 6 target biomarkers from the
disease_classification parquet shards into a flat image folder plus a
multi-label CSV, reusing the patient-level split already fixed in
data/olives_split/splits.json (the same split used for disease classification
-- not recomputed here).

Why a flat folder + CSV, not one ImageFolder tree per biomarker: biomarkers
can co-occur in the same B-scan (unlike DR/DME) -- e.g. Partially/Fully
Attached Vitreous Face are two states of the same underlying process, and
Intraretinal Fluid often co-occurs with Diffuse Retinal Thickening/DME. A
single multi-label classifier lets the shared backbone represent that
correlation instead of training 6 independent binary models. This needs
training/util/datasets.py's MultiLabelImageDataset (not torchvision's
single-label ImageFolder) -- see training/main_finetune.py's --multilabel
flag.

The 6 biomarkers (of 16 annotated in the parquet) and their source columns:
  IRHRF  (B1, Intraretinal Hyperreflective Foci)   <- "IR HRF"
  PAVF   (B2, Partially Attached Vitreous Face)     <- "Partially attached vitreous face"
  FAVF   (B3, Fully Attached Vitreous Face)         <- "Fully attached vitreous face"
  IRF    (B4, Intraretinal Fluid)                   <- "Fluid (IRF)"
  DRTME  (B5, Diffuse Retinal Thickening / DME)     <- "DRT/ME"
  VD     (B6, Vitreous Debris)                      <- "Vitreous debris"

Only a patient's first and last visit carry biomarker annotations -- every
other visit's rows have NaN across all 16 biomarker columns (17,493 of
78,822 total rows are annotated; ANNOTATION_FLAG_COLUMN is never NaN on an
annotated row). The parquet also duplicates a large fraction of rows
byte-for-byte (same image bytes, same metadata, back-to-back row pairs) --
these are dropped before materializing. A further ~12 rows are missing just
one of the 6 target columns despite being otherwise annotated; those rows
are dropped entirely (a multi-label row needs all 6 labels, unlike the old
per-biomarker-folder approach which could skip a row for one biomarker
only). Net: 9,396 unique, fully-labeled B-scans.

Basic command (from project root, after split_olives_dataset.py has already
produced data/olives_split/splits.json):
  python data/materialize_olives_biomarkers.py

Output:
  olives_biomarker_imagefolder/
    train/*.png  val/*.png  test/*.png
  data/olives_split/olives_biomarker_manifest.csv
    filename, split, Patient_ID, uid, IRHRF, PAVF, FAVF, IRF, DRTME, VD (0/1)
    -- this file doubles as the --labels_csv for training/main_finetune.py's
    --multilabel mode.

Image bytes in the parquet are already PNG-encoded, so they're written
straight to disk -- no re-encoding, no quality loss, no extra decode cost.
"""
import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq

from oct_zarr import BIOMARKER_COLUMNS

BIOMARKER_TARGETS = {
    "IRHRF": "IR HRF",
    "PAVF": "Partially attached vitreous face",
    "FAVF": "Fully attached vitreous face",
    "IRF": "Fluid (IRF)",
    "DRTME": "DRT/ME",
    "VD": "Vitreous debris",
}
ANNOTATION_FLAG_COLUMN = "IR HRF"
META_COLUMNS = ["Patient_ID", "Eye_ID", "Scan (n/49)", "BCVA", "CST", "Disease Label"]
DEDUP_KEY = META_COLUMNS + BIOMARKER_COLUMNS


def load_annotated_manifest(data_path):
    files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_path}")
    print(f"Found {len(files)} parquet shards.")

    rows = []
    for fp in files:
        t = pq.read_table(fp, columns=META_COLUMNS + BIOMARKER_COLUMNS)
        df = t.to_pandas()
        df["shard_file"] = os.path.basename(fp)
        df["row_idx"] = range(len(df))
        rows.append(df)
    manifest = pd.concat(rows, ignore_index=True)
    print(f"Total B-scans (all visits): {len(manifest)}")

    annotated = manifest[manifest[ANNOTATION_FLAG_COLUMN].notna()].copy()
    print(f"Annotated B-scans (first/last visit only): {len(annotated)}")

    deduped = annotated.drop_duplicates(subset=DEDUP_KEY, keep="first")
    print(f"Annotated B-scans after dropping exact duplicate rows: {len(deduped)}")

    target_cols = list(BIOMARKER_TARGETS.values())
    complete = deduped.dropna(subset=target_cols).reset_index(drop=True)
    print(f"B-scans with all 6 target biomarkers present: {len(complete)} "
          f"(dropped {len(deduped) - len(complete)} missing just one of the 6)")

    complete["uid"] = complete.index
    return complete


def assign_splits(manifest, splits_json):
    with open(splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    pid_to_split = {pid: split for split, ids in splits.items() for pid in ids}

    manifest = manifest.copy()
    manifest["split"] = manifest["Patient_ID"].map(pid_to_split)
    unmapped = manifest[manifest["split"].isna()]["Patient_ID"].unique()
    if len(unmapped):
        raise ValueError(f"Patient IDs with biomarker annotations but no split assignment "
                          f"in {splits_json}: {sorted(int(p) for p in unmapped)}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="./olives/disease_classification")
    ap.add_argument("--splits_json", type=str, default="./data/olives_split/splits.json")
    ap.add_argument("--manifest_out", type=str, default="./data/olives_split/olives_biomarker_manifest.csv")
    ap.add_argument("--out_dir", type=str, default="./olives_biomarker_imagefolder")
    args = ap.parse_args()

    manifest = load_annotated_manifest(args.data_path)
    manifest = assign_splits(manifest, args.splits_json)
    manifest["filename"] = "p" + manifest["Patient_ID"].astype(str) + "_" + manifest["uid"].astype(str) + ".png"

    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    manifest_cols = ["filename", "split", "Patient_ID", "uid"] + list(BIOMARKER_TARGETS.values())
    out_manifest = manifest[manifest_cols].rename(columns={v: k for k, v in BIOMARKER_TARGETS.items()})
    for short_name in BIOMARKER_TARGETS:
        out_manifest[short_name] = out_manifest[short_name].astype(int)
    out_manifest.to_csv(args.manifest_out, index=False)
    print(f"Saved manifest: {args.manifest_out}")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    written = 0
    for shard_file, group in manifest.groupby("shard_file"):
        fp = os.path.join(args.data_path, shard_file)
        table = pq.read_table(fp, columns=["Image"])
        images = table.column("Image").to_pylist()

        for _, row in group.iterrows():
            img_bytes = images[row["row_idx"]]["bytes"]
            out_path = os.path.join(args.out_dir, row["split"], row["filename"])
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            written += 1

        print(f"{shard_file}: done ({written}/{len(manifest)} total)")

    print(f"\nWrote {written} images to {args.out_dir}")
    for split in ("train", "val", "test"):
        d = os.path.join(args.out_dir, split)
        print(f"  {split}: {len(os.listdir(d))}")
    print("\nPositive counts per biomarker (whole dataset):")
    for short_name in BIOMARKER_TARGETS:
        print(f"  {short_name}: {int(out_manifest[short_name].sum())} / {len(out_manifest)}")


if __name__ == "__main__":
    main()
