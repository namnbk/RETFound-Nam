"""
Patient-level, disease-stratified train/val/test split for the OLIVES
disease_classification parquet shards.

Why patient-level: OLIVES is longitudinal (repeated visits/B-scans per eye).
Splitting at the row (B-scan) level lets near-duplicate scans of the same
patient leak across train/val/test. Instead we split whole patients (all
visits, all scans) 70/10/20, stratified by Disease Label (0=DR, 1=DME).

This script only reads the lightweight metadata columns (Patient_ID, Eye_ID,
Disease Label, Scan) from every shard -- not the image bytes -- so it's fast.

Basic command (from project root):
  python split_olives_dataset.py --data_path ./olives/disease_classification

Outputs (under --out_dir, default ./data/olives_split):
  splits.json           patient IDs per split (train/val/test), for reuse
                         across every model you compare
  olives_manifest.csv   one row per B-scan: which shard/row it lives at,
                         its Patient_ID/Eye_ID/Disease Label, and assigned
                         split -- consumed by materialize_olives.py
"""
import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

LABEL_NAMES = {0.0: "DR", 1.0: "DME"}


def load_manifest(data_path):
    files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_path}")
    print(f"Found {len(files)} parquet shards.")

    rows = []
    for fp in files:
        t = pq.read_table(fp, columns=["Patient_ID", "Eye_ID", "Disease Label", "Scan (n/49)"])
        df = t.to_pandas()
        df["shard_file"] = os.path.basename(fp)
        df["row_idx"] = range(len(df))
        rows.append(df)
    manifest = pd.concat(rows, ignore_index=True)
    manifest = manifest.rename(columns={"Disease Label": "Disease_Label", "Scan (n/49)": "Scan"})
    print(f"Total B-scans: {len(manifest)}")
    return manifest


def split_patients(manifest, val_size=0.10, test_size=0.20, seed=42):
    patients = manifest.groupby("Patient_ID", as_index=False)["Disease_Label"].first()
    n_per_patient = manifest.groupby("Patient_ID")["Disease_Label"].nunique()
    assert n_per_patient.max() == 1, "Some patients have multiple disease labels."

    temp_size = val_size + test_size
    train_p, temp_p = train_test_split(
        patients, test_size=temp_size, stratify=patients["Disease_Label"], random_state=seed
    )
    val_p, test_p = train_test_split(
        temp_p, test_size=test_size / temp_size, stratify=temp_p["Disease_Label"], random_state=seed
    )

    train_ids, val_ids, test_ids = (
        set(train_p["Patient_ID"]), set(val_p["Patient_ID"]), set(test_p["Patient_ID"])
    )
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    return train_ids, val_ids, test_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="./olives/disease_classification")
    ap.add_argument("--out_dir", type=str, default="./data/olives_split")
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = load_manifest(args.data_path)
    train_ids, val_ids, test_ids = split_patients(manifest, args.val_size, args.test_size, args.seed)

    def assign(pid):
        if pid in train_ids:
            return "train"
        if pid in val_ids:
            return "val"
        return "test"

    manifest["split"] = manifest["Patient_ID"].map(assign)
    manifest["label_name"] = manifest["Disease_Label"].map(LABEL_NAMES)
    manifest["uid"] = manifest.index  # globally unique across shards (row_idx alone resets per shard)

    manifest_path = os.path.join(args.out_dir, "olives_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    splits = {
        "train": sorted(int(p) for p in train_ids),
        "val": sorted(int(p) for p in val_ids),
        "test": sorted(int(p) for p in test_ids),
    }
    splits_path = os.path.join(args.out_dir, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("\nPatients per split:", {k: len(v) for k, v in splits.items()})
    print("B-scans per split:\n", manifest["split"].value_counts())
    print("\nB-scans per (split, label):\n", manifest.groupby(["split", "label_name"]).size())
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved patient-ID splits: {splits_path}")


if __name__ == "__main__":
    main()
