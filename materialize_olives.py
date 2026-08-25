"""
Extract OLIVES B-scans from the parquet shards into an ImageFolder-style
directory tree, using the split assignment from split_olives_dataset.py.

util/datasets.py's build_dataset() calls torchvision.datasets.ImageFolder on
<data_path>/<train|val|test>, so this produces exactly the layout
main_finetune.py / train.sh already expect -- no changes needed there.

Basic command (from project root, after running split_olives_dataset.py):
  python materialize_olives.py --data_path ./olives/disease_classification \
    --manifest ./olives_split/olives_manifest.csv --out_dir ./olives_imagefolder

Output:
  olives_imagefolder/
    train/DR/*.png  train/DME/*.png
    val/DR/*.png    val/DME/*.png
    test/DR/*.png   test/DME/*.png

Image bytes in the parquet are already PNG-encoded, so they're written
straight to disk -- no re-encoding, no quality loss, no extra decode cost.
"""
import argparse
import os

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="./olives/disease_classification")
    ap.add_argument("--manifest", type=str, default="./olives_split/olives_manifest.csv")
    ap.add_argument("--out_dir", type=str, default="./olives_imagefolder")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest rows: {len(manifest)}")

    for split in ("train", "val", "test"):
        for label_name in ("DR", "DME"):
            os.makedirs(os.path.join(args.out_dir, split, label_name), exist_ok=True)

    written = 0
    for shard_file, group in manifest.groupby("shard_file"):
        fp = os.path.join(args.data_path, shard_file)
        table = pq.read_table(fp, columns=["Image"])
        images = table.column("Image").to_pylist()

        for _, row in group.iterrows():
            img_bytes = images[row["row_idx"]]["bytes"]
            fname = f"p{row['Patient_ID']}_{row['uid']}.png"
            out_path = os.path.join(args.out_dir, row["split"], row["label_name"], fname)
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            written += 1

        print(f"{shard_file}: done ({written}/{len(manifest)} total)")

    print(f"\nWrote {written} images to {args.out_dir}")
    for split in ("train", "val", "test"):
        for label_name in ("DR", "DME"):
            d = os.path.join(args.out_dir, split, label_name)
            print(f"  {split}/{label_name}: {len(os.listdir(d))}")


if __name__ == "__main__":
    main()
