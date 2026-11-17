"""
Download the OLIVES dataset (Ophthalmic Labels for Investigating Visual Eye
Semantics) from Hugging Face into a local folder under this workspace.

Repo: https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset (MIT license, public)
Two configs live as top-level folders in the repo:
  - disease_classification  (78k+ OCT scans, disease labels)  ~ default
  - biomarker_detection     (curated split, 6 biomarkers)

Basic command (from project root):
  python olives.py --out_dir ./olives

To grab both configs:
  python olives.py --out_dir ./olives --config all
"""
import argparse

from huggingface_hub import snapshot_download

REPO_ID = "gOLIVES/OLIVES_Dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./olives", help="Local folder to download into.")
    ap.add_argument("--config", type=str, default="disease_classification",
                    choices=["disease_classification", "biomarker_detection", "all"])
    args = ap.parse_args()

    allow_patterns = None if args.config == "all" else [f"{args.config}/*"]

    print(f"Downloading {REPO_ID} ({args.config}) into {args.out_dir} ...")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out_dir,
        allow_patterns=allow_patterns,
    )
    print(f"Done. Files are under: {path}")


if __name__ == "__main__":
    main()
