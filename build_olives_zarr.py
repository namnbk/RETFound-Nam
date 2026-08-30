"""
Convert a handful of OLIVES patients (raw .tif/.jpg B-scans + fundus photos,
plus the clinical/biomarker spreadsheets) into per-patient Zarr v3 stores.

All the actual read/write logic lives in oct_zarr.py (writer functions +
reader API reused by the retinal-data/retinal-ai MCP servers) -- this file is
just the CLI: which patients to build, and where from/to.

Source layout (patient folder is relative to --source_dir; the "original"
batch's --source_dir is olives_original/ which already strips the trial/arm
wrapper, while "batch2" uses OLIVES_ROOT/OLIVES/ and folders keep it, e.g.
"TREX DME/GILA/0234GOD"):
  <source_dir>/<patient_folder>/<visit>/<OD|OS>/*.tif
  <clinical_xlsx>  (sheet "clinical_data_images": File_Path, BCVA, CST, Eye_ID, Patient_ID)
  <biomarker_csv>  (only needed when --include_biomarkers)

Output layout, one .zarr store per patient -- see oct_zarr.py's module
docstring for the full on-disk structure.

Basic commands (from project root):
  python build_olives_zarr.py --batch original --source_dir ./olives_original --out_dir ./olives_zarr

  python build_olives_zarr.py --batch batch2 --include_biomarkers=False \
      --source_dir "C:/Users/lamata/Documents/temp/OLIVES_ROOT/OLIVES" \
      --clinical_xlsx "C:/Users/lamata/Documents/temp/OLIVES_ROOT/OLIVES_Dataset_Labels/full_labels/Clinical_Data_Images.xlsx" \
      --out_dir ./olives_zarr

Each B-scan .tif is 496x504 grayscale; both fundus photos (near-IR localizer
and true-color) are 768x768. Files are told apart by (size, mode) rather than
filename, since the source studies (Prime_FULL vs TREX DME's several arms)
name their files differently -- e.g. bare digits (0.tif), 6-digit-index with
a study prefix (TREXU_000000.tif), or with an extra constant segment
(TREXA_000004_000000.tif). A patient's clinical/biomarker rows are matched by
Patient_ID *and* their source folder's basename, since one patient (234) has
two unrelated enrollments (GILA + TREX arms) with independently numbered
visits -- matching by Patient_ID alone would cross-contaminate them.
"""
import argparse
from pathlib import Path

from oct_zarr import build_patient_zarr, load_clinical, load_biomarkers

BATCHES = {
    "original": [
        {"patient_id": 60, "folder": "01-012"},
        {"patient_id": 251, "folder": "0251GOD"},
    ],
    "batch2": [
        {"patient_id": 64, "folder": "Prime_FULL/01-023"},
        {"patient_id": 67, "folder": "Prime_FULL/01-027"},
        {"patient_id": 71, "folder": "Prime_FULL/01-038"},
        {"patient_id": 75, "folder": "Prime_FULL/02-004"},
        {"patient_id": 77, "folder": "Prime_FULL/02-008"},
        {"patient_id": 79, "folder": "Prime_FULL/02-016"},
        {"patient_id": 95, "folder": "Prime_FULL/02-045"},
        {"patient_id": 206, "folder": "TREX DME/Monthly/0206MOD"},
        {"patient_id": 209, "folder": "TREX DME/TREX/0209TOD"},
        {"patient_id": 211, "folder": "TREX DME/TREX/0211TOD"},
        {"patient_id": 217, "folder": "TREX DME/GILA/0217GOS"},
        {"patient_id": 220, "folder": "TREX DME/TREX/0220TOD"},
        {"patient_id": 221, "folder": "TREX DME/GILA/0221GOD"},
        {"patient_id": 230, "folder": "TREX DME/TREX/0230TOS"},
        {"patient_id": 231, "folder": "TREX DME/TREX/0231TOD"},
        # Patient 234 also has an independent TREX-arm enrollment (TREX
        # DME/TREX/0234TOS, study eye OS) with its own V1..V11 numbering;
        # per user's choice we only materialize the GILA enrollment below.
        {"patient_id": 234, "folder": "TREX DME/GILA/0234GOD"},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default="original", choices=list(BATCHES.keys()))
    ap.add_argument("--source_dir", type=str, default="./olives_original")
    ap.add_argument("--out_dir", type=str, default="./olives_zarr")
    ap.add_argument("--clinical_xlsx", type=str, default=None,
                     help="Defaults to <source_dir>/Clinical_Data_Images.xlsx")
    ap.add_argument("--biomarker_csv", type=str, default=None,
                     help="Defaults to <source_dir>/Biomarker_Clinical_Data_Images.csv")
    ap.add_argument("--include_biomarkers", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--overwrite", action="store_true", default=True)
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clinical_xlsx = Path(args.clinical_xlsx) if args.clinical_xlsx else source_dir / "Clinical_Data_Images.xlsx"
    print("Loading clinical spreadsheet...")
    clinical_df = load_clinical(clinical_xlsx)

    biomarker_df = None
    if args.include_biomarkers:
        biomarker_csv = Path(args.biomarker_csv) if args.biomarker_csv else source_dir / "Biomarker_Clinical_Data_Images.csv"
        print("Loading biomarker spreadsheet...")
        biomarker_df = load_biomarkers(biomarker_csv)

    for patient in BATCHES[args.batch]:
        pid, folder = patient["patient_id"], patient["folder"]
        patient_dir = source_dir / folder
        print(f"\nBuilding Zarr for patient {pid} ({patient_dir}) ...")
        out_path = build_patient_zarr(pid, patient_dir, clinical_df, biomarker_df, out_dir, args.overwrite)
        print(f"  -> {out_path}")

    print(f"\nDone. Zarr stores written to {out_dir}")


if __name__ == "__main__":
    main()
