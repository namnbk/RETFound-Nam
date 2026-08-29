"""
Convert a handful of OLIVES patients (raw .tif B-scans + fundus photos, plus
the clinical/biomarker spreadsheets) into per-patient Zarr v3 stores.

Source layout (patient folder is relative to --source_dir; the "original"
batch's --source_dir is olives_original/ which already strips the trial/arm
wrapper, while "batch2" uses OLIVES_ROOT/OLIVES/ and folders keep it, e.g.
"TREX DME/GILA/0234GOD"):
  <source_dir>/<patient_folder>/<visit>/<OD|OS>/*.tif
  <clinical_xlsx>  (sheet "clinical_data_images": File_Path, BCVA, CST, Eye_ID, Patient_ID)
  <biomarker_csv>  (only needed when --include_biomarkers)

Output layout, one .zarr store per patient:
  patient_<id>.zarr
    zarr.json                       -- patient-level attrs (id, sex, age)
    eyes/<OD|OS>/visit_NN/
      oct_image                     -- array (B, H, W) uint8, chunks (1, H, W)
      fundus/near-IR                -- array (H, W) uint8, chunks = full shape
      fundus/color                  -- array (H, W, 3) uint8, chunks = full shape
      clinical                      -- group, attrs = {bcva, cst_um, eye_id}
      biomarkers/values             -- array (B, 16) int8, only first/last visit
                                        (only when --include_biomarkers)

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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

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

BIOMARKER_COLUMNS = [
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM",
]

TRAILING_DIGITS = re.compile(r"(\d+)(?=\.[^.]+$)", re.IGNORECASE)
IMAGE_GLOBS = ("*.tif", "*.tiff", "*.jpg", "*.jpeg")


def visit_sort_key(visit_dir_name: str) -> int:
    m = re.search(r"\d+", visit_dir_name)
    return int(m.group()) if m else 0


def get_visits(patient_dir: Path):
    """Return [(visit_label, visit_path), ...] sorted chronologically."""
    visits = [p for p in patient_dir.iterdir() if p.is_dir()]
    visits.sort(key=lambda p: visit_sort_key(p.name))
    return [(p.name, p) for p in visits]


def classify_eye_files(eye_dir: Path):
    """Split a visit/eye folder's image files into bscans / near-IR / color fundus.

    Discriminated by (size, mode) rather than filename, since Prime_FULL uses
    bare digits (0.tif..48.tif, fundus_OD_W0.tif, fundus_W0.tif) while TREX DME
    uses prefixed 6-digit indices (TREXU_000000.tif.., fundus_OD_V1.tif,
    TREXU_001.tif) for the same three roles. Most visits use .tif; a handful
    (e.g. patient 209's V2/V3/V5/V6/V8/V10) use .jpg for the same content.
    """
    bscans = []  # (index, path)
    near_ir = None
    color = None

    image_paths = [p for pattern in IMAGE_GLOBS for p in eye_dir.glob(pattern)]
    for img_path in image_paths:
        with Image.open(img_path) as im:
            size, mode = im.size, im.mode

        if size == (768, 768) and mode == "RGB":
            color = img_path
        elif size == (768, 768) and mode == "L":
            near_ir = img_path
        else:
            m = TRAILING_DIGITS.search(img_path.name)
            if m is None:
                continue
            bscans.append((int(m.group()), img_path))

    bscans.sort(key=lambda t: t[0])
    return bscans, near_ir, color


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_clinical(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="clinical_data_images")
    parts = df["File_Path"].str.strip("/").str.split("/")
    df["visit"] = parts.str[-3]
    df["eye"] = parts.str[-2]
    df["folder_basename"] = parts.str[-4]
    return df


def load_biomarkers(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parts = df["Path (Trial/Arm/Folder/Visit/Eye/Image Name)"].str.strip("/").str.split("/")
    df["visit"] = parts.str[-3]
    df["eye"] = parts.str[-2]
    df["folder_basename"] = parts.str[-4]
    return df


def build_patient_zarr(patient_id: int, patient_dir: Path, clinical_df: pd.DataFrame,
                        biomarker_df: pd.DataFrame | None, out_dir: Path, overwrite: bool):
    out_path = out_dir / f"patient_{patient_id}.zarr"
    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store, overwrite=overwrite)
    root.attrs.update({
        "patient_id": patient_id,
        "sex": "",
        "age": "",
        "source_dataset": "OLIVES",
        "source_folder": patient_dir.name,
    })

    eyes_group = root.create_group("eyes")
    visits = get_visits(patient_dir)
    # Match by Patient_ID *and* source folder basename: a patient can have more
    # than one independent enrollment (e.g. 234's GILA and TREX arms) whose
    # visits are numbered independently, so Patient_ID alone is ambiguous.
    pid_clinical = clinical_df[(clinical_df["Patient_ID"] == patient_id) &
                                (clinical_df["folder_basename"] == patient_dir.name)]
    if biomarker_df is not None:
        pid_biomarker = biomarker_df[(biomarker_df["Patient_ID"] == patient_id) &
                                      (biomarker_df["folder_basename"] == patient_dir.name)]
    else:
        pid_biomarker = None

    for eye in ("OD", "OS"):
        eye_group = eyes_group.create_group(eye)
        visit_num = 0

        for visit_label, visit_path in visits:
            eye_dir = visit_path / eye
            if not eye_dir.is_dir():
                continue

            bscans, near_ir_path, color_path = classify_eye_files(eye_dir)
            if not bscans:
                continue
            visit_num += 1

            visit_group = eye_group.create_group(f"visit_{visit_num:02d}")
            visit_group.attrs.update({"source_visit": visit_label, "eye": eye})

            stack = np.stack([load_gray(p) for _, p in bscans], axis=0)
            b, h, w = stack.shape
            oct_arr = visit_group.create_array(
                "oct_image", shape=stack.shape, chunks=(1, h, w), dtype=stack.dtype,
            )
            oct_arr[:] = stack
            oct_arr.attrs.update({
                "dims": ["bscan", "height", "width"],
                "n_bscans": b,
                "bscan_source_files": [p.name for _, p in bscans],
            })

            fundus_group = visit_group.create_group("fundus")
            if near_ir_path is not None:
                img = load_gray(near_ir_path)
                arr = fundus_group.create_array(
                    "near-IR", shape=img.shape, chunks=img.shape, dtype=img.dtype,
                )
                arr[:] = img
                arr.attrs["source_file"] = near_ir_path.name
            if color_path is not None:
                img = load_rgb(color_path)
                arr = fundus_group.create_array(
                    "color", shape=img.shape, chunks=img.shape, dtype=img.dtype,
                )
                arr[:] = img
                arr.attrs["source_file"] = color_path.name

            crow = pid_clinical[(pid_clinical["visit"] == visit_label) & (pid_clinical["eye"] == eye)]
            if len(crow):
                r = crow.iloc[0]
                clinical_group = visit_group.create_group("clinical")
                clinical_group.attrs.update({
                    "bcva": float(r["BCVA"]),
                    "cst_um": float(r["CST"]),
                    "eye_id": int(r["Eye_ID"]),
                })

            brows = pid_biomarker[(pid_biomarker["visit"] == visit_label) & (pid_biomarker["eye"] == eye)] \
                if pid_biomarker is not None else pid_biomarker
            if brows is not None and len(brows):
                brows = brows.sort_values("Scan (n/49)")
                values = brows[BIOMARKER_COLUMNS].to_numpy(dtype=np.int8)
                biomarkers_group = visit_group.create_group("biomarkers")
                values_arr = biomarkers_group.create_array(
                    "values", shape=values.shape, chunks=values.shape, dtype=values.dtype,
                )
                values_arr[:] = values
                values_arr.attrs.update({
                    "dims": ["bscan", "biomarker"],
                    "labels": BIOMARKER_COLUMNS,
                    "bscan_index_note": "row i corresponds to bscan index i in oct_image (Scan (n/49) is 1-indexed)",
                })

    return out_path


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
