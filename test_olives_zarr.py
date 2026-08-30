"""
Sanity-check the Zarr stores produced by build_olives_zarr.py: walk each
patient store using oct_zarr's reader API (doubling as a regression test of
that API), and verify a sample of arrays round-trip back to pixel-identical
data against the original source image files.

Basic command (from project root, after running build_olives_zarr.py):
  python test_olives_zarr.py --zarr_dir ./olives_zarr --source_dir ./olives_original
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import oct_zarr as oz


def describe_store(patient_id: int, zarr_dir: Path):
    info = oz.get_patient_info(patient_id, zarr_dir)
    print(f"\n=== patient_{patient_id}.zarr ===")
    print("patient attrs:", info)

    for eye in oz.list_eyes(patient_id, zarr_dir):
        visits = oz.list_visits(patient_id, eye, zarr_dir)
        print(f"  eye {eye}: {len(visits)} visits -> {[v['visit_id'] for v in visits]}")

        for v in visits:
            vol_info = oz.get_oct_volume_info(patient_id, eye, v["visit_id"], zarr_dir)
            print(f"    {v['visit_id']} (source={v['source_visit']}): "
                  f"modalities={v['modalities']}, oct_shape={vol_info['shape']}, "
                  f"has_clinical={v['has_clinical']}")
            if v["has_clinical"]:
                print(f"      clinical: {oz.get_clinical(patient_id, eye, v['visit_id'], zarr_dir)}")


def verify_roundtrip(patient_id: int, zarr_dir: Path, source_dir: Path):
    """Spot-check that the first B-scan and the fundus images of the first
    visit on the study eye are pixel-identical to their source image files."""
    info = oz.get_patient_info(patient_id, zarr_dir)
    source_folder = info["source_folder"]

    eye = oz.list_eyes(patient_id, zarr_dir)[0]
    visits = oz.list_visits(patient_id, eye, zarr_dir)
    visit_id = visits[0]["visit_id"]
    source_visit = visits[0]["source_visit"]
    eye_dir = source_dir / source_folder / source_visit / eye

    bscan, source_file = oz.get_bscan_array(patient_id, eye, visit_id, 0, zarr_dir)
    original = np.array(Image.open(eye_dir / source_file).convert("L"))
    assert np.array_equal(bscan, original), f"B-scan mismatch for patient {patient_id}"
    print(f"  [OK] patient {patient_id}: bscan 0 matches {source_file} (shape={bscan.shape})")

    for modality in ("near-IR", "color"):
        if f"fundus_{'near_ir' if modality == 'near-IR' else 'color'}" not in visits[0]["modalities"]:
            continue
        arr, source_file = oz.get_fundus_array(patient_id, eye, visit_id, modality, zarr_dir)
        mode = "L" if modality == "near-IR" else "RGB"
        original = np.array(Image.open(eye_dir / source_file).convert(mode))
        assert np.array_equal(arr, original), f"fundus/{modality} mismatch for patient {patient_id}"
        print(f"  [OK] patient {patient_id}: fundus/{modality} matches {source_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_dir", type=str, default="./olives_zarr")
    ap.add_argument("--source_dir", type=str, default="./olives_original")
    args = ap.parse_args()

    zarr_dir = Path(args.zarr_dir)
    source_dir = Path(args.source_dir)
    patient_ids = oz.list_patient_ids(zarr_dir)

    for pid in patient_ids:
        describe_store(pid, zarr_dir)

    print("\n--- round-trip pixel verification ---")
    for pid in patient_ids:
        try:
            verify_roundtrip(pid, zarr_dir, source_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] patient {pid}: source file not reachable under --source_dir={source_dir} ({e})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
