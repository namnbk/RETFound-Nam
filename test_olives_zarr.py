"""
Sanity-check the Zarr stores produced by build_olives_zarr.py: walk each
patient store, print its hierarchy, and verify a sample of arrays round-trip
back to pixel-identical data against the original .tif files.

Basic command (from project root, after running build_olives_zarr.py):
  python test_olives_zarr.py --zarr_dir ./olives_zarr --source_dir ./olives_original
"""
import argparse
from pathlib import Path

import numpy as np
import zarr
from PIL import Image


def describe_store(store_path: Path):
    root = zarr.open_group(zarr.storage.LocalStore(str(store_path)), mode="r")
    print(f"\n=== {store_path.name} ===")
    print("patient attrs:", dict(root.attrs))

    for eye in ("OD", "OS"):
        if eye not in root["eyes"]:
            continue
        eye_group = root["eyes"][eye]
        visit_names = sorted(eye_group.keys())
        print(f"  eye {eye}: {len(visit_names)} visits -> {visit_names}")

        for visit_name in visit_names:
            visit_group = eye_group[visit_name]
            members = list(visit_group.keys())
            oct_shape = visit_group["oct_image"].shape if "oct_image" in members else None
            has_clinical = "clinical" in members
            has_biomarkers = "biomarkers" in members
            print(f"    {visit_name} (source={visit_group.attrs.get('source_visit')}): "
                  f"members={members}, oct_image.shape={oct_shape}, "
                  f"clinical={has_clinical}, biomarkers={has_biomarkers}")
            if has_clinical:
                print(f"      clinical attrs: {dict(visit_group['clinical'].attrs)}")
            if has_biomarkers:
                vals = visit_group["biomarkers"]["values"]
                print(f"      biomarkers/values.shape={vals.shape}, dtype={vals.dtype}, "
                      f"labels[:3]={vals.attrs['labels'][:3]}")


def verify_roundtrip(store_path: Path, source_dir: Path):
    """Spot-check that the first B-scan and the fundus images of visit_01/OD
    are pixel-identical to their source .tif files."""
    root = zarr.open_group(zarr.storage.LocalStore(str(store_path)), mode="r")
    patient_id = root.attrs["patient_id"]
    source_folder = root.attrs["source_folder"]

    visit_group = root["eyes"]["OD"]["visit_01"]
    source_visit = visit_group.attrs["source_visit"]
    eye_dir = source_dir / source_folder / source_visit / "OD"

    oct_arr = visit_group["oct_image"]
    bscan_files = oct_arr.attrs["bscan_source_files"]
    first_slice = np.asarray(oct_arr[0])
    original = np.array(Image.open(eye_dir / bscan_files[0]).convert("L"))
    assert np.array_equal(first_slice, original), f"B-scan mismatch for patient {patient_id}"
    print(f"  [OK] patient {patient_id}: oct_image[0] matches {bscan_files[0]} "
          f"(shape={first_slice.shape})")

    if "near-IR" in visit_group["fundus"]:
        near_ir = np.asarray(visit_group["fundus"]["near-IR"][:])
        src_name = visit_group["fundus"]["near-IR"].attrs["source_file"]
        original = np.array(Image.open(eye_dir / src_name).convert("L"))
        assert np.array_equal(near_ir, original), f"near-IR fundus mismatch for patient {patient_id}"
        print(f"  [OK] patient {patient_id}: fundus/near-IR matches {src_name}")

    if "color" in visit_group["fundus"]:
        color = np.asarray(visit_group["fundus"]["color"][:])
        src_name = visit_group["fundus"]["color"].attrs["source_file"]
        original = np.array(Image.open(eye_dir / src_name).convert("RGB"))
        assert np.array_equal(color, original), f"color fundus mismatch for patient {patient_id}"
        print(f"  [OK] patient {patient_id}: fundus/color matches {src_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_dir", type=str, default="./olives_zarr")
    ap.add_argument("--source_dir", type=str, default="./olives_original")
    args = ap.parse_args()

    zarr_dir = Path(args.zarr_dir)
    source_dir = Path(args.source_dir)
    store_paths = sorted(zarr_dir.glob("*.zarr"))

    for store_path in store_paths:
        describe_store(store_path)

    print("\n--- round-trip pixel verification ---")
    for store_path in store_paths:
        verify_roundtrip(store_path, source_dir)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
