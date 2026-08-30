"""
Offline, evaluator-only ground-truth builder for the OLIVES agent benchmark.

Imports oct_zarr.py and grad_cam_retfound.py directly (no MCP, no agent) and
reads the out-of-band label sources (olives_original/, olives_split/) that
the retinal-data / retinal-ai MCP servers structurally never touch. Output
(eval/answer_keys.json) is the hidden answer key generate_tasks.py and
grade.py consume -- the agent under test must never see this file or the
sources it reads.

Usage (from repo root, with the retfound conda env):
  python eval/build_answer_keys.py --patients 60,206,251 --out eval/answer_keys.json
  python eval/build_answer_keys.py --patients all --out eval/answer_keys.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import grad_cam_retfound as gc  # noqa: E402
import oct_zarr as oz  # noqa: E402

# Ground-truth sources live outside the repo tree (moved to GROUND_TRUTH_ROOT) so
# that, once the agent's file tools are confined to the repo's own working
# directory, there is nothing sensitive left for them to reach even if a tool
# were mistakenly re-enabled -- the sandbox no longer relies solely on
# permission scoping to keep these out of the agent's reach.
GROUND_TRUTH_ROOT = Path(os.environ.get("OLIVES_GROUND_TRUTH_DIR", r"D:\Nam\Olive_temp_move"))
BIOMARKER_CSV = GROUND_TRUTH_ROOT / "olives_original" / "Biomarker_Clinical_Data_Images.csv"
MANIFEST_CSV = GROUND_TRUTH_ROOT / "olives_split" / "olives_manifest.csv"
CHECKPOINT = REPO_ROOT / "output_dir" / "retfound_mae_OLIVES_lp" / "checkpoint-best.pth"
CLASS_NAMES = ["DME", "DR"]
INPUT_SIZE = 224
MARGIN_THRESHOLD = 0.6  # empirical cutoff already used by evidence-grounded-oct-assessment's driver.py
TOP_K = 3


def load_biomarker_index(patients: list[int]) -> pd.DataFrame:
    df = oz.load_biomarkers(BIOMARKER_CSV)
    df["filename"] = (
        df["Path (Trial/Arm/Folder/Visit/Eye/Image Name)"].str.strip("/").str.split("/").str[-1]
    )
    return df[df["Patient_ID"].isin(patients)]


def load_disease_labels(patients: list[int]) -> dict[int, str]:
    """Disease_Label/label_name in olives_manifest.csv is constant per patient
    for this 18-patient test cohort (verified empirically: each patient's
    rows carry exactly one label_name, matching its trial -- Prime_FULL
    patients are all "DR", TREX DME patients are all "DME"). So the ground
    truth "disease" for a patient is a single fixed value, not a per-visit
    vote."""
    df = pd.read_csv(MANIFEST_CSV)
    df = df[df["Patient_ID"].isin(patients)]
    per_patient = df.drop_duplicates("Patient_ID").set_index("Patient_ID")["label_name"]
    labels = per_patient.to_dict()
    missing = [p for p in patients if p not in labels]
    if missing:
        raise SystemExit(f"No disease label found in {MANIFEST_CSV} for patients: {missing}")
    return {int(k): v for k, v in labels.items()}


def biomarker_rows_for_visit(biomarker_df, patient_id, eye, source_folder, source_visit):
    return biomarker_df[
        (biomarker_df["Patient_ID"] == patient_id)
        & (biomarker_df["eye"] == eye)
        & (biomarker_df["folder_basename"] == source_folder)
        & (biomarker_df["visit"] == source_visit)
    ]


def build_biomarker_block(biomarker_df, patient_id, eye, source_folder, visit_info, bscan_source_files):
    """Returns {bscan_index: {"flags": {...}, "count": int, "positive": bool}}
    for the bscans covered by Biomarker_Clinical_Data_Images.csv at this
    visit, or {} if this visit isn't one of the two the CSV covers."""
    rows = biomarker_rows_for_visit(
        biomarker_df, patient_id, eye, source_folder, visit_info["source_visit"]
    )
    if rows.empty:
        return {}
    filename_to_index = {fn: i for i, fn in enumerate(bscan_source_files)}
    block = {}
    unmatched = []
    for _, row in rows.iterrows():
        idx = filename_to_index.get(row["filename"])
        if idx is None:
            unmatched.append(row["filename"])
            continue
        flags = {col: int(row[col]) if pd.notna(row[col]) else 0 for col in oz.BIOMARKER_COLUMNS}
        count = sum(flags.values())
        block[idx] = {"flags": flags, "count": count, "positive": count > 0}
    if unmatched:
        print(
            f"  [warn] patient={patient_id} eye={eye} visit={visit_info['visit_id']}: "
            f"{len(unmatched)} biomarker CSV rows didn't match any bscan_source_files entry "
            f"(e.g. {unmatched[:3]})",
            file=sys.stderr,
        )
    return block


def build_task12_fixture(model, patient_id, eye, latest_visit_id):
    """Reproduce screen_oct_volume's own logic offline (mean-prob volume
    call + top-k ranking) to label this patient/eye's latest visit
    "confident" or "uncertain", the same way evidence-grounded-oct-assessment's
    driver.py does (MARGIN_THRESHOLD = 0.6, its own stated empirical cutoff).
    Deterministic (no sampling), so this is a stable fixture.

    The fixture is decided by margin alone, not by cross-checking one slice
    outside the top-k: per-slice predictions naturally include a few outliers
    even in a highly confident volume (e.g. patient 60/OD's 92%-confident DR
    call still has ~2/49 slices individually predicting DME), so picking a
    single outside-top-k slice -- especially an adversarially "worst" one --
    would mislabel confident cases as uncertain. Margin matches the skill's
    own two worked examples exactly: patient 60/OD -> 0.845 (confident),
    patient 206/OD -> 0.495 (uncertain). A few slices *spread* across the
    volume are still recorded (not decision-driving) as reference context
    and for grading whether the agent checked beyond the initial top-k."""
    volume, resolved_visit_id = oz.get_oct_volume_array(patient_id, eye, latest_visit_id)
    n_bscans = volume.shape[0]
    images = [Image.fromarray(volume[i]) for i in range(n_bscans)]
    per_slice = gc.classify_batch(model, CLASS_NAMES, "cpu", images, INPUT_SIZE)

    mean_probs = {
        name: sum(r["probabilities"][name] for r in per_slice) / len(per_slice) for name in CLASS_NAMES
    }
    volume_prediction = max(mean_probs, key=mean_probs.get)
    other_class = [c for c in CLASS_NAMES if c != volume_prediction][0]
    margin = mean_probs[volume_prediction] - mean_probs[other_class]

    ranked = sorted(
        range(len(per_slice)),
        key=lambda i: per_slice[i]["probabilities"][volume_prediction],
        reverse=True,
    )
    top_k_indices = ranked[:TOP_K]

    # A handful of indices spread evenly across the volume, excluding the
    # top-k -- informational only (used by grade.py to check the agent
    # looked beyond the initial top-k), not used to decide the fixture.
    spread = sorted(set(int(round(x)) for x in np.linspace(0, n_bscans - 1, 5)) - set(top_k_indices))
    reference_checks = [
        {"index": i, "predicted_class": per_slice[i]["predicted_class"]} for i in spread
    ]

    fixture = "confident" if margin >= MARGIN_THRESHOLD else "uncertain"
    correct_assessment = volume_prediction if fixture == "confident" else "uncertain"

    return {
        "visit_id": resolved_visit_id,
        "volume_prediction": volume_prediction,
        "mean_probabilities": mean_probs,
        "margin": margin,
        "top_k_indices": top_k_indices,
        "reference_checks_outside_top_k": reference_checks,
        "fixture": fixture,
        "correct_assessment": correct_assessment,
    }


def build_patient(biomarker_df, disease_labels, model, patient_id: int) -> dict:
    info = oz.get_patient_info(patient_id)
    source_folder = info["source_folder"]
    entry = {
        "source_folder": source_folder,
        "disease_label": disease_labels[patient_id],
        "eyes": {},
    }
    for eye in oz.list_eyes(patient_id):
        visits = oz.list_visits(patient_id, eye)
        if not visits:
            continue
        visit_ids = [v["visit_id"] for v in visits]
        first_id, latest_id = visit_ids[0], visit_ids[-1]
        eye_entry = {"first_visit": first_id, "latest_visit": latest_id, "visits": {}}

        for v in visits:
            vid = v["visit_id"]
            vol_info = oz.get_oct_volume_info(patient_id, eye, vid)
            clinical, _ = oz.get_clinical(patient_id, eye, vid) if v["has_clinical"] else (None, vid)
            visit_group, _ = oz._get_visit_group(patient_id, eye, vid)
            bscan_source_files = list(visit_group["oct_image"].attrs.get("bscan_source_files", []))

            biomarkers = {}
            if vid in (first_id, latest_id):
                biomarkers = build_biomarker_block(
                    biomarker_df, patient_id, eye, source_folder, v, bscan_source_files
                )

            eye_entry["visits"][vid] = {
                "source_visit": v["source_visit"],
                "bcva": clinical["bcva"] if clinical else None,
                "cst_um": clinical["cst_um"] if clinical else None,
                "bscan_count": vol_info["bscan_count"],
                "biomarkers": biomarkers,  # {bscan_index: {"flags","count","positive"}}, {} if not covered
            }

        eye_entry["task12"] = build_task12_fixture(model, patient_id, eye, latest_id)
        entry["eyes"][eye] = eye_entry

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help='Comma-separated patient ids, or "all"')
    ap.add_argument("--out", default=str(REPO_ROOT / "eval" / "answer_keys.json"))
    args = ap.parse_args()

    all_ids = oz.list_patient_ids()
    if args.patients == "all":
        patients = all_ids
    else:
        patients = [int(p) for p in args.patients.split(",")]
        unknown = [p for p in patients if p not in all_ids]
        if unknown:
            raise SystemExit(f"Unknown patient ids (not in olives_zarr): {unknown}. Available: {all_ids}")

    print(f"Building answer keys for {len(patients)} patient(s): {patients}", file=sys.stderr)
    biomarker_df = load_biomarker_index(patients)
    disease_labels = load_disease_labels(patients)

    print(f"Loading RETFound classifier from {CHECKPOINT} ...", file=sys.stderr)
    model, n_classes = gc.load_classifier(str(CHECKPOINT), INPUT_SIZE, "cpu")

    answer = {}
    for pid in patients:
        print(f"  patient {pid} ...", file=sys.stderr)
        answer[str(pid)] = build_patient(biomarker_df, disease_labels, model, pid)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
