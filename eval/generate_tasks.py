"""
Builds eval/tasks.jsonl: one row per (patient, task) instance, each with a
rendered prompt to send to the agent under test, the JSON Schema to pass via
`claude -p --json-schema`, and the hidden `expected` answer + any extra
`grading` metadata grade.py needs (tool-trace checks that aren't part of the
structured answer itself). `expected`/`grading` must never be sent to the
agent -- run_harness.py only ever sends `prompt`.

Usage (from repo root):
  python eval/generate_tasks.py --patients 60,206,251 --out eval/tasks.jsonl
  python eval/generate_tasks.py --patients all --out eval/tasks.jsonl
"""
import argparse
import json
from pathlib import Path

from schemas import TASK_META, TASK_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_BSCAN_INDEX = 25


def study_eye(patient_entry: dict) -> str:
    for eye, e in patient_entry["eyes"].items():
        if e["visits"][e["first_visit"]]["bcva"] is not None:
            return eye
    raise ValueError("No study eye (no visit with clinical data) found for this patient")


def clamp_bscan(visit_entry: dict) -> int:
    return min(FIXED_BSCAN_INDEX, visit_entry["bscan_count"] - 1)


def positive_bscans(visit_entry: dict, top_n: int | None = None) -> list[int]:
    items = [(int(idx), v["count"]) for idx, v in visit_entry["biomarkers"].items() if v["positive"]]
    items.sort(key=lambda t: -t[1])
    if top_n is not None:
        items = items[:top_n]
    return [idx for idx, _ in items]


def direction(delta: float, lower_is_better: bool) -> str:
    if delta == 0:
        return "unchanged"
    improved = (delta < 0) if lower_is_better else (delta > 0)
    return "improved" if improved else "worsened"


def build_instance(task_no, patient_id, eye, prompt, expected, grading=None):
    category, task_name = TASK_META[task_no]
    return {
        "instance_id": f"p{patient_id}_{eye}_t{task_no:02d}",
        "task_no": task_no,
        "category": category,
        "task_name": task_name,
        "patient_id": patient_id,
        "eye": eye,
        "prompt": prompt,
        "json_schema": TASK_SCHEMAS[task_no],
        "expected": expected,
        "grading": grading or {},
    }


def tasks_for_patient(patient_id: int, patient_entry: dict) -> list[dict]:
    eye = study_eye(patient_entry)
    eye_entry = patient_entry["eyes"][eye]
    first_id = eye_entry["first_visit"]
    latest_id = eye_entry["latest_visit"]
    first_v = eye_entry["visits"][first_id]
    latest_v = eye_entry["visits"][latest_id]
    first_source = first_v["source_visit"]
    disease = patient_entry["disease_label"]
    task12 = eye_entry["task12"]

    out = []

    # 1. Exact B-scan retrieval -- give the raw source-visit label (e.g. "W0"/"V1"),
    # not the canonical visit_NN, so the agent must actually resolve it.
    bscan_1 = clamp_bscan(first_v)
    out.append(build_instance(
        1, patient_id, eye,
        f"Retrieve B-scan {bscan_1} from patient {patient_id}, eye {eye}, visit {first_source}.",
        {"patient_id": patient_id, "eye": eye, "visit_id": first_id, "bscan_index": bscan_1},
    ))

    # 2. Latest-visit retrieval
    bscan_2 = clamp_bscan(latest_v)
    out.append(build_instance(
        2, patient_id, eye,
        f"Retrieve B-scan {bscan_2} from the latest available visit for patient {patient_id}, eye {eye}.",
        {"visit_id": latest_id, "bscan_index": bscan_2},
    ))

    # 3. Longitudinal retrieval (BCVA/CST at first and latest visit)
    out.append(build_instance(
        3, patient_id, eye,
        f"Retrieve the first and latest OCT examinations for patient {patient_id}, eye {eye}, "
        f"and report their BCVA and CST.",
        {
            "first_visit_id": first_id, "latest_visit_id": latest_id,
            "bcva_first": first_v["bcva"], "cst_first": first_v["cst_um"],
            "bcva_latest": latest_v["bcva"], "cst_latest": latest_v["cst_um"],
        },
    ))

    # 4. Find abnormal/evidence-rich slices (latest visit -- has biomarker ground truth)
    top3_latest = positive_bscans(latest_v, top_n=3)
    out.append(build_instance(
        4, patient_id, eye,
        f"Review the OCT volume for patient {patient_id}, eye {eye}, at the latest available visit, "
        f"and return the three B-scans that provide the strongest evidence of abnormality.",
        {"bscan_indices": top3_latest},
    ))

    # 5. DR vs DME assessment
    out.append(build_instance(
        5, patient_id, eye,
        f"For patient {patient_id}, eye {eye}, latest available visit, determine whether this OCT "
        f"examination is more consistent with DR or DME.",
        {"disease": disease},
    ))

    # 6. Disease + supporting slices
    out.append(build_instance(
        6, patient_id, eye,
        f"For patient {patient_id}, eye {eye}, latest available visit, determine whether this OCT "
        f"examination is more consistent with DR or DME, and return the three B-scans that most "
        f"strongly support your conclusion.",
        {"disease": disease, "bscan_indices": top3_latest},
    ))

    # 7. Disease + saliency evidence
    out.append(build_instance(
        7, patient_id, eye,
        f"For patient {patient_id}, eye {eye}, latest available visit, determine the likely disease "
        f"and return the single most relevant B-scan along with its saliency map.",
        {"disease": disease, "bscan_index": top3_latest[0] if top3_latest else None},
        grading={"requires_gradcam": True, "bscan_in_positive_set": top3_latest},
    ))

    # 8. Evidence-first assessment (tool-order check: evidence before disease)
    out.append(build_instance(
        8, patient_id, eye,
        f"Review the OCT volume for patient {patient_id}, eye {eye}, latest available visit. "
        f"First identify the most relevant abnormal B-scans, then determine the likely disease "
        f"based on that evidence.",
        {"bscan_indices": top3_latest, "disease": disease},
        grading={"ordering_check": "evidence_before_disease"},
    ))

    # 9. Compound: latest exam -> diagnosis -> evidence (Strict Workflow Success)
    out.append(build_instance(
        9, patient_id, eye,
        f"Find the latest OCT examination for patient {patient_id}, eye {eye}, determine whether it "
        f"is more consistent with DR or DME, and return the three most relevant supporting B-scans.",
        {"visit_id": latest_id, "disease": disease, "bscan_indices": top3_latest},
    ))

    # 10. Longitudinal comparison (direction of BCVA/CST change)
    bcva_delta = latest_v["bcva"] - first_v["bcva"]
    cst_delta = latest_v["cst_um"] - first_v["cst_um"]
    out.append(build_instance(
        10, patient_id, eye,
        f"Compare the first and latest OCT examinations for patient {patient_id}, eye {eye}, and "
        f"report whether BCVA and CST improved or worsened. (Higher BCVA means better vision; "
        f"lower CST means less macular edema.)",
        {
            "first_visit_id": first_id, "latest_visit_id": latest_id,
            "bcva_direction": direction(bcva_delta, lower_is_better=False),
            "cst_direction": direction(cst_delta, lower_is_better=True),
        },
    ))

    # 11. Evidence verification (evidence-before-answer ordering, like task 8)
    out.append(build_instance(
        11, patient_id, eye,
        f"Assess the latest OCT examination for patient {patient_id}, eye {eye}. Determine the "
        f"likely disease, and verify your conclusion using relevant B-scans as evidence before "
        f"giving your final answer.",
        {"disease": disease, "bscan_indices": top3_latest},
        grading={"ordering_check": "evidence_before_disease"},
    ))

    # 12. Uncertain-case evidence escalation
    out.append(build_instance(
        12, patient_id, eye,
        f"RETFound is uncertain on the latest OCT examination for patient {patient_id}, eye {eye}. "
        f"Inspect additional B-scans and supporting saliency evidence, then provide the "
        f"best-supported assessment or report that the case remains uncertain.",
        {"assessment": task12["correct_assessment"], "evidence_bscans": top3_latest},
        grading={
            "fixture": task12["fixture"],
            "top_k_indices": task12["top_k_indices"],
            "requires_gradcam": True,
            "requires_classify_outside_topk": True,
        },
    ))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help='Comma-separated patient ids, or "all"')
    ap.add_argument("--answer-keys", default=str(REPO_ROOT / "eval" / "answer_keys.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "eval" / "tasks.jsonl"))
    args = ap.parse_args()

    answer_keys = json.loads(Path(args.answer_keys).read_text())
    if args.patients == "all":
        patients = [int(p) for p in answer_keys.keys()]
    else:
        patients = [int(p) for p in args.patients.split(",")]
        missing = [p for p in patients if str(p) not in answer_keys]
        if missing:
            raise SystemExit(f"Patients {missing} not in {args.answer_keys}; run build_answer_keys.py first")

    rows = []
    for pid in patients:
        rows.extend(tasks_for_patient(pid, answer_keys[str(pid)]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} task instances ({len(patients)} patients x 12 tasks) to {out_path}")


if __name__ == "__main__":
    main()
