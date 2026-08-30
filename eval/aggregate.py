"""
Rolls up results/graded.csv into results/summary.md (the category/task table)
and results/summary.json (same data, machine-readable).

Usage (from repo root):
  python eval/aggregate.py --graded results/graded.csv --out-md results/summary.md --out-json results/summary.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Which metrics key represents "the" headline score for each task, for the
# summary table's Score column. Everything in `metrics` is still kept in
# summary.json for drill-down.
HEADLINE_METRIC = {
    1: "exact_match", 2: "exact_match", 3: "correct", 4: "evidence_precision_at_3",
    5: "disease_correct", 6: "joint_pass", 7: "joint_pass", 8: "joint_pass",
    9: "strict_workflow_success", 10: "correct", 11: "joint_pass",
    12: "uncertainty_handling_accuracy",
}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", default=str(REPO_ROOT / "results" / "graded.csv"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "results" / "summary.md"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "results" / "summary.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.graded, encoding="utf-8")))
    for r in rows:
        r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else {}
        r["harness_error"] = r["harness_error"] == "True"
        r["malformed_output"] = r["malformed_output"] == "True"

    by_task = defaultdict(list)
    for r in rows:
        by_task[int(r["task_no"])].append(r)

    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    task_summaries = []
    for task_no in sorted(by_task):
        task_rows = by_task[task_no]
        headline_key = HEADLINE_METRIC[task_no]
        headline_vals = [float(r["metrics"][headline_key]) for r in task_rows
                          if headline_key in r["metrics"] and not r["malformed_output"] and not r["harness_error"]]
        n_total = len(task_rows)
        n_scored = len(headline_vals)
        n_malformed = sum(1 for r in task_rows if r["malformed_output"])
        n_harness_error = sum(1 for r in task_rows if r["harness_error"])

        all_metric_keys = sorted({k for r in task_rows for k in r["metrics"].keys()})
        submetrics = {}
        for key in all_metric_keys:
            vals = [float(r["metrics"][key]) for r in task_rows
                     if key in r["metrics"] and not r["malformed_output"] and not r["harness_error"]]
            submetrics[key] = round(mean(vals), 3) if vals else None

        task_summaries.append({
            "task_no": task_no, "category": task_rows[0]["category"], "task_name": task_rows[0]["task_name"],
            "n_total": n_total, "n_scored": n_scored, "n_malformed_output": n_malformed,
            "n_harness_error": n_harness_error,
            "headline_metric": headline_key, "headline_score": round(mean(headline_vals), 3) if headline_vals else None,
            "submetrics": submetrics,
        })

    patient_summaries = []
    for pid, prows in sorted(by_patient.items()):
        n_malformed = sum(1 for r in prows if r["malformed_output"])
        n_harness_error = sum(1 for r in prows if r["harness_error"])
        patient_summaries.append({"patient_id": pid, "n_instances": len(prows),
                                   "n_malformed_output": n_malformed, "n_harness_error": n_harness_error})

    summary = {"n_instances": len(rows), "tasks": task_summaries, "patients": patient_summaries}
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    lines = [f"# Benchmark summary ({len(rows)} instances)\n"]
    current_category = None
    for t in task_summaries:
        if t["category"] != current_category:
            current_category = t["category"]
            lines.append(f"\n## {current_category}\n")
            lines.append("| Task | Score (N) | Malformed | Harness error | Sub-metrics |")
            lines.append("|---|---|---|---|---|")
        score_str = f"{t['headline_score']}" if t["headline_score"] is not None else "n/a"
        submetric_str = ", ".join(f"{k}={v}" for k, v in t["submetrics"].items() if k != t["headline_metric"])
        lines.append(
            f"| {t['task_name']} | {score_str} ({t['n_scored']}/{t['n_total']}) | "
            f"{t['n_malformed_output']} | {t['n_harness_error']} | {submetric_str} |"
        )

    lines.append("\n## Per-patient\n")
    lines.append("| Patient | N instances | Malformed output | Harness errors |")
    lines.append("|---|---|---|---|")
    for p in patient_summaries:
        lines.append(f"| {p['patient_id']} | {p['n_instances']} | {p['n_malformed_output']} | {p['n_harness_error']} |")

    lines.append("\n## Limitations\n")
    lines.append(
        "- Any run recorded before the MCP servers were changed to emit real image "
        "content blocks measured an agent that could not visually inspect anything: "
        "get_bscan/generate_gradcam used to return base64 image data inside a plain "
        "JSON TextContent block, so the model saw literal base64 text and never "
        "pixels. Evidence Precision@3 and \"abnormality\"/saliency metrics from such "
        "a run reflect reasoning from indirect text signals only (screen_oct_volume "
        "rankings, classify_bscan_retfound probabilities, generate_gradcam activation "
        "stats). get_bscan, get_fundus_image and generate_gradcam now return a "
        "metadata TextContent block plus a real ImageContent block (downscaled to 224 "
        "on the longest side by default), so runs after that change do have genuine "
        "visual grounding -- check a transcript's tool results before comparing "
        "scores across the two regimes."
    )
    lines.append(
        "- Evidence-precision tasks are restricted to each patient's first/last "
        "visit because that's the only ground-truth coverage "
        "Biomarker_Clinical_Data_Images.csv has."
    )
    lines.append(
        "- Task 12 (\"Uncertain-case evidence escalation\") measures single-model "
        "(volume-vs-slice) confidence/consistency, since no second classifier is "
        "deployed -- there is no \"disagreement between models\" framing in this task."
    )
    lines.append(
        "- \"Balanced accuracy\" for task 5 as a class-balanced metric only makes "
        "sense with enough instances to have both classes represented (the full "
        "18-patient run, not a small pilot) -- small-sample scores are directional only."
    )

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
