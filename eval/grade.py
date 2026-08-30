"""
Parses each results/raw/<instance_id>.jsonl stream-json transcript and
scores it against eval/tasks.jsonl's hidden `expected`/`grading` fields, per
the metric defined for that task. Writes results/graded.csv.

Usage (from repo root):
  python eval/grade.py --tasks eval/tasks.jsonl --raw-dir results/raw --out results/graded.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_TOOLS = {
    "mcp__retinal-ai__screen_oct_volume",
    "mcp__retinal-ai__classify_bscan_retfound",
    "mcp__retinal-ai__generate_gradcam",
}
DISEASE_WORD_RE = re.compile(r"\b(DR|DME)\b")


def load_transcript(path: Path) -> dict:
    """Returns {"events": [...], "structured_output": dict|None, "final_text": str|None,
    "completed": bool} from a stream-json transcript. `events` is the ordered
    list of parsed JSON lines (excluding blank ones)."""
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    structured_output = None
    final_text = None
    completed = False
    for e in events:
        if e.get("type") == "result":
            completed = e.get("subtype") == "success"
            structured_output = e.get("structured_output")
            final_text = e.get("result")
    return {"events": events, "structured_output": structured_output, "final_text": final_text, "completed": completed}


def tool_calls(events: list) -> list:
    """Ordered list of {"seq", "name", "input", "result_text", "is_error"}.
    `seq` is the position in `events` so callers can compare ordering.
    A tool_result counts as an error if the block's own is_error flag is set,
    OR its content starts with "Error:" -- the CLI's own tool-result size cap
    surfaces as an "Error: ... exceeds maximum allowed tokens" string without
    necessarily setting is_error, so both must be checked. (That cap used to
    fire on get_bscan/generate_gradcam's base64 image payloads; those now
    return small metadata plus an ImageContent block, but get_oct_volume's
    .npy blob can still trip it, and old transcripts still contain the
    string.)"""
    calls = []
    pending = {}  # tool_use_id -> index into calls
    for seq, e in enumerate(events):
        if e.get("type") == "assistant":
            for block in e.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    calls.append({
                        "seq": seq, "id": block.get("id"), "name": block.get("name"),
                        "input": block.get("input", {}), "result_text": None, "is_error": None,
                    })
                    pending[block.get("id")] = len(calls) - 1
        elif e.get("type") == "user":
            for block in e.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    idx = pending.get(block.get("tool_use_id"))
                    if idx is None:
                        continue
                    content = block.get("content")
                    text = content if isinstance(content, str) else json.dumps(content)
                    is_error = bool(block.get("is_error")) or text.strip().startswith("Error:")
                    calls[idx]["result_text"] = text
                    calls[idx]["is_error"] = is_error
                    calls[idx]["result_seq"] = seq
    return calls


def assistant_text_blocks(events: list) -> list:
    """Ordered list of (seq, text) for plain assistant text blocks (not thinking, not tool_use)."""
    out = []
    for seq, e in enumerate(events):
        if e.get("type") == "assistant":
            for block in e.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    out.append((seq, block["text"]))
    return out


def evidence_before_disease_ok(events: list) -> bool:
    calls = tool_calls(events)
    evidence_seqs = [c["result_seq"] for c in calls
                     if c["name"] in EVIDENCE_TOOLS and not c["is_error"] and "result_seq" in c]
    if not evidence_seqs:
        return False
    first_evidence = min(evidence_seqs)

    disease_mention_seqs = [seq for seq, text in assistant_text_blocks(events) if DISEASE_WORD_RE.search(text)]
    if not disease_mention_seqs:
        return True  # disease only ever stated in the structured final answer -- fine
    return first_evidence <= min(disease_mention_seqs)


def num_close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def precision_at_3(predicted, truth) -> float:
    if not isinstance(predicted, list):
        return 0.0
    top3 = predicted[:3]
    return len(set(top3) & set(truth)) / 3.0


def grade_instance(instance: dict, transcript: dict) -> dict:
    row = {
        "instance_id": instance["instance_id"], "task_no": instance["task_no"],
        "category": instance["category"], "task_name": instance["task_name"],
        "patient_id": instance["patient_id"], "eye": instance["eye"],
    }
    so = transcript["structured_output"]
    row["harness_error"] = not transcript["completed"]
    row["malformed_output"] = so is None
    row["actual"] = json.dumps(so) if so is not None else ""
    row["expected"] = json.dumps(instance["expected"])

    metrics = {}
    task_no = instance["task_no"]
    expected = instance["expected"]
    grading = instance.get("grading", {})
    events = transcript["events"]
    calls = tool_calls(events)

    if so is None:
        row["metrics"] = json.dumps(metrics)
        return row

    if task_no == 1:
        metrics["exact_match"] = all(so.get(k) == expected[k] for k in
                                      ("patient_id", "eye", "visit_id", "bscan_index"))
    elif task_no == 2:
        metrics["exact_match"] = so.get("visit_id") == expected["visit_id"] and \
            so.get("bscan_index") == expected["bscan_index"]
    elif task_no == 3:
        visits_ok = so.get("first_visit_id") == expected["first_visit_id"] and \
            so.get("latest_visit_id") == expected["latest_visit_id"]
        values_ok = all(num_close(so.get(k), expected[k]) for k in
                         ("bcva_first", "cst_first", "bcva_latest", "cst_latest"))
        metrics["visits_correct"] = visits_ok
        metrics["values_correct"] = values_ok
        metrics["correct"] = visits_ok and values_ok
    elif task_no == 4:
        metrics["evidence_precision_at_3"] = precision_at_3(so.get("bscan_indices"), expected["bscan_indices"])
    elif task_no == 5:
        metrics["disease_correct"] = so.get("disease") == expected["disease"]
    elif task_no == 6:
        disease_ok = so.get("disease") == expected["disease"]
        prec = precision_at_3(so.get("bscan_indices"), expected["bscan_indices"])
        metrics["disease_correct"] = disease_ok
        metrics["evidence_precision_at_3"] = prec
        metrics["joint_pass"] = disease_ok and prec == 1.0
    elif task_no == 7:
        disease_ok = so.get("disease") == expected["disease"]
        bscan_ok = so.get("bscan_index") in grading.get("bscan_in_positive_set", [])
        gradcam_ok = any(c["name"] == "mcp__retinal-ai__generate_gradcam" and not c["is_error"] for c in calls)
        metrics["disease_correct"] = disease_ok
        metrics["evidence_bscan_positive"] = bscan_ok
        metrics["gradcam_generated"] = gradcam_ok
        metrics["joint_pass"] = disease_ok and bscan_ok and gradcam_ok
    elif task_no == 8:
        disease_ok = so.get("disease") == expected["disease"]
        prec = precision_at_3(so.get("bscan_indices"), expected["bscan_indices"])
        order_ok = evidence_before_disease_ok(events)
        metrics["disease_correct"] = disease_ok
        metrics["evidence_precision_at_3"] = prec
        metrics["order_ok"] = order_ok
        metrics["joint_pass"] = disease_ok and prec == 1.0 and order_ok
    elif task_no == 9:
        visit_ok = so.get("visit_id") == expected["visit_id"]
        disease_ok = so.get("disease") == expected["disease"]
        evidence_ok = len(set(so.get("bscan_indices", [])) & set(expected["bscan_indices"])) >= 1
        metrics["visit_correct"] = visit_ok
        metrics["disease_correct"] = disease_ok
        metrics["has_valid_evidence"] = evidence_ok
        metrics["strict_workflow_success"] = visit_ok and disease_ok and evidence_ok
    elif task_no == 10:
        visits_ok = so.get("first_visit_id") == expected["first_visit_id"] and \
            so.get("latest_visit_id") == expected["latest_visit_id"]
        bcva_ok = so.get("bcva_direction") == expected["bcva_direction"]
        cst_ok = so.get("cst_direction") == expected["cst_direction"]
        metrics["visits_correct"] = visits_ok
        metrics["bcva_direction_correct"] = bcva_ok
        metrics["cst_direction_correct"] = cst_ok
        metrics["correct"] = visits_ok and bcva_ok and cst_ok
    elif task_no == 11:
        disease_ok = so.get("disease") == expected["disease"]
        prec = precision_at_3(so.get("bscan_indices"), expected["bscan_indices"])
        order_ok = evidence_before_disease_ok(events)
        metrics["disease_correct"] = disease_ok
        metrics["evidence_precision_at_3"] = prec
        metrics["order_ok"] = order_ok
        metrics["joint_pass"] = disease_ok and prec == 1.0 and order_ok
    elif task_no == 12:
        assessment_ok = so.get("assessment") == expected["assessment"]
        prec = precision_at_3(so.get("evidence_bscans"), expected["evidence_bscans"])
        top_k = set(grading.get("top_k_indices", []))
        inspected = set()
        for c in calls:
            if c["name"] == "mcp__retinal-data__get_bscan" and "bscan_index" in c["input"]:
                inspected.add(c["input"]["bscan_index"])
            if c["name"] == "mcp__retinal-ai__classify_bscan_retfound" and c["input"].get("bscan_id") is not None:
                inspected.add(c["input"]["bscan_id"])
            if c["name"] == "mcp__retinal-ai__generate_gradcam" and "bscan_index" in c["input"]:
                inspected.add(c["input"]["bscan_index"])
        checked_beyond_topk = bool(inspected - top_k)
        gradcam_ok = any(c["name"] == "mcp__retinal-ai__generate_gradcam" and not c["is_error"] for c in calls)
        metrics["uncertainty_handling_accuracy"] = assessment_ok
        metrics["evidence_precision_at_3"] = prec
        metrics["checked_beyond_topk"] = checked_beyond_topk
        metrics["gradcam_generated"] = gradcam_ok
        metrics["appropriate_tool_use"] = checked_beyond_topk and gradcam_ok
        if grading.get("fixture") == "uncertain":
            metrics["correct_abstention"] = so.get("assessment") == "uncertain"

    row["metrics"] = json.dumps(metrics)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(REPO_ROOT / "eval" / "tasks.jsonl"))
    ap.add_argument("--raw-dir", default=str(REPO_ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "graded.csv"))
    args = ap.parse_args()

    instances = [json.loads(line) for line in Path(args.tasks).open()]
    raw_dir = Path(args.raw_dir)

    rows = []
    for instance in instances:
        transcript_path = raw_dir / f"{instance['instance_id']}.jsonl"
        transcript = load_transcript(transcript_path)
        rows.append(grade_instance(instance, transcript))

    fieldnames = ["instance_id", "task_no", "category", "task_name", "patient_id", "eye",
                  "harness_error", "malformed_output", "metrics", "actual", "expected"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} graded rows to {out_path}")


if __name__ == "__main__":
    main()
