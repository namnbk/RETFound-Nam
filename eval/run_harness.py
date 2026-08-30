r"""
Drives the `claude` CLI, one fresh headless session per task instance, and
saves the full stream-json transcript for grade.py to parse.

Sandboxing (see the plan's rationale): --tools "Skill" removes every other
built-in file/shell/web tool (Read, Write, Edit, Glob, Grep, Bash,
PowerShell, WebFetch, WebSearch, NotebookEdit, Agent) so there is
structurally no way to read olives_original/olives_split/imagefolder
(the evaluator-only ground truth, since moved out of the repo tree to
OLIVES_GROUND_TRUTH_DIR, as defense in depth -- but --tools "Skill" alone already made this unreachable regardless
of where those files live). --strict-mcp-config --mcp-config .mcp.json
pins the MCP surface to exactly retinal-data + retinal-ai (a default
session also picks up unrelated user-level MCP servers, e.g. Google Drive
-- confirmed during harness development). MAX_MCP_OUTPUT_TOKENS is raised
because get_oct_volume's base64 .npy blob is large enough to hit the CLI's
default tool-result size cap. It was originally raised for get_bscan/
generate_gradcam too, whose responses were then ~100-150K chars of base64
image text and errored out on every call; those tools now return a small
metadata block plus a real ImageContent block, so they no longer come
anywhere near the cap. The raised limit is kept for the volume tool and as
headroom.

Read was tried (allowing "Skill,Read") to let the agent retrieve its own
persisted large tool outputs, then reverted, and it stays reverted: (1)
plain Read is NOT confined to the working directory by default -- it
happily attempted an absolute path outside the repo
(D:\Nam\Olive_temp_move\...), only failing because that particular file
exceeded the 256KB single-read cap, not because of any directory boundary
-- a real leak surface if the agent ever learns/guesses where ground truth
was moved to. (2) --restricted does properly confine file tools to the
working directory (verified: hard permission_denied on the same absolute
path), but it also ignores project-level settings entirely, which silently
drops both project skills (evidence-grounded-oct-assessment,
retinal-data-operations) from the session -- unusable here since
skill-guided behavior is what's being benchmarked.

Visual grounding no longer depends on any of that. get_bscan,
get_fundus_image and generate_gradcam return a metadata TextContent block
followed by a real MCP ImageContent block (downscaled to 224 on the longest
side by default), so the agent sees actual pixels through the tool surface
itself. Runs recorded before that change did not, and their
"evidence"/"abnormality"/saliency scores measure text-only reasoning --
see results/summary.md's Limitations, and don't pool scores across the two
regimes without saying so.

Usage (from repo root):
  python eval/run_harness.py --tasks eval/tasks.jsonl --out-dir results/raw
  python eval/run_harness.py --tasks eval/tasks.jsonl --out-dir results/raw --only p60_OD_t01
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_CONFIG = REPO_ROOT / ".mcp.json"
TIMEOUT_SECONDS = 240
MAX_BUDGET_USD = "1.00"


def run_one(instance: dict, out_dir: Path) -> dict:
    out_path = out_dir / f"{instance['instance_id']}.jsonl"
    cmd = [
        "claude", "-p", instance["prompt"],
        "--output-format", "stream-json",
        "--verbose",
        "--json-schema", json.dumps(instance["json_schema"]),
        "--tools", "Skill",
        "--allowedTools", "Skill,mcp__retinal-data__*,mcp__retinal-ai__*",
        "--permission-mode", "bypassPermissions",
        "--strict-mcp-config",
        "--mcp-config", str(MCP_CONFIG),
        "--no-session-persistence",
        "--max-budget-usd", MAX_BUDGET_USD,
    ]
    env = dict(os.environ)
    env["MAX_MCP_OUTPUT_TOKENS"] = "200000"

    status = "ok"
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
        out_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            status = f"nonzero_exit_{proc.returncode}"
            (out_dir / f"{instance['instance_id']}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        out_path.write_text(e.stdout or "", encoding="utf-8")
    duration = time.time() - t0

    return {"instance_id": instance["instance_id"], "status": status, "duration_s": round(duration, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(REPO_ROOT / "eval" / "tasks.jsonl"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "raw"))
    ap.add_argument("--only", default=None, help="Comma-separated instance_ids to run (default: all)")
    ap.add_argument("--force", action="store_true", help="Re-run instances even if a transcript already exists")
    args = ap.parse_args()

    instances = [json.loads(line) for line in Path(args.tasks).open()]
    if args.only:
        wanted = set(args.only.split(","))
        instances = [i for i in instances if i["instance_id"] in wanted]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force:
        before = len(instances)
        instances = [i for i in instances
                     if not (out_dir / f"{i['instance_id']}.jsonl").exists()
                     or (out_dir / f"{i['instance_id']}.jsonl").stat().st_size == 0]
        skipped = before - len(instances)
        if skipped:
            print(f"Skipping {skipped} instance(s) with existing non-empty transcripts (use --force to re-run).", file=sys.stderr)

    results = []
    for i, instance in enumerate(instances, 1):
        print(f"[{i}/{len(instances)}] {instance['instance_id']} ({instance['task_name']}) ...", file=sys.stderr)
        r = run_one(instance, out_dir)
        print(f"  -> {r['status']} in {r['duration_s']}s", file=sys.stderr)
        results.append(r)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done: {n_ok}/{len(results)} completed without harness-level errors.", file=sys.stderr)
    (out_dir / "_run_log.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
