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

Read is enabled, but only behind explicit deny rules (DENY_READ globs
below), because plain Read is NOT confined to the working directory and the
answer key lives inside it. Both halves of that were re-verified live
against this CLI (2.1.251), not assumed:

  * Without a deny rule, a bypassPermissions session reads
    eval/tasks.jsonl -- which carries the hidden `expected`/`grading`
    fields -- and reads D:\Nam\Olive_temp_move\olives_split\splits.json,
    an absolute path outside the repo. Both succeeded. So Read alone makes
    the benchmark self-scoring.
  * With the deny rules, both reads come back
    "File is in a directory that is denied by your permission settings"
    with is_error set. Deny rules DO survive --permission-mode
    bypassPermissions.

Rule syntax is load-bearing and not obvious: `Read(D:/Nam/...)` (drive
letter, forward slashes) and `Read(//d/Nam/...)` both match, while
`Read(//D:/Nam/...)` and a bare `Read(**/Olive_temp_move/**)` silently do
NOT -- they look identical in the config and simply never fire. If you
change GROUND_TRUTH_DIR, re-run the deny check rather than trusting the
pattern by eye.

--restricted would confine file tools to the working directory, but it
ignores project-level settings entirely and silently drops both project
skills (evidence-grounded-oct-assessment, retinal-data-operations) from the
session -- unusable here, since skill-guided behavior is what's being
benchmarked. Hence deny rules rather than --restricted.

Note the deny rules do not (and cannot) close the source_path channel:
patients 206/209/211/220/251 have B-scan filenames like
TREX_DME_000024.tif, which the retinal-data MCP tools return as provenance.
That leak predates Read being enabled and is served by the tool surface
itself.

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
EFFORT = os.environ.get("EVAL_EFFORT", "low")

# Where the evaluator-only OLIVES originals/splits were moved to. Matches
# build_answer_keys.py's env var so both agree on one location.
GROUND_TRUTH_DIR = os.environ.get("OLIVES_GROUND_TRUTH_DIR", r"D:\Nam\Olive_temp_move")

# Everything the agent must not read while being scored. See the module
# docstring on why these are deny rules and why the syntax matters.
DENY_READ = [
    "Read(./eval/**)",      # tasks.jsonl's hidden expected/grading fields, answer_keys.json, grade.py
    "Read(./results/**)",   # other instances' transcripts and their scores
    f"Read({GROUND_TRUTH_DIR.replace(chr(92), '/')}/**)",  # OLIVES originals, splits, biomarker CSV
]


def run_one(instance: dict, out_dir: Path) -> dict:
    out_path = out_dir / f"{instance['instance_id']}.jsonl"
    cmd = [
        "claude", "-p", instance["prompt"],
        "--output-format", "stream-json",
        "--verbose",
        "--json-schema", json.dumps(instance["json_schema"]),
        "--tools", "Skill,Read",
        "--allowedTools", "Skill,Read,mcp__retinal-data__*,mcp__retinal-ai__*",
        "--disallowedTools", *DENY_READ,
        "--effort", EFFORT,
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
