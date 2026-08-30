---
name: retinal-data-operations
description: Retrieve, search, reconstruct, compare, or manipulate retinal OCT/fundus data from the OLIVES Zarr stores via the retinal-data (and, when interpretation is actually needed, retinal-ai) MCP servers. Use whenever a request involves reconstructed OCT volumes, individual B-scans, patient/eye/visit navigation, longitudinal exams, fundus images, or clinical metadata (BCVA/CST) -- looking up, pulling, comparing, or navigating retinal exam data.
---

This is the general data-handling skill for the OLIVES retinal dataset: it
governs how to use the `retinal-data` MCP server (patient/eye/visit
navigation, B-scans, OCT volumes, fundus images, clinical metadata) and,
sparingly, the `retinal-ai` MCP server (classification, volume screening,
Grad-CAM) that these servers expose. Both are already-built, already-running
MCP servers registered in `.mcp.json` at the repo root -- there is nothing to
build or launch. The handle for exercising them programmatically (for
verification, or as a reference implementation of the workflow below) is
`.claude/skills/retinal-data-operations/driver.py`, which drives them over
the real stdio MCP transport, the same way Claude Code does.

All paths below are relative to the repo root (`D:\Nam\RETFound-Nam`).

## Prerequisites

Nothing to install. The servers run in the `retfound` conda env
(`C:/Users/lamata/Miniconda3/envs/retfound/python.exe`, Python 3.11), which
already has `mcp==1.29.1`, `torch==2.5.1+cu121`, and `zarr==3.1.5` (pinned
below `numpy 2.0` -- see Gotchas). `.mcp.json` registers two servers:

```json
{
  "mcpServers": {
    "retinal-data": {"command": "C:/Users/lamata/Miniconda3/envs/retfound/python.exe",
                      "args": ["D:/Nam/RETFound-Nam/mcp_servers/retinal_data/server.py"]},
    "retinal-ai":   {"command": "C:/Users/lamata/Miniconda3/envs/retfound/python.exe",
                      "args": ["D:/Nam/RETFound-Nam/mcp_servers/retinal_ai/server.py"]}
  }
}
```

If these tools (`mcp__retinal-data__*` / `mcp__retinal-ai__*`) aren't showing
up as available, the current Claude Code session started before `.mcp.json`
was last edited -- restart the session; MCP servers are only loaded at
startup.

## Run (agent path)

**As Claude Code, you almost never invoke `driver.py` directly** -- once the
servers are connected you call `mcp__retinal-data__<tool>` /
`mcp__retinal-ai__<tool>` the normal way. `driver.py` exists for two
purposes: (1) it's runnable, verified evidence that the servers work end to
end over the real protocol, independent of whether this session's MCP
connection is live; (2) it's a literal, working implementation of the
resolve -> retrieve minimum -> verify -> preserve-provenance workflow below,
one section per rule -- read it, don't just read about it.

```bash
"C:/Users/lamata/Miniconda3/envs/retfound/python.exe" .claude/skills/retinal-data-operations/driver.py
# add --with-ai to also exercise retinal-ai (rule 7 demo; loads a ~1.2GB checkpoint, +4s)
# add --patient 251 --eye OD to point at a different patient
```

Verified output (this session, this container, both servers, full run):

```
======================================================================
RULE 1 -- resolve patient/eye/visit/modality before operating
======================================================================
patient 60 eye OD: 17 visits
  first visit: visit_01 (source=W0)
  latest visit: visit_17 (source=W104)
latest volume shape=[49, 496, 504] dtype=uint8 bscan_count=49

======================================================================
RULE 2 -- retrieve only what's necessary (one bscan, not the whole volume)
======================================================================
fetched bscan 24/48 as a real MCP image block: 1x image/png at 224x220 (downscaled from 504x496; not the full 49-slice volume)

======================================================================
RULE 3 -- longitudinal comparison stays within the same patient+eye
======================================================================
visit visit_01 -> visit_17: delta_BCVA=-4.0, delta_CST=-8.0

======================================================================
RULE 4 -- multimodal retrieval from the SAME patient/eye/visit
======================================================================
OCT bscan and fundus both resolved to visit_id='visit_17' / 'visit_17' (must match -- they do)

======================================================================
RULE 5/6 -- verify before answering, preserve provenance
======================================================================
bscan provenance: source_path='24.tif', bscan_index=24, visit_id=visit_17, eye=OD
fundus provenance: source_path='fundus_W104.tif', modality='color'

======================================================================
RULE 5 (continued) -- an unverifiable request returns an explicit error, not a guess
======================================================================
clinical metadata for the non-study eye (OS): {'error': "No clinical data recorded for patient 60 eye OS visit 'visit_01' -- BCVA/CST are only recorded for the study eye of each patient."}

======================================================================
RULE 8 -- ground-truth/biomarker data is structurally unreachable
======================================================================
get_clinical_metadata fields: ['patient_id', 'eye', 'visit_id', 'BCVA', 'CST', 'eye_id']
(no 'biomarkers', 'DR', or 'DME' field exists on any retinal-data tool response --
 the reader module has no function that reads the biomarkers/ zarr group at all)

======================================================================
RULE 7 -- diagnostic models are a SEPARATE, opt-in step (retinal-ai), never automatic
======================================================================
retinal-data alone (above) never touched retinal-ai. Only call these when the
user's request actually requires interpretation, e.g. "is this DME or DR?":
screen_oct_volume -> volume_prediction='DR', top candidates: [{'index': 45, 'DR_probability': 0.985}, {'index': 42, 'DR_probability': 0.981}, {'index': 44, 'DR_probability': 0.981}]

All steps completed against the real retinal-data and retinal-ai MCP server(s) over stdio.
```

## Core logic

```
Understand request
      |
Identify required data (patient / eye / visit / modality / scan)
      |
Find the minimum necessary source data
      |
Retrieve or process it
      |
Verify identity and provenance
      |
Return result
```

Concretely, with the actual tools:

1. **Resolve first.** Call `list_patient_examinations(patient_id, eye)`
   before touching anything else -- it tells you which visits exist, which
   modalities each has, and whether clinical data is present, all without
   loading a single pixel. For a specific volume, follow with
   `get_oct_volume_info(patient_id, eye, visit_id)` to get shape/dtype/
   bscan_count before fetching -- this validates a `bscan_index` or range up
   front instead of discovering it's out of bounds after a wasted call.
   `visit_id` accepts an exact `"visit_NN"`, `"latest"`, `"first"`, or a raw
   source label (`"W0"`, `"V1"`) -- resolve with whichever is natural for
   the request, but see Gotchas re: what to report back afterward.
2. **Retrieve the minimum.** One B-scan -> `get_bscan`. A range -> `get_oct_volume`
   with `start_bscan`/`end_bscan` (not the whole thing if only a few slices
   are needed). The whole volume -> `get_oct_volume` with no range. Never
   fetch a full volume just to look at one slice.
   `get_bscan`/`get_fundus_image` return a *viewable* MCP image block, sized
   by `max_dimension` (default 224, the classifier's own input resolution).
   Leave that default alone unless a question genuinely needs finer detail:
   224 costs ~70 image tokens, while the untouched 504x496 scan is ~215K
   base64 characters, and the extra pixels rarely change the answer.
3. **Longitudinal requests stay within one patient+eye.** `compare_visits(patient_id, eye, visit_a, visit_b)`
   computes BCVA/CST deltas deterministically -- don't do that arithmetic by
   hand, and don't compare visits across different eyes or patients (there is
   no tool that lets you do this by accident -- every tool takes a single
   `eye` argument).
4. **Multimodal retrieval must share patient/eye/visit.** When pairing OCT
   and fundus, call `get_bscan`/`get_oct_volume` and `get_fundus_image` with
   the *same* `patient_id`, `eye`, `visit_id` -- and check the resolved
   `visit_id` field in both responses actually matches (see Rule 4 in the
   verified output above).
5. **Verify before answering.** `get_clinical_metadata` for the non-study
   eye returns `{"error": "..."}`, not zeros or nulls -- treat any `"error"`
   key in a tool result as "this can't be answered as asked," not as data to
   paraphrase around.
6. **Preserve provenance in your answer.** Every image-returning tool
   includes `source_path` (the original filename) and the *resolved*
   `visit_id`/`bscan_index`/`eye`/`patient_id` -- carry these into your
   response so a human can trace exactly which file, which slice, which
   visit an answer came from.
7. **Diagnostic models are opt-in, not automatic.** `classify_bscan_retfound`,
   `screen_oct_volume`, and `generate_gradcam` live in the separate
   `retinal-ai` server. Calling any `retinal-data` tool never touches them.
   Only reach for `retinal-ai` when the user's request actually asks for
   interpretation ("what does this look like," "screen this volume," "is
   this DME") -- a request to just *see* or *compare* data doesn't need
   classification.
8. **Ground-truth is not reachable, and you don't need to route around it.**
   No tool on either server exposes DR/DME labels or OLIVES biomarker
   annotations -- there is no function anywhere in `oct_zarr.py`'s reader
   API that reads the `biomarkers/` Zarr group. If a request implies "tell
   me the correct diagnosis," the honest answer is what `retinal-ai`'s
   *predicted* class is (clearly labeled as a prediction), never a
   ground-truth lookup, because there is no path to one.

## Gotchas

- **A tool's `visit_id` output is the *resolved* canonical id, not whatever
  you passed in.** Pass `visit_id="latest"` to `get_bscan` and the response's
  `visit_id` field comes back as e.g. `"visit_17"`, not `"latest"` -- this
  was a real bug caught while building this skill's driver (early versions
  echoed the raw input alias, which silently broke provenance: `"latest"` is
  not a reproducible identifier). Always read the resolved id back out of
  the response rather than assuming your input alias round-trips.
- **Every tool puts its JSON in `content[0]`, and `structuredContent` is
  always `None`** -- but for two different reasons, so don't over-generalize
  from one to the other. Dict-returning tools (`list_patient_examinations`,
  `get_oct_volume_info`, `get_oct_volume`, `get_clinical_metadata`,
  `compare_visits`) get there via FastMCP, which serializes a plain dict as
  one pretty-printed `TextContent` block. The image tools (`get_bscan`,
  `get_fundus_image`, and retinal-ai's `generate_gradcam`) return an explicit
  `CallToolResult` whose blocks are ordered metadata-first, image-second,
  precisely so `json.loads(result.content[0].text)` keeps working. When
  driving these servers with the `mcp` client SDK, read `content[0].text`;
  never look for `structuredContent`.
- **An image tool's pixels are an `ImageContent` block, not a base64 field.**
  There is no `"image"` key in `get_bscan`/`get_fundus_image`/`generate_gradcam`
  metadata any more, and nothing to pipe from one tool into another: a
  multimodal client (Claude Code) renders the block as an image you can
  actually look at, and `classify_bscan_retfound`/`generate_gradcam` re-read
  the slice from Zarr at full resolution when given
  `patient_id`/`eye`/`visit_id`/`bscan_index`. Over the SDK the blocks are
  `[c for c in result.content if c.type == "image"]` (see `driver.py`'s
  `call_images()`).
- **Two files are both literally named `server.py`** (`mcp_servers/retinal_data/server.py`,
  `mcp_servers/retinal_ai/server.py`). Importing both with `sys.path.insert`
  + `import server` in the same Python process returns the *same* cached
  module the second time -- silently, no error. `driver.py` sidesteps this
  by spawning each as its own subprocess (which is also how they really run
  in production, so it's not a workaround so much as the correct thing to
  do). If you ever need both loaded in one process, use `importlib.util`
  with an explicit module name per file.
- **`zarr` in the `retfound` conda env is pinned to `==3.1.5`, not a `~=3.3`
  range.** `zarr>=3.2` requires Python >=3.12 (this env is 3.11), and even
  `zarr==3.1.6` silently requires `numpy>=2.0`, which breaks `opencv-python`
  4.9.0.80 (compiled against numpy 1.x -- `AttributeError: _ARRAY_API not
  found`). `3.1.5` is the newest release still compatible with the pinned
  `numpy~=1.26.4`. Don't `pip install --upgrade zarr` in this env without
  re-checking this.
- **Clinical data exists for exactly one eye per patient, never both** (the
  OLIVES "study eye"). This isn't a data gap to work around -- calling
  `get_clinical_metadata`/`compare_visits` for the other eye is expected to
  fail with a clear error, every time, for every patient.
- **Biomarker data doesn't currently exist in any of the 18 on-disk Zarr
  stores at all** (a separate, unrelated build-time detail), which makes
  Rule 8 trivially true today -- but the guarantee is structural (no reader
  function touches that group), not incidental, so it holds even if a future
  rebuild adds biomarkers back for some patients.

## Troubleshooting

- **`mcp__retinal-data__*` / `mcp__retinal-ai__*` tools not available**:
  `.mcp.json` was edited after this session started. Restart the session.
- **A tool call returns `{"error": "No Zarr store for patient_id=... "}`**:
  wrong patient id -- the error message lists every valid id currently on
  disk; use one of those.
- **`{"error": "bscan_index N out of range..."}`**: call
  `get_oct_volume_info` first to get `bscan_count`, then pick an index in
  `0..bscan_count-1`.
