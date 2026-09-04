"""
MCP server exposing read-only access to the OLIVES OCT-Zarr stores
(olives_zarr/patient_<id>.zarr) so an LLM coding agent can navigate patient
exams, pull B-scans/OCT volumes/fundus images, and read permitted clinical
metadata (BCVA, CST) -- without hand-rolling zarr access.

This server deliberately never reads or exposes DR/DME ground-truth labels
or OLIVES biomarker annotations (see oct_zarr.py's reader API, which has no
function that touches the biomarkers/ group at all). Those remain
evaluator-only for the agent benchmark this supports.

Registration:
  Claude Code -- see .mcp.json at the repo root (auto-discovered). Use an
    absolute path to the Python interpreter that has `mcp`/`zarr` installed
    (e.g. a conda env's python.exe) -- a bare "python" resolves against
    whatever environment spawns the subprocess, which may not be the one
    you developed against.
  Codex CLI   -- add to ~/.codex/config.toml:
      [mcp_servers.retinal-data]
      command = "<absolute path to python.exe with mcp/zarr installed>"
      args = ["<absolute path to this file>"]

Config via environment variables (all optional):
  OCT_ZARR_DIR  default: <repo root>/olives_zarr
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO_ROOT, "data"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import CallToolResult, ImageContent, TextContent  # noqa: E402

import oct_zarr as oz  # noqa: E402

mcp = FastMCP("retinal-data")

# Upper bound on the `max_dimension` a caller may request, so a single call
# can't blow the client's tool-result size cap. 1024 keeps a full-resolution
# B-scan (496x504) and fundus photo (768x768) reachable while still refusing
# anything unbounded.
MAX_REQUESTABLE_DIMENSION = 1024


def _image_result(arr: np.ndarray, metadata: dict, max_dimension: int) -> CallToolResult:
    """Package a retrieved image as a real MCP result: one TextContent block
    holding the provenance JSON, then one ImageContent block holding the
    pixels.

    Returning the PNG as an ImageContent block -- rather than a base64 string
    inside a dict, which FastMCP serializes as literal text -- is what lets a
    multimodal client actually *see* the scan instead of reading ~150K
    characters of base64. The metadata block stays first so callers driving
    these servers with the `mcp` client SDK can keep reading
    `json.loads(result.content[0].text)` unchanged."""
    img = Image.fromarray(arr)
    display = oz.downscale_for_display(img, max_dimension)
    metadata = {
        **metadata,
        "image_width": display.width,
        "image_height": display.height,
        "source_width": img.width,
        "source_height": img.height,
        "downscaled": display.size != img.size,
    }
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps(metadata, indent=2)),
        ImageContent(type="image", data=oz.pil_to_base64_png(display), mimeType="image/png"),
    ])


def _error_result(message: str) -> CallToolResult:
    """Errors keep the historical `{"error": ...}` JSON-text shape (and stay
    isError=False) so the documented "treat an `error` key as unanswerable"
    rule reads identically whether a tool returns a dict or a CallToolResult."""
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": message}, indent=2))])


def _resolve_max_dimension(max_dimension: int | None) -> int:
    if max_dimension is None:
        return oz.MODEL_INPUT_SIZE
    if not (1 <= max_dimension <= MAX_REQUESTABLE_DIMENSION):
        raise oz.OctZarrError(f"max_dimension must be between 1 and {MAX_REQUESTABLE_DIMENSION}, "
                               f"got {max_dimension}")
    return max_dimension


@mcp.tool()
def list_patient_examinations(patient_id: int, eye: str) -> dict:
    """List a patient's visits for one eye, with which imaging modalities and
    clinical data are available at each. Use this to find the latest visit,
    check what's available before fetching, or plan a multi-visit comparison.

    Args:
        patient_id: Patient identifier (e.g. 60).
        eye: "OD" (right) or "OS" (left).

    Note: OLIVES has no real calendar dates -- visits are only labeled with
    relative study codes (e.g. "W0"/"W104" = weeks since baseline, "V1"/"V11"
    = visit number). Each visit's "source_visit" field carries that label;
    there is no "date" field since one doesn't exist in the source data.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        visits = oz.list_visits(patient_id, eye_n, None)
        return {"patient_id": patient_id, "eye": eye_n, "visits": visits}
    except oz.OctZarrError as e:
        return {"error": str(e)}


@mcp.tool()
def get_oct_volume_info(patient_id: int, eye: str, visit_id: str) -> dict:
    """Get an OCT volume's shape/dtype/location without loading any pixel
    data -- use this to validate a bscan_index or a start/end range before
    calling get_bscan or get_oct_volume.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").
    """
    try:
        return oz.get_oct_volume_info(patient_id, eye, visit_id)
    except oz.OctZarrError as e:
        return {"error": str(e)}


@mcp.tool()
def get_bscan(patient_id: int, eye: str, visit_id: str, bscan_index: int,
              max_dimension: int | None = None) -> CallToolResult:
    """Fetch a single OCT B-scan slice as a viewable image: a provenance JSON
    block followed by the scan itself as an MCP image, so you can actually
    look at the retina rather than read a base64 string.

    To classify or explain this slice, call the retinal-ai server's
    classify_bscan_retfound / generate_gradcam with the same
    patient_id/eye/visit_id/bscan_index -- they read the full-resolution scan
    from the Zarr store themselves, so there is no image to hand along.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").
        bscan_index: 0-based index into the volume (check get_oct_volume_info
            for the valid range first).
        max_dimension: Longest side of the returned image, in pixels.
            Defaults to 224 -- the classifier's own input resolution, and
            enough to read gross retinal structure. Raise it (up to 1024;
            504 is the untouched scan) only when a question needs finer
            detail; the payload grows roughly with its square.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        max_dim = _resolve_max_dimension(max_dimension)
        arr, source_file, resolved_visit_id = oz.get_bscan_array(patient_id, eye_n, visit_id, bscan_index)
        return _image_result(arr, {
            "bscan_index": bscan_index,
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            **oz.source_path_field(source_file),
        }, max_dim)
    except oz.OctZarrError as e:
        return _error_result(str(e))


@mcp.tool()
def get_oct_volume(patient_id: int, eye: str, visit_id: str,
                    start_bscan: int | None = None, end_bscan: int | None = None) -> dict:
    """Fetch a range of B-scans (or the whole volume) as a single base64-
    encoded .npy blob -- preserves Zarr's partial-read efficiency for large
    ranges. For a single visual slice, use get_bscan instead; this tool does
    NOT return per-slice images (up to 49 slices would multiply payload
    size ~50x for no benefit).

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").
        start_bscan: 0-based inclusive start index. Defaults to 0.
        end_bscan: 0-based exclusive end index. Defaults to the full volume.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        arr, resolved_visit_id = oz.get_oct_volume_array(patient_id, eye_n, visit_id, start_bscan, end_bscan)
        return {
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            "start_bscan": 0 if start_bscan is None else start_bscan,
            "end_bscan": arr.shape[0] if end_bscan is None else end_bscan,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "volume_base64": oz.array_to_base64_npy(arr),
            "encoding": "npy",
        }
    except oz.OctZarrError as e:
        return {"error": str(e)}


@mcp.tool()
def get_fundus_image(patient_id: int, eye: str, visit_id: str, modality: str = "color",
                     max_dimension: int | None = None) -> CallToolResult:
    """Fetch a fundus photo (not an OCT B-scan) as a viewable image: a
    provenance JSON block followed by the photo as an MCP image. For paired
    retrieval / multimodal display / modality association -- not for
    automated diagnosis (there is no fundus model on the retinal-ai server).

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").
        modality: "color" (true-color fundus photo, default) or "near-IR"
            (the near-infrared localizer image acquired alongside the OCT).
        max_dimension: Longest side of the returned image, in pixels.
            Defaults to 224. Raise it (up to 1024; 768 is the untouched
            photo) when a question needs finer detail; the payload grows
            roughly with its square.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        max_dim = _resolve_max_dimension(max_dimension)
        arr, source_file, resolved_visit_id = oz.get_fundus_array(patient_id, eye_n, visit_id, modality)
        return _image_result(arr, {
            "modality": modality,
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            **oz.source_path_field(source_file),
        }, max_dim)
    except oz.OctZarrError as e:
        return _error_result(str(e))


@mcp.tool()
def get_clinical_metadata(patient_id: int, eye: str, visit_id: str) -> dict:
    """Get permitted clinical variables (BCVA, CST) for one visit. Does NOT
    expose DR/DME labels or OLIVES biomarker annotations -- those are
    evaluator-only and never available through this tool.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").

    Note: each OLIVES patient has clinical data recorded for exactly one
    "study eye" -- calling this for the other eye returns an explanatory
    error, not zero/null values.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        clinical, resolved_visit_id = oz.get_clinical(patient_id, eye_n, visit_id)
        if clinical is None:
            return {"error": f"No clinical data recorded for patient {patient_id} eye {eye_n} "
                              f"visit {resolved_visit_id!r} -- BCVA/CST are only recorded for the study eye "
                              f"of each patient."}
        return {
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            "BCVA": clinical["bcva"],
            "CST": clinical["cst_um"],
            "eye_id": clinical["eye_id"],
        }
    except oz.OctZarrError as e:
        return {"error": str(e)}


@mcp.tool()
def compare_visits(patient_id: int, eye: str, visit_a: str, visit_b: str) -> dict:
    """Deterministically compute BCVA/CST deltas between two visits (visit_b
    minus visit_a), so the agent doesn't need to do this arithmetic itself.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_a: First visit id/label (baseline).
        visit_b: Second visit id/label (comparison).
    """
    try:
        eye_n = oz.normalize_eye(eye)
        return oz.compare_visits(patient_id, eye_n, visit_a, visit_b)
    except oz.OctZarrError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()  # stdio transport -- works with Claude Code, Codex CLI, Claude Desktop
