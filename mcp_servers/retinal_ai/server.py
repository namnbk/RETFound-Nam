"""
MCP server exposing the fine-tuned RETFound OCT classifier and Grad-CAM
explainer to an LLM coding agent: classify a single B-scan, screen a whole
OCT volume for the most diagnostically relevant slices in one call (instead
of the agent making 49 individual classify calls), and generate a Grad-CAM
saliency map for a specific slice.

This server never returns OLIVES ground-truth labels, and generate_gradcam
never interprets its heatmap as a specific biomarker (e.g. "this is IRF") --
there is no model here establishing that relationship, only class-logit
attribution. Supersedes the old gradcam_mcp_server.py.

Note: the only usable trained classifier in this repo is a 2-class DME-vs-DR
RETFound linear probe (output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth)
-- there is no "normal" class. A DINO classifier was investigated and is not
usable (its checkpoints were never saved), so no classify_bscan_dino tool
exists here.

Registration:
  Claude Code -- see .mcp.json at the repo root (auto-discovered). Use an
    absolute path to the Python interpreter that has `mcp`/`torch`/`zarr`
    installed (e.g. a conda env's python.exe) -- a bare "python" resolves
    against whatever environment spawns the subprocess, which may not be the
    one you developed against.
  Codex CLI   -- add to ~/.codex/config.toml:
      [mcp_servers.retinal-ai]
      command = "<absolute path to python.exe with mcp/torch/zarr installed>"
      args = ["<absolute path to this file>"]

Config via environment variables (all optional):
  RETFOUND_CHECKPOINT   default: output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth
  RETFOUND_CLASS_NAMES  default: "DME,DR"
  RETFOUND_DEVICE       default: cuda if available, else cpu
  OCT_ZARR_DIR          default: <repo root>/olives_zarr
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO_ROOT, "interpretability"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import CallToolResult, ImageContent, TextContent  # noqa: E402

import grad_cam_retfound as gc  # noqa: E402
import oct_zarr as oz  # noqa: E402

DEFAULT_CHECKPOINT = os.environ.get(
    "RETFOUND_CHECKPOINT",
    os.path.join(REPO_ROOT, "output_dir", "retfound_mae_OLIVES_lp", "checkpoint-best.pth"))
DEFAULT_CLASS_NAMES = os.environ.get("RETFOUND_CLASS_NAMES", "DME,DR")
DEFAULT_DEVICE = os.environ.get("RETFOUND_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
MODEL_NAME = "retfound-olives-v1"

mcp = FastMCP("retinal-ai")

# Loading the ~300M-param ViT checkpoint is slow; this process is long-lived
# (one instance per client session), so cache the loaded model.
_model_cache = {}


def _get_model():
    key = (DEFAULT_CHECKPOINT, INPUT_SIZE, DEFAULT_DEVICE)
    if key not in _model_cache:
        _model_cache[key] = gc.load_classifier(DEFAULT_CHECKPOINT, INPUT_SIZE, DEFAULT_DEVICE)
    return _model_cache[key]


def _class_names():
    return [c.strip() for c in DEFAULT_CLASS_NAMES.split(",")]


def _image_result(img: Image.Image, metadata: dict) -> CallToolResult:
    """Package a rendered image as a real MCP result: a metadata TextContent
    block first, then the pixels as an ImageContent block.

    A base64 string inside a returned dict is serialized by FastMCP as
    literal text, which a multimodal client can only read, never see -- an
    ImageContent block is what makes a saliency map actually inspectable.
    Metadata stays in block 0 so SDK callers can keep reading
    `json.loads(result.content[0].text)`."""
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps({
            **metadata, "image_width": img.width, "image_height": img.height,
        }, indent=2)),
        ImageContent(type="image", data=oz.pil_to_base64_png(img), mimeType="image/png"),
    ])


def _error_result(message: str) -> CallToolResult:
    """Mirrors retinal-data's error shape: the historical `{"error": ...}`
    JSON text, isError left False, so "an `error` key means unanswerable"
    holds across both servers regardless of a tool's return type."""
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": message}, indent=2))])


@mcp.tool()
def classify_bscan_retfound(patient_id: int | None = None, eye: str | None = None,
                             visit_id: str = "latest", bscan_index: int | None = None,
                             image: str | None = None, bscan_id: int | None = None) -> dict:
    """Classify a single OCT B-scan with the fine-tuned RETFound classifier.

    Normally identify the slice by patient_id/eye/visit_id/bscan_index and
    this tool reads it from the Zarr store at full resolution -- do NOT try
    to forward pixels from get_bscan, which returns a downscaled MCP image
    block for viewing, not a base64 string to pipe. `image` remains available
    only for an image that exists nowhere in the store.

    Args:
        patient_id: Patient identifier. Required unless `image` is given.
        eye: "OD" or "OS". Required unless `image` is given.
        visit_id: An exact visit id ("visit_01"), "latest" (default),
            "first", or a raw source visit label (e.g. "W0", "V1").
        bscan_index: 0-based index into the volume. Required unless `image`
            is given.
        image: Escape hatch -- a base64-encoded PNG to classify directly,
            for an image that isn't in the Zarr store.
        bscan_id: Optional index to echo back in the response for
            bookkeeping (not used for classification itself). Defaults to
            `bscan_index` on the Zarr path.
    """
    try:
        model, n_classes = _get_model()
        class_names = _class_names()
        provenance = {}

        if image is not None:
            pil_img = oz.base64_png_to_pil(image)
        else:
            missing = [n for n, v in (("patient_id", patient_id), ("eye", eye),
                                       ("bscan_index", bscan_index)) if v is None]
            if missing:
                return {"error": f"classify_bscan_retfound needs {', '.join(missing)} to locate a "
                                  f"B-scan (or an `image` to classify directly)."}
            eye_n = oz.normalize_eye(eye)
            arr, source_file, resolved_visit_id = oz.get_bscan_array(patient_id, eye_n, visit_id, bscan_index)
            pil_img = Image.fromarray(arr)
            if bscan_id is None:
                bscan_id = bscan_index
            provenance = {"patient_id": patient_id, "eye": eye_n, "visit_id": resolved_visit_id,
                          "bscan_index": bscan_index, **oz.source_path_field(source_file)}

        result = gc.classify_image(model, class_names, DEFAULT_DEVICE, pil_img, INPUT_SIZE)
        return {
            "model": MODEL_NAME,
            "prediction": result["predicted_class"],
            "probabilities": result["probabilities"],
            "bscan_id": bscan_id,
            **provenance,
        }
    except oz.OctZarrError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"classify_bscan_retfound failed: {e}"}


@mcp.tool()
def screen_oct_volume(patient_id: int, eye: str, visit_id: str = "latest", top_k: int = 3) -> dict:
    """Screen an entire OCT volume in one call: classify every B-scan with a
    single batched forward pass, then rank slices by how strongly they
    support the volume-level prediction. Use this instead of calling
    classify_bscan_retfound 49 times.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest" (default),
            "first", or a raw source visit label (e.g. "W0", "V1").
        top_k: How many top-ranked B-scan indices to return.

    Note: the underlying model is a 2-class DME-vs-DR classifier with no
    "normal" class, so ranking reflects "which slices most support the
    predicted diagnosis" -- this is not a general anomaly/abnormality
    detector.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        volume, resolved_visit_id = oz.get_oct_volume_array(patient_id, eye_n, visit_id)
        model, n_classes = _get_model()
        class_names = _class_names()

        pil_images = [Image.fromarray(volume[i]) for i in range(volume.shape[0])]
        per_slice = gc.classify_batch(model, class_names, DEFAULT_DEVICE, pil_images, INPUT_SIZE)

        mean_probs = {name: sum(r["probabilities"][name] for r in per_slice) / len(per_slice)
                      for name in class_names}
        volume_prediction = max(mean_probs, key=mean_probs.get)

        ranked = sorted(range(len(per_slice)), key=lambda i: per_slice[i]["probabilities"][volume_prediction],
                         reverse=True)
        candidate_bscans = [
            {"index": i, f"{volume_prediction}_probability": per_slice[i]["probabilities"][volume_prediction]}
            for i in ranked[:top_k]
        ]

        return {
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            "volume_prediction": volume_prediction,
            "candidate_bscans": candidate_bscans,
        }
    except oz.OctZarrError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"screen_oct_volume failed: {e}"}


@mcp.tool()
def generate_gradcam(patient_id: int, eye: str, visit_id: str, bscan_index: int,
                      target_class: str | None = None) -> CallToolResult:
    """Generate a Grad-CAM saliency map explaining a classification decision
    for one B-scan. Returns a metadata JSON block (target class, activation
    statistics, provenance) followed by the heatmap overlay as a viewable MCP
    image -- so the saliency can actually be inspected against the retina it
    sits on, not just summarized numerically.

    The overlay is 224x224: the model's own eval view of the slice (short
    side resized to 256, center-cropped to 224), so the heatmap and the
    pixels underneath it are in exact correspondence. Note this is a
    center crop -- it is not the same framing as get_bscan's full-width
    downscale of the same slice.

    Reading it: the overlay says which regions moved the class logit, and
    nothing more. It does NOT identify a biomarker (e.g. "this is IRF") --
    no model here establishes that relationship, so do not narrate one.

    Args:
        patient_id: Patient identifier.
        eye: "OD" or "OS".
        visit_id: An exact visit id ("visit_01"), "latest", "first", or a
            raw source visit label (e.g. "W0", "V1").
        bscan_index: 0-based index into the volume.
        target_class: Class name to explain (e.g. "DME"). Defaults to the
            model's own predicted class.
    """
    try:
        eye_n = oz.normalize_eye(eye)
        arr, source_file, resolved_visit_id = oz.get_bscan_array(patient_id, eye_n, visit_id, bscan_index)
        model, n_classes = _get_model()
        class_names = _class_names()
        pil_img = Image.fromarray(arr)

        result = gc.run_gradcam(
            pil_img, class_names=class_names, target_class=target_class,
            input_size=INPUT_SIZE, device=DEFAULT_DEVICE, model=model, n_classes=n_classes,
        )
        return _image_result(result["overlay_image"], {
            "bscan_index": bscan_index,
            "patient_id": patient_id,
            "eye": eye_n,
            "visit_id": resolved_visit_id,
            **oz.source_path_field(source_file),
            "target_class": result["target_class"],
            "model": MODEL_NAME,
            "activation_max": result["activation_max"],
            "activation_mean": result["activation_mean"],
        })
    except oz.OctZarrError as e:
        return _error_result(str(e))
    except Exception as e:
        return _error_result(f"generate_gradcam failed: {e}")


if __name__ == "__main__":
    mcp.run()  # stdio transport -- works with Claude Code, Codex CLI, Claude Desktop
