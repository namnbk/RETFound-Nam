"""
MCP server exposing grad_cam_retfound.run_gradcam() as a tool so an LLM
coding agent (Claude Code, Codex CLI, etc.) can classify an OCT image and get
back a Grad-CAM heatmap it can actually see -- returned as an embedded
base64 image content block, not just a file path.

Registration:
  Claude Code -- see .mcp.json at the repo root (auto-discovered). Use an
    absolute path to the Python interpreter that has `mcp`/`torch` installed
    (e.g. a conda env's python.exe) -- a bare "python" resolves against
    whatever environment spawns the subprocess, which may not be the one
    you developed against.
  Codex CLI   -- add to ~/.codex/config.toml:
      [mcp_servers.retfound-gradcam]
      command = "<absolute path to python.exe with mcp/torch installed>"
      args = ["<absolute path to this file>"]

Config via environment variables (all optional):
  RETFOUND_CHECKPOINT   default: output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth
  RETFOUND_CLASS_NAMES  default: "DME,DR"
  RETFOUND_DEVICE       default: cuda if available, else cpu
"""
import io
import os

import torch
from mcp.server.fastmcp import FastMCP, Image

import grad_cam_retfound as gc

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.environ.get(
    "RETFOUND_CHECKPOINT",
    os.path.join(REPO_ROOT, "output_dir", "retfound_mae_OLIVES_lp", "checkpoint-best.pth"))
DEFAULT_CLASS_NAMES = os.environ.get("RETFOUND_CLASS_NAMES", "DME,DR")
DEFAULT_DEVICE = os.environ.get("RETFOUND_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

mcp = FastMCP("retfound-gradcam")

# Loading the ~300M-param ViT checkpoint is slow; this process is long-lived
# (one instance per client session), so cache loaded models across calls.
_model_cache = {}


def _get_model(finetune, input_size, device):
    key = (finetune, input_size, device)
    if key not in _model_cache:
        _model_cache[key] = gc.load_classifier(finetune, input_size, device)
    return _model_cache[key]


def _png_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool()
def explain_oct_prediction(
    image_path: str,
    finetune: str = DEFAULT_CHECKPOINT,
    class_names: str = DEFAULT_CLASS_NAMES,
    target_class: str | None = None,
    alpha: float = 0.45,
) -> list:
    """Classify a retinal OCT image with the fine-tuned RETFound ViT classifier
    and return the predicted diagnosis plus a Grad-CAM heatmap image showing
    which regions of the scan drove that prediction.

    Args:
        image_path: Path to the OCT image file to classify.
        finetune: Path to a fine-tuned checkpoint with a trained classifier head.
        class_names: Comma-separated class names in prediction-index order.
        target_class: Class index or name to explain. Defaults to the model's
            own predicted class; pass a different class to see what evidence
            would have supported that alternative diagnosis instead.
        alpha: Heatmap blend strength in the overlay image (0-1).
    """
    input_size = 224
    try:
        model, n_classes = _get_model(finetune, input_size, DEFAULT_DEVICE)
    except Exception as e:
        return [f"Failed to load checkpoint '{finetune}': {e}"]

    try:
        result = gc.run_gradcam(
            image_path, class_names=class_names, target_class=target_class,
            input_size=input_size, device=DEFAULT_DEVICE, alpha=alpha,
            model=model, n_classes=n_classes,
        )
    except Exception as e:
        return [f"Grad-CAM failed: {e}"]

    lines = [f"Predicted: {result['predicted_class']} "
             f"({result['probabilities'][result['predicted_class']]:.1%})"]
    if result["target_class"] != result["predicted_class"]:
        lines.append(f"Explaining: {result['target_class']} ({result['confidence']:.1%})")
    lines.append("Class probabilities:")
    for name, p in result["probabilities"].items():
        lines.append(f"  {name}: {p:.1%}")

    return ["\n".join(lines), Image(data=_png_bytes(result["overlay_image"]), format="png")]


if __name__ == "__main__":
    mcp.run()  # stdio transport -- works with Claude Code, Codex CLI, Claude Desktop
