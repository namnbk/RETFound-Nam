"""
Inference-only engine for the fine-tuned multi-label RETFound biomarker
classifier (see training/main_finetune.py's --multilabel path and
data/materialize_olives_biomarkers.py for how it was trained).

Mirrors interpretability/grad_cam_retfound.py's load_classifier /
classify_image / classify_batch shape -- same checkpoint format, same eval
transform, same "reuse an in-memory model across many calls" contract for a
long-lived caller like the retinal-ai MCP server -- but:
  - sigmoid per class instead of softmax: biomarkers co-occur (e.g. IRF and
    DRT/ME often together), they are not mutually exclusive like DME/DR, so
    each of the 6 gets its own independent probability (see
    training/util/datasets.py's MultiLabelImageDataset).
  - returns per-label probabilities + a threshold-derived "present" list,
    not a single predicted_class.
  - no Grad-CAM / saliency here, on purpose -- this model has no attribution
    tooling built for it. classify_bscan_biomarics / screen_oct_volume_biomarkers
    in mcp_servers/retinal_ai/server.py only classify.

The checkpoint's own `args` (nb_classes, global_pool, label_columns) are used
by default, so a caller normally only needs `finetune` -- label_columns only
needs overriding for a checkpoint trained before --label_columns existed.

Basic command (from project root):
  python inference/biomarker_classifier.py --image_path ./some_bscan.png \
      --finetune output_dir/retfound_mae_OLIVESBiomarkers_finetune/checkpoint-best.pth

`load_classifier()`, `classify_image()`, and `classify_batch()` are also
importable for programmatic use (e.g. by mcp_servers/retinal_ai/server.py).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import models_vit as models
from util.pos_embed import interpolate_pos_embed

DEFAULT_THRESHOLD = 0.5


def load_classifier(finetune, input_size=224, device="cpu", nb_classes=None,
                     global_pool=None, label_columns=None):
    checkpoint = torch.load(finetune, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]
    ckpt_args = checkpoint.get("args")

    if nb_classes is None:
        nb_classes = getattr(ckpt_args, "nb_classes", None) if ckpt_args is not None else None
        if nb_classes is None:
            raise ValueError("Checkpoint has no 'args.nb_classes'; pass --nb_classes explicitly.")
    if global_pool is None:
        global_pool = getattr(ckpt_args, "global_pool", True) if ckpt_args is not None else True
    if label_columns is None:
        raw = getattr(ckpt_args, "label_columns", None) if ckpt_args is not None else None
        if not raw:
            raise ValueError("Checkpoint has no 'args.label_columns'; pass label_columns explicitly.")
        label_columns = [c.strip() for c in raw.split(",") if c.strip()]
    if len(label_columns) != nb_classes:
        raise ValueError(f"label_columns has {len(label_columns)} names but checkpoint has "
                          f"{nb_classes} classes: {label_columns}")

    model = models.RETFound_mae(img_size=input_size, num_classes=nb_classes,
                                 drop_path_rate=0.0, global_pool=global_pool)

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded {finetune} (nb_classes={nb_classes}, global_pool={global_pool}, "
          f"label_columns={label_columns}). Missing: {missing} | Unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return model, label_columns


def build_eval_transform(input_size=224):
    # Identical to grad_cam_retfound.py's copy of the same transform (mirrors
    # util/datasets.py's build_transform for is_train != 'train') -- predictions
    # must match what the model saw during evaluation.
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_size = int(input_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def _to_result(probs_row, label_columns, threshold) -> dict:
    probabilities = {name: float(p) for name, p in zip(label_columns, probs_row.tolist())}
    present = [name for name, p in probabilities.items() if p >= threshold]
    return {"probabilities": probabilities, "present": present}


def classify_image(model, label_columns, device, image, input_size=224,
                    threshold=DEFAULT_THRESHOLD) -> dict:
    """Classify a single image with a plain no-grad forward pass. `image` is
    a PIL.Image. Returns {"probabilities": {name: float}, "present": [name, ...]}
    -- "present" is every label whose probability is >= threshold, independent
    of the others (unlike a single-label argmax)."""
    tfm = build_eval_transform(input_size)
    img_tensor = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor))[0].cpu()
    return _to_result(probs, label_columns, threshold)


def classify_batch(model, label_columns, device, images, input_size=224, batch_size=32,
                    threshold=DEFAULT_THRESHOLD) -> list:
    """Classify many images with chunked no-grad batched forward passes --
    one (or a few) forward pass instead of one per image. Used by
    screen_oct_volume_biomarkers in the retinal-ai MCP server to score a
    whole OCT volume (up to 49 B-scans) without 49 separate model calls.
    `images` is a list of PIL.Image; returns a list of
    {"probabilities","present"} dicts in the same order as `images`."""
    tfm = build_eval_transform(input_size)
    results = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        tensor = torch.stack([tfm(im.convert("RGB")) for im in chunk]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor)).cpu()
        for row in probs:
            results.append(_to_result(row, label_columns, threshold))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--finetune", type=str, required=True,
                     help="Path to a fine-tuned multi-label classifier checkpoint (.pth).")
    ap.add_argument("--label_columns", type=str, default=None,
                     help="Comma-separated label names, in prediction-index order. "
                          "Defaults to the checkpoint's own args.label_columns.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--nb_classes", type=int, default=None,
                     help="Override number of classes if the checkpoint lacks 'args'.")
    ap.add_argument("--global_pool", action="store_true", default=None,
                     help="Override global_pool if the checkpoint lacks 'args'.")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    label_columns = [c.strip() for c in args.label_columns.split(",")] if args.label_columns else None
    model, label_columns = load_classifier(
        args.finetune, args.input_size, args.device,
        nb_classes=args.nb_classes, global_pool=args.global_pool, label_columns=label_columns)

    image = Image.open(args.image_path)
    result = classify_image(model, label_columns, args.device, image, args.input_size, args.threshold)

    present = ", ".join(result["present"]) if result["present"] else "(none above threshold)"
    print(f"Present: {present}")
    print("Probabilities:")
    for name, p in result["probabilities"].items():
        print(f"  {name}: {p:.1%}")


if __name__ == "__main__":
    main()
