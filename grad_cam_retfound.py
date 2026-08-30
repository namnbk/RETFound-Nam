"""
Grad-CAM saliency map for a fine-tuned RETFound_mae ViT classifier.

Unlike attention_map_retfound.py (which visualizes raw backbone attention),
this attributes a specific classification decision to image regions: it
backprops the predicted (or chosen) class logit to the output of the last
transformer block -- the ViT analog of "the last conv feature map before
global-average-pool + FC" in classic Grad-CAM -- and weights each channel by
its average gradient.

Requires a fine-tuned checkpoint with a trained classification head (e.g.
output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth), not the bare
pretrained backbone.

Basic command (from project root):
  python grad_cam_retfound.py --image_path ./some_image.png \
      --finetune output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth

`run_gradcam()`, `classify_image()`, and `classify_batch()` are also
importable for programmatic use (e.g. by mcp_servers/retinal_ai/server.py)
-- run_gradcam() returns the prediction and both PIL images in-memory,
optionally also saving them to disk.
"""
import argparse
import os

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models_vit as models
from util.pos_embed import interpolate_pos_embed


def load_classifier(finetune, input_size=224, device="cpu", nb_classes=None, global_pool=None):
    checkpoint = torch.load(finetune, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]
    ckpt_args = checkpoint.get("args")

    if nb_classes is None:
        nb_classes = getattr(ckpt_args, "nb_classes", None) if ckpt_args is not None else None
        if nb_classes is None:
            raise ValueError("Checkpoint has no 'args.nb_classes'; pass --nb_classes explicitly.")
    if global_pool is None:
        global_pool = getattr(ckpt_args, "global_pool", True) if ckpt_args is not None else True

    model = models.RETFound_mae(img_size=input_size, num_classes=nb_classes,
                                 drop_path_rate=0.0, global_pool=global_pool)

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded {finetune} (nb_classes={nb_classes}, global_pool={global_pool}). "
          f"Missing: {missing} | Unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return model, nb_classes


def build_eval_transform(input_size=224):
    # Mirrors util/datasets.py's build_transform for is_train != 'train', so
    # predictions match what the model actually saw during evaluation.
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_size = int(input_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def unnormalize_to_uint8(img_tensor):
    """img_tensor: (3, H, W) normalized. Returns (H, W, 3) uint8 RGB array."""
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def compute_gradcam(model, img_tensor, device, target_class=None):
    """Returns (cam over patch tokens, target_class index, predicted_class
    index, class probabilities, raw_max). target_class is the class explained
    (the argmax prediction unless the caller overrides it); predicted_class is
    always the argmax, regardless of any override. raw_max is the
    pre-normalization peak activation (cam is itself normalized to [0,1], so
    its own max is trivially ~1.0 -- raw_max is the only meaningful "how
    strong was the peak evidence" signal)."""
    activation = {}

    def fwd_hook(_module, _input, output):
        output.retain_grad()
        activation["value"] = output

    handle = model.blocks[-1].register_forward_hook(fwd_hook)
    logits = model(img_tensor.to(device))  # no torch.no_grad() -- we need gradients
    handle.remove()

    probs = logits.softmax(dim=-1)[0]
    predicted_class = int(probs.argmax().item())
    if target_class is None:
        target_class = predicted_class

    model.zero_grad()
    logits[0, target_class].backward()

    act = activation["value"]
    grad = act.grad[0, 1:, :]          # (num_patches, C) -- drop CLS token
    acts = act.detach()[0, 1:, :]      # (num_patches, C)

    weights = grad.mean(dim=0)                              # (C,)
    cam = torch.relu((weights * acts).sum(dim=-1))           # (num_patches,)
    raw_max = float(cam.max().item())
    cam = cam / (raw_max + 1e-8)

    return cam.detach().cpu(), target_class, predicted_class, probs.detach().cpu(), raw_max


def render_gradcam(cam, grid_size, base_img_uint8, alpha, out_dir=None):
    """Returns (heatmap_image, overlay_image) as PIL Images. Also saves them
    to out_dir/heatmap.png and out_dir/overlay.png when out_dir is given."""
    gh, gw = grid_size
    out_size = (base_img_uint8.shape[1], base_img_uint8.shape[0])  # (W, H) for PIL resize

    amap = cam.reshape(gh, gw).numpy()
    heatmap_img = Image.fromarray((cm.jet(amap)[:, :, :3] * 255).astype(np.uint8)) \
        .resize(out_size, resample=Image.BICUBIC)

    heatmap_arr = np.array(heatmap_img).astype(np.float32)
    base_arr = base_img_uint8.astype(np.float32)
    overlay_img = Image.fromarray((alpha * heatmap_arr + (1 - alpha) * base_arr).astype(np.uint8))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        heatmap_img.save(os.path.join(out_dir, "heatmap.png"))
        overlay_img.save(os.path.join(out_dir, "overlay.png"))

    return heatmap_img, overlay_img


def classify_image(model, class_names, device, image, input_size=224) -> dict:
    """Classify a single image with a plain no-grad forward pass -- no
    backward pass, no Grad-CAM rendering. Fast path for callers that only
    need a prediction (e.g. classify_bscan_retfound in the retinal-ai MCP
    server). `image` is a PIL.Image."""
    tfm = build_eval_transform(input_size)
    img_tensor = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(img_tensor).softmax(dim=-1)[0].cpu()
    predicted_idx = int(probs.argmax().item())
    return {
        "predicted_class": class_names[predicted_idx],
        "probabilities": {name: float(p) for name, p in zip(class_names, probs.tolist())},
    }


def classify_batch(model, class_names, device, images, input_size=224, batch_size=32) -> list:
    """Classify many images with chunked no-grad batched forward passes --
    one (or a few) forward pass instead of one per image. Used by
    screen_oct_volume in the retinal-ai MCP server to score a whole OCT
    volume (up to 49 B-scans) without 49 separate model calls. `images` is a
    list of PIL.Image; returns a list of {"predicted_class","probabilities"}
    dicts in the same order as `images`."""
    tfm = build_eval_transform(input_size)
    results = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        tensor = torch.stack([tfm(im.convert("RGB")) for im in chunk]).to(device)
        with torch.no_grad():
            probs = model(tensor).softmax(dim=-1).cpu()
        for row in probs:
            predicted_idx = int(row.argmax().item())
            results.append({
                "predicted_class": class_names[predicted_idx],
                "probabilities": {name: float(p) for name, p in zip(class_names, row.tolist())},
            })
    return results


def resolve_target_class(target_class_arg, class_names):
    if target_class_arg is None:
        return None
    if isinstance(target_class_arg, int):
        return target_class_arg
    if target_class_arg.isdigit():
        return int(target_class_arg)
    if target_class_arg in class_names:
        return class_names.index(target_class_arg)
    raise ValueError(f"--target_class '{target_class_arg}' is not a valid index or one of {class_names}")


def run_gradcam(image_path, finetune=None, class_names="DME,DR", target_class=None,
                 input_size=224, device=None, alpha=0.45, out_dir=None,
                 nb_classes=None, global_pool=None, model=None, n_classes=None):
    """End-to-end Grad-CAM: load the classifier (or reuse a pre-loaded one),
    run one image through it, and return the prediction plus both rendered
    images in-memory. Optionally also saves heatmap.png/overlay.png to
    out_dir. Used by both the CLI (main(), below) and gradcam_mcp_server.py.

    Pass `model`/`n_classes` (e.g. from a previous load_classifier() call) to
    skip re-loading the checkpoint from disk -- useful for a long-lived
    caller that serves many requests, like the MCP server.

    `image_path` may also be an in-memory PIL.Image (e.g. a B-scan fetched
    from a Zarr store rather than a file on disk)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if isinstance(class_names, str):
        class_names = [c.strip() for c in class_names.split(",")]

    if model is None:
        if finetune is None:
            raise ValueError("Either 'finetune' or a pre-loaded 'model' must be provided.")
        model, n_classes = load_classifier(finetune, input_size, device,
                                            nb_classes=nb_classes, global_pool=global_pool)
    if len(class_names) != n_classes:
        raise ValueError(f"class_names has {len(class_names)} names but checkpoint has "
                          f"{n_classes} classes: {class_names}")

    tfm = build_eval_transform(input_size)
    img = image_path.convert("RGB") if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
    img_tensor = tfm(img).unsqueeze(0)

    target_idx = resolve_target_class(target_class, class_names)
    cam, target_idx, predicted_idx, probs, raw_max = compute_gradcam(model, img_tensor, device, target_idx)

    patch_size = model.patch_embed.patch_size[0]
    grid = input_size // patch_size

    base_img_uint8 = unnormalize_to_uint8(img_tensor[0])
    heatmap_img, overlay_img = render_gradcam(cam, (grid, grid), base_img_uint8, alpha, out_dir)

    probabilities = {name: float(p) for name, p in zip(class_names, probs.tolist())}
    return {
        "predicted_class": class_names[predicted_idx],
        "target_class": class_names[target_idx],
        "confidence": probabilities[class_names[target_idx]],
        "probabilities": probabilities,
        "overlay_image": overlay_img,
        "heatmap_image": heatmap_img,
        "activation_max": raw_max,
        "activation_mean": float(cam.mean().item()),
        "out_dir": out_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--finetune", type=str, required=True,
                     help="Path to a fine-tuned classifier checkpoint (.pth) with a trained head.")
    ap.add_argument("--class_names", type=str, default="DME,DR",
                     help="Comma-separated class names in prediction-index order.")
    ap.add_argument("--target_class", type=str, default=None,
                     help="Class index or class name to explain. Defaults to the predicted class.")
    ap.add_argument("--nb_classes", type=int, default=None,
                     help="Override number of classes if the checkpoint lacks 'args'.")
    ap.add_argument("--global_pool", action="store_true", default=None,
                     help="Override global_pool if the checkpoint lacks 'args'.")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_root", type=str, default="tmp_gradcam")
    ap.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend weight in the overlay.")
    args = ap.parse_args()

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_dir = os.path.join(args.out_root, image_name)

    result = run_gradcam(args.image_path, args.finetune, args.class_names, args.target_class,
                          args.input_size, args.device, args.alpha, out_dir,
                          args.nb_classes, args.global_pool)

    lines = [f"Predicted: {result['predicted_class']} ({result['probabilities'][result['predicted_class']]:.1%})"]
    if result["target_class"] != result["predicted_class"]:
        lines.append(f"Explaining: {result['target_class']} ({result['confidence']:.1%})")
    lines.append("Class probabilities:")
    for name, p in result["probabilities"].items():
        lines.append(f"  {name}: {p:.1%}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "prediction.txt"), "w") as f:
        f.write(report + "\n")

    print(f"Saved heatmap.png / overlay.png / prediction.txt to {out_dir}")


if __name__ == "__main__":
    main()
