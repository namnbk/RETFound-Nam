"""
Visualize last-layer self-attention (CLS token -> patch tokens) of a
pretrained DINOv3 backbone -- base model, no fine-tuning -- loaded through
Hugging Face `transformers` (AutoImageProcessor / AutoModel) rather than
Meta's torch.hub repo, so weights download automatically.

Saves one PNG per attention head under tmp/<image_name>/head_<i>.png

Model sizing note: "vitl16" (~300M params) is the size comparable to
RETFound's ViT-Large backbones. "vith16plus" is a much larger ~840M variant.

DINOv3ViTModel supports output_attentions=True natively (returns a tuple of
per-layer (batch, num_heads, seq_len, seq_len) tensors, RoPE already
correctly applied) -- no manual attention-recomputation needed here, unlike
the torch.hub path in the models_vit.py Dinov3() function. attn_implementation
is forced to "eager" since fused/SDPA attention paths don't materialize
attention weights.

Basic command (from project root):
  python attention_map_dinov3.py --image_path ./some_image.png
  python attention_map_dinov3.py --image_path ./some_image.png --model_id facebook/dinov3-vith16plus-pretrain-lvd1689m
"""
import argparse
import os

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_model(model_id, device):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def get_last_layer_cls_attention(model, inputs):
    """Returns (num_heads, num_patches) attention from the CLS token to every
    patch token (register tokens excluded), in the last layer."""
    outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[-1]  # (B, num_heads, N, N)

    prefix = 1 + model.config.num_register_tokens  # CLS + register tokens
    return attn[0, :, 0, prefix:].cpu()  # CLS -> patch tokens only


def save_attention_maps(cls_attn, grid_size, out_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gh, gw = grid_size
    for h in range(cls_attn.shape[0]):
        amap = cls_attn[h].reshape(gh, gw).numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        colored = (cm.jet(amap)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).resize(out_size, resample=Image.BICUBIC) \
            .save(os.path.join(out_dir, f"head_{h}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                    help="e.g. facebook/dinov3-{vits16,vitb16,vitl16,vith16plus,vit7b16}-pretrain-lvd1689m")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_root", type=str, default="tmp")
    args = ap.parse_args()

    device = torch.device(args.device)
    processor, model = load_model(args.model_id, device)

    img = Image.open(args.image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    cls_attn = get_last_layer_cls_attention(model, inputs)

    patch_size = model.config.patch_size
    _, _, ph, pw = inputs.pixel_values.shape
    grid = (ph // patch_size, pw // patch_size)

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_dir = os.path.join(args.out_root, image_name)
    save_attention_maps(cls_attn, grid, img.size, out_dir)

    print(f"Saved {cls_attn.shape[0]} attention-head maps ({grid[0]}x{grid[1]} grid) to {out_dir}")


if __name__ == "__main__":
    main()
