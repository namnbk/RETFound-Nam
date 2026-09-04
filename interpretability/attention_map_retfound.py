"""
Visualize last-layer self-attention (CLS token -> patch tokens) of the
pretrained RETFound_mae_natureOCT backbone -- base model, no fine-tuning.

Saves one PNG per attention head under tmp/<image_name>/head_<i>.png

Basic command (from project root):
  python attention_map.py --image_path ./some_image.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from matplotlib import cm
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import models_vit as models
from util.pos_embed import interpolate_pos_embed


def load_backbone(finetune="RETFound_mae_natureOCT", input_size=224, device="cpu"):
    # num_classes=0 -> no head; global_pool doesn't matter here, we only read
    # the last block's attention, never the pooled output.
    model = models.RETFound_mae(img_size=input_size, num_classes=0, drop_path_rate=0.0, global_pool=False)

    # A local path to a fine-tuned checkpoint takes precedence over downloading
    # a named pretrained backbone from the hub.
    if os.path.isfile(finetune):
        ckpt_path = finetune
    else:
        ckpt_path = hf_hub_download(repo_id=f"YukunZhou/{finetune}", filename=f"{finetune}.pth")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded {ckpt_path}. Missing: {missing} | Unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_last_layer_cls_attention(model, img_tensor, device):
    """Returns (num_heads, num_patches) attention from the CLS token to every
    patch token, in the last transformer block. Computed by hand from the
    qkv projection's raw output, so it doesn't depend on timm's fused-SDPA
    attention path (which never materializes attention weights)."""
    last_block = model.blocks[-1]
    num_heads = last_block.attn.num_heads
    captured = {}

    def hook(_module, _input, output):
        captured["qkv"] = output.detach()

    handle = last_block.attn.qkv.register_forward_hook(hook)
    model.forward_features(img_tensor.to(device))
    handle.remove()

    qkv = captured["qkv"]  # (B, N, 3*C)
    B, N, three_c = qkv.shape
    head_dim = (three_c // 3) // num_heads

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)  # (B, num_heads, N, N)

    return attn[0, :, 0, 1:].cpu()  # (num_heads, num_patches) -- CLS -> patches


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
    ap.add_argument("--finetune", type=str, default="RETFound_mae_natureOCT",
                     help="Either a named pretrained backbone to download from the "
                          "YukunZhou hub (e.g. RETFound_mae_natureOCT), or a local "
                          "path to a fine-tuned checkpoint .pth file.")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_root", type=str, default="tmp")
    args = ap.parse_args()

    device = torch.device(args.device)
    tfm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    img = Image.open(args.image_path).convert("RGB")
    img_tensor = tfm(img).unsqueeze(0)

    model = load_backbone(args.finetune, args.input_size, device)
    cls_attn = get_last_layer_cls_attention(model, img_tensor, device)

    patch_size = model.patch_embed.patch_size[0]
    grid = args.input_size // patch_size  # e.g. 224 / 16 = 14

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_dir = os.path.join(args.out_root, image_name)
    save_attention_maps(cls_attn, (grid, grid), img.size, out_dir)

    print(f"Saved {cls_attn.shape[0]} attention-head maps ({grid}x{grid} grid) to {out_dir}")


if __name__ == "__main__":
    main()
