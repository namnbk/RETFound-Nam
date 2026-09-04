"""
CLS-token feature extraction with a pretrained RETFound backbone
(RETFound_dinov2 or RETFound_mae). No fine-tuning, no classification head —
just: image -> transformer blocks -> CLS token -> 1024-dim vector.

Basic command (from project root):
  python extract_cls_features.py --data_path ./some_images --out_csv cls_features.csv \
    --model RETFound_mae --finetune RETFound_mae_natureOCT

  python extract_cls_features.py --data_path ./some_images --out_csv cls_features.csv \
    --model RETFound_dinov2 --finetune RETFound_dinov2_meh

The pretrained weights are pulled automatically from the matching
YukunZhou/<finetune> repo via huggingface_hub (requires being logged in
and having access approved on the model page).
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import models_vit as models
from util.pos_embed import interpolate_pos_embed

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_FINETUNE = {
    "RETFound_dinov2": "RETFound_dinov2_meh",
    "RETFound_mae": "RETFound_mae_natureOCT",
}


def load_backbone(model_name, finetune, input_size=224, device="cpu"):
    if model_name == "RETFound_dinov2":
        # num_classes=0 -> timm builds this with head = Identity(), no classifier at all
        model = models.RETFound_dinov2(args=None, num_classes=0, drop_path_rate=0.0)
        ckpt_key = "teacher"
    elif model_name == "RETFound_mae":
        # global_pool=False -> forward_features returns self.norm(x)[:, 0], i.e. the CLS
        # token itself (models_vit.py's custom VisionTransformer already does this split).
        model = models.RETFound_mae(
            img_size=input_size, num_classes=0, drop_path_rate=0.0, global_pool=False,
        )
        ckpt_key = "model"
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Expected RETFound_dinov2 or RETFound_mae.")

    ckpt_path = hf_hub_download(repo_id=f"YukunZhou/{finetune}", filename=f"{finetune}.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = checkpoint[ckpt_key]

    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

    interpolate_pos_embed(model, checkpoint_model)
    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded {finetune}. Missing: {missing} | Unexpected: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_cls_features(model, model_name, image_paths, tfm, device, batch_size=32):
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch]).to(device)
        out = model.forward_features(imgs)
        if model_name == "RETFound_dinov2":
            cls = out[:, 0, :]   # raw timm ViT: (B, num_patches+1, D) -> take CLS token
        else:
            cls = out            # custom MAE class already returns the pooled CLS vector: (B, D)
        feats.append(cls.cpu())
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="Folder of images (flat, no subfolders needed).")
    ap.add_argument("--model", type=str, default="RETFound_dinov2",
                    choices=["RETFound_dinov2", "RETFound_mae"])
    ap.add_argument("--finetune", type=str, default=None,
                    help="HF repo name under YukunZhou/. Defaults per --model if omitted.")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", type=str, default="cls_features.csv")
    args = ap.parse_args()
    finetune = args.finetune or DEFAULT_FINETUNE[args.model]

    device = torch.device(args.device)
    tfm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    image_paths = [p for p in sorted(Path(args.data_path).iterdir()) if p.suffix.lower() in IMG_EXT]
    if not image_paths:
        print(f"No images found under {args.data_path}")
        return
    print(f"Found {len(image_paths)} images.")

    model = load_backbone(args.model, finetune, args.input_size, device)
    feats = get_cls_features(model, args.model, image_paths, tfm, device, args.batch_size)
    print(f"Feature shape: {tuple(feats.shape)}")  # (num_images, 1024)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path"] + [f"feature_{i}" for i in range(feats.shape[1])])
        for p, vec in zip(image_paths, feats.tolist()):
            w.writerow([str(p)] + [f"{v:.6f}" for v in vec])
    print(f"Saved CLS features: {args.out_csv}")


if __name__ == "__main__":
    main()
