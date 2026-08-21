"""
Cross-dataset inference script.

Purpose:
  Use a Kermany 2017 fine-tuned RETFound model to run inference on OCTDL,
  mapping overlapping classes (DME, NORMAL/NO) and collapsing the rest into OTHER.

Basic command (from project root):
  python inference_cross_dataset.py
    (expects:
       checkpoint: output_dir/retfound_dinov2_OCT2017_finetune/checkpoint-best.pth
       octdl data: ./octdl/<CLASS>/*.png)

Explicit example:
  python inference_cross_dataset.py ^
    --checkpoint output_dir/retfound_dinov2_OCT2017_finetune/checkpoint-best.pth ^
    --octdl_path C:\data\octdl ^
    --out_csv octdl_preds.csv ^
    --save_json_metrics octdl_metrics.json ^
    --only_overlapping_report

If training head class order differs, override:
  python inference_cross_dataset.py --source_classes CNV,DME,DRUSEN,NORMAL

Outputs:
  - CSV (per-image): path, raw GT, mapped GT, raw prediction, mapped prediction, per-class probs
  - JSON: confusion matrix, per-class precision/recall/F1, overall & overlapping accuracies

To force CPU:
  python inference_cross_dataset.py --device cpu

"""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import csv
from collections import Counter
from pathlib import Path

# Local imports (assumes running from project root)
import models_vit as models  # same module used in training

# ---------------------------
# Label / mapping utilities
# ---------------------------
SOURCE_CLASSES_DEFAULT = ["CNV", "DME", "DRUSEN", "NORMAL"]  # Adjust if your training order differs.
MAPPED_ORDER = ["DME", "NORMAL", "OTHER"]  # Fixed order for confusion matrix / reporting.


def map_source_label(lbl: str) -> str:
    u = lbl.upper()
    if u == "DME":
        return "DME"
    if u == "NORMAL":
        return "NORMAL"
    return "OTHER"


def map_target_label(lbl: str) -> str:
    u = lbl.upper()
    if u == "DME":
        return "DME"
    if u in ("NORMAL", "NO"):
        return "NORMAL"
    return "OTHER"


# ---------------------------
# Dataset loader (simple)
# ---------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(root: Path) -> List[Tuple[Path, str]]:
    samples = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                samples.append((p, label))
    return samples


# ---------------------------
# Metrics
# ---------------------------
def confusion_matrix(preds: List[str], gts: List[str], labels: List[str]) -> torch.Tensor:
    idx = {l: i for i, l in enumerate(labels)}
    mat = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    for p, g in zip(preds, gts):
        if p not in idx or g not in idx:
            continue
        mat[idx[g], idx[p]] += 1
    return mat


def precision_recall_f1(cm: torch.Tensor, labels: List[str]) -> Dict[str, Dict[str, float]]:
    res = {}
    for i, lab in enumerate(labels):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        res[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": cm[i, :].sum().item()}
    return res


# ---------------------------
# Build model & load weights
# ---------------------------
def load_model(args, device):
    model = models.__dict__[args.model](
        num_classes=len(args.source_classes),
        drop_path_rate=0.0,
        args=args
    )
    # Attempt safer load first; fallback if any exception (e.g., UnpicklingError or missing support)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[Info] weights_only load failed ({e.__class__.__name__}: {e}). Falling back to full load.")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    return model


# ---------------------------
# Inference loop
# ---------------------------
@torch.no_grad()
def run_inference(model, samples, tfm, device, source_classes, batch_size=32):
    probs_all = []
    pred_idx_all = []
    for i in range(0, len(samples), batch_size):
        batch_paths = samples[i:i + batch_size]
        imgs = []
        for (p, _) in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"Warn: cannot open {p}: {e}")
                continue
            imgs.append(tfm(im))
        if not imgs:
            continue
        tensor = torch.stack(imgs).to(device)
        logits = model(tensor)
        prob = torch.softmax(logits, dim=-1)
        probs_all.append(prob.cpu())
        pred_idx_all.extend(prob.argmax(dim=-1).cpu().tolist())

    if probs_all:
        probs_all = torch.cat(probs_all, dim=0)
    else:
        probs_all = torch.empty(0, len(source_classes))
    preds_source = [source_classes[i] for i in pred_idx_all]
    return probs_all, preds_source


# ---------------------------
# Main
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser("Cross-dataset inference (Kermany model -> OCTDL)")
    ap.add_argument("--octdl_path", type=str, default="octdl/test", help="Root of OCTDL dataset (class subfolders).")
    ap.add_argument("--checkpoint", type=str,
                    default="output_dir/retfound_dinov2_OCT2017_finetune/checkpoint-best.pth")
    ap.add_argument("--model", type=str, default="RETFound_dinov2")
    ap.add_argument("--model_arch", type=str, default="retfound_dinov2")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source_classes", type=str,
                    default=",".join(SOURCE_CLASSES_DEFAULT),
                    help="Comma list; order MUST match training head order.")
    ap.add_argument("--out_csv", type=str, default="output_dir/refound_dinov2_finetune_OCT2017_to_OCTDL/octdl_inference_mapped.csv")
    ap.add_argument("--save_json_metrics", type=str, default="output_dir/refound_dinov2_finetune_OCT2017_to_OCTDL/octdl_inference_metrics.json")
    ap.add_argument("--only_overlapping_report", action="store_true",
                    help="If set, also print metrics restricted to overlapping classes (DME,NORMAL).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.source_classes = [c.strip() for c in args.source_classes.split(",")]
    device = torch.device(args.device)

    # NEW: ensure output directories exist
    def _ensure_output_dirs():
        for p in [args.out_csv, args.save_json_metrics]:
            parent = Path(p).parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
    _ensure_output_dirs()

    # Transforms (approximate training norm: ImageNet)
    tfm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # Collect OCTDL images
    samples = collect_images(Path(args.octdl_path))
    if not samples:
        print(f"No images found under {args.octdl_path}")
        return
    print(f"Collected {len(samples)} images from {args.octdl_path}")

    model = load_model(args, device)

    probs, preds_source = run_inference(
        model, samples, tfm, device, args.source_classes, batch_size=args.batch_size
    )

    # Map labels
    gt_raw = [lbl for (_, lbl) in samples]
    mapped_gt = [map_target_label(l) for l in gt_raw]
    mapped_pred = [map_source_label(p) for p in preds_source]

    # Metrics
    cm = confusion_matrix(mapped_pred, mapped_gt, MAPPED_ORDER)
    prf = precision_recall_f1(cm, MAPPED_ORDER)
    total = len(mapped_gt)
    overall_acc = sum(p == g for p, g in zip(mapped_pred, mapped_gt)) / total

    # Overlapping subset (exclude OTHER in GT)
    overlap_indices = [i for i, g in enumerate(mapped_gt) if g in ("DME", "NORMAL")]
    if overlap_indices:
        overlap_acc = sum(
            mapped_pred[i] == mapped_gt[i] for i in overlap_indices
        ) / len(overlap_indices)
    else:
        overlap_acc = float("nan")

    # Print summary
    print("\n=== Confusion Matrix (rows=GT, cols=Pred) order:", MAPPED_ORDER)
    print(cm)
    print("\nPer-class metrics:")
    for lab in MAPPED_ORDER:
        m = prf[lab]
        print(f"{lab}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"\nOverall mapped accuracy: {overall_acc:.3f}")
    print(f"Overlapping (DME,NORMAL) accuracy: {overlap_acc:.3f}")
    dist_pred = Counter(mapped_pred)
    print("Prediction distribution:", dict(dist_pred))

    # Save per-image CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_path", "gt_raw", "gt_mapped",
            "pred_source", "pred_mapped"
        ] + [f"prob_{c}" for c in args.source_classes])
        for (path, gt), ps, pm, gtm, prob in zip(
            samples, preds_source, mapped_pred, mapped_gt, probs
        ):
            w.writerow([
                str(path), gt, gtm, ps, pm
            ] + [f"{p:.6f}" for p in prob.tolist()])
    print(f"Saved per-image predictions: {args.out_csv}")

    # RE-ADD metrics_payload (was missing -> NameError)
    metrics_payload = {
        "confusion_matrix_order": MAPPED_ORDER,
        "confusion_matrix": cm.tolist(),
        "per_class": prf,
        "overall_mapped_accuracy": overall_acc,
        "overlap_accuracy_DME_NORMAL": overlap_acc,
        "source_classes": args.source_classes,
        "prediction_distribution": Counter(mapped_pred),
        "total_samples": total,
        "notes": "Model trained on Kermany 2017. Mapped overlapping classes (DME,NORMAL/NO); others -> OTHER."
    }

    # Save metrics JSON
    with open(args.save_json_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics JSON: {args.save_json_metrics}")

    if args.only_overlapping_report:
        # Filter rows where GT is overlapping
        print("\nRestricted report (GT in overlapping classes):")
        keep = [i for i, g in enumerate(mapped_gt) if g in ("DME", "NORMAL")]
        if keep:
            kept_preds = [mapped_pred[i] for i in keep]
            kept_gts = [mapped_gt[i] for i in keep]
            cm_overlap = confusion_matrix(kept_preds, kept_gts, ["DME", "NORMAL"])
            prf_overlap = precision_recall_f1(cm_overlap, ["DME", "NORMAL"])
            print("Confusion (overlap):\n", cm_overlap)
            for lab in ["DME", "NORMAL"]:
                m = prf_overlap[lab]
                print(f"{lab}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        else:
            print("No overlapping GT samples found.")

if __name__ == "__main__":
    main()
