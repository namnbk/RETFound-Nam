#!/usr/bin/env python3
"""
Evaluate a trained (fine-tuned or linear-probe) RETFound checkpoint on one or more
splits of an ImageFolder dataset, and record a full set of binary metrics.

Designed for the 2-class (DME vs NORMAL) in-domain / cross-domain experiments:
  - Class index 0 = DME (positive / "disease"), index 1 = NORMAL. Both datasets use
    sorted folder names DME, NORMAL, so ImageFolder already gives this order; the
    script asserts it.
  - Preprocessing is the SAME eval transform used during training/validation
    (util.datasets.build_transform: bicubic resize -> center crop -> ImageNet norm).
  - Several splits can be pooled (e.g. --splits train val test = "whole dataset").

Metrics (with 95% bootstrap CI): accuracy, balanced accuracy, AUC-ROC, average
precision, sensitivity (recall of DME), specificity, PPV, NPV, F1, Cohen's kappa,
MCC, plus confusion matrix and cross-entropy loss. Per-image probabilities are saved
so thresholds / extra analyses can be done later without re-running inference.

Example:
  python cross_domain_eval.py \
      --checkpoint output_dir/retfound_dinov2_oct2017_2cls_finetune/checkpoint-best.pth \
      --data_path ./octdl --splits train val test --test_name octdl_all
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

import models_vit as models
from util.datasets import build_transform

CLASSES = ["DME", "NORMAL"]  # index 0 = DME (positive), index 1 = NORMAL


def parse_args():
    ap = argparse.ArgumentParser("Cross-domain evaluation (binary DME vs NORMAL)")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint-best.pth")
    ap.add_argument("--data_path", required=True, help="Dataset root with <split>/<class>/ images")
    ap.add_argument("--splits", nargs="+", default=["test"], help="Splits to pool, e.g. test or train val test")
    ap.add_argument("--test_name", default=None, help="Label for the evaluation set (default: <data dir>_<splits>)")
    ap.add_argument("--out_root", default="output_dir/cross_domain_2cls")
    ap.add_argument("--run_name", default=None, help="Output folder name (default: <train task>__on__<test_name>)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for 95%% CIs")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]  # training-time args: model, input_size, nb_classes, task, ...
    if targs.model == "RETFound_mae":
        model = models.__dict__[targs.model](
            img_size=targs.input_size, num_classes=targs.nb_classes,
            drop_path_rate=0.0, global_pool=targs.global_pool)
    else:
        model = models.__dict__[targs.model](
            num_classes=targs.nb_classes, drop_path_rate=0.0, args=targs)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, targs, ckpt.get("epoch", -1)


def build_eval_set(data_path, splits, targs):
    # Identical eval preprocessing as training-time validation/test.
    tf = build_transform("test", targs)
    parts, split_of = [], []
    for s in splits:
        ds = datasets.ImageFolder(os.path.join(data_path, s), transform=tf)
        assert ds.classes == CLASSES, f"{s}: classes {ds.classes} != {CLASSES}"
        parts.append(ds)
        split_of += [s] * len(ds)
    paths = [p for ds in parts for p, _ in ds.samples]
    return ConcatDataset(parts), paths, split_of


@torch.no_grad()
def infer(model, dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(args.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
            out = model(x)
        logits_all.append(out.float().cpu())
        y_all.append(y)
    return torch.cat(logits_all), torch.cat(y_all).numpy()


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def compute_metrics(y, p_dme, pred):
    """y, pred: class indices (0 = DME = positive); p_dme: P(DME)."""
    pos, ppos = (y == 0), (pred == 0)
    tp = int((pos & ppos).sum()); fn = int((pos & ~ppos).sum())
    fp = int((~pos & ppos).sum()); tn = int((~pos & ~ppos).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    ppv, npv = _div(tp, tp + fp), _div(tn, tn + fn)
    both = pos.any() and (~pos).any()
    return {
        "accuracy": _div(tp + tn, len(y)),
        "balanced_accuracy": np.nanmean([sens, spec]),
        "auc_roc": float(roc_auc_score(pos, p_dme)) if both else float("nan"),
        "average_precision": float(average_precision_score(pos, p_dme)) if pos.any() else float("nan"),
        "sensitivity": sens,
        "specificity": spec,
        "ppv_precision": ppv,
        "npv": npv,
        "f1": _div(2 * ppv * sens, ppv + sens) if not np.isnan(ppv + sens) else float("nan"),
        "kappa": float(cohen_kappa_score(y, pred)) if both else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if both else float("nan"),
    }, [[tp, fn], [fp, tn]]


def bootstrap_ci(y, p_dme, pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    n, rows = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rows.append(compute_metrics(y[idx], p_dme[idx], pred[idx])[0])
    ci = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        ci[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [None, None]
    return ci


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, targs, epoch = load_model(args.checkpoint, args.device)

    test_name = args.test_name or f"{Path(args.data_path).name}_{'+'.join(args.splits)}"
    run_name = args.run_name or f"{targs.task}__on__{test_name}"
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset, paths, split_of = build_eval_set(args.data_path, args.splits, targs)
    print(f"[{run_name}] {len(dataset)} images from {args.data_path} splits={args.splits}")
    logits, y = infer(model, dataset, args)

    prob = F.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    p_dme = prob[:, 0]
    loss = float(F.cross_entropy(logits, torch.from_numpy(y)).item())

    metrics, cm = compute_metrics(y, p_dme, pred)
    ci = bootstrap_ci(y, p_dme, pred, args.n_boot, args.seed)

    result = {
        "run_name": run_name,
        "train_task": targs.task,
        "adaptation": getattr(targs, "adaptation", None),
        "train_data": os.path.basename(os.path.normpath(targs.data_path)),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(epoch),
        "test_data": os.path.basename(os.path.normpath(args.data_path)),
        "test_splits": args.splits,
        "test_name": test_name,
        "input_size": targs.input_size,
        "n_images": int(len(y)),
        "n_dme": int((y == 0).sum()),
        "n_normal": int((y == 1).sum()),
        "positive_class": "DME",
        "threshold": 0.5,
        "loss": loss,
        "metrics": metrics,
        "ci95_bootstrap": ci,
        "n_boot": args.n_boot,
        "confusion_matrix": {"order": "rows=true [DME, NORMAL], cols=pred [DME, NORMAL]", "matrix": cm},
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "split", "true", "pred", "prob_DME", "prob_NORMAL"])
        for p, s, t, pr, pb in zip(paths, split_of, y, pred, prob):
            w.writerow([p, s, CLASSES[t], CLASSES[pr], f"{pb[0]:.6f}", f"{pb[1]:.6f}"])

    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=1))
    print(f"confusion [[TP,FN],[FP,TN]] = {cm}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
