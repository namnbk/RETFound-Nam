# filepath: bench_infer.py
import argparse
import os
import time
import warnings
import csv

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import models_vit as models
from util.datasets import build_dataset

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser("Single-image inference latency benchmark", add_help=True)

    # Model
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--model_arch", type=str, default="retfound_dinov2")
    p.add_argument("--nb_classes", type=int, required=True)
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--global_pool", action="store_true")

    # Data
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--task", type=str, default="benchmark")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_mem", action="store_true", default=True)

    # Checkpoint
    p.add_argument("--resume", type=str, required=True, help="Path to checkpoint-best.pth")

    # Device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    # Output
    p.add_argument("--output_dir", type=str, default="./output_dir")

    # Dataset/transforms defaults (to satisfy build_dataset)
    p.add_argument("--color_jitter", type=float, default=None)
    p.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--reprob", type=float, default=0.25)
    p.add_argument("--remode", type=str, default="pixel")
    p.add_argument("--recount", type=int, default=1)
    p.add_argument("--resplit", action="store_true", default=False)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    p.add_argument("--mixup_prob", type=float, default=1.0)
    p.add_argument("--mixup_switch_prob", type=float, default=0.5)
    p.add_argument("--mixup_mode", type=str, default="batch")
    p.add_argument("--norm", type=str, default="IMAGENET")
    p.add_argument("--enhance", action="store_true", default=False)
    p.add_argument("--dataratio", type=str, default="1.0")
    p.add_argument("--stratified", action="store_true", default=False)
    p.add_argument("--datasets_seed", type=int, default=2026)

    # Benchmark
    p.add_argument("--bench_warmup", type=int, default=20)
    p.add_argument("--bench_iters", type=int, default=100)

    return p.parse_args()


def build_model(args):
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    return model


def _benchmark_inference(model, dataset_test, device, pin_mem=True, warmup=20, iters=100, num_workers=0):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    # Warmup
    with torch.no_grad():
        seen = 0
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            with torch.cuda.amp.autocast():
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seen += 1
            if seen >= warmup:
                break

    # Measure
    times = []
    with torch.no_grad():
        done = 0
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast():
                _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            done += 1
            if done >= iters:
                break

    times = np.array(times, dtype=np.float64)
    if times.size == 0:
        print("No samples available for benchmarking.")
        return None
    avg = times.mean()
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    thr = 1.0 / avg
    print(f"Latency (single image): avg={avg*1000:.2f} ms | p50={p50*1000:.2f} ms | p95={p95*1000:.2f} ms | throughput={thr:.2f} img/s")
    return avg, p50, p95, thr


def main():
    args = parse_args()

    # Device and seeds
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Build model and load checkpoint
    model = build_model(args)
    ckpt = torch.load(args.resume, map_location="cpu")
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    _ = model.load_state_dict(state, strict=False)
    model.to(device)

    # Dataset (test split)
    dataset_test = build_dataset(is_train="test", args=args)

    # Run benchmark
    result = _benchmark_inference(
        model,
        dataset_test,
        device,
        pin_mem=args.pin_mem,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
        num_workers=args.num_workers,
    )

    # Write CSV
    out_dir = os.path.join(args.output_dir, "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"latency_{args.task}.csv")
    if result is not None:
        avg, p50, p95, thr = result
        header = [
            "timestamp", "model", "model_arch", "task", "nb_classes", "input_size", "device",
            "warmup", "iters", "avg_ms", "p50_ms", "p95_ms", "throughput_img_s"
        ]
        row = [
            time.strftime("%Y-%m-%d %H:%M:%S"), args.model, args.model_arch, args.task,
            args.nb_classes, args.input_size, args.device,
            args.bench_warmup, args.bench_iters,
            round(avg * 1000, 4), round(p50 * 1000, 4), round(p95 * 1000, 4), round(thr, 4)
        ]
        write_header = not os.path.isfile(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        print(f"Saved benchmark CSV: {out_csv}")


if __name__ == "__main__":
    main()
