#!/usr/bin/env python3
"""Collect every output_dir/cross_domain_2cls/*/metrics.json into summary.csv and summary.md."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "output_dir/cross_domain_2cls")
KEYS = ["accuracy", "balanced_accuracy", "auc_roc", "sensitivity", "specificity",
        "ppv_precision", "npv", "f1", "kappa", "mcc", "average_precision"]

rows = []
for p in sorted(root.glob("*/metrics.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    rows.append(r)

order = {"finetune": 0, "lp": 1}
rows.sort(key=lambda r: (order.get(r["adaptation"], 9), r["train_data"], r["test_data"], r["test_name"]))


def fmt(r, k):
    v, (lo, hi) = r["metrics"][k], r["ci95_bootstrap"][k]
    if v != v:
        return "nan"
    return f"{v:.3f} [{lo:.3f}-{hi:.3f}]" if lo is not None else f"{v:.3f}"


with open(root / "summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["adaptation", "train_data", "test_name", "n_images", "n_dme", "n_normal", "loss"]
               + KEYS + [k + "_ci_lo" for k in KEYS] + [k + "_ci_hi" for k in KEYS])
    for r in rows:
        w.writerow([r["adaptation"], r["train_data"], r["test_name"], r["n_images"], r["n_dme"],
                    r["n_normal"], f"{r['loss']:.4f}"]
                   + [r["metrics"][k] for k in KEYS]
                   + [r["ci95_bootstrap"][k][0] for k in KEYS] + [r["ci95_bootstrap"][k][1] for k in KEYS])

with open(root / "summary.md", "w", encoding="utf-8") as f:
    f.write("| Adaptation | Train | Test set | n (DME/NORMAL) | " + " | ".join(KEYS) + " |\n")
    f.write("|" + "---|" * (4 + len(KEYS)) + "\n")
    for r in rows:
        f.write(f"| {r['adaptation']} | {r['train_data']} | {r['test_name']} | {r['n_images']} "
                f"({r['n_dme']}/{r['n_normal']}) | " + " | ".join(fmt(r, k) for k in KEYS) + " |\n")
print(f"{len(rows)} runs -> {root/'summary.csv'}, {root/'summary.md'}")
