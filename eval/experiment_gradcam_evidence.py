"""
One-off research script (not part of the eval harness) to test whether
per-slice Grad-CAM activation strength is a better proxy for "evidence of
abnormality" than the classify-probability ranking screen_oct_volume
currently uses.

For each patient/eye's latest visit (the only visits with biomarker ground
truth), computes for every B-scan:
  - classify probability for the volume's predicted class (what
    screen_oct_volume ranks by today)
  - Grad-CAM raw_max and activation_mean for that same class (what
    generate_gradcam already computes per-call, just batched here offline)

Then reports Spearman rank correlation of each signal against the
biomarker-flag count, and precision@3 against the answer key's chosen
top-3, to see whether Grad-CAM strength would have picked better evidence
slices than model confidence.

Usage (repo root, retfound conda env):
  python eval/experiment_gradcam_evidence.py --patients 60,206,251
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "interpretability"))
sys.path.insert(0, str(REPO_ROOT / "data"))

import grad_cam_retfound as gc  # noqa: E402
import oct_zarr as oz  # noqa: E402

CHECKPOINT = REPO_ROOT / "output_dir" / "retfound_mae_OLIVES_lp" / "checkpoint-best.pth"
CLASS_NAMES = ["DME", "DR"]
INPUT_SIZE = 224
DEVICE = "cuda"


def precision_at_3(predicted_top3, truth):
    return len(set(predicted_top3) & set(truth)) / 3.0


def run_patient(model, answer_keys, patient_id):
    entry = answer_keys[str(patient_id)]
    eye = next(iter(entry["eyes"]))  # single study eye per this cohort
    eye_entry = entry["eyes"][eye]
    latest_id = eye_entry["latest_visit"]
    visit = eye_entry["visits"][latest_id]
    biomarkers = visit["biomarkers"]
    if not biomarkers:
        print(f"patient {patient_id}: no biomarker ground truth at latest visit, skipping")
        return None

    volume, resolved_visit_id = oz.get_oct_volume_array(patient_id, eye, latest_id)
    n = volume.shape[0]
    images = [Image.fromarray(volume[i]) for i in range(n)]

    tfm = gc.build_eval_transform(INPUT_SIZE)
    device = torch.device(DEVICE)

    counts = np.array([biomarkers.get(str(i), {}).get("count", 0) for i in range(n)])
    top3_truth = [int(i) for i in sorted(range(n), key=lambda i: -counts[i])[:3]]
    # ties at the max count -- report the whole tied group too, since the
    # answer key's top3 is an arbitrary slice of it
    max_count = counts.max()
    tied_max = [int(i) for i in range(n) if counts[i] == max_count]

    # 1. classify probabilities (batched, no grad) -- same signal
    # screen_oct_volume uses today.
    per_slice = gc.classify_batch(model, CLASS_NAMES, DEVICE, images, INPUT_SIZE)
    mean_probs = {c: sum(r["probabilities"][c] for r in per_slice) / n for c in CLASS_NAMES}
    volume_prediction = max(mean_probs, key=mean_probs.get)
    probs = np.array([r["probabilities"][volume_prediction] for r in per_slice])

    # 2. Grad-CAM activation strength for the same predicted class, one
    # slice at a time (needs a backward pass, can't batch as easily).
    raw_maxes, act_means = [], []
    target_idx = CLASS_NAMES.index(volume_prediction)
    for img in images:
        img_tensor = tfm(img.convert("RGB")).unsqueeze(0)
        cam, t_idx, pred_idx, sp, raw_max = gc.compute_gradcam(model, img_tensor, device, target_idx)
        raw_maxes.append(raw_max)
        act_means.append(float(cam.mean().item()))
    raw_maxes = np.array(raw_maxes)
    act_means = np.array(act_means)

    top3_prob = [int(i) for i in np.argsort(-probs)[:3]]
    top3_rawmax = [int(i) for i in np.argsort(-raw_maxes)[:3]]
    top3_actmean = [int(i) for i in np.argsort(-act_means)[:3]]

    result = {
        "patient_id": patient_id, "eye": eye, "visit_id": resolved_visit_id,
        "disease": entry["disease_label"], "volume_prediction": volume_prediction,
        "n_bscans": n,
        "truth_top3 (answer key)": top3_truth,
        "truth_tied_at_max_count": tied_max,
        "max_biomarker_count": int(max_count),
        "classify_prob": {
            "top3": top3_prob,
            "precision@3_vs_key": precision_at_3(top3_prob, top3_truth),
            "precision@3_vs_tied_max": precision_at_3(top3_prob, tied_max),
            "spearman_vs_count": round(spearmanr(probs, counts).correlation, 3),
        },
        "gradcam_raw_max": {
            "top3": top3_rawmax,
            "precision@3_vs_key": precision_at_3(top3_rawmax, top3_truth),
            "precision@3_vs_tied_max": precision_at_3(top3_rawmax, tied_max),
            "spearman_vs_count": round(spearmanr(raw_maxes, counts).correlation, 3),
        },
        "gradcam_activation_mean": {
            "top3": top3_actmean,
            "precision@3_vs_key": precision_at_3(top3_actmean, top3_truth),
            "precision@3_vs_tied_max": precision_at_3(top3_actmean, tied_max),
            "spearman_vs_count": round(spearmanr(act_means, counts).correlation, 3),
        },
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True)
    ap.add_argument("--answer-keys", default=str(REPO_ROOT / "eval" / "answer_keys.json"))
    args = ap.parse_args()

    answer_keys = json.loads(Path(args.answer_keys).read_text())
    patients = [int(p) for p in args.patients.split(",")]

    print(f"Loading RETFound classifier on {DEVICE} ...", file=sys.stderr)
    model, n_classes = gc.load_classifier(str(CHECKPOINT), INPUT_SIZE, DEVICE)

    for pid in patients:
        print(f"\n=== patient {pid} ===")
        r = run_patient(model, answer_keys, pid)
        if r:
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    import torch  # noqa: E402
    main()
