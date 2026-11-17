# In-Domain and Cross-Domain Classification Experiments

Summary of the experiments already run in this repo: in-domain training on two OCT datasets, plus one cross-dataset (device-shift) test.

Sources: `train.sh`, `inference_cross_dataset.py`, `output_dir/*` (metrics CSV/JSON, logs), `octdl/` folder.

## 1. Datasets (2)

| Dataset | Classes | Size | Used for |
|---|---|---|---|
| Kermany OCT2017 (stored outside the repo) | 4: CNV, DME, DRUSEN, NORMAL | Test set about 1,000 images (inferred from accuracy granularity, not verified) | Training and in-domain test |
| OCTDL (local `octdl/`) | 7: AMD, DME, ERM, NO (normal), RAO, RVO, VID | Train 1,647; val 206; test 211 | Training and in-domain test; external set for the cross-dataset test |

Only DME and normal exist in both datasets. OCT2017's CNV and DRUSEN have no OCTDL counterpart, and OCTDL's AMD, ERM, RAO, RVO and VID have none in OCT2017.

OCTDL split counts (images per class):

| Class | Train | Val | Test |
|---|---|---|---|
| AMD | 984 | 123 | 124 |
| DME | 117 | 15 | 15 |
| ERM | 124 | 15 | 16 |
| NO | 265 | 33 | 34 |
| RAO | 17 | 2 | 3 |
| RVO | 80 | 10 | 11 |
| VID | 60 | 8 | 8 |

## 2. Model

Single architecture: **RETFound DINOv2 (ViT-Large, about 303M parameters)**, initialised from `RETFound_dinov2_meh`. All runs use 50 epochs, 224x224 input, batch size 24 (`train.sh`). They differ only in adaptation:

- **Fine-tune:** all weights trained.
- **Linear probe (LP):** only the classification head trained (4,100 parameters for OCT2017; 7,175 for OCTDL).

No other architectures were run (README mentions DINOv3 and MAE weights, but nothing was run with them).

## 3. Experiments

- **A. In-domain training.** Four runs: {OCT2017, OCTDL} x {fine-tune, LP}. Each is evaluated on its own val and test splits.
- **B. Cross-dataset (device-shift) test.** The OCT2017 fine-tuned model is applied to OCTDL with no retraining (`inference_cross_dataset.py`).
  - The 4-class output is collapsed to DME / NORMAL / OTHER, and OCTDL's 7 classes are collapsed the same way (`NO` becomes NORMAL).
  - Only one direction was run (OCT2017 to OCTDL). The OCT2017 LP model and the OCTDL-trained models were not tested cross-dataset.

## 4. Results

### A. In-domain (last row of each `metrics_*.csv`)

| Run | Val acc | Test acc | Test F1 | Test AUC | Test kappa |
|---|---|---|---|---|---|
| OCT2017 fine-tune | 0.977 | **0.993** | 0.993 | 1.000 | 0.991 |
| OCT2017 LP | 0.794 | 0.831 | 0.823 | 0.983 | 0.775 |
| OCTDL fine-tune | 0.942 | **0.924** | 0.861 | 0.994 | 0.876 |
| OCTDL LP | 0.767 | 0.777 | 0.569 | 0.946 | 0.598 |

Fine-tuning clearly beats the linear probe on both datasets.

### B. Cross-dataset (OCT2017 fine-tuned model on OCTDL, 206 images)

Overall mapped accuracy: **0.369** (vs 0.993 in-domain). Accuracy on the DME/NORMAL subset only: **0.979**.

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| DME | 0.139 | 0.933 | 0.241 | 15 |
| NORMAL | 0.434 | 1.000 | 0.606 | 33 |
| OTHER | 1.000 | 0.184 | 0.310 | 158 |

Confusion matrix (rows = ground truth, columns = prediction):

| GT \ Pred | DME | NORMAL | OTHER |
|---|---|---|---|
| DME | 14 | 1 | 0 |
| NORMAL | 0 | 33 | 0 |
| OTHER | 87 | 42 | 29 |

Where the 158 OTHER images went:

- AMD (123): 65 DME, 29 NORMAL, 29 CNV
- ERM (15): 10 DME, 5 NORMAL
- RVO (10): 6 DME, 4 NORMAL
- VID (8): 6 DME, 2 NORMAL
- RAO (2): 2 NORMAL

Outputs: `output_dir/refound_dinov2_finetune_OCT2017_to_OCTDL/octdl_inference_mapped.csv` (per-image) and `octdl_inference_metrics.json` (summary).

## 5. Caveats

- **Likely val split, not test.** The 206 images match the OCTDL **val** split exactly (123+15+15+33+10+8+2). The script's default path is `octdl/test`, so the run was probably pointed at `octdl/val`. Verify before reporting.
- **Label-space mismatch dominates the drop.** The source model has no classes for ERM, RVO and the other OCTDL pathologies, so it is forced into DME or NORMAL for them. The DME/NORMAL accuracy (0.979) is the cleaner signal, but it rests on only 48 images (15 DME).
- **Missing for a true device-shift study:** the reverse direction (OCTDL to OCT2017), fine-tune vs LP across datasets, and metrics restricted to shared classes. New cross-dataset results should go in `output_dir/<source>_to_<target>/`.

(Single-image latency benchmark omitted for now.)
