# In-Domain and Cross-Domain Classification Experiments (v2: binary DME vs NORMAL)

This is the redesigned experiment. It replaces the earlier one in `EXPERIMENTS_IN_CROSS_DOMAIN_CLASSIFICATION.md`, which scored a 4-class model on OCTDL after collapsing labels into DME / NORMAL / OTHER. Here both datasets are restricted to the two classes they share, so every model has the same label space and any drop in performance is easier to attribute to dataset shift.

## 1. Method

- **Model:** RETFound DINOv2 (ViT-Large, about 303M parameters), initialised from `RETFound_dinov2_meh`, with a 2-class head (DME, NORMAL).
- **Two adaptation strategies:**
  - **Fine-tune:** all weights are trained.
  - **Linear probe (LP):** only the classification head is trained.
- **Training setup (identical for all 4 training runs):** 50 epochs, batch size 24, 224x224 input, AdamW with base LR 5e-3 and layer decay 0.65, standard timm training augmentation, seed 0, one run each. The checkpoint with the best validation score (mean of macro-F1, AUC and kappa) is kept. Training uses the dataset's `train` split and selects the checkpoint on its `val` split. The test split is never used for training or model selection.
- **Evaluation:** each trained model is evaluated on its own dataset (in-domain) and on the other dataset (cross-domain), with no retraining.
- **Preprocessing (identical for both datasets, at validation and at inference):** bicubic resize to 256, centre crop to 224, ImageNet mean/std normalisation. Cross-domain inference uses the same function (`util.datasets.build_transform`) as training-time validation, so no preprocessing differs between datasets.
- **Four train/test pairs, each for fine-tune and LP (8 results total):**

| # | Train on (train + val) | Evaluate on | Type |
|---|---|---|---|
| 1 | OCT2017 | OCT2017 test | in-domain |
| 2 | OCT2017 | whole OCTDL (train + val + test) | cross-domain |
| 3 | OCTDL | OCTDL test | in-domain |
| 4 | OCTDL | OCT2017 test | cross-domain |

In pair 2 the whole OCTDL set is used because the OCT2017 model never saw any OCTDL images, so there is no leakage.

## 2. Datasets

Both datasets are restricted to two classes, **DME** and **NORMAL**, stored as ImageFolder folders with `train/`, `val/` and `test/` splits. Class order is the same in both (DME = index 0, NORMAL = index 1). DME is treated as the positive class in all metrics.

| Dataset | Folder | Source | Classes | Train | Val | Test | Total |
|---|---|---|---|---|---|---|---|
| Kermany OCT2017 (2-class subset) | `OCT2017/` | Kermany et al. | DME, NORMAL | 33,898 (DME 10,214 / NORMAL 23,684) | 3,765 (DME 1,134 / NORMAL 2,631) | 500 (250 / 250) | 38,163 |
| OCTDL (2-class subset) | `octdl/` | OCTDL | DME, NORMAL | 382 (DME 117 / NORMAL 265) | 48 (DME 15 / NORMAL 33) | 49 (DME 15 / NORMAL 34) | 479 |

- Patient-level splitting is assumed to be correct in both folders, and no leakage check was done.
- `OCT2017/` is gitignored (`.gitignore`), as `octdl/` already was.

## 3. Scripts, run order and where results are stored

**Scripts**

| Script | Role |
|---|---|
| `run_cross_domain.sh` | Driver. Runs everything below in order, for linear probe and then fine-tune. Skips training if a checkpoint already exists. |
| `main_finetune.py` (with `engine_finetune.py`, `models_vit.py`, `util/datasets.py`) | Trains one model (fine-tune or LP) and writes its in-domain validation/test metrics. |
| `cross_domain_eval.py` | Evaluates one checkpoint on one or more splits of one dataset. Computes all metrics with bootstrap confidence intervals and saves per-image predictions. |
| `summarize_cross_domain.py` | Collects every evaluation into `summary.csv` and `summary.md`. |

**Order of execution** (`bash run_cross_domain.sh`)

1. For each adaptation (LP first, then fine-tune):
   1. Train on OCT2017, then train on OCTDL (`main_finetune.py`, 4 training runs in total).
   2. Evaluate: OCT2017 model on OCT2017 test, then on whole OCTDL. OCTDL model on OCTDL test, then on OCT2017 test (`cross_domain_eval.py`, 8 evaluations in total).
2. Summarise all evaluations (`summarize_cross_domain.py`).

Shortcuts: `SKIP_TRAIN=1 bash run_cross_domain.sh` evaluates only, and `ADAPTATIONS="lp" bash run_cross_domain.sh` runs a subset.

**Where results are stored**

| What | Location |
|---|---|
| Trained models, training logs, in-domain validation/test metrics | `output_dir/retfound_dinov2_{oct2017,octdl}_2cls_{lp,finetune}/` (the `.pth` checkpoints are gitignored) |
| TensorBoard logs | `output_logs/retfound_dinov2_{oct2017,octdl}_2cls_{lp,finetune}/` |
| One folder per evaluation, with `metrics.json` (all metrics, CIs, confusion matrix) and `predictions.csv` (per-image probabilities) | `output_dir/cross_domain_2cls/<train task>__on__<test set>/` |
| Combined results table | `output_dir/cross_domain_2cls/summary.md` and `summary.csv` |
| Full console log of the run | `output_dir/cross_domain_2cls/run.log` |

## 4. How each metric is measured

DME is the positive class. A prediction is DME when P(DME) is at least 0.5 (that is, the argmax of the two-class softmax). With TP = DME predicted as DME, FN = DME predicted as NORMAL, FP = NORMAL predicted as DME and TN = NORMAL predicted as NORMAL:

| Metric | Definition |
|---|---|
| **Accuracy** | (TP + TN) / all images. Fraction of images classified correctly. |
| **AUC (ROC AUC)** | Area under the ROC curve built from the predicted probability of DME against the true label, computed over all thresholds. It measures how well the model ranks DME above NORMAL, independent of the 0.5 cutoff. 1.0 is perfect and 0.5 is chance. |
| **Sensitivity (recall)** | TP / (TP + FN). Of the images that really are DME, the fraction found. |
| **Specificity** | TN / (TN + FP). Of the images that really are NORMAL, the fraction correctly called NORMAL. |
| **Precision (PPV)** | TP / (TP + FP). Of the images the model called DME, the fraction that really are DME. |
| **F1** | 2 x precision x sensitivity / (precision + sensitivity). Harmonic mean of precision and sensitivity for DME. |

The evaluator also records balanced accuracy, NPV, Cohen's kappa, MCC, average precision and the confusion matrix. Every metric has a 95% confidence interval from 1,000 bootstrap resamples of the test images; the intervals are in `summary.md`. Precision and F1 are undefined (NaN) if the model never predicts DME.

## 5. Results

`n` is the number of images used for **inference** (evaluation). `Train / val` is the number of images used to train the model and to select its best checkpoint.

| Adaptation | Train → Test | Train / val images | n (DME/NORMAL) | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Fine-tune | OCT2017 → OCT2017 test | 33,898 / 3,765 | 500 (250/250) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Fine-tune | OCT2017 → whole OCTDL | 33,898 / 3,765 | 479 (147/332) | 0.987 | 1.000 | 0.973 | 0.994 | 0.986 | 0.979 |
| Fine-tune | OCTDL → OCTDL test | 382 / 48 | 49 (15/34) | 0.980 | 0.994 | 0.933 | 1.000 | 1.000 | 0.966 |
| Fine-tune | OCTDL → OCT2017 test | 382 / 48 | 500 (250/250) | **0.694** | 0.957 | **0.388** | 1.000 | 1.000 | 0.559 |
| Linear probe | OCT2017 → OCT2017 test | 33,898 / 3,765 | 500 (250/250) | 0.968 | 0.996 | 0.948 | 0.988 | 0.988 | 0.967 |
| Linear probe | OCT2017 → whole OCTDL | 33,898 / 3,765 | 479 (147/332) | 0.906 | 0.976 | 0.735 | 0.982 | 0.947 | 0.828 |
| Linear probe | OCTDL → OCTDL test | 382 / 48 | 49 (15/34) | 0.918 | 0.976 | 0.867 | 0.941 | 0.867 | 0.867 |
| Linear probe | OCTDL → OCT2017 test | 382 / 48 | 500 (250/250) | **0.664** | 0.917 | **0.328** | 1.000 | 1.000 | 0.494 |

95% bootstrap confidence intervals for the headline metrics (full set of intervals is in `output_dir/cross_domain_2cls/summary.md`):

| Adaptation | Train → Test | Accuracy | AUC | Sensitivity |
|---|---|---|---|---|
| Fine-tune | OCT2017 → OCT2017 test | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] |
| Fine-tune | OCT2017 → whole OCTDL | 0.987 [0.977–0.996] | 1.000 [0.999–1.000] | 0.973 [0.943–0.994] |
| Fine-tune | OCTDL → OCTDL test | 0.980 [0.939–1.000] | 0.994 [0.976–1.000] | 0.933 [0.800–1.000] |
| Fine-tune | OCTDL → OCT2017 test | 0.694 [0.654–0.734] | 0.957 [0.938–0.973] | 0.388 [0.328–0.448] |
| Linear probe | OCT2017 → OCT2017 test | 0.968 [0.950–0.982] | 0.996 [0.992–0.998] | 0.948 [0.920–0.974] |
| Linear probe | OCT2017 → whole OCTDL | 0.906 [0.879–0.931] | 0.976 [0.965–0.987] | 0.735 [0.662–0.804] |
| Linear probe | OCTDL → OCTDL test | 0.918 [0.837–0.980] | 0.976 [0.928–1.000] | 0.867 [0.667–1.000] |
| Linear probe | OCTDL → OCT2017 test | 0.664 [0.622–0.704] | 0.917 [0.891–0.940] | 0.328 [0.271–0.385] |

**Main observations**

- **The shift is asymmetric.** OCT2017-trained models transfer well to OCTDL, while OCTDL-trained models transfer poorly to OCT2017 (accuracy about 0.66–0.69).
- **The OCTDL → OCT2017 failure is mostly a threshold problem.** Specificity is 1.000 and sensitivity is only 0.33–0.39, but AUC stays at 0.92–0.96. The models still rank DME above NORMAL fairly well, but at the 0.5 cutoff they call too many DME images NORMAL, which suggests calibration shift. The saved `predictions.csv` files allow a re-thresholding analysis without re-running inference.
- **Fine-tuning beats linear probe** in-domain and in most cross-domain cells. In OCTDL → OCT2017 it improves AUC but barely changes accuracy.

**Caveats**

- Each result comes from a single run (seed 0), so small differences between fine-tune and LP may be within run-to-run noise.
- The OCT2017 test set is saturated for the fine-tuned model (accuracy 1.000, test loss about 0.001), so it cannot show any drop for that model.
- The OCTDL test set has only 49 images (15 DME), so its intervals are wide.
- The experiment cannot separate device shift from other dataset differences (population, annotation, preprocessing, training-set size). The asymmetry may partly reflect that OCT2017 provides 34k training images against 382 for OCTDL.
