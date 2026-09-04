# Benchmark summary (36 instances)


## 2. OLIVES retrieval/navigation

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| Exact B-scan retrieval | 1.0 (2/3) | 1 | 1 |  |
| Latest-visit retrieval | 1.0 (2/3) | 1 | 1 |  |
| Longitudinal retrieval | 1.0 (2/3) | 1 | 1 | values_correct=1.0, visits_correct=1.0 |
| Find abnormal/evidence-rich slices | 0.0 (2/3) | 1 | 1 |  |

## 3. Disease assessment + evidence grounding

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| DR vs DME assessment | 1.0 (2/3) | 1 | 1 |  |
| Disease + supporting slices | 0.0 (2/3) | 1 | 1 | disease_correct=1.0, evidence_precision_at_3=0.167 |
| Disease + saliency evidence | 0.0 (2/3) | 1 | 1 | disease_correct=1.0, evidence_bscan_positive=0.0, gradcam_generated=1.0 |
| Evidence-first assessment | 0.0 (2/3) | 1 | 1 | disease_correct=1.0, evidence_precision_at_3=0.167, order_ok=1.0 |

## 4. Compound workflow

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| Latest exam -> diagnosis -> evidence | 0.5 (2/3) | 1 | 1 | disease_correct=1.0, has_valid_evidence=0.5, visit_correct=1.0 |
| Longitudinal comparison | 1.0 (2/3) | 1 | 1 | bcva_direction_correct=1.0, cst_direction_correct=1.0, visits_correct=1.0 |
| Evidence verification | 0.0 (2/3) | 1 | 1 | disease_correct=1.0, evidence_precision_at_3=0.167, order_ok=1.0 |
| Uncertain-case evidence escalation | 0.5 (2/3) | 1 | 1 | appropriate_tool_use=0.0, checked_beyond_topk=0.0, correct_abstention=0.0, evidence_precision_at_3=0.167, gradcam_generated=1.0 |

## Per-patient

| Patient | N instances | Malformed output | Harness errors |
|---|---|---|---|
| 206 | 12 | 0 | 0 |
| 251 | 12 | 12 | 12 |
| 60 | 12 | 0 | 0 |

## Limitations

- Any run recorded before the MCP servers were changed to emit real image content blocks measured an agent that could not visually inspect anything: get_bscan/generate_gradcam used to return base64 image data inside a plain JSON TextContent block, so the model saw literal base64 text and never pixels. Evidence Precision@3 and "abnormality"/saliency metrics from such a run reflect reasoning from indirect text signals only (screen_oct_volume rankings, classify_bscan_retfound probabilities, generate_gradcam activation stats). get_bscan, get_fundus_image and generate_gradcam now return a metadata TextContent block plus a real ImageContent block (downscaled to 224 on the longest side by default), so runs after that change do have genuine visual grounding -- check a transcript's tool results before comparing scores across the two regimes.
- Evidence-precision tasks are restricted to each patient's first/last visit because that's the only ground-truth coverage Biomarker_Clinical_Data_Images.csv has.
- Task 12 ("Uncertain-case evidence escalation") measures single-model (volume-vs-slice) confidence/consistency, since no second classifier is deployed -- there is no "disagreement between models" framing in this task.
- "Balanced accuracy" for task 5 as a class-balanced metric only makes sense with enough instances to have both classes represented (the full 18-patient run, not a small pilot) -- small-sample scores are directional only.
