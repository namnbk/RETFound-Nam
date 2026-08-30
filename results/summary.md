# Benchmark summary (36 instances)


## 2. OLIVES retrieval/navigation

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| Exact B-scan retrieval | 1.0 (3/3) | 0 | 0 |  |
| Latest-visit retrieval | 1.0 (3/3) | 0 | 0 |  |
| Longitudinal retrieval | 1.0 (3/3) | 0 | 0 | values_correct=1.0, visits_correct=1.0 |
| Find abnormal/evidence-rich slices | 0.0 (3/3) | 0 | 0 |  |

## 3. Disease assessment + evidence grounding

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| DR vs DME assessment | 0.667 (3/3) | 0 | 0 |  |
| Disease + supporting slices | 0.0 (3/3) | 0 | 0 | disease_correct=0.667, evidence_precision_at_3=0.0 |
| Disease + saliency evidence | 0.0 (3/3) | 0 | 0 | disease_correct=0.667, evidence_bscan_positive=0.0, gradcam_generated=1.0 |
| Evidence-first assessment | 0.0 (3/3) | 0 | 0 | disease_correct=0.667, evidence_precision_at_3=0.0, order_ok=1.0 |

## 4. Compound workflow

| Task | Score (N) | Malformed | Harness error | Sub-metrics |
|---|---|---|---|---|
| Latest exam -> diagnosis -> evidence | 0.0 (3/3) | 0 | 0 | disease_correct=0.667, has_valid_evidence=0.0, visit_correct=1.0 |
| Longitudinal comparison | 1.0 (3/3) | 0 | 0 | bcva_direction_correct=1.0, cst_direction_correct=1.0, visits_correct=1.0 |
| Evidence verification | 0.0 (3/3) | 0 | 0 | disease_correct=0.667, evidence_precision_at_3=0.0, order_ok=1.0 |
| Uncertain-case evidence escalation | 1.0 (3/3) | 0 | 0 | appropriate_tool_use=0.333, checked_beyond_topk=0.667, correct_abstention=1.0, evidence_precision_at_3=0.0, gradcam_generated=0.667 |

## Per-patient

| Patient | N instances | Malformed output | Harness errors |
|---|---|---|---|
| 206 | 12 | 0 | 0 |
| 251 | 12 | 0 | 0 |
| 60 | 12 | 0 | 0 |

## Limitations

- The agent cannot visually inspect a B-scan or saliency image through these tools as currently implemented: FastMCP returns get_bscan/generate_gradcam's base64 image data as plain JSON TextContent, never a real MCP ImageContent block, so the model only ever sees literal base64 text -- never actual pixels -- regardless of size limits or file-tool access. Evidence Precision@3 and "abnormality"/saliency-based metrics reflect this: the agent can only reason from indirect text signals (screen_oct_volume rankings, classify_bscan_retfound probabilities, generate_gradcam activation stats), never from viewing the image itself. A real fix requires the retinal-ai/retinal-data MCP servers to return proper image content blocks (out of scope for this eval harness).
- Evidence-precision tasks are restricted to each patient's first/last visit because that's the only ground-truth coverage Biomarker_Clinical_Data_Images.csv has.
- Task 12 ("Uncertain-case evidence escalation") measures single-model (volume-vs-slice) confidence/consistency, since no second classifier is deployed -- there is no "disagreement between models" framing in this task.
- "Balanced accuracy" for task 5 as a class-balanced metric only makes sense with enough instances to have both classes represented (the full 18-patient run, not a small pilot) -- small-sample scores are directional only.
