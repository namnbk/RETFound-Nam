---
name: evidence-grounded-oct-assessment
description: Clinical/model reasoning policy for OCT questions that require interpretation, not just data retrieval -- assessing DR vs DME, identifying which B-scans support a prediction, inspecting Grad-CAM saliency, comparing findings across visits, resolving disagreement between model calls, or any compound "what does this look like and why" workflow. Governs HOW to use the retinal-data and retinal-ai MCP tools together for a clinical answer: when to invoke a model at all, how many B-scans to check, how to detect and escalate on disagreement, when to abstain, and how to structure a final answer (prediction / evidence / confidence / tools used / provenance). Use whenever a request asks for an interpretation, diagnosis, evidence, saliency, or a longitudinal/comparative clinical judgment -- not for plain "show me" / "fetch" / "compare BCVA" requests, which stay in retinal-data-operations.
---

This is the clinical-reasoning skill layered on top of the mechanical
data-access rules in `retinal-data-operations`: it governs *when* to call a
model at all, how to pick which B-scans get checked, how to notice when
`retinal-ai`'s outputs don't hold together, and how a final clinical answer
must be shaped. It does not repeat the resolve/retrieve/provenance rules --
see `.claude/skills/retinal-data-operations/SKILL.md` for those; this skill
assumes them and adds the interpretation layer on top.

Both MCP servers are already-built and already-running (registered in
`.mcp.json`) -- there is nothing to build or launch. The handle for
exercising the *decision policy* itself (not just the tools) is
`.claude/skills/evidence-grounded-oct-assessment/driver.py`, which drives
both servers together over the real stdio transport and runs four scenarios
that each take a genuinely different path through the policy below.

All paths below are relative to the repo root (`D:\Nam\RETFound-Nam`).

## The key design principle: this is a decision policy, not a pipeline

There is no fixed sequence like `Zarr -> RETFound -> Grad-CAM -> answer` that
every request runs. What gets called depends on what the question actually
needs -- and calling more tools than a question needs is itself a policy
violation (it manufactures evidence that wasn't asked for and slows down a
simple answer). `driver.py`'s four scenarios exist specifically to prove
this branches, not just to show more tool calls:

| Scenario | Question | Path taken |
|---|---|---|
| A | "show me the fundus photo" | `retinal-data` only -- `retinal-ai` is never touched |
| B | "is this DR or DME, and what's the evidence?" (patient 60/OD) | screen -> cross-check top slice -> **agree, wide margin** -> Grad-CAM as auxiliary evidence -> confident answer |
| C | "is this DR or DME?" (patient 206/OD) | screen -> cross-check -> **weak margin + real disagreement** -> escalate -> still conflicting -> explicit uncertainty, no Grad-CAM (there's no settled prediction to explain) |
| D | "does this B-scan show IRF?" | no model call at all -- no tool on either server predicts biomarkers, so the answer is "not determinable," decided before touching `retinal-ai` |

## Run (agent path)

**As Claude Code, you almost never invoke `driver.py` directly** -- once the
servers are connected you call `mcp__retinal-data__<tool>` /
`mcp__retinal-ai__<tool>` yourself, applying the policy below. `driver.py`
exists for two reasons: (1) it's runnable, verified evidence that all four
policy branches actually work against the real servers and real patient
data, not just in prose; (2) it's a literal reference implementation, one
scenario per row of the table above -- read it when a request doesn't map
cleanly onto B, C, or D.

```bash
"C:/Users/lamata/Miniconda3/envs/retfound/python.exe" .claude/skills/evidence-grounded-oct-assessment/driver.py
```

Takes ~30-60s on CPU (loads the ~1.2GB RETFound checkpoint once, then runs
~10 classify calls + 1 Grad-CAM across two real patient volumes). Verified
output (this session, this container, both servers, full run; trimmed to
the decision points -- full JSON answers print at the end of B and C):

```
SCENARIO A -- pure data question: retinal-ai is never touched
  get_fundus_image -> visit_id='visit_17', source_path='fundus_W104.tif'

SCENARIO B -- diagnostic question, patient 60/OD: sufficient, consistent evidence
  screen_oct_volume -> volume_prediction='DR', top candidates=[{'index': 45, 'DR_probability': 0.985}, ...]
  classify_bscan_retfound(slice 45) -> prediction='DR' (volume-level agrees: True); slice-level margin=0.970
  consistency check: margin 0.970 >= threshold 0.6 and predictions agree -> evidence sufficient
  generate_gradcam(slice 45) -> target_class='DR', activation_max=0.018 (auxiliary evidence only)
  confidence.label = "high (volume/slice agree, wide margin)"

SCENARIO C -- diagnostic question, patient 206/OD: weak margin + real disagreement
  screen_oct_volume -> volume_prediction='DME' (mean over 49 slices: DME=0.748, DR=0.252, margin=0.495)
  slices 48/47/46 (top-3 by DME_probability): all individually agree (DME)
  slice 0 (checked outside the top-k): individually predicts DR (0.745) -- genuine disagreement
  consistency check: margin 0.495 < 0.6 AND a cross-checked slice disagrees -> evidence CONFLICTING
  confidence.label = "insufficient / conflicting -- do not treat as a diagnosis"
  (no Grad-CAM call -- there is no settled prediction to explain)

SCENARIO D -- biomarker question: no validated tool exists to answer it
  generate_gradcam response fields: ['activation_max', 'activation_mean', 'bscan_index', 'model',
    'saliency_image', 'target_class'] -- no 'biomarker'/'IRF' field on any tool, on either server

All four scenarios completed against the real retinal-data and retinal-ai MCP servers over
stdio -- no scenario followed the same tool sequence as any other.
```

## Core logic

```
Understand clinical question
        |
Acquire correct OCT examination        <- retinal-data-operations Rule 1
        |
Determine what evidence is required    <- does this question even need retinal-ai? (Scenario A/D: no)
        |
Select informative B-scans             <- screen_oct_volume, not every slice
        |
Call appropriate models/tools          <- classify_bscan_retfound / screen_oct_volume / generate_gradcam
        |
Collect predictions + evidence
        |
Check consistency                      <- margin + cross-check agreement, not just "did it return"
        |
   Enough evidence?
    /          \
  yes            no
   |              |
 answer      acquire more (cross-check more slices) or abstain -- never guess
```

Concretely, with the actual tools:

1. **Never jump from a classifier output straight to a clinical
   conclusion.** Every `screen_oct_volume` or `classify_bscan_retfound`
   result is one input to the consistency check (rule 8), not the answer
   itself. A single high-probability number from one call is not evidence
   of sufficiency on its own -- see Scenario B, where the *agreement between
   two independent calls* (volume-level and slice-level) is what justifies
   confidence, not either call alone.
2. **Resolve patient/eye/visit/volume first.** This is
   `retinal-data-operations` Rule 1 (`list_patient_examinations`, then
   `get_oct_volume_info`) -- don't skip it because the request "sounds
   diagnostic." A confident-looking answer about the wrong visit is worse
   than a slow one about the right visit.
3. **Select B-scans when the request needs image-level evidence, don't
   grade every slice.** `screen_oct_volume(top_k=...)` ranks slices by how
   strongly they support its own volume-level call in one batched pass --
   use its `candidate_bscans` instead of calling `classify_bscan_retfound`
   in a loop over the whole volume. Reach for individual
   `classify_bscan_retfound` calls only to (a) cross-check a specific
   candidate, or (b) when a request names a specific slice.
4. **`classify_bscan_retfound`/`screen_oct_volume` output is model
   evidence, not ground truth.** State it as "the model predicts X," with
   the model name (`retfound-olives-v1`, a 2-class DME-vs-DR linear probe --
   there is no "normal" class), never as an unqualified diagnosis. Today
   there is exactly one classifier registered on `retinal-ai` -- a DINO
   classifier was investigated for this project and has no usable saved
   checkpoint, so no `classify_bscan_dino` tool exists (see
   `mcp_servers/retinal_ai/server.py`'s module docstring). If a second
   model is ever added, treat its output identically: evidence to be
   cross-checked, not a tiebreaker to defer to.
5. **`generate_gradcam` is auxiliary evidence, called after a prediction
   already stands, to explain it -- never to produce or rescue one.** In
   Scenario B it's called only once the volume/slice predictions already
   agree with a wide margin. In Scenario C, where the evidence stays
   conflicting, it is *not* called -- there is no settled prediction for a
   saliency map to explain, and running it anyway would dress up an
   unresolved disagreement as if it had visual support.
6. **Never claim a specific biomarker.** No tool on either server predicts
   one -- `generate_gradcam`'s own docstring states it does not interpret
   its heatmap as a biomarker (e.g. "this is IRF"), and its response has no
   `biomarker` field (verified live in Scenario D:
   `['activation_max', 'activation_mean', 'bscan_index', 'model', 'saliency_image', 'target_class']`).
   A high-activation region can be described by location ("saliency
   concentrated near slice 45's outer retina"), never labeled with a named
   finding no tool actually predicted.
7. **OLIVES biomarker/disease ground truth is structurally unreachable
   during inference, not just withheld by convention.** No function in
   `oct_zarr.py`'s reader layer touches the `biomarkers/` Zarr group or any
   DR/DME label column (`retinal-data-operations` Rule 8) -- there is
   nothing to accidentally look up even under pressure to "just check the
   real answer."
8. **When calls disagree, escalate -- don't arbitrarily pick one.**
   `screen_oct_volume`'s top-k candidates are ranked by probability *for
   its own predicted class*, so they mostly agree with it by construction
   (Scenario C: all of slices 48/47/46 individually agreed with the DME
   call). Real disagreement shows up by cross-checking slices *outside*
   the top-k, or by noting a weak volume-level margin (below ~0.6, versus
   the 0.75-0.98 seen in confident cases across this store) as a signal to
   look further before answering, not as the answer's caveat footnote.
9. **If evidence stays insufficient or contradictory after escalation,
   report uncertainty explicitly -- don't force an answer.** Scenario C's
   final answer sets `confidence.label` to
   `"insufficient / conflicting -- do not treat as a diagnosis"` and lists
   the specific disagreeing evidence in `uncertainty_notes`, rather than
   silently reporting the volume-level majority as if it were settled.
10. **A final clinical answer keeps five things separate, never folded
    into one paragraph:** model prediction (with model name, explicitly
    labeled as a prediction), supporting B-scans (index + source_path +
    why each was selected), confidence/uncertainty (a label plus the
    numbers that justify it), tools used (the actual call sequence, which
    will differ by scenario -- see the table above), and provenance
    (patient/eye/visit as *resolved*, not as typed). `build_answer()` in
    `driver.py` is the reference shape; both Scenario B and C print a full
    example.

## Gotchas

- **`screen_oct_volume`'s `top_k` ranking is close to circular as a
  disagreement check.** It ranks by probability *for the class it already
  picked*, so the top-ranked slices will almost always individually agree
  with it (confirmed: patient 206/OD's top-3 all agreed with the DME call).
  To find real disagreement you have to look at slices the ranking
  *excludes* -- Scenario C checks slice 0 specifically because it was
  verified offline to disagree; in a live request without a precomputed
  answer, sample a few slices spread across the volume (not just the
  bottom of the top-k) rather than trusting agreement among the top-3 alone.
- **`MARGIN_THRESHOLD = 0.6` in `driver.py` is an empirical heuristic, not
  a validated clinical cutoff.** It's set from observed volume-level
  margins across all 18 patients currently on disk (confident cases run
  0.75-0.98; the one genuinely ambiguous case, patient 206/OD, sits at
  0.495) -- there's a real gap there today, but it's a property of this
  checkpoint and this small patient set, not a clinically validated
  boundary. Recheck it if the checkpoint is retrained or more patients are
  added.
- **There is only one classifier, so "disagreement between models" today
  means volume-level vs. slice-level disagreement within the *same*
  RETFound model, not RETFound vs. DINO.** Don't call a
  `classify_bscan_dino` tool -- it doesn't exist. If asked to "check with
  another model," the honest response is that only one is currently
  available, and more evidence means more slices from it, exactly as
  Scenario C's `uncertainty_notes` states.
- **`activation_max`/`activation_mean` from `generate_gradcam` have no
  established clinical meaning or threshold.** They're raw post-hook
  activation statistics from `grad_cam_retfound.py`, useful only to
  describe *where* saliency concentrated -- don't invent a confidence score
  or biomarker likelihood from their magnitude.
- **The DME-vs-DR classifier has no "normal" class**, so it will confidently
  output DME or DR even for a scan a clinician would call unremarkable.
  If asked "is this eye normal," that's structurally the same guardrail as
  the biomarker case (rule 6/Scenario D) -- no tool here can answer it, say
  so rather than reporting the lower-probability class as "normal."
- **`FastMCP` returns a plain-dict tool result as a single pretty-printed
  JSON `TextContent` block** (`structuredContent` is `None`) -- when
  driving these servers directly, as `driver.py` does,
  `json.loads(result.content[0].text)`, same as `retinal-data-operations`
  already documents.

## Troubleshooting

Shared with `retinal-data-operations` (`.mcp.json` edited after session
start -> restart; `{"error": "No Zarr store..."}` -> wrong patient id, the
message lists valid ones; `bscan_index out of range` -> call
`get_oct_volume_info` first). One addition specific to this skill:

- **Driver takes 30-60s and looks stuck partway through** -- expected. It
  loads the ~1.2GB checkpoint once at `retinal-ai` startup, then Scenario C
  alone makes four individual `classify_bscan_retfound` calls (three
  top-k candidates plus slice 0) on CPU. Not a hang; let it finish.
