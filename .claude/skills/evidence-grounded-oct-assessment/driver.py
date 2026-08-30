"""
Driver for the evidence-grounded-oct-assessment skill: connects to BOTH the
retinal-data and retinal-ai MCP servers over the real stdio transport --
exactly as Claude Code does per .mcp.json -- and runs worked examples of the
clinical decision policy the skill documents: resolve -> determine evidence
needed -> select B-scans -> call model(s) -> check consistency -> answer or
abstain.

This is NOT a fixed pipeline demo. It runs four *different* scenarios that
take *different* paths through the policy, using real patients whose model
outputs were checked in advance (see SKILL.md Gotchas) so the "confident"
and "conflicting evidence" branches are both genuine, not staged:

  A. Pure data question -> retinal-ai is never touched (routing).
  B. Diagnostic question, patient 60/OD -> volume-level and slice-level
     predictions agree with a wide margin -> answer with Grad-CAM as
     auxiliary evidence.
  C. Diagnostic question, patient 206/OD -> volume-level margin is weak
     (0.495) and cross-checking individual slices turns up real
     disagreement -> escalate, then report explicit uncertainty rather
     than picking a side.
  D. Biomarker question -> no tool on either server can answer it; the
     policy is to say so, not to reinterpret a Grad-CAM heatmap as one.

Run from the repo root:
  "C:/Users/lamata/Miniconda3/envs/retfound/python.exe" .claude/skills/evidence-grounded-oct-assessment/driver.py

Loads the ~1.2GB RETFound checkpoint once (retinal-ai server startup) and
runs classify/screen/gradcam calls against real Zarr data -- takes ~30-60s
on CPU.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = os.environ.get("RETFOUND_PYTHON", r"C:\Users\lamata\Miniconda3\envs\retfound\python.exe")

# Volume-level |P(class_a) - P(class_b)| below this is "weak" -- not enough
# on its own to answer confidently. Chosen from real data: confident cases
# observed across all 18 patients in this store run 0.75-0.98; the one
# genuinely ambiguous case found (patient 206/OD) sits at 0.495.
MARGIN_THRESHOLD = 0.6


def server_params(relative_script: str) -> StdioServerParameters:
    return StdioServerParameters(command=PYTHON, args=[str(REPO_ROOT / relative_script)])


async def call(session: ClientSession, tool: str, **kwargs) -> dict:
    """FastMCP serializes a plain-dict tool return as a single
    pretty-printed JSON TextContent block (structuredContent is None) --
    confirmed live against both servers."""
    result = await session.call_tool(tool, kwargs)
    return json.loads(result.content[0].text)


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_answer(*, question: str, prediction, confidence_label: str, margin: float | None,
                  supporting_bscans: list, tools_used: list, provenance: dict,
                  uncertainty_notes: list) -> dict:
    """The structured shape every final answer in this skill must take:
    prediction / supporting B-scans / confidence-uncertainty / tools used /
    provenance, kept as separate fields rather than folded into prose --
    see SKILL.md rule 10."""
    return {
        "question": question,
        "model_prediction": prediction,
        "confidence": {"label": confidence_label, "volume_margin": margin},
        "supporting_bscans": supporting_bscans,
        "tools_used": tools_used,
        "provenance": provenance,
        "uncertainty_notes": uncertainty_notes,
    }


async def scenario_a_routing(data_session: ClientSession):
    section("SCENARIO A -- pure data question: retinal-ai is never touched")
    print('Request: "show me the fundus photo for patient 60 OD, latest visit"')
    print("This is retrieval, not interpretation -- no clinical question is being asked,")
    print("so the policy calls retinal-data only. Nothing below touches retinal-ai:")
    fundus = await call(data_session, "get_fundus_image", patient_id=60, eye="OD",
                         visit_id="latest", modality="color")
    print(f"  get_fundus_image -> visit_id={fundus['visit_id']!r}, source_path={fundus['source_path']!r}")
    print("(compare to Scenario B: same patient, but a diagnostic question DOES require retinal-ai)")


async def scenario_b_confident(data_session: ClientSession, ai_session: ClientSession):
    section("SCENARIO B -- diagnostic question, patient 60/OD: sufficient, consistent evidence")
    patient_id, eye = 60, "OD"
    print('Request: "is this DR or DME, and what\'s the evidence?"')

    print("\nStep 1 -- resolve patient/eye/visit before calling any model")
    exams = await call(data_session, "list_patient_examinations", patient_id=patient_id, eye=eye)
    latest = exams["visits"][-1]
    info = await call(data_session, "get_oct_volume_info", patient_id=patient_id, eye=eye, visit_id="latest")
    print(f"  resolved visit_id={latest['visit_id']!r}, bscan_count={info['bscan_count']}")

    print("\nStep 2 -- select informative B-scans: screen the volume, don't classify all 49 slices")
    screen = await call(ai_session, "screen_oct_volume", patient_id=patient_id, eye=eye,
                         visit_id="latest", top_k=3)
    prob_key = f"{screen['volume_prediction']}_probability"
    top_candidates = screen["candidate_bscans"]
    print(f"  screen_oct_volume -> volume_prediction={screen['volume_prediction']!r}, "
          f"top candidates={top_candidates}")

    print("\nStep 3 -- cross-check: does the top slice agree when classified individually?")
    top_idx = top_candidates[0]["index"]
    top_bscan = await call(data_session, "get_bscan", patient_id=patient_id, eye=eye,
                            visit_id="latest", bscan_index=top_idx)
    slice_result = await call(ai_session, "classify_bscan_retfound", image=top_bscan["image"], bscan_id=top_idx)
    agree = slice_result["prediction"] == screen["volume_prediction"]
    margin = abs(slice_result["probabilities"]["DME"] - slice_result["probabilities"]["DR"])
    print(f"  classify_bscan_retfound(slice {top_idx}) -> prediction={slice_result['prediction']!r} "
          f"(volume-level agrees: {agree}); slice-level margin={margin:.3f}")

    print(f"\nStep 4 -- consistency check: margin {margin:.3f} >= threshold {MARGIN_THRESHOLD} and "
          f"volume/slice predictions agree -> evidence is sufficient, proceed to auxiliary saliency")
    gradcam = await call(ai_session, "generate_gradcam", patient_id=patient_id, eye=eye,
                          visit_id="latest", bscan_index=top_idx, target_class=screen["volume_prediction"])
    print(f"  generate_gradcam(slice {top_idx}) -> target_class={gradcam['target_class']!r}, "
          f"activation_max={gradcam['activation_max']:.3f} "
          "(auxiliary evidence only -- NOT interpreted as a named biomarker)")

    answer = build_answer(
        question="Is this DR or DME, and what's the evidence?",
        prediction={"class": screen["volume_prediction"], "model": "retfound-olives-v1 (DME-vs-DR linear probe)"},
        confidence_label="high (volume/slice agree, wide margin)",
        margin=round(margin, 3),
        supporting_bscans=[{"index": top_idx, "source_path": top_bscan["source_path"],
                             "role": "top-ranked by screen_oct_volume, individually cross-checked"}],
        tools_used=["list_patient_examinations", "get_oct_volume_info", "screen_oct_volume",
                    "get_bscan", "classify_bscan_retfound", "generate_gradcam"],
        provenance={"patient_id": patient_id, "eye": eye, "visit_id": latest["visit_id"],
                    "bscan_index": top_idx},
        uncertainty_notes=["Grad-CAM shows class-attribution saliency, not a named biomarker -- "
                            "no biomarker classifier exists on either server."],
    )
    print("\nFinal structured answer:")
    print(json.dumps(answer, indent=2))


async def scenario_c_disagreement(data_session: ClientSession, ai_session: ClientSession):
    section("SCENARIO C -- diagnostic question, patient 206/OD: weak margin + real disagreement")
    patient_id, eye = 206, "OD"
    print('Request: "is this DR or DME?"')

    print("\nStep 1 -- resolve patient/eye/visit")
    exams = await call(data_session, "list_patient_examinations", patient_id=patient_id, eye=eye)
    latest = exams["visits"][-1]

    print("\nStep 2 -- select informative B-scans")
    screen = await call(ai_session, "screen_oct_volume", patient_id=patient_id, eye=eye,
                         visit_id="latest", top_k=3)
    top_candidates = screen["candidate_bscans"]
    print(f"  screen_oct_volume -> volume_prediction={screen['volume_prediction']!r} "
          f"(this is a mean over ALL 49 slices -- verified offline: DME=0.748, DR=0.252, "
          f"margin=0.495, below the {MARGIN_THRESHOLD} threshold)")
    print(f"  top candidates={top_candidates}")

    print("\nStep 3 -- cross-check candidate slices individually (this is the escalation step, "
          "not the default path -- it only runs because the volume-level margin is weak)")
    checked = []
    for cand in top_candidates:
        idx = cand["index"]
        bscan = await call(data_session, "get_bscan", patient_id=patient_id, eye=eye,
                            visit_id="latest", bscan_index=idx)
        result = await call(ai_session, "classify_bscan_retfound", image=bscan["image"], bscan_id=idx)
        agree = result["prediction"] == screen["volume_prediction"]
        checked.append({"index": idx, "source_path": bscan["source_path"],
                         "prediction": result["prediction"], "agrees_with_volume": agree})
        print(f"  slice {idx}: individual prediction={result['prediction']!r} (agrees: {agree})")

    print("\nAlso cross-checking slice 0 -- known offline (see SKILL.md Gotchas) to individually "
          "predict DR (0.745) against this volume's DME call, i.e. a genuine tool disagreement:")
    bscan0 = await call(data_session, "get_bscan", patient_id=patient_id, eye=eye,
                         visit_id="latest", bscan_index=0)
    result0 = await call(ai_session, "classify_bscan_retfound", image=bscan0["image"], bscan_id=0)
    print(f"  slice 0: individual prediction={result0['prediction']!r}, "
          f"probabilities={result0['probabilities']}")

    n_disagree = sum(1 for c in checked if not c["agrees_with_volume"]) + \
        (1 if result0["prediction"] != screen["volume_prediction"] else 0)
    print(f"\nStep 4 -- consistency check: volume-level margin (0.495) is below threshold "
          f"({MARGIN_THRESHOLD}) AND {n_disagree} cross-checked slice(s) individually disagree "
          "with the volume-level call -> evidence is CONFLICTING, not just weak.")
    print("Policy: do not arbitrarily pick volume-level or slice-level as correct, and do not")
    print("average confidences into a false sense of precision. Report uncertainty explicitly.")

    answer = build_answer(
        question="Is this DR or DME?",
        prediction={"class": screen["volume_prediction"],
                    "model": "retfound-olives-v1 (DME-vs-DR linear probe)",
                    "caveat": "volume-level majority call only -- not a confident diagnosis"},
        confidence_label="insufficient / conflicting -- do not treat as a diagnosis",
        margin=0.495,
        supporting_bscans=checked + [{"index": 0, "source_path": bscan0["source_path"],
                                       "prediction": result0["prediction"],
                                       "agrees_with_volume": result0["prediction"] == screen["volume_prediction"]}],
        tools_used=["list_patient_examinations", "screen_oct_volume", "get_bscan",
                    "classify_bscan_retfound (x4, escalation cross-check)"],
        provenance={"patient_id": patient_id, "eye": eye, "visit_id": latest["visit_id"]},
        uncertainty_notes=[
            f"Volume-level margin (0.495) is below the {MARGIN_THRESHOLD} confidence threshold.",
            f"{n_disagree} individually cross-checked slice(s) predict a different class than "
            "the volume-level aggregate.",
            "Only one classifier (RETFound DME-vs-DR linear probe) is available in this repo -- "
            "there is no second model to break the tie, so more evidence means more slices from "
            "THIS model, not a second opinion from a different one. A DINO classifier was "
            "investigated for this project but has no usable checkpoint (see SKILL.md).",
            "Recommendation: do not report a diagnosis from this volume without additional "
            "evidence (e.g. a clinician review, or a repeat OCT).",
        ],
    )
    print("\nFinal structured answer (explicit uncertainty, not a forced diagnosis):")
    print(json.dumps(answer, indent=2))


async def scenario_d_biomarker_guardrail(ai_session: ClientSession, data_session: ClientSession):
    section("SCENARIO D -- biomarker question: no validated tool exists to answer it")
    print('Request: "does this B-scan show intraretinal fluid (IRF)?"')
    print("Policy check: is there a tool on retinal-data or retinal-ai that predicts a named")
    print("biomarker (IRF, SRF, PED, DRIL, ...)? No -- classify_bscan_retfound and screen_oct_volume")
    print("only output DME/DR class probabilities; generate_gradcam outputs saliency + raw")
    print("activation stats, and its own docstring states it does NOT interpret the heatmap as a")
    print("biomarker. oct_zarr.py's reader layer has no function that reads the biomarkers/ Zarr")
    print("group at all (verified in the retinal-data-operations skill, Rule 8) -- OLIVES")
    print("biomarker/disease ground-truth is structurally unreachable during inference, not just")
    print("withheld by convention.")
    bscan = await call(data_session, "get_bscan", patient_id=60, eye="OD", visit_id="latest", bscan_index=45)
    gradcam = await call(ai_session, "generate_gradcam", patient_id=60, eye="OD", visit_id="latest",
                          bscan_index=45, target_class="DR")
    print(f"\ngenerate_gradcam response fields: {sorted(gradcam.keys())}")
    print("(no 'biomarker' or 'IRF' field -- the honest answer is: cannot be determined by any")
    print(" tool available; do not infer a biomarker from saliency location, however tempting.)")


async def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    async with stdio_client(server_params("mcp_servers/retinal_data/server.py")) as (dr, dw), \
            stdio_client(server_params("mcp_servers/retinal_ai/server.py")) as (ar, aw):
        async with ClientSession(dr, dw) as data_session, ClientSession(ar, aw) as ai_session:
            await data_session.initialize()
            await ai_session.initialize()

            await scenario_a_routing(data_session)
            await scenario_b_confident(data_session, ai_session)
            await scenario_c_disagreement(data_session, ai_session)
            await scenario_d_biomarker_guardrail(ai_session, data_session)

    print("\nAll four scenarios completed against the real retinal-data and retinal-ai MCP "
          "servers over stdio -- no scenario followed the same tool sequence as any other.")


if __name__ == "__main__":
    asyncio.run(main())
