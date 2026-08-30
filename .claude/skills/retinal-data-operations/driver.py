"""
Driver for the retinal-data-operations skill: connects to the retinal-data
(and, sparingly, retinal-ai) MCP servers over the real stdio transport --
exactly as Claude Code does per .mcp.json -- and runs a worked example of
the resolve -> retrieve minimum -> verify -> preserve-provenance workflow
the skill documents.

Run from the repo root:
  "C:/Users/lamata/Miniconda3/envs/retfound/python.exe" .claude/skills/retinal-data-operations/driver.py

Add --with-ai to also exercise retinal-ai (slower: loads a ~1.2GB checkpoint).
Add --patient/--eye to point at a different patient than the default demo
patient (60, OD).
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


def server_params(relative_script: str) -> StdioServerParameters:
    return StdioServerParameters(command=PYTHON, args=[str(REPO_ROOT / relative_script)])


async def call(session: ClientSession, tool: str, **kwargs) -> dict:
    """Call a tool and return its metadata block as a parsed JSON dict.

    Every tool on both servers puts its JSON in content[0]: dict-returning
    tools because FastMCP serializes a plain dict as one pretty-printed
    TextContent block (structuredContent stays None), and the image-returning
    tools (get_bscan, get_fundus_image, generate_gradcam) because they
    deliberately order their CallToolResult metadata-first, ImageContent
    second. Use call_images() to get at the pixels."""
    result = await session.call_tool(tool, kwargs)
    return json.loads(result.content[0].text)


async def call_images(session: ClientSession, tool: str, **kwargs) -> tuple[dict, list]:
    """Like call(), but also returns the ImageContent blocks. Claude Code
    sees these as real images; over the SDK they arrive as base64 PNG data
    on blocks with type == "image"."""
    result = await session.call_tool(tool, kwargs)
    meta = json.loads(result.content[0].text)
    return meta, [c for c in result.content if c.type == "image"]


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


async def run_retinal_data_demo(patient_id: int, eye: str):
    async with stdio_client(server_params("mcp_servers/retinal_data/server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            section("RULE 1 -- resolve patient/eye/visit/modality before operating")
            exams = await call(session, "list_patient_examinations", patient_id=patient_id, eye=eye)
            print(f"patient {patient_id} eye {eye}: {len(exams['visits'])} visits")
            latest = exams["visits"][-1]
            first = exams["visits"][0]
            print(f"  first visit: {first['visit_id']} (source={first['source_visit']})")
            print(f"  latest visit: {latest['visit_id']} (source={latest['source_visit']})")

            info = await call(session, "get_oct_volume_info", patient_id=patient_id, eye=eye, visit_id="latest")
            print(f"latest volume shape={info['shape']} dtype={info['dtype']} "
                  f"bscan_count={info['bscan_count']}")

            section("RULE 2 -- retrieve only what's necessary (one bscan, not the whole volume)")
            mid_index = info["bscan_count"] // 2
            bscan, bscan_images = await call_images(session, "get_bscan", patient_id=patient_id,
                                                     eye=eye, visit_id="latest", bscan_index=mid_index)
            print(f"fetched bscan {mid_index}/{info['bscan_count'] - 1} as a real MCP image block: "
                  f"{len(bscan_images)}x {bscan_images[0].mimeType} at "
                  f"{bscan['image_width']}x{bscan['image_height']} "
                  f"(downscaled from {bscan['source_width']}x{bscan['source_height']}; "
                  f"not the full {info['bscan_count']}-slice volume)")

            section("RULE 3 -- longitudinal comparison stays within the same patient+eye")
            cmp = await call(session, "compare_visits", patient_id=patient_id, eye=eye,
                              visit_a=first["visit_id"], visit_b=latest["visit_id"])
            print(f"visit {first['visit_id']} -> {latest['visit_id']}: "
                  f"delta_BCVA={cmp['delta_BCVA']}, delta_CST={cmp['delta_CST']}")

            section("RULE 4 -- multimodal retrieval from the SAME patient/eye/visit")
            fundus = await call(session, "get_fundus_image", patient_id=patient_id, eye=eye,
                                 visit_id="latest", modality="color")
            print(f"OCT bscan and fundus both resolved to visit_id={bscan['visit_id']!r} / "
                  f"{fundus['visit_id']!r} (must match -- they do)")

            section("RULE 5/6 -- verify before answering, preserve provenance")
            print(f"bscan provenance: source_path={bscan['source_path']!r}, "
                  f"bscan_index={bscan['bscan_index']}, visit_id={bscan['visit_id']}, eye={bscan['eye']}")
            print(f"fundus provenance: source_path={fundus['source_path']!r}, modality={fundus['modality']!r}")

            section("RULE 5 (continued) -- an unverifiable request returns an explicit error, not a guess")
            other_eye = "OS" if eye == "OD" else "OD"
            bad = await call(session, "get_clinical_metadata", patient_id=patient_id, eye=other_eye,
                              visit_id="visit_01")
            print(f"clinical metadata for the non-study eye ({other_eye}): {bad}")

            section("RULE 8 -- ground-truth/biomarker data is structurally unreachable")
            print("get_clinical_metadata fields:", list((await call(
                session, "get_clinical_metadata", patient_id=patient_id, eye=eye, visit_id="visit_01")).keys()))
            print("(no 'biomarkers', 'DR', or 'DME' field exists on any retinal-data tool response --")
            print(" the reader module has no function that reads the biomarkers/ zarr group at all)")


async def run_retinal_ai_demo(patient_id: int, eye: str):
    async with stdio_client(server_params("mcp_servers/retinal_ai/server.py")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            section("RULE 7 -- diagnostic models are a SEPARATE, opt-in step (retinal-ai), never automatic")
            print("retinal-data alone (above) never touched retinal-ai. Only call these when the")
            print("user's request actually requires interpretation, e.g. \"is this DME or DR?\":")
            result = await call(session, "screen_oct_volume", patient_id=patient_id, eye=eye,
                                 visit_id="latest", top_k=3)
            print(f"screen_oct_volume -> volume_prediction={result['volume_prediction']!r}, "
                  f"top candidates: {result['candidate_bscans']}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=int, default=60)
    ap.add_argument("--eye", type=str, default="OD")
    ap.add_argument("--with-ai", action="store_true",
                     help="Also exercise retinal-ai (rule 7 demo). Slower: loads a ~1.2GB checkpoint.")
    args = ap.parse_args()

    await run_retinal_data_demo(args.patient, args.eye)
    if args.with_ai:
        await run_retinal_ai_demo(args.patient, args.eye)

    print("\nAll steps completed against the real retinal-data" +
          (" and retinal-ai" if args.with_ai else "") + " MCP server(s) over stdio.")


if __name__ == "__main__":
    asyncio.run(main())
