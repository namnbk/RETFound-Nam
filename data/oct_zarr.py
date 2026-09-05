"""
Reusable read/write layer for the OLIVES per-patient Zarr v3 stores
(patient_<id>.zarr under olives_zarr/).

Writer functions build the on-disk layout consumed by build_olives_zarr.py's
CLI. Reader functions back the retinal-data and retinal-ai MCP servers
(mcp_servers/retinal_data/server.py, mcp_servers/retinal_ai/server.py).

Reader functions deliberately never touch the biomarkers/ group or any
DR/DME ground-truth label -- there is no function anywhere in this module
that reads it, by design, since those must stay evaluator-only for the
agent benchmark this supports.

On-disk structure (see build_patient_zarr()):
  patient_<id>.zarr
    zarr.json                     -- attrs: {patient_id, sex, age, source_dataset, source_folder}
    eyes/<OD|OS>/visit_NN/        -- attrs: {source_visit, eye}; NN is a zero-padded
                                      sequential index, not the raw source visit label
      oct_image                   -- array (B, H, W) uint8, chunks (1, H, W)
                                      attrs: {dims, n_bscans, bscan_source_files}
      fundus/near-IR              -- array (H, W) uint8, chunks = full shape (may be absent)
      fundus/color                -- array (H, W, 3) uint8, chunks = full shape (may be absent)
      clinical                    -- group, attrs: {bcva, cst_um, eye_id} (only the study eye)
      biomarkers/values           -- array (B, 16) int8 (only first/last visit, some builds)
                                      -- NEVER read by any function below
"""
import base64
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

# ---------------------------------------------------------------------------
# Writer side (moved from build_olives_zarr.py; bodies unchanged)
# ---------------------------------------------------------------------------

BIOMARKER_COLUMNS = [
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM",
]

TRAILING_DIGITS = re.compile(r"(\d+)(?=\.[^.]+$)", re.IGNORECASE)
IMAGE_GLOBS = ("*.tif", "*.tiff", "*.jpg", "*.jpeg")


def visit_sort_key(visit_dir_name: str) -> int:
    m = re.search(r"\d+", visit_dir_name)
    return int(m.group()) if m else 0


def get_visits(patient_dir: Path):
    """Return [(visit_label, visit_path), ...] sorted chronologically."""
    visits = [p for p in patient_dir.iterdir() if p.is_dir()]
    visits.sort(key=lambda p: visit_sort_key(p.name))
    return [(p.name, p) for p in visits]


def classify_eye_files(eye_dir: Path):
    """Split a visit/eye folder's image files into bscans / near-IR / color fundus.

    Discriminated by (size, mode) rather than filename, since Prime_FULL uses
    bare digits (0.tif..48.tif, fundus_OD_W0.tif, fundus_W0.tif) while TREX DME
    uses prefixed 6-digit indices (TREXU_000000.tif.., fundus_OD_V1.tif,
    TREXU_001.tif) for the same three roles. Most visits use .tif; a handful
    (e.g. patient 209's V2/V3/V5/V6/V8/V10) use .jpg for the same content.
    """
    bscans = []  # (index, path)
    near_ir = None
    color = None

    image_paths = [p for pattern in IMAGE_GLOBS for p in eye_dir.glob(pattern)]
    for img_path in image_paths:
        with Image.open(img_path) as im:
            size, mode = im.size, im.mode

        if size == (768, 768) and mode == "RGB":
            color = img_path
        elif size == (768, 768) and mode == "L":
            near_ir = img_path
        else:
            m = TRAILING_DIGITS.search(img_path.name)
            if m is None:
                continue
            bscans.append((int(m.group()), img_path))

    bscans.sort(key=lambda t: t[0])
    return bscans, near_ir, color


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_clinical(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="clinical_data_images")
    parts = df["File_Path"].str.strip("/").str.split("/")
    df["visit"] = parts.str[-3]
    df["eye"] = parts.str[-2]
    df["folder_basename"] = parts.str[-4]
    return df


def load_biomarkers(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parts = df["Path (Trial/Arm/Folder/Visit/Eye/Image Name)"].str.strip("/").str.split("/")
    df["visit"] = parts.str[-3]
    df["eye"] = parts.str[-2]
    df["folder_basename"] = parts.str[-4]
    return df


def build_patient_zarr(patient_id: int, patient_dir: Path, clinical_df: pd.DataFrame,
                        biomarker_df: pd.DataFrame | None, out_dir: Path, overwrite: bool):
    out_path = out_dir / f"patient_{patient_id}.zarr"
    store = zarr.storage.LocalStore(str(out_path))
    root = zarr.create_group(store, overwrite=overwrite)
    root.attrs.update({
        "patient_id": patient_id,
        "sex": "",
        "age": "",
        "source_dataset": "OLIVES",
        "source_folder": patient_dir.name,
    })

    eyes_group = root.create_group("eyes")
    visits = get_visits(patient_dir)
    # Match by Patient_ID *and* source folder basename: a patient can have more
    # than one independent enrollment (e.g. 234's GILA and TREX arms) whose
    # visits are numbered independently, so Patient_ID alone is ambiguous.
    pid_clinical = clinical_df[(clinical_df["Patient_ID"] == patient_id) &
                                (clinical_df["folder_basename"] == patient_dir.name)]
    if biomarker_df is not None:
        pid_biomarker = biomarker_df[(biomarker_df["Patient_ID"] == patient_id) &
                                      (biomarker_df["folder_basename"] == patient_dir.name)]
    else:
        pid_biomarker = None

    for eye in ("OD", "OS"):
        eye_group = eyes_group.create_group(eye)
        visit_num = 0

        for visit_label, visit_path in visits:
            eye_dir = visit_path / eye
            if not eye_dir.is_dir():
                continue

            bscans, near_ir_path, color_path = classify_eye_files(eye_dir)
            if not bscans:
                continue
            visit_num += 1

            visit_group = eye_group.create_group(f"visit_{visit_num:02d}")
            visit_group.attrs.update({"source_visit": visit_label, "eye": eye})

            stack = np.stack([load_gray(p) for _, p in bscans], axis=0)
            b, h, w = stack.shape
            oct_arr = visit_group.create_array(
                "oct_image", shape=stack.shape, chunks=(1, h, w), dtype=stack.dtype,
            )
            oct_arr[:] = stack
            oct_arr.attrs.update({
                "dims": ["bscan", "height", "width"],
                "n_bscans": b,
                "bscan_source_files": [p.name for _, p in bscans],
            })

            fundus_group = visit_group.create_group("fundus")
            if near_ir_path is not None:
                img = load_gray(near_ir_path)
                arr = fundus_group.create_array(
                    "near-IR", shape=img.shape, chunks=img.shape, dtype=img.dtype,
                )
                arr[:] = img
                arr.attrs["source_file"] = near_ir_path.name
            if color_path is not None:
                img = load_rgb(color_path)
                arr = fundus_group.create_array(
                    "color", shape=img.shape, chunks=img.shape, dtype=img.dtype,
                )
                arr[:] = img
                arr.attrs["source_file"] = color_path.name

            crow = pid_clinical[(pid_clinical["visit"] == visit_label) & (pid_clinical["eye"] == eye)]
            if len(crow):
                r = crow.iloc[0]
                clinical_group = visit_group.create_group("clinical")
                clinical_group.attrs.update({
                    "bcva": float(r["BCVA"]),
                    "cst_um": float(r["CST"]),
                    "eye_id": int(r["Eye_ID"]),
                })

            brows = pid_biomarker[(pid_biomarker["visit"] == visit_label) & (pid_biomarker["eye"] == eye)] \
                if pid_biomarker is not None else pid_biomarker
            if brows is not None and len(brows):
                brows = brows.sort_values("Scan (n/49)")
                values = brows[BIOMARKER_COLUMNS].to_numpy(dtype=np.int8)
                biomarkers_group = visit_group.create_group("biomarkers")
                values_arr = biomarkers_group.create_array(
                    "values", shape=values.shape, chunks=values.shape, dtype=values.dtype,
                )
                values_arr[:] = values
                values_arr.attrs.update({
                    "dims": ["bscan", "biomarker"],
                    "labels": BIOMARKER_COLUMNS,
                    "bscan_index_note": "row i corresponds to bscan index i in oct_image (Scan (n/49) is 1-indexed)",
                })

    return out_path


# ---------------------------------------------------------------------------
# Reader side
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZARR_DIR = os.environ.get("OCT_ZARR_DIR", os.path.join(REPO_ROOT, "olives_zarr"))

VALID_EYES = ("OD", "OS")


class OctZarrError(ValueError):
    """Raised for any patient/eye/visit/bscan lookup that can't be resolved.
    Kept as a single plain exception type so MCP tool wrappers can catch one
    thing and turn it into a readable error string."""


def _zarr_dir(zarr_dir=None) -> Path:
    return Path(zarr_dir) if zarr_dir else Path(DEFAULT_ZARR_DIR)


def list_patient_ids(zarr_dir=None) -> list[int]:
    """Discover which patient_<id>.zarr stores exist in zarr_dir."""
    ids = []
    for p in _zarr_dir(zarr_dir).glob("patient_*.zarr"):
        m = re.match(r"patient_(\d+)\.zarr$", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def open_patient_store(patient_id, zarr_dir=None) -> zarr.Group:
    store_path = _zarr_dir(zarr_dir) / f"patient_{patient_id}.zarr"
    if not store_path.exists():
        available = list_patient_ids(zarr_dir)
        raise OctZarrError(f"No Zarr store for patient_id={patient_id!r} at {store_path}. "
                            f"Available patient ids: {available}")
    return zarr.open_group(zarr.storage.LocalStore(str(store_path)), mode="r")


def get_patient_info(patient_id, zarr_dir=None) -> dict:
    root = open_patient_store(patient_id, zarr_dir)
    return dict(root.attrs)


def normalize_eye(eye: str) -> str:
    eye_u = str(eye).upper()
    if eye_u not in VALID_EYES:
        raise OctZarrError(f"eye must be one of {VALID_EYES}, got {eye!r}")
    return eye_u


def list_eyes(patient_id, zarr_dir=None) -> list[str]:
    root = open_patient_store(patient_id, zarr_dir)
    eyes_group = root["eyes"]
    return [eye for eye in VALID_EYES if eye in eyes_group and len(list(eyes_group[eye].keys())) > 0]


def _get_eye_group(patient_id, eye, zarr_dir=None):
    root = open_patient_store(patient_id, zarr_dir)
    eye = normalize_eye(eye)
    eyes_group = root["eyes"]
    if eye not in eyes_group or len(list(eyes_group[eye].keys())) == 0:
        available = list_eyes(patient_id, zarr_dir)
        raise OctZarrError(f"No visits for patient {patient_id} eye {eye}. "
                            f"Eyes with data: {available}")
    return eyes_group[eye]


def _visit_modalities(visit_group) -> list[str]:
    modalities = ["oct"]
    if "fundus" in visit_group:
        fundus_group = visit_group["fundus"]
        if "near-IR" in fundus_group:
            modalities.append("fundus_near_ir")
        if "color" in fundus_group:
            modalities.append("fundus_color")
    return modalities


def list_visits(patient_id, eye, zarr_dir=None) -> list[dict]:
    eye_group = _get_eye_group(patient_id, eye, zarr_dir)
    out = []
    for visit_id in sorted(eye_group.keys()):
        vg = eye_group[visit_id]
        out.append({
            "visit_id": visit_id,
            "source_visit": vg.attrs.get("source_visit"),
            "modalities": _visit_modalities(vg),
            "has_clinical": "clinical" in vg,
        })
    return out


def resolve_visit_id(eye_group, visit_id) -> str:
    """Accepts an exact "visit_NN", the keywords "latest"/"first", or a raw
    source-visit label (e.g. "W0"/"V1") and returns the canonical "visit_NN"."""
    visit_ids = sorted(eye_group.keys())
    if not visit_ids:
        raise OctZarrError("This eye has no visits.")

    if visit_id in visit_ids:
        return visit_id
    if visit_id == "latest":
        return visit_ids[-1]
    if visit_id == "first":
        return visit_ids[0]

    for vid in visit_ids:
        if eye_group[vid].attrs.get("source_visit") == visit_id:
            return vid

    labels = [(vid, eye_group[vid].attrs.get("source_visit")) for vid in visit_ids]
    raise OctZarrError(
        f"Unknown visit_id {visit_id!r}. Use 'latest', 'first', an exact visit id, "
        f"or a source visit label. Available: {labels}")


def _get_visit_group(patient_id, eye, visit_id, zarr_dir=None):
    eye_group = _get_eye_group(patient_id, eye, zarr_dir)
    resolved = resolve_visit_id(eye_group, visit_id)
    return eye_group[resolved], resolved


def get_oct_volume_info(patient_id, eye, visit_id, zarr_dir=None) -> dict:
    visit_group, resolved_visit_id = _get_visit_group(patient_id, eye, visit_id, zarr_dir)
    oct_arr = visit_group["oct_image"]
    return {
        "visit_id": resolved_visit_id,
        "shape": list(oct_arr.shape),
        "dtype": str(oct_arr.dtype),
        "bscan_count": int(oct_arr.shape[0]),
        "zarr_path": str(_zarr_dir(zarr_dir) / f"patient_{patient_id}.zarr" /
                          "eyes" / normalize_eye(eye) / resolved_visit_id / "oct_image"),
        "acquisition_metadata": {
            "source_visit": visit_group.attrs.get("source_visit"),
            "eye": visit_group.attrs.get("eye"),
            "source_dataset": "OLIVES",
        },
    }


def get_bscan_array(patient_id, eye, visit_id, bscan_index, zarr_dir=None):
    """Returns (np.ndarray (H,W) uint8, source_filename: str, resolved_visit_id: str).

    The resolved visit_id is the canonical "visit_NN" -- callers should
    report THAT as provenance, not whatever alias ("latest", a raw source
    label) the caller passed in, since an alias isn't a stable identifier."""
    visit_group, resolved_visit_id = _get_visit_group(patient_id, eye, visit_id, zarr_dir)
    oct_arr = visit_group["oct_image"]
    n_bscans = oct_arr.shape[0]
    if not (0 <= bscan_index < n_bscans):
        raise OctZarrError(f"bscan_index {bscan_index} out of range for volume with "
                            f"{n_bscans} bscans (valid: 0-{n_bscans - 1})")
    source_files = oct_arr.attrs.get("bscan_source_files", [])
    source_file = source_files[bscan_index] if bscan_index < len(source_files) else None
    return np.asarray(oct_arr[bscan_index]), source_file, resolved_visit_id


def get_oct_volume_array(patient_id, eye, visit_id, start_bscan=None, end_bscan=None, zarr_dir=None):
    """Returns (np.ndarray, resolved_visit_id: str) -- see get_bscan_array's
    docstring on why the resolved id, not the input alias, is the one to
    report as provenance."""
    visit_group, resolved_visit_id = _get_visit_group(patient_id, eye, visit_id, zarr_dir)
    oct_arr = visit_group["oct_image"]
    n_bscans = oct_arr.shape[0]
    start = 0 if start_bscan is None else start_bscan
    end = n_bscans if end_bscan is None else end_bscan
    if not (0 <= start < end <= n_bscans):
        raise OctZarrError(f"Invalid range start_bscan={start_bscan}, end_bscan={end_bscan} "
                            f"for volume with {n_bscans} bscans")
    return np.asarray(oct_arr[start:end]), resolved_visit_id


def get_fundus_array(patient_id, eye, visit_id, modality="color", zarr_dir=None):
    """Returns (np.ndarray, source_filename: str, resolved_visit_id: str). modality is "color" or "near-IR"."""
    visit_group, resolved_visit_id = _get_visit_group(patient_id, eye, visit_id, zarr_dir)
    if "fundus" not in visit_group:
        raise OctZarrError(f"No fundus images for this visit.")
    fundus_group = visit_group["fundus"]
    if modality not in fundus_group:
        available = list(fundus_group.keys())
        raise OctZarrError(f"modality {modality!r} not available for this visit. "
                            f"Available: {available}")
    arr = fundus_group[modality]
    return np.asarray(arr[:]), arr.attrs.get("source_file"), resolved_visit_id


def get_clinical(patient_id, eye, visit_id, zarr_dir=None):
    """Reads ONLY the clinical group's bcva/cst_um/eye_id. Returns
    (None, resolved_visit_id) if this visit has no clinical data (expected
    for the non-study eye -- every patient has clinical data on exactly one
    eye, not both), else (dict, resolved_visit_id)."""
    visit_group, resolved_visit_id = _get_visit_group(patient_id, eye, visit_id, zarr_dir)
    if "clinical" not in visit_group:
        return None, resolved_visit_id
    return dict(visit_group["clinical"].attrs), resolved_visit_id


def compare_visits(patient_id, eye, visit_a, visit_b, zarr_dir=None) -> dict:
    clinical_a, resolved_a = get_clinical(patient_id, eye, visit_a, zarr_dir)
    clinical_b, resolved_b = get_clinical(patient_id, eye, visit_b, zarr_dir)
    if clinical_a is None or clinical_b is None:
        missing = visit_a if clinical_a is None else visit_b
        raise OctZarrError(f"No clinical data for visit {missing!r} on patient {patient_id} eye {eye}.")
    return {
        "visit_a": resolved_a,
        "visit_b": resolved_b,
        "BCVA_a": clinical_a["bcva"],
        "BCVA_b": clinical_b["bcva"],
        "CST_a": clinical_a["cst_um"],
        "CST_b": clinical_b["cst_um"],
        "delta_BCVA": clinical_b["bcva"] - clinical_a["bcva"],
        "delta_CST": clinical_b["cst_um"] - clinical_a["cst_um"],
    }


# ---------------------------------------------------------------------------
# Provenance disclosure (shared by both MCP servers)
# ---------------------------------------------------------------------------

# TEMPORARY WITHHOLDING. OLIVES B-scan filenames for patients 206/209/211/220/
# 251 look like "TREX_DME_000024.tif" -- the diagnosis is spelled out in the
# filename. Returning that as provenance hands the answer to anything being
# asked "is this DR or DME", through the tool surface itself, with no file
# access needed. Until the filenames are sanitized at build time (or the eval
# stops asking that question), the field is withheld.
#
# Set RETINAL_EXPOSE_SOURCE_PATH=1 in the MCP server's environment to restore
# it -- nothing else needs editing. This is the whole switch.
EXPOSE_SOURCE_PATH = os.environ.get("RETINAL_EXPOSE_SOURCE_PATH", "0").strip().lower() \
    not in ("0", "", "false", "no", "off")


def source_path_field(source_file) -> dict:
    """Returns `{"source_path": ...}` when disclosure is enabled, else `{}`.

    Spread this into a tool's metadata dict (`**oz.source_path_field(f)`) so
    the key is *absent* rather than present-and-null when withheld -- a null
    still advertises that a filename exists and invites a caller to go hunting
    for it, which is the opposite of the point."""
    return {"source_path": source_file} if EXPOSE_SOURCE_PATH else {}


# ---------------------------------------------------------------------------
# Image codec helpers (shared by both MCP servers)
# ---------------------------------------------------------------------------

# The classifier's input resolution. Images handed to an LLM are downscaled to
# this by default: a raw 496x504 B-scan is ~151K base64 chars (~38K tokens) and
# carries no detail the 224x224 model input preserves anyway, while the same
# scan at 224 costs ~70 image tokens. See downscale_for_display().
MODEL_INPUT_SIZE = 224


def downscale_for_display(img: Image.Image, max_side: int = MODEL_INPUT_SIZE) -> Image.Image:
    """Shrink `img` so its longest side is at most `max_side`, preserving
    aspect ratio. Never upscales -- an image already within the bound is
    returned unchanged.

    This is a plain aspect-preserving resize, NOT the model's eval transform
    (which resizes the short side to 256 then center-crops to 224). Callers
    that need the exact pixels the classifier saw should render from the
    transformed tensor instead (see grad_cam_retfound.unnormalize_to_uint8);
    this is for giving a viewer something to look at."""
    if max_side <= 0:
        raise OctZarrError(f"max_side must be positive, got {max_side}")
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def array_to_base64_png(arr: np.ndarray) -> str:
    return pil_to_base64_png(Image.fromarray(arr))


def base64_png_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def array_to_base64_npy(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")
