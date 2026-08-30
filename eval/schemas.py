"""JSON Schemas for each task's structured final answer, passed to `claude -p
--json-schema` at run time and reused by grade.py to know which fields to
compare. Kept intentionally minimal/strict (additionalProperties: false) so
a malformed or evasive answer fails schema validation rather than needing
lenient parsing."""

DISEASE_ENUM = {"type": "string", "enum": ["DR", "DME"]}
ASSESSMENT_ENUM = {"type": "string", "enum": ["DR", "DME", "uncertain"]}
DIRECTION_ENUM = {"type": "string", "enum": ["improved", "worsened", "unchanged"]}

TASK_SCHEMAS = {
    1: {
        "type": "object",
        "properties": {
            "patient_id": {"type": "integer"},
            "eye": {"type": "string"},
            "visit_id": {"type": "string"},
            "bscan_index": {"type": "integer"},
        },
        "required": ["patient_id", "eye", "visit_id", "bscan_index"],
        "additionalProperties": False,
    },
    2: {
        "type": "object",
        "properties": {
            "visit_id": {"type": "string"},
            "bscan_index": {"type": "integer"},
        },
        "required": ["visit_id", "bscan_index"],
        "additionalProperties": False,
    },
    3: {
        "type": "object",
        "properties": {
            "first_visit_id": {"type": "string"},
            "latest_visit_id": {"type": "string"},
            "bcva_first": {"type": "number"},
            "cst_first": {"type": "number"},
            "bcva_latest": {"type": "number"},
            "cst_latest": {"type": "number"},
        },
        "required": ["first_visit_id", "latest_visit_id", "bcva_first", "cst_first", "bcva_latest", "cst_latest"],
        "additionalProperties": False,
    },
    4: {
        "type": "object",
        "properties": {
            "bscan_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
        },
        "required": ["bscan_indices"],
        "additionalProperties": False,
    },
    5: {
        "type": "object",
        "properties": {"disease": DISEASE_ENUM},
        "required": ["disease"],
        "additionalProperties": False,
    },
    6: {
        "type": "object",
        "properties": {
            "disease": DISEASE_ENUM,
            "bscan_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
        },
        "required": ["disease", "bscan_indices"],
        "additionalProperties": False,
    },
    7: {
        "type": "object",
        "properties": {
            "disease": DISEASE_ENUM,
            "bscan_index": {"type": "integer"},
        },
        "required": ["disease", "bscan_index"],
        "additionalProperties": False,
    },
    8: {
        "type": "object",
        "properties": {
            "bscan_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
            "disease": DISEASE_ENUM,
        },
        "required": ["bscan_indices", "disease"],
        "additionalProperties": False,
    },
    9: {
        "type": "object",
        "properties": {
            "visit_id": {"type": "string"},
            "disease": DISEASE_ENUM,
            "bscan_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
        },
        "required": ["visit_id", "disease", "bscan_indices"],
        "additionalProperties": False,
    },
    10: {
        "type": "object",
        "properties": {
            "first_visit_id": {"type": "string"},
            "latest_visit_id": {"type": "string"},
            "bcva_direction": DIRECTION_ENUM,
            "cst_direction": DIRECTION_ENUM,
        },
        "required": ["first_visit_id", "latest_visit_id", "bcva_direction", "cst_direction"],
        "additionalProperties": False,
    },
    11: {
        "type": "object",
        "properties": {
            "disease": DISEASE_ENUM,
            "bscan_indices": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5},
        },
        "required": ["disease", "bscan_indices"],
        "additionalProperties": False,
    },
    12: {
        "type": "object",
        "properties": {
            "assessment": ASSESSMENT_ENUM,
            "evidence_bscans": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3},
        },
        "required": ["assessment", "evidence_bscans"],
        "additionalProperties": False,
    },
}

TASK_META = {
    1: ("2. OLIVES retrieval/navigation", "Exact B-scan retrieval"),
    2: ("2. OLIVES retrieval/navigation", "Latest-visit retrieval"),
    3: ("2. OLIVES retrieval/navigation", "Longitudinal retrieval"),
    4: ("2. OLIVES retrieval/navigation", "Find abnormal/evidence-rich slices"),
    5: ("3. Disease assessment + evidence grounding", "DR vs DME assessment"),
    6: ("3. Disease assessment + evidence grounding", "Disease + supporting slices"),
    7: ("3. Disease assessment + evidence grounding", "Disease + saliency evidence"),
    8: ("3. Disease assessment + evidence grounding", "Evidence-first assessment"),
    9: ("4. Compound workflow", "Latest exam -> diagnosis -> evidence"),
    10: ("4. Compound workflow", "Longitudinal comparison"),
    11: ("4. Compound workflow", "Evidence verification"),
    12: ("4. Compound workflow", "Uncertain-case evidence escalation"),
}
