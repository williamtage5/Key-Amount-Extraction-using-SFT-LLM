from __future__ import annotations

import json
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract first JSON object from text."""
    if not text:
        return {"citations": []}
    # strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # prefer object, fallback to array
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")

    block = None
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start and (obj_start == -1 or arr_start < obj_start):
        block = cleaned[arr_start : arr_end + 1]
    elif obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        block = cleaned[obj_start : obj_end + 1]
    else:
        return {"citations": []}
    try:
        obj = json.loads(block)
        if isinstance(obj, dict) and "citations" in obj:
            return obj
        if isinstance(obj, list):
            return {"citations": obj}
    except Exception:
        pass
    return {"citations": []}
