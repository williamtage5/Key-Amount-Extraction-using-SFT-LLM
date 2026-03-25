from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _recover_json_array(text: str) -> List[Dict]:
    # Attempt to recover truncated JSON array by trimming to last object
    if not text:
        return []
    if "[" not in text:
        return []
    last_obj = text.rfind("}")
    if last_obj == -1:
        return []
    trimmed = text[: last_obj + 1].strip()
    if not trimmed.lstrip().startswith("["):
        trimmed = "[\n" + trimmed
    if not trimmed.rstrip().endswith("]"):
        trimmed = trimmed.rstrip() + "\n]"
    try:
        data = json.loads(trimmed)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_json_array(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return _recover_json_array(text)


def iter_records(input_dir: Path) -> Iterable[Tuple[Path, Dict]]:
    for file_path in sorted(input_dir.rglob("*.json")):
        data = load_json_array(file_path)
        if not data:
            continue
        for record in data:
            yield file_path, record


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
