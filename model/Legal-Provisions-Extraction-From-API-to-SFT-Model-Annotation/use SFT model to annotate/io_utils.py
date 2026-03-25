from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_json_array(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def iter_records(input_dir: Path) -> Iterable[Tuple[Path, Dict]]:
    for file_path in sorted(input_dir.rglob("*.json")):
        data = load_json_array(file_path)
        if not data:
            continue
        for record in data:
            yield file_path, record


def ensure_output_path(output_root: Path, input_file: Path) -> Path:
    year = input_file.parent.name
    stem = input_file.stem  # e.g., 2013-05
    year_dir = output_root / year
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir / f"{stem}.json"


def write_json_array(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
