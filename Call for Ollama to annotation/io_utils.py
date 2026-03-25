from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def read_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    except Exception:
        return []
    return rows


def load_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() != ".json":
        return []

    parsed = read_json_file(path)
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def iter_records(input_dir: Path, exclude_dirs: Optional[List[Path]] = None) -> Iterator[Tuple[Path, Dict]]:
    excludes = [p.resolve() for p in (exclude_dirs or [])]

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        file_resolved = file_path.resolve()
        skip = False
        for ex in excludes:
            if ex == file_resolved or ex in file_resolved.parents:
                skip = True
                break
        if skip:
            continue
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        for record in load_records(file_path):
            yield file_path, record


def write_json_array(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_array(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        pass
    return []


def build_output_path(input_root: Path, output_run_dir: Path, input_file: Path) -> Path:
    strip_first_level = {"input_data"}
    try:
        rel = input_file.relative_to(input_root)
        if rel.parts and rel.parts[0].lower() in strip_first_level:
            rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(input_file.name)
        out = output_run_dir / rel
    except ValueError:
        out = output_run_dir / input_file.name

    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")

    return out
