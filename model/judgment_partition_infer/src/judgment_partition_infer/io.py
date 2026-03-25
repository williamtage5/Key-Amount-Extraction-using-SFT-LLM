from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_text(record: Dict) -> str:
    return record.get("text") or record.get("full_text") or ""


def passthrough_fields(record: Dict) -> Dict:
    out: Dict = {}
    for k in ("sample_id", "case_no", "case_name"):
        if k in record and record.get(k) is not None:
            out[k] = record.get(k)
    return out

