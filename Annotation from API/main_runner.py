from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, TextIO, Tuple

from tqdm import tqdm

from config import API_KEYS, INPUT_DIR, OUTPUT_ROOT
from extraction_service import extract_target_amount


def iter_records(input_dir: Path) -> Iterable[Tuple[Path, Dict]]:
    for file_path in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            if isinstance(record, dict):
                yield file_path, record


def iter_unique_case_records(input_dir: Path) -> Iterable[Tuple[Path, Dict, Optional[str]]]:
    seen_case_no = set()
    for file_path, record in iter_records(input_dir):
        case_no = record.get("case_no")
        case_no = str(case_no).strip() if case_no is not None else None
        if case_no:
            if case_no in seen_case_no:
                continue
            seen_case_no.add(case_no)
        yield file_path, record, case_no


def get_year_month_from_path(file_path: Path) -> Tuple[str, str]:
    stem = file_path.stem  # e.g. 2013-05
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[0], f"{parts[0]}-{parts[1]}"
    return "unknown", stem


def get_zone_text(record: Dict, zone_key: str) -> str:
    direct = record.get(zone_key)
    if isinstance(direct, str):
        return direct

    zones = record.get("zones")
    if isinstance(zones, dict):
        zone_obj = zones.get(zone_key)
        if isinstance(zone_obj, dict):
            text = zone_obj.get("text")
            if isinstance(text, str):
                return text
        if isinstance(zone_obj, str):
            return zone_obj

    return ""


def load_completed_cases_from_index(index_path: Path) -> set[str]:
    completed = set()
    if not index_path.exists():
        return completed
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                completed.add(line)
    except Exception:
        return set()
    return completed


def load_completed_cases_from_outputs(output_dir: Path) -> set[str]:
    completed = set()
    if not output_dir.exists():
        return completed

    case_no_re = re.compile(r'"case_no"\s*:\s*"([^"]+)"')

    for json_path in output_dir.rglob("*.json"):
        if json_path.name == "run_meta.json":
            continue
        try:
            content = json_path.read_text(encoding="utf-8")
        except Exception:
            continue

        stripped = content.strip()
        if not stripped:
            continue

        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        case_no = item.get("case_no")
                        if case_no:
                            completed.add(str(case_no).strip())
                continue
        except Exception:
            pass

        for m in case_no_re.finditer(content):
            completed.add(m.group(1).strip())

    return completed


def ensure_output_handle(out_path: Path) -> Tuple[TextIO, bool]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        fh = out_path.open("w", encoding="utf-8")
        fh.write("[\n")
        return fh, True

    raw = out_path.read_text(encoding="utf-8")
    stripped = raw.strip()

    if stripped in ("[]", "[\n]"):
        out_path.write_text("[\n", encoding="utf-8")
        fh = out_path.open("a", encoding="utf-8")
        return fh, True

    # If file ends with ], remove it for append.
    if stripped.endswith("]"):
        idx = len(raw) - 1
        while idx >= 0 and raw[idx].isspace():
            idx -= 1
        if idx >= 0 and raw[idx] == "]":
            raw = raw[:idx]

    # If broken/truncated mid-object, truncate to last complete object.
    trimmed = raw.rstrip()
    if trimmed and trimmed[-1] not in ("}", "["):
        last_obj = raw.rfind("}")
        if last_obj != -1:
            raw = raw[: last_obj + 1]
        else:
            raw = "[\n"

    has_object = '"case_no"' in raw
    out_path.write_text(raw, encoding="utf-8")
    fh = out_path.open("a", encoding="utf-8")
    return fh, not has_object


def finalize_output_file(out_path: Path) -> None:
    raw = out_path.read_text(encoding="utf-8")
    if not raw.strip().endswith("]"):
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write("\n]\n")


def count_remaining_records(input_dir: Path, completed_cases: set[str]) -> int:
    count = 0
    for _, _record, case_no in iter_unique_case_records(input_dir):
        if case_no and case_no in completed_cases:
            continue
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference for target amount extraction (resume + dedupe).")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Folder name under data/loan_amount_extraction. If omitted, auto-create timestamp folder.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional full output path (backward compatibility). Prefer --run-name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Target total records in this output-root. Supports resume.",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_root:
        run_root = Path(args.output_root)
        run_name = run_root.name
    elif args.run_name:
        run_name = args.run_name
        run_root = Path(OUTPUT_ROOT) / run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = Path(OUTPUT_ROOT) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    if not API_KEYS:
        raise RuntimeError("API_KEYS is empty. Please check create train dataset/api_keys.json")

    index_path = run_root / "completed_case_no.txt"
    completed_cases = set()
    completed_cases.update(load_completed_cases_from_index(index_path))
    completed_cases.update(load_completed_cases_from_outputs(run_root))

    already_done = len(completed_cases)
    target_total = args.limit
    remaining_target = max(target_total - already_done, 0)
    remaining_available = count_remaining_records(input_dir, completed_cases)
    to_process = min(remaining_target, remaining_available)

    print(f"[Resume] run_name={run_name}")
    print(f"[Resume] output={run_root}")
    print(f"[Resume] completed={already_done}, target_total={target_total}, to_process={to_process}")

    if to_process <= 0:
        print("[Resume] Nothing to do.")
        return

    max_workers = args.max_workers if args.max_workers is not None else len(API_KEYS)
    max_workers = max(1, min(max_workers, len(API_KEYS), to_process))
    key_cycle = cycle(API_KEYS)

    handles: Dict[Path, TextIO] = {}
    first_flags: Dict[Path, bool] = {}
    stats = {"success": 0, "failed": 0, "skipped": 0}

    pbar = tqdm(total=to_process, desc="Batch amount infer", unit="sample")
    futures = {}
    scheduled_case_no = set()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scheduled = 0
            for file_path, record, case_no in iter_unique_case_records(input_dir):
                if case_no and case_no in completed_cases:
                    continue
                if case_no and case_no in scheduled_case_no:
                    continue

                z4_text = get_zone_text(record, "Z4_Reasoning")
                z3_text = get_zone_text(record, "Z3_Fact")
                assigned_key = next(key_cycle)
                future = executor.submit(extract_target_amount, z4_text, z3_text, assigned_key)
                futures[future] = (file_path, record, case_no)
                if case_no:
                    scheduled_case_no.add(case_no)

                scheduled += 1
                if scheduled >= to_process:
                    break

            for future in as_completed(futures):
                file_path, record, case_no = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "data": {"target_amount": "", "amount_type": "unknown", "source_zone": "none"},
                        "error_msg": str(exc),
                    }

                status = result.get("status", "failed")
                if status == "success":
                    stats["success"] += 1
                elif status == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1

                data = result.get("data", {})

                year, year_month = get_year_month_from_path(file_path)
                year_dir = run_root / year
                year_dir.mkdir(parents=True, exist_ok=True)
                out_path = year_dir / f"{year_month}.json"

                if out_path not in handles:
                    fh, is_first = ensure_output_handle(out_path)
                    handles[out_path] = fh
                    first_flags[out_path] = is_first

                out_record = {
                    "case_no": record.get("case_no"),
                    "target_amount": data.get("target_amount", ""),
                    "amount_type": data.get("amount_type", "unknown"),
                    "source_zone": data.get("source_zone", "none"),
                    "status": status,
                }

                if not first_flags[out_path]:
                    handles[out_path].write(",\n")
                handles[out_path].write(json.dumps(out_record, ensure_ascii=False))
                first_flags[out_path] = False

                # Success/skipped are deterministic enough to mark completed.
                if case_no and status in ("success", "skipped"):
                    with index_path.open("a", encoding="utf-8") as idx:
                        idx.write(f"{case_no}\n")
                    completed_cases.add(case_no)

                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={stats['success']} fail={stats['failed']} skip={stats['skipped']}"
                )
    finally:
        for out_path, fh in handles.items():
            fh.close()
            finalize_output_file(out_path)
        pbar.close()

    run_meta = {
        "input_dir": str(input_dir),
        "run_root": str(run_root),
        "run_name": run_name,
        "limit": target_total,
        "max_workers": max_workers,
        "completed_before": already_done,
        "processed_this_run": to_process,
        "completed_after": len(completed_cases),
        "stats": stats,
    }
    (run_root / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        latest_path = Path(OUTPUT_ROOT) / "latest_run.txt"
        latest_path.write_text(run_name, encoding="utf-8")
    except Exception:
        pass

    print(
        f"Done. processed={to_process} success={stats['success']} failed={stats['failed']} "
        f"skipped={stats['skipped']} completed_after={len(completed_cases)} run_name={run_name} output={run_root}"
    )


if __name__ == "__main__":
    main()
