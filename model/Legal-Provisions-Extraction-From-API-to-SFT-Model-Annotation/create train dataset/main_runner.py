import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
from pathlib import Path

from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_ROOT, API_KEYS
from extraction_service import extract_citations


def iter_records(input_dir):
    files = sorted(input_dir.rglob("*.json"))
    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            yield file_path, record


def get_year_month_from_path(file_path: Path):
    stem = file_path.stem  # e.g., 2013-05
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[0], f"{parts[0]}-{parts[1]}"
    return "unknown", stem


def count_valid_records(input_dir, completed_cases=None):
    total = 0
    completed_cases = completed_cases or set()
    for _, record in iter_records(input_dir):
        z4_text = record.get("Z4_Reasoning") or ""
        if not z4_text.strip():
            continue
        case_no = record.get("case_no")
        if case_no and case_no in completed_cases:
            continue
        total += 1
    return total


def load_completed_cases_from_outputs(output_dir: Path):
    completed = set()
    if not output_dir.exists():
        return completed

    for json_path in output_dir.rglob("*.json"):
        try:
            content = json_path.read_text(encoding="utf-8")
            stripped = content.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if isinstance(data, list):
                    for item in data:
                        case_no = item.get("case_no")
                        if case_no:
                            completed.add(case_no)
                    continue
            except Exception:
                pass
            # Fallback for broken JSON: regex extract case_no
            import re

            for match in re.finditer(r'"case_no"\s*:\s*"([^"]+)"', content):
                completed.add(match.group(1))
        except Exception:
            continue
    return completed


def load_completed_cases_from_index(index_path: Path):
    completed = set()
    if not index_path.exists():
        return completed
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            completed.add(line)
    except Exception:
        return completed
    return completed


def ensure_output_handle(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not out_path.exists():
        fh = out_path.open("w", encoding="utf-8")
        fh.write("[\n")
        return fh, True

    raw = out_path.read_text(encoding="utf-8")
    stripped = raw.strip()
    is_empty_array = stripped in ("[]", "[\n]")

    # If file ends with ], remove it for append
    if stripped.endswith("]"):
        idx = len(raw) - 1
        while idx >= 0 and raw[idx].isspace():
            idx -= 1
        if idx >= 0 and raw[idx] == "]":
            raw = raw[:idx]

    # If file ends mid-object, truncate to last complete object
    trimmed = raw.rstrip()
    if trimmed and trimmed[-1] not in ("}", "["):
        last_obj = raw.rfind("}")
        if last_obj != -1:
            raw = raw[: last_obj + 1]
        else:
            raw = "[\n"
            is_empty_array = True

    out_path.write_text(raw, encoding="utf-8")

    # Detect if file already has at least one object
    has_object = '"case_no"' in raw
    fh = out_path.open("a", encoding="utf-8")
    return fh, not has_object


def finalize_output_file(out_path: Path):
    # Ensure JSON array closure
    raw = out_path.read_text(encoding="utf-8")
    if not raw.strip().endswith("]"):
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write("\n]\n")


def main():
    parser = argparse.ArgumentParser(description="Full-run citation extraction with progress bar.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (e.g., 10000).",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume from existing output dir (e.g., data\\law_citation_extraction\\20260302_120000)",
    )
    args = parser.parse_args()

    if not API_KEYS:
        raise RuntimeError("API_KEYS is empty. Please fill create train dataset/api_keys.json")

    if args.resume_dir:
        out_root = Path(args.resume_dir)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = OUTPUT_ROOT / "law_citation_extraction" / timestamp
        out_root.mkdir(parents=True, exist_ok=True)

    index_path = out_root / "completed_case_no.txt"
    completed_cases = set()
    completed_cases.update(load_completed_cases_from_index(index_path))
    completed_cases.update(load_completed_cases_from_outputs(out_root))

    remaining_available = count_valid_records(INPUT_DIR, completed_cases)
    if args.limit is not None:
        remaining_target = max(args.limit - len(completed_cases), 0)
        total_records = min(remaining_available, remaining_target)
    else:
        remaining_target = None
        total_records = remaining_available

    handles = {}
    first_flags = {}

    stats = {"success": 0, "failed": 0, "skipped": 0}
    skipped_completed = 0
    processed = 0

    print(f"[Resume] Loaded completed case_no: {len(completed_cases)}")
    print(f"[Resume] Remaining to process (after resume & limit): {total_records}")

    if args.limit is not None and total_records <= 0:
        print("[Resume] Limit already satisfied. Nothing to do.")
        return

    pbar = tqdm(total=total_records, desc="Extracting", unit="sample")

    max_workers = max(1, len(API_KEYS))
    key_cycle = cycle(API_KEYS)

    futures = {}

    def submit_task(executor, record):
        z4_text = record.get("Z4_Reasoning") or ""
        if not z4_text.strip():
            return None, None
        assigned_key = next(key_cycle)
        future = executor.submit(extract_citations, z4_text, assigned_key)
        return future, z4_text

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for file_path, record in iter_records(INPUT_DIR):
                z4_text = record.get("Z4_Reasoning") or ""
                if not z4_text.strip():
                    stats["skipped"] += 1
                    continue

                case_no = record.get("case_no")
                if case_no and case_no in completed_cases:
                    skipped_completed += 1
                    continue

                year, ym = get_year_month_from_path(file_path)
                year_dir = out_root / year
                year_dir.mkdir(parents=True, exist_ok=True)
                out_path = year_dir / f"{ym}.json"

                if out_path not in handles:
                    fh, is_first = ensure_output_handle(out_path)
                    handles[out_path] = fh
                    first_flags[out_path] = is_first

                future, z4_text = submit_task(executor, record)
                if future is None:
                    stats["skipped"] += 1
                    continue
                futures[future] = (record, out_path, z4_text)

                if remaining_target is not None and len(futures) + processed >= remaining_target:
                    break

            for future in as_completed(futures):
                record, out_path, z4_text = futures[future]
                result = future.result()

                status = result.get("status", "failed") if result else "failed"
                if status == "success":
                    stats["success"] += 1
                elif status == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1

                citations = result.get("data", {}).get("citations", []) if result else []

                out_record = {
                    "source_file": record.get("source_file"),
                    "case_no": record.get("case_no"),
                    "case_name": record.get("case_name"),
                    "Z4_Reasoning": z4_text,
                    "citations": citations,
                }

                if not first_flags[out_path]:
                    handles[out_path].write(",\n")
                handles[out_path].write(json.dumps(out_record, ensure_ascii=False))
                first_flags[out_path] = False

                case_no = record.get("case_no")
                if case_no:
                    with index_path.open("a", encoding="utf-8") as idx:
                        idx.write(f"{case_no}\n")
                    completed_cases.add(case_no)

                processed += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={stats['success']} fail={stats['failed']} "
                    f"skip={stats['skipped']} done={skipped_completed}"
                )

                if remaining_target is not None and processed >= remaining_target:
                    break
    finally:
        for out_path, fh in handles.items():
            fh.close()
            finalize_output_file(out_path)
        pbar.close()

    print(
        f"Done. Processed: {processed}. Success: {stats['success']}. "
        f"Failed: {stats['failed']}. Skipped: {stats['skipped']}. "
        f"SkippedCompleted: {skipped_completed}. Output: {out_root}"
    )


if __name__ == "__main__":
    main()
