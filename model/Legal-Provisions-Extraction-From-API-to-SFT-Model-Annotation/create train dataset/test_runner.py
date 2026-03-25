import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
from pathlib import Path

from config import INPUT_DIR, OUTPUT_ROOT, DEFAULT_TEST_LIMIT, API_KEYS
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


def main():
    if not API_KEYS:
        raise RuntimeError("API_KEYS is empty. Please fill create train dataset/api_keys.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = OUTPUT_ROOT / "prompt_test" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)

    handles = {}
    first_flags = {}

    total = 0
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path, record in iter_records(INPUT_DIR):
            z4_text = record.get("Z4_Reasoning") or ""
            if not z4_text.strip():
                continue

            _, ym = get_year_month_from_path(file_path)
            out_path = out_root / f"result_{ym}.json"

            if out_path not in handles:
                fh = out_path.open("w", encoding="utf-8")
                fh.write("[\n")
                handles[out_path] = fh
                first_flags[out_path] = True

            future, z4_text = submit_task(executor, record)
            if future is None:
                continue
            futures[future] = (record, out_path, z4_text)

            if DEFAULT_TEST_LIMIT is not None and len(futures) >= DEFAULT_TEST_LIMIT:
                break

        for future in as_completed(futures):
            record, out_path, z4_text = futures[future]
            result = future.result()

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

            total += 1
            if DEFAULT_TEST_LIMIT is not None and total >= DEFAULT_TEST_LIMIT:
                break

    for out_path, fh in handles.items():
        fh.write("\n]\n")
        fh.close()

    print(f"Test done. Total samples: {total}. Output: {out_root}")


if __name__ == "__main__":
    main()
