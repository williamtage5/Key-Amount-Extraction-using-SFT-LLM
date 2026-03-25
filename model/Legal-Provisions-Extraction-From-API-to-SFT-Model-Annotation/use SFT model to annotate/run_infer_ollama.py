from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from config import INPUT_DIR, OUTPUT_ROOT, OLLAMA_BASE_URL, OLLAMA_MODEL, PROMPT_TEMPLATE
from io_utils import ensure_output_path, iter_records, load_json_array, write_json_array
from parser_utils import extract_json_object


def build_prompt(text: str) -> str:
    # Avoid str.format to prevent conflicts with JSON braces in the template.
    return PROMPT_TEMPLATE.replace("{text}", text)


def ollama_chat(model: str, prompt: str, base_url: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    message = parsed.get("message") or {}
    return (message.get("content") or "").strip()


def request_with_retry(args, prompt: str) -> str:
    last_exc: Optional[Exception] = None
    for _ in range(max(1, args.max_retries + 1)):
        try:
            return ollama_chat(args.model, prompt, args.base_url, args.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
            time.sleep(args.retry_wait)
    if last_exc:
        raise last_exc
    return ""


def build_record_key(record: Dict) -> str:
    source_file = record.get("source_file") or ""
    case_no = record.get("case_no") or ""
    if not source_file and not case_no:
        return ""
    return f"{source_file}::{case_no}"


def load_existing_keys(resume_dir: Optional[str]) -> Set[str]:
    if not resume_dir:
        return set()
    base_path = Path(resume_dir)
    if not base_path.exists():
        return set()
    keys: Set[str] = set()
    for path in base_path.rglob("*.json"):
        data = load_json_array(path)
        for record in data:
            key = build_record_key(record)
            if key:
                keys.add(key)
    return keys


def merge_and_write(path: Path, new_records: List[Dict], existing_keys: Set[str]) -> None:
    if not new_records:
        return
    existing = load_json_array(path) if path.exists() else []
    merged = list(existing)
    for rec in new_records:
        key = build_record_key(rec)
        if key and key in existing_keys:
            continue
        merged.append(rec)
        if key:
            existing_keys.add(key)
    write_json_array(path, merged)


def iter_tasks(
    records: Iterable[Tuple[Path, Dict]],
    existing_keys: Set[str],
    limit: Optional[int],
) -> Iterable[Tuple[Path, Dict]]:
    count = 0
    for file_path, record in records:
        if limit is not None and count >= limit:
            break
        z4 = (record.get("Z4_Reasoning") or "").strip()
        if not z4:
            continue
        key = build_record_key(record)
        if key and key in existing_keys:
            continue
        yield file_path, record
        count += 1


def count_remaining(records: Iterable[Tuple[Path, Dict]], existing_keys: Set[str]) -> int:
    remaining = 0
    for _, record in records:
        z4 = (record.get("Z4_Reasoning") or "").strip()
        if not z4:
            continue
        key = build_record_key(record)
        if key and key in existing_keys:
            continue
        remaining += 1
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate Z4_Reasoning with Ollama model.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--resume-dir", type=str, default=None, help="Resume from existing output directory")
    parser.add_argument("--write-batch-size", type=int, default=200, help="Buffer size per output file")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for failed requests")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Wait seconds between retries")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable progress bar")
    parser.set_defaults(progress=True)
    parser.add_argument("--debug", action="store_true", help="Print raw model output for debugging")
    args = parser.parse_args()

    existing_keys = load_existing_keys(args.resume_dir) if args.resume_dir else set()
    if args.resume_dir:
        remaining_total = count_remaining(iter_records(INPUT_DIR), existing_keys)
        remaining_after_limit = (
            min(args.limit, remaining_total) if args.limit is not None else remaining_total
        )
        print(f"[Resume] Loaded completed case_no: {len(existing_keys)}")
        print(f"[Resume] Remaining available (no limit): {remaining_total}")
        print(f"[Resume] Remaining to process (after resume & limit): {remaining_after_limit}")

    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    total = args.limit if args.limit is not None else None
    pbar = tqdm(total=total, desc="Annotating", unit="sample") if args.progress and tqdm else None

    buffer_by_path: Dict[str, List[Dict]] = {}
    count = 0
    skipped = 0
    failed = 0
    succeeded = 0

    def flush_buffer(out_path: str) -> None:
        buf = buffer_by_path.get(out_path) or []
        if not buf:
            return
        merge_and_write(Path(out_path), buf, existing_keys)
        buffer_by_path[out_path] = []

    try:
        records = iter_records(INPUT_DIR)
        task_iter = iter_tasks(records, existing_keys, args.limit)

        def submit_next(executor) -> Optional[Tuple]:
            nonlocal count, skipped
            try:
                file_path, record = next(task_iter)
            except StopIteration:
                return None
            z4 = (record.get("Z4_Reasoning") or "").strip()
            if not z4:
                skipped += 1
                return None
            prompt = build_prompt(z4)
            future = executor.submit(request_with_retry, args, prompt)
            count += 1
            return future, file_path, record

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            pending: Dict = {}
            while len(pending) < max(1, args.workers):
                item = submit_next(ex)
                if item is None:
                    break
                future, file_path, record = item
                pending[future] = (file_path, record)

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    file_path, record = pending.pop(future)
                    try:
                        raw = future.result()
                        request_failed = False
                    except Exception as exc:  # noqa: BLE001
                        print(f"[Error] Ollama request failed: {exc}")
                        raw = ""
                        request_failed = True

                    if args.debug:
                        print("[Debug] Raw output:", raw)
                    result = extract_json_object(raw)
                    if args.debug:
                        print("[Debug] Parsed citations:", result.get("citations"))
                    citations = result.get("citations", []) if isinstance(result, dict) else []

                    out_path = str(ensure_output_path(OUTPUT_ROOT, file_path))
                    buffer_by_path.setdefault(out_path, []).append(
                        {
                            "source_file": record.get("source_file"),
                            "case_no": record.get("case_no"),
                            "case_name": record.get("case_name"),
                            "Z4_Reasoning": (record.get("Z4_Reasoning") or "").strip(),
                            "citations": citations,
                        }
                    )

                    if request_failed:
                        failed += 1
                    else:
                        succeeded += 1

                    if len(buffer_by_path.get(out_path, [])) >= max(1, args.write_batch_size):
                        flush_buffer(out_path)

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(ok=succeeded, fail=failed, skip=skipped)

                    item = submit_next(ex)
                    if item is not None:
                        future2, file_path2, record2 = item
                        pending[future2] = (file_path2, record2)
    finally:
        for out_path in list(buffer_by_path):
            flush_buffer(out_path)
        if pbar is not None:
            pbar.close()

    print(
        f"Done. Samples: {count}. Success: {succeeded}. Failed: {failed}. "
        f"Skipped: {skipped}. Output: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
