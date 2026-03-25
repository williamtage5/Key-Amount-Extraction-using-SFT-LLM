from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from config_amount import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_BASE_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from io_utils import iter_records, load_json_array, write_json_array
from parser_utils import parse_amount_json
from prompt_utils import build_prompt


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


def get_zone_text(record: Dict, key: str) -> str:
    value = record.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    zones = record.get("zones")
    if isinstance(zones, dict):
        z = zones.get(key)
        if isinstance(z, dict):
            text = z.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    # For SFT-formatted rows where question contains Z3/Z4 JSON.
    question = record.get("question")
    if question:
        try:
            qobj = question if isinstance(question, dict) else json.loads(str(question))
            qv = qobj.get(key)
            if isinstance(qv, str) and qv.strip():
                return qv.strip()
        except Exception:
            pass
    return ""


def build_record_key(record: Dict) -> str:
    source_file = str(record.get("source_file") or "")
    case_no = str(record.get("case_no") or "")
    sample_id = str(record.get("sample_id") or "")
    if not source_file and not case_no and not sample_id:
        return ""
    return f"{source_file}::{case_no}::{sample_id}"


def load_existing_keys(resume_dir: Optional[str]) -> Set[str]:
    if not resume_dir:
        return set()
    base = Path(resume_dir)
    if not base.exists():
        return set()
    keys: Set[str] = set()
    for path in base.rglob("*.json"):
        records = load_json_array(path)
        for rec in records:
            key = build_record_key(rec)
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
) -> Iterable[Tuple[Path, Dict, str, str]]:
    count = 0
    for file_path, record in records:
        if limit is not None and count >= limit:
            break

        z4 = get_zone_text(record, "Z4_Reasoning")
        z3 = get_zone_text(record, "Z3_Fact")
        if not z4 and not z3:
            continue

        key = build_record_key(record)
        if key and key in existing_keys:
            continue

        yield file_path, record, z4, z3
        count += 1


def run_output_dir(output_base_dir: Path, run_ts: Optional[str]) -> Path:
    task_dir = output_base_dir
    task_dir.mkdir(parents=True, exist_ok=True)
    ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = task_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def infer_year_month_output_path(output_run_dir: Path, input_file: Path) -> Path:
    stem = input_file.stem

    year = None
    m = re.match(r"^(\d{4})", stem)
    if m:
        year = m.group(1)
    if year is None:
        for part in reversed(input_file.parts):
            if re.fullmatch(r"\d{4}", part):
                year = part
                break
    if year is None:
        year = "unknown_year"

    return output_run_dir / year / f"{stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama Q4 amount extraction for Z4/Z3 records.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Input directory (json/jsonl)")
    parser.add_argument("--output-base-dir", type=str, default=str(DEFAULT_OUTPUT_BASE_DIR), help="Base output folder")
    parser.add_argument("--run-ts", type=str, default=None, help="Run timestamp folder name override")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for smoke testing")
    parser.add_argument("--random-sample", action="store_true", help="Randomly sample records before inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --random-sample")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N completed samples")
    parser.add_argument("--write-batch-size", type=int, default=100, help="Buffered write size per output file")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry attempts per request")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Retry wait seconds")
    parser.add_argument("--resume-dir", type=str, default=None, help="Directory to load completed keys from")
    parser.add_argument("--preview-max", type=int, default=20, help="Max preview rows printed")
    parser.add_argument("--debug", action="store_true", help="Print raw model output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_base_dir)
    output_run_dir = run_output_dir(output_base_dir, args.run_ts)

    existing_keys = load_existing_keys(args.resume_dir) if args.resume_dir else set()

    print(f"[Input] {input_dir}")
    print(f"[Output] {output_run_dir}")
    print(f"[Model] {args.model}")
    if args.resume_dir:
        print(f"[Resume] loaded keys: {len(existing_keys)} from {args.resume_dir}")

    buffer_by_path: Dict[str, List[Dict]] = {}
    total = 0
    ok = 0
    fail = 0
    skip = 0
    done_count = 0
    started_at = time.time()

    def flush_one(path_key: str) -> None:
        rows = buffer_by_path.get(path_key) or []
        if not rows:
            return
        merge_and_write(Path(path_key), rows, existing_keys)
        buffer_by_path[path_key] = []

    exclude_dirs: List[Path] = []
    try:
        input_resolved = input_dir.resolve()
        output_resolved = output_base_dir.resolve()
        if output_resolved == input_resolved or input_resolved in output_resolved.parents:
            exclude_dirs.append(output_resolved)
    except Exception:
        pass

    records = iter_records(input_dir, exclude_dirs=exclude_dirs)
    if args.random_sample:
        # Build full candidate pool first, then sample by limit.
        all_tasks = list(iter_tasks(records, existing_keys, None))
        rng = random.Random(args.seed)
        rng.shuffle(all_tasks)
        if args.limit is not None:
            all_tasks = all_tasks[: args.limit]
        tasks = iter(all_tasks)
        print(f"[Sampling] random_sample=True seed={args.seed} candidates={len(all_tasks)}")
    else:
        tasks = iter_tasks(records, existing_keys, args.limit)

    def submit_next(executor):
        nonlocal total, skip
        try:
            file_path, record, z4, z3 = next(tasks)
        except StopIteration:
            return None
        if not z4 and not z3:
            skip += 1
            return None
        prompt = build_prompt(z4, z3)
        future = executor.submit(request_with_retry, args, prompt)
        total += 1
        return future, file_path, record, z4, z3

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            pending = {}
            while len(pending) < max(1, args.workers):
                item = submit_next(ex)
                if item is None:
                    break
                future, file_path, record, z4, z3 = item
                pending[future] = (file_path, record, z4, z3)

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    file_path, record, z4, z3 = pending.pop(future)
                    raw = ""
                    request_failed = False
                    try:
                        raw = future.result()
                    except Exception as exc:  # noqa: BLE001
                        request_failed = True
                        raw = ""
                        print(f"[Error] request failed: {exc}")

                    parsed = parse_amount_json(raw)
                    if request_failed:
                        fail += 1
                    else:
                        ok += 1
                    done_count += 1

                    output_row = {
                        "source_file": str(file_path),
                        "case_no": record.get("case_no"),
                        "target_amount": parsed.get("target_amount", ""),
                        "amount_type": parsed.get("amount_type", "unknown"),
                        "source_zone": parsed.get("source_zone", "none"),
                    }

                    out_path = infer_year_month_output_path(output_run_dir, file_path)
                    out_key = str(out_path)
                    buffer_by_path.setdefault(out_key, []).append(output_row)
                    if len(buffer_by_path[out_key]) >= max(1, args.write_batch_size):
                        flush_one(out_key)

                    if done_count % max(1, args.progress_every) == 0 or request_failed:
                        elapsed = max(time.time() - started_at, 1e-6)
                        avg_sec = elapsed / max(done_count, 1)
                        print(
                            f"[Progress] done={done_count}, submitted={total}, ok={ok}, "
                            f"fail={fail}, skip={skip}, elapsed={elapsed:.1f}s, avg={avg_sec:.2f}s/sample"
                        )

                    item = submit_next(ex)
                    if item is not None:
                        future2, file_path2, record2, z4_2, z3_2 = item
                        pending[future2] = (file_path2, record2, z4_2, z3_2)
    finally:
        for path_key in list(buffer_by_path.keys()):
            flush_one(path_key)

    print(
        "[Done]",
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_run_dir": str(output_run_dir),
                "model": args.model,
                "workers": args.workers,
                "limit": args.limit,
                "total_submitted": total,
                "success": ok,
                "failed": fail,
                "skipped": skip,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
