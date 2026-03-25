from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from amount_config import (
    AMBIGUOUS_CASE_FILENAME,
    DATASET_FILENAME,
    DATASET_JSONL_FILENAME,
    DEFAULT_AMOUNT_INPUT_BASE,
    DEFAULT_PARTITION_DIR,
    DUPLICATE_AMOUNT_CASE_FILENAME,
    INVALID_AMOUNT_FILENAME,
    MISSING_CASE_FILENAME,
    OUTPUT_BASE,
    OVERLENGTH_FILENAME,
    STATS_FILENAME,
)
from amount_dataset_builder import (
    ANSWER_PREFIX,
    INSTRUCTION_PREFIX,
    QUESTION_PREFIX,
    build_sample,
    is_over_length,
    is_valid_amount_record,
    normalize_amount_record,
)
from io_utils import iter_records, write_jsonl


def normalize_case_no(case_no: object) -> str:
    return str(case_no or "").strip()


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


def resolve_amount_input_dir(args: argparse.Namespace) -> Path:
    if args.amount_input_dir:
        return Path(args.amount_input_dir)
    if args.run_name:
        return DEFAULT_AMOUNT_INPUT_BASE / args.run_name

    latest_file = DEFAULT_AMOUNT_INPUT_BASE / "latest_run.txt"
    if latest_file.exists():
        run_name = latest_file.read_text(encoding="utf-8").strip()
        if run_name:
            return DEFAULT_AMOUNT_INPUT_BASE / run_name
    raise ValueError("Please provide --amount-input-dir or --run-name.")


def resolve_output_dir(amount_input_dir: Path) -> Path:
    return OUTPUT_BASE / amount_input_dir.name


def build_partition_index(partition_dir: Path) -> Tuple[Dict[str, List[Dict]], int]:
    index: Dict[str, List[Dict]] = {}
    no_case_no_count = 0
    for _, record in iter_records(partition_dir):
        case_no = normalize_case_no(record.get("case_no"))
        if not case_no:
            no_case_no_count += 1
            continue
        index.setdefault(case_no, []).append(record)
    return index, no_case_no_count


def iter_amount_records(amount_input_dir: Path) -> Iterable[Tuple[Path, Dict]]:
    return iter_records(amount_input_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset for amount extraction from batch inference outputs.")
    parser.add_argument(
        "--amount-input-dir",
        type=str,
        default=None,
        help="Amount extraction run directory, e.g. data\\loan_amount_extraction\\20260322_215921",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Folder name under data\\loan_amount_extraction. Used when --amount-input-dir is not provided.",
    )
    parser.add_argument(
        "--partition-dir",
        type=str,
        default=str(DEFAULT_PARTITION_DIR),
        help="Directory of partition records with Z3/Z4, default: data\\yishen_partition",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include unknown samples in dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Build at most N valid samples (for quick check).",
    )
    args = parser.parse_args()

    amount_input_dir = resolve_amount_input_dir(args)
    partition_dir = Path(args.partition_dir)

    if not amount_input_dir.exists():
        raise FileNotFoundError(f"Amount input dir not found: {amount_input_dir}")
    if not partition_dir.exists():
        raise FileNotFoundError(f"Partition dir not found: {partition_dir}")

    output_dir = resolve_output_dir(amount_input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / DATASET_FILENAME
    dataset_jsonl_path = output_dir / DATASET_JSONL_FILENAME
    missing_case_path = output_dir / MISSING_CASE_FILENAME
    ambiguous_case_path = output_dir / AMBIGUOUS_CASE_FILENAME
    invalid_amount_path = output_dir / INVALID_AMOUNT_FILENAME
    overlength_path = output_dir / OVERLENGTH_FILENAME
    duplicate_amount_case_path = output_dir / DUPLICATE_AMOUNT_CASE_FILENAME
    stats_path = output_dir / STATS_FILENAME

    partition_index, partition_no_case_no_count = build_partition_index(partition_dir)

    missing_case: List[Dict] = []
    ambiguous_case: List[Dict] = []
    invalid_amount: List[Dict] = []
    overlength: List[Dict] = []
    duplicate_amount_case: List[Dict] = []

    valid_count = 0
    seen_amount_case_no = set()
    total_amount_records = 0

    with dataset_path.open("w", encoding="utf-8") as txt_out, dataset_jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_out:
        for file_path, raw_record in iter_amount_records(amount_input_dir):
            total_amount_records += 1

            case_no = normalize_case_no(raw_record.get("case_no"))
            if not case_no:
                invalid_amount.append(
                    {
                        "source_file": str(file_path),
                        "case_no": raw_record.get("case_no"),
                        "reason": "empty_case_no",
                        "record": raw_record,
                    }
                )
                continue

            if case_no in seen_amount_case_no:
                duplicate_amount_case.append(
                    {
                        "source_file": str(file_path),
                        "case_no": case_no,
                    }
                )
                continue
            seen_amount_case_no.add(case_no)

            source_candidates = partition_index.get(case_no)
            if not source_candidates:
                missing_case.append(
                    {
                        "source_file": str(file_path),
                        "case_no": case_no,
                    }
                )
                continue
            if len(source_candidates) > 1:
                ambiguous_case.append(
                    {
                        "source_file": str(file_path),
                        "case_no": case_no,
                        "candidate_count": len(source_candidates),
                    }
                )
                continue

            normalized_amount = normalize_amount_record(raw_record)
            if not is_valid_amount_record(normalized_amount, include_unknown=args.include_unknown):
                invalid_amount.append(
                    {
                        "source_file": str(file_path),
                        "case_no": case_no,
                        "reason": "invalid_or_filtered_amount",
                        "record": normalized_amount,
                    }
                )
                continue

            source_record = source_candidates[0]
            z4_text = (get_zone_text(source_record, "Z4_Reasoning") or "").strip()
            z3_text = (get_zone_text(source_record, "Z3_Fact") or "").strip()

            instruction, question, answer = build_sample(
                z4_text=z4_text,
                z3_text=z3_text,
                target_amount=normalized_amount["target_amount"],
                amount_type=normalized_amount["amount_type"],
                source_zone=normalized_amount["source_zone"],
            )

            if is_over_length(instruction, question, answer):
                overlength.append(
                    {
                        "source_file": str(file_path),
                        "case_no": case_no,
                        "instruction_len": len(instruction),
                        "question_len": len(question),
                        "answer_len": len(answer),
                    }
                )
                continue

            txt_out.write(instruction + "\n")
            txt_out.write(question + "\n")
            txt_out.write(answer + "\n")
            txt_out.write("\n")

            jsonl_item = {
                "case_no": case_no,
                "instruction": instruction[len(INSTRUCTION_PREFIX) :],
                "question": question[len(QUESTION_PREFIX) :],
                "answer": answer[len(ANSWER_PREFIX) :],
                "meta": {
                    "amount_type": normalized_amount["amount_type"],
                    "source_zone": normalized_amount["source_zone"],
                },
            }
            jsonl_out.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")

            valid_count += 1
            if args.max_samples is not None and valid_count >= args.max_samples:
                break

    write_jsonl(missing_case_path, missing_case)
    write_jsonl(ambiguous_case_path, ambiguous_case)
    write_jsonl(invalid_amount_path, invalid_amount)
    write_jsonl(overlength_path, overlength)
    write_jsonl(duplicate_amount_case_path, duplicate_amount_case)

    stats = {
        "amount_input_dir": str(amount_input_dir),
        "partition_dir": str(partition_dir),
        "output_dir": str(output_dir),
        "include_unknown": args.include_unknown,
        "max_samples": args.max_samples,
        "total_amount_records": total_amount_records,
        "unique_amount_case_no": len(seen_amount_case_no),
        "valid_samples": valid_count,
        "missing_case_no_in_partition": len(missing_case),
        "ambiguous_case_no_in_partition": len(ambiguous_case),
        "invalid_amount_or_filtered": len(invalid_amount),
        "overlength": len(overlength),
        "duplicate_case_no_in_amount_input": len(duplicate_amount_case),
        "partition_records_without_case_no": partition_no_case_no_count,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 48)
    print(f"Output dir: {output_dir}")
    print(f"Dataset txt: {dataset_path}")
    print(f"Dataset jsonl: {dataset_jsonl_path}")
    print(f"Valid samples: {valid_count}")
    print(f"Missing case_no: {len(missing_case)}")
    print(f"Ambiguous case_no: {len(ambiguous_case)}")
    print(f"Invalid/filtered amount: {len(invalid_amount)}")
    print(f"Overlength: {len(overlength)}")
    print(f"Duplicate amount case_no: {len(duplicate_amount_case)}")
    print("-" * 48)


if __name__ == "__main__":
    main()
