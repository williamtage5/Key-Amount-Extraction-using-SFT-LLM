from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from align_checker import filter_aligned_citations
from config import (
    ALIGNMENT_FAILED_FILENAME,
    AMBIGUOUS_PRONOUN_FILENAME,
    DATASET_FILENAME,
    DATA_DIR,
    OUTPUT_BASE,
)
from dataset_builder import build_sample, clean_citations, is_over_length
from io_utils import iter_records, write_jsonl
from pronoun_resolver import resolve_pronouns


def resolve_output_dir(input_dir: Path) -> Path:
    # Keep only the last directory name (e.g., 20260302_111906)
    return OUTPUT_BASE / input_dir.name


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from law citation extraction outputs.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory, e.g. data\\law_citation_extraction\\20260302_111906",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = resolve_output_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / DATASET_FILENAME
    align_failed_path = output_dir / ALIGNMENT_FAILED_FILENAME
    ambiguous_path = output_dir / AMBIGUOUS_PRONOUN_FILENAME

    align_failed: List[Dict] = []
    ambiguous_list: List[Dict] = []

    valid_count = 0
    skipped_count = 0

    with dataset_path.open("w", encoding="utf-8") as out:
        for _, record in iter_records(input_dir):
            z4_text = (record.get("Z4_Reasoning") or "").strip()
            if not z4_text:
                continue

            citations = record.get("citations") or []
            if isinstance(citations, dict):
                citations = [citations]
            if isinstance(citations, str):
                citations = []
            if isinstance(citations, list):
                citations = [c for c in citations if isinstance(c, dict)]
            if not citations:
                continue

            resolved, ambiguous = resolve_pronouns(citations)
            if ambiguous:
                ambiguous_list.append({
                    "source_file": record.get("source_file"),
                    "case_no": record.get("case_no"),
                    "case_name": record.get("case_name"),
                    "ambiguous": ambiguous,
                })

            aligned = filter_aligned_citations(z4_text, resolved)
            if len(aligned) < len(resolved):
                align_failed.append({
                    "source_file": record.get("source_file"),
                    "case_no": record.get("case_no"),
                    "case_name": record.get("case_name"),
                    "citations": resolved,
                    "aligned": aligned,
                    "Z4_Reasoning": z4_text,
                })

            cleaned = clean_citations(resolved)
            if not cleaned:
                continue

            instruction, question, answer = build_sample(z4_text, cleaned)
            if is_over_length(instruction, question, answer):
                skipped_count += 1
                continue

            out.write(instruction + "\n")
            out.write(question + "\n")
            out.write(answer + "\n")
            out.write("\n")
            valid_count += 1

    write_jsonl(align_failed_path, align_failed)
    write_jsonl(ambiguous_path, ambiguous_list)

    print("-" * 40)
    print(f"Output dir: {output_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Valid samples: {valid_count}")
    print(f"Skipped (overlength): {skipped_count}")
    print(f"Alignment failed: {len(align_failed)}")
    print(f"Ambiguous pronoun: {len(ambiguous_list)}")
    print("-" * 40)


if __name__ == "__main__":
    main()
