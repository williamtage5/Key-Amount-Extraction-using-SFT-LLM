from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Output root
OUTPUT_BASE = DATA_DIR / "sft_dataset"

# Training output filename
DATASET_FILENAME = "dataset_for_autodl.txt"
ALIGNMENT_FAILED_FILENAME = "alignment_failed.jsonl"
AMBIGUOUS_PRONOUN_FILENAME = "ambiguous_pronoun.jsonl"

# Length filter (instruction + question + answer)
MAX_TOTAL_LENGTH = 8000

SYSTEM_INSTRUCTION = (
    "你是法律条文抽取器。请从裁判理由中抽取引用的法律与条款，"
    "仅输出JSON数组，每项包含law_name和article。"
)
