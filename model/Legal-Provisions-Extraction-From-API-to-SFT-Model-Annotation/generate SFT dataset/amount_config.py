from __future__ import annotations

from pathlib import Path


# Path layout:
# .../Important_number_extraction/model/Legal-Provisions-.../generate SFT dataset/amount_config.py
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = WORKSPACE_ROOT / "data"

# Inputs
DEFAULT_AMOUNT_INPUT_BASE = DATA_ROOT / "loan_amount_extraction"
DEFAULT_PARTITION_DIR = DATA_ROOT / "yishen_partition"

# Output
OUTPUT_BASE = DATA_ROOT / "sft_amount_dataset"
DATASET_FILENAME = "dataset_for_autodl.txt"
DATASET_JSONL_FILENAME = "dataset_for_autodl.jsonl"
MISSING_CASE_FILENAME = "missing_case_no.jsonl"
AMBIGUOUS_CASE_FILENAME = "ambiguous_case_no.jsonl"
INVALID_AMOUNT_FILENAME = "invalid_amount.jsonl"
OVERLENGTH_FILENAME = "overlength.jsonl"
DUPLICATE_AMOUNT_CASE_FILENAME = "duplicate_amount_case_no.jsonl"
STATS_FILENAME = "stats.json"

# Length filter (instruction + question + answer)
MAX_TOTAL_LENGTH = 12000

SYSTEM_INSTRUCTION = (
    "你是民事借贷案件金额抽取器。"
    "给定判决文本分区（先Z4后Z3），输出最能决定审判结果的金额。"
    "只输出JSON对象，包含target_amount、amount_type、source_zone。"
    "target_amount必须是阿拉伯数字字符串。"
)
