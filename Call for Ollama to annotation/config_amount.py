from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input can be changed by CLI --input-dir.
# Default points to the canonical dataset root.
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "yishen_partition"

# Output rule:
# data/ollama annotation/<run_timestamp>/
DEFAULT_OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "ollama annotation"

OLLAMA_MODEL = "Law-Qwen-4bit"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

PROMPT_TEMPLATE = (
    "你是民事借贷案件金额抽取器。\n"
    "请根据输入中的 Z4_Reasoning 和 Z3_Fact，抽取“最能决定裁判结果的一个金额”。\n"
    "规则：\n"
    "1. 先看 Z4_Reasoning。若能明确识别应支付/偿还金额，则 amount_type=交付金额，source_zone=Z4。\n"
    "2. 若 Z4 不能明确识别，则看 Z3_Fact，提取涉案借款主金额，amount_type=涉案金额，source_zone=Z3。\n"
    "3. target_amount 只保留阿拉伯数字（不带单位，不带逗号）。\n"
    "4. 若无法判断，返回 target_amount 为空字符串，amount_type=unknown，source_zone=none。\n"
    "只输出 JSON 对象，不要输出解释。JSON schema:\n"
    "{\"target_amount\":\"...\",\"amount_type\":\"交付金额|涉案金额|unknown\",\"source_zone\":\"Z4|Z3|none\"}\n\n"
    "输入：\n"
    "{question_json}\n"
)
