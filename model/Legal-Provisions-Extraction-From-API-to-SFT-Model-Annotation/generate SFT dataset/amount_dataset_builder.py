from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from amount_config import MAX_TOTAL_LENGTH, SYSTEM_INSTRUCTION


AMOUNT_RE = re.compile(r"^\d+(?:\.\d+)?$")
ALLOWED_AMOUNT_TYPE = {"交付金额", "涉案金额", "unknown"}
ALLOWED_SOURCE_ZONE = {"Z4", "Z3", "none"}
INSTRUCTION_PREFIX = "指令："
QUESTION_PREFIX = "问："
ANSWER_PREFIX = "答："


def build_instruction() -> str:
    return f"{INSTRUCTION_PREFIX}{SYSTEM_INSTRUCTION}"


def _flat(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def build_question(z4_text: str, z3_text: str) -> str:
    payload = {
        "Z4_Reasoning": z4_text or "",
        "Z3_Fact": z3_text or "",
    }
    return f"{QUESTION_PREFIX}{json.dumps(payload, ensure_ascii=False)}"


def build_answer(target_amount: str, amount_type: str, source_zone: str) -> str:
    payload = {
        "target_amount": target_amount,
        "amount_type": amount_type,
        "source_zone": source_zone,
    }
    return f"{ANSWER_PREFIX}{json.dumps(payload, ensure_ascii=False)}"


def build_sample(z4_text: str, z3_text: str, target_amount: str, amount_type: str, source_zone: str) -> Tuple[str, str, str]:
    instruction = build_instruction()
    question = build_question(_flat(z4_text or ""), _flat(z3_text or ""))
    answer = build_answer(target_amount, amount_type, source_zone)
    return instruction, question, answer


def normalize_amount_record(record: Dict) -> Dict:
    target_amount = str(record.get("target_amount") or "").strip()
    amount_type = str(record.get("amount_type") or "unknown").strip()
    source_zone = str(record.get("source_zone") or "none").strip()
    status = str(record.get("status") or "").strip()

    if amount_type not in ALLOWED_AMOUNT_TYPE:
        amount_type = "unknown"
    if source_zone not in ALLOWED_SOURCE_ZONE:
        source_zone = "none"
    if not AMOUNT_RE.fullmatch(target_amount):
        target_amount = ""
        amount_type = "unknown"
        source_zone = "none"

    return {
        "case_no": record.get("case_no"),
        "target_amount": target_amount,
        "amount_type": amount_type,
        "source_zone": source_zone,
        "status": status,
    }


def is_valid_amount_record(record: Dict, include_unknown: bool = False) -> bool:
    target_amount = (record.get("target_amount") or "").strip()
    amount_type = (record.get("amount_type") or "").strip()
    source_zone = (record.get("source_zone") or "").strip()

    if include_unknown and amount_type == "unknown":
        return True
    if not target_amount:
        return False
    if not AMOUNT_RE.fullmatch(target_amount):
        return False
    if amount_type not in {"交付金额", "涉案金额"}:
        return False
    if source_zone not in {"Z4", "Z3"}:
        return False
    return True


def is_over_length(instruction: str, question: str, answer: str) -> bool:
    total_len = len(instruction) + len(question) + len(answer)
    return total_len > MAX_TOTAL_LENGTH
