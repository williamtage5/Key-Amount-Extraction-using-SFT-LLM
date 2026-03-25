from __future__ import annotations

import json
from typing import Dict, List, Tuple

from config import MAX_TOTAL_LENGTH, SYSTEM_INSTRUCTION

INSTRUCTION_PREFIX = "指令："
QUESTION_PREFIX = "问："
ANSWER_PREFIX = "答："


def build_instruction() -> str:
    return f"{INSTRUCTION_PREFIX}{SYSTEM_INSTRUCTION}"


def build_question(z4_text: str) -> str:
    # Flatten to one line, keep logical newlines as \n
    flat = z4_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return f"{QUESTION_PREFIX}{flat}"


def build_answer(citations: List[Dict]) -> str:
    return f"{ANSWER_PREFIX}{json.dumps(citations, ensure_ascii=False)}"


def clean_citations(citations: List[Dict]) -> List[Dict]:
    cleaned = []
    seen = set()
    for c in citations:
        law = (c.get("law_name") or "").strip()
        article = (c.get("article") or "").strip()
        if not law or not article:
            continue
        key = (law, article)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"law_name": law, "article": article})
    return cleaned


def build_sample(z4_text: str, citations: List[Dict]) -> Tuple[str, str, str]:
    instruction = build_instruction()
    question = build_question(z4_text)
    answer = build_answer(citations)
    return instruction, question, answer


def is_over_length(instruction: str, question: str, answer: str) -> bool:
    total_len = len(instruction) + len(question) + len(answer)
    return total_len > MAX_TOTAL_LENGTH
