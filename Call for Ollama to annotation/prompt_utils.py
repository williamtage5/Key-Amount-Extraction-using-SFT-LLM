from __future__ import annotations

import json
from typing import Dict

from config_amount import PROMPT_TEMPLATE


def build_prompt(z4_text: str, z3_text: str) -> str:
    question_payload: Dict[str, str] = {
        "Z4_Reasoning": z4_text or "",
        "Z3_Fact": z3_text or "",
    }
    question_json = json.dumps(question_payload, ensure_ascii=False)
    return PROMPT_TEMPLATE.replace("{question_json}", question_json)

