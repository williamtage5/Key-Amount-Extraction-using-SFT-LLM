from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any, Dict

AMOUNT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
UNIT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(万|亿)")

ALLOWED_AMOUNT_TYPE = {"交付金额", "涉案金额", "unknown", "payment", "loan"}
ALLOWED_SOURCE_ZONE = {"Z4", "Z3", "none"}


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        i = t.find("\n")
        if i != -1:
            t = t[i + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _find_first_json_object(text: str) -> Dict[str, Any]:
    t = _strip_code_fence(text)
    if not t:
        return {}

    try:
        parsed = json.loads(t)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = -1
    depth = 0
    for idx, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    chunk = t[start : idx + 1]
                    try:
                        parsed = json.loads(chunk)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        start = -1
                        continue
    return {}


def _normalize_amount(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().replace(",", "")
    if not s:
        return ""

    m_unit = UNIT_RE.search(s)
    if m_unit:
        base = Decimal(m_unit.group(1))
        mul = Decimal("10000") if m_unit.group(2) == "万" else Decimal("100000000")
        val = base * mul
        if val == val.to_integral():
            return str(int(val))
        return format(val, "f").rstrip("0").rstrip(".")

    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        return s

    m = AMOUNT_RE.search(s)
    if m:
        return m.group(0)
    return ""


def _normalize_amount_type(raw: Any) -> str:
    s = str(raw).strip() if raw is not None else ""
    if s == "payment":
        return "交付金额"
    if s == "loan":
        return "涉案金额"
    if s in {"交付金额", "涉案金额", "unknown"}:
        return s
    if s in ALLOWED_AMOUNT_TYPE:
        return s
    return "unknown"


def _normalize_source_zone(raw: Any) -> str:
    s = str(raw).strip() if raw is not None else ""
    if s in ALLOWED_SOURCE_ZONE:
        return s
    return "none"


def parse_amount_json(text: str) -> Dict[str, str]:
    data = _find_first_json_object(text)

    amount = _normalize_amount(data.get("target_amount"))
    amount_type = _normalize_amount_type(data.get("amount_type"))
    source_zone = _normalize_source_zone(data.get("source_zone"))

    if amount and amount_type == "unknown":
        if source_zone == "Z4":
            amount_type = "交付金额"
        elif source_zone == "Z3":
            amount_type = "涉案金额"

    if amount and source_zone == "none":
        source_zone = "Z4" if amount_type == "交付金额" else ("Z3" if amount_type == "涉案金额" else "none")

    if not amount:
        return {"target_amount": "", "amount_type": "unknown", "source_zone": "none"}

    return {"target_amount": amount, "amount_type": amount_type, "source_zone": source_zone}
