from __future__ import annotations

from decimal import Decimal, InvalidOperation
import random
import re
import time
from typing import Any, Dict

from api_client import call_llm_extraction
from config import (
    API_KEYS,
    MAX_TOKENS,
    MODEL_NAME,
    RATE_LIMIT_BACKOFF_FACTOR,
    RATE_LIMIT_INITIAL_DELAY,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MAX_RETRIES,
    TEMPERATURE,
)
from prompt_manager import build_payload


AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?")
ARABIC_UNIT_RE = re.compile(r"(?:人民币)?\s*([0-9]+(?:\.[0-9]+)?)\s*(亿元|亿|万元|万|元)")
CN_UNIT_RE = re.compile(
    r"(?:人民币)?\s*([零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟萬億貳兩]+)\s*(亿元|亿|万元|万|元)"
)
KEYWORD_NUMBER_RE = re.compile(
    r"(?:借款|欠款|本金|偿还|归还|返还|支付|给付|清偿)[^。；;，,\n]{0,12}?([0-9]+(?:\.[0-9]+)?)"
)

UNIT_MULTIPLIER = {
    "元": Decimal("1"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
}

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
CN_REPLACEMENTS = {
    "壹": "一",
    "贰": "二",
    "叁": "三",
    "肆": "四",
    "伍": "五",
    "陆": "六",
    "柒": "七",
    "捌": "八",
    "玖": "九",
    "貳": "二",
    "兩": "两",
    "拾": "十",
    "佰": "百",
    "仟": "千",
    "萬": "万",
    "億": "亿",
    "圆": "元",
}

STRONG_PAYMENT_KEYWORDS = ("判令", "应当", "支付", "给付", "偿还", "归还", "返还", "清偿", "判决如下")
PRINCIPAL_KEYWORDS = ("本金", "借款", "欠款", "借款本金", "应偿还")
NEGATIVE_KEYWORDS = ("诉讼费", "受理费", "保全", "撤诉", "驳回起诉", "移送", "利息", "违约金", "罚息")
WEAK_FALLBACK_KEYWORDS = ("借款", "欠款", "本金", "金额", "标的", "冻结", "偿还", "归还", "支付", "给付")

ALLOWED_AMOUNT_TYPES = {"交付金额", "涉案金额", "unknown", "payment", "loan"}
ALLOWED_SOURCE_ZONE = {"Z4", "Z3", "none"}


def is_rate_limit_error(error_msg: str) -> bool:
    if not error_msg:
        return False
    lowered = str(error_msg).lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def compute_rate_limit_delay(retry_count: int) -> int:
    delay = RATE_LIMIT_INITIAL_DELAY * (RATE_LIMIT_BACKOFF_FACTOR**retry_count)
    return min(delay, RATE_LIMIT_MAX_DELAY)


def _parse_cn_section(section: str) -> int:
    total = 0
    num = 0
    for ch in section:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch in CN_SMALL_UNITS:
            unit = CN_SMALL_UNITS[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
    return total + num


def _cn_to_int(text: str) -> int | None:
    if not text:
        return None
    s = text.strip()
    for src, dst in CN_REPLACEMENTS.items():
        s = s.replace(src, dst)
    s = s.replace("元", "").replace("整", "").replace("正", "").strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)
    if not re.search(r"[零〇一二两三四五六七八九十百千万亿]", s):
        return None

    total = 0
    if "亿" in s:
        parts = s.split("亿")
        for part in parts[:-1]:
            total += (_parse_cn_section(part) if part else 1) * 100000000
        s = parts[-1]

    if "万" in s:
        left, right = s.split("万", 1)
        total += (_parse_cn_section(left) if left else 1) * 10000
        s = right

    total += _parse_cn_section(s)
    return total


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    s = format(value, "f")
    return s.rstrip("0").rstrip(".")


def _to_yuan(value_text: str, unit_text: str) -> str:
    unit = unit_text.strip()
    mul = UNIT_MULTIPLIER.get(unit)
    if mul is None:
        return ""

    value_text = value_text.strip().replace(",", "")
    try:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value_text):
            value = Decimal(value_text)
        else:
            cn = _cn_to_int(value_text)
            if cn is None:
                return ""
            value = Decimal(cn)
    except (InvalidOperation, ValueError):
        return ""

    return _format_decimal(value * mul)


def _normalize_amount(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().replace(",", "")
    if not s:
        return ""

    m = ARABIC_UNIT_RE.search(s)
    if m:
        return _to_yuan(m.group(1), m.group(2))

    m = CN_UNIT_RE.search(s)
    if m:
        return _to_yuan(m.group(1), m.group(2))

    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        return s

    m = AMOUNT_RE.search(s)
    return m.group(0) if m else ""


def _normalize_amount_type(raw: Any) -> str:
    s = str(raw).strip() if raw is not None else ""
    if s == "payment":
        return "交付金额"
    if s == "loan":
        return "涉案金额"
    return s if s in ALLOWED_AMOUNT_TYPES else "unknown"


def _normalize_source_zone(raw: Any) -> str:
    s = str(raw).strip() if raw is not None else ""
    return s if s in ALLOWED_SOURCE_ZONE else "none"


def _normalize_response(data: Dict[str, Any]) -> Dict[str, str]:
    target_amount = _normalize_amount(data.get("target_amount"))
    amount_type = _normalize_amount_type(data.get("amount_type"))
    source_zone = _normalize_source_zone(data.get("source_zone"))

    if target_amount and amount_type == "unknown":
        if source_zone == "Z4":
            amount_type = "交付金额"
        elif source_zone == "Z3":
            amount_type = "涉案金额"

    if target_amount and source_zone == "none":
        source_zone = "Z4" if amount_type == "交付金额" else ("Z3" if amount_type == "涉案金额" else "none")

    if not target_amount:
        amount_type = "unknown"
        source_zone = "none"

    return {
        "target_amount": target_amount,
        "amount_type": amount_type,
        "source_zone": source_zone,
    }


def _sentence_of(text: str, start_idx: int) -> str:
    left = max(
        text.rfind("。", 0, start_idx),
        text.rfind("；", 0, start_idx),
        text.rfind(";", 0, start_idx),
        text.rfind("\n", 0, start_idx),
    )
    right_candidates = [
        i
        for i in (text.find("。", start_idx), text.find("；", start_idx), text.find(";", start_idx), text.find("\n", start_idx))
        if i != -1
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right]


def _score_sentence(sentence: str, zone: str) -> int:
    score = 1 if zone == "Z4" else 0
    if any(k in sentence for k in STRONG_PAYMENT_KEYWORDS):
        score += 5
    if any(k in sentence for k in PRINCIPAL_KEYWORDS):
        score += 3
    if any(k in sentence for k in NEGATIVE_KEYWORDS):
        score -= 4
    return score


def _best_amount_from_zone(text: str, zone: str) -> str:
    text = text or ""
    if not text.strip():
        return ""

    candidates: list[tuple[int, str]] = []

    for m in ARABIC_UNIT_RE.finditer(text):
        amount = _to_yuan(m.group(1), m.group(2))
        if not amount:
            continue
        sent = _sentence_of(text, m.start())
        candidates.append((_score_sentence(sent, zone), amount))

    for m in CN_UNIT_RE.finditer(text):
        amount = _to_yuan(m.group(1), m.group(2))
        if not amount:
            continue
        sent = _sentence_of(text, m.start())
        candidates.append((_score_sentence(sent, zone), amount))

    for m in KEYWORD_NUMBER_RE.finditer(text):
        amount = _normalize_amount(m.group(1))
        if not amount:
            continue
        sent = _sentence_of(text, m.start())
        candidates.append((_score_sentence(sent, zone) + 1, amount))

    if not candidates:
        return ""

    def _sort_key(item: tuple[int, str]) -> tuple[int, Decimal]:
        score, amount_str = item
        try:
            amount_num = Decimal(amount_str)
        except InvalidOperation:
            amount_num = Decimal("0")
        return score, amount_num

    candidates.sort(key=_sort_key, reverse=True)
    best_score, best_amount = candidates[0]
    if zone == "Z4" and best_score < 1:
        return ""
    if zone == "Z3" and best_score < 0:
        weak = _weak_number_from_zone(text)
        return weak
    return best_amount


def _weak_number_from_zone(text: str) -> str:
    text = text or ""
    weak_candidates: list[Decimal] = []
    for m in AMOUNT_RE.finditer(text):
        try:
            value = Decimal(m.group(0))
        except InvalidOperation:
            continue
        # avoid years/dates and tiny counts
        if value < Decimal("5000"):
            continue
        left = max(0, m.start() - 10)
        right = min(len(text), m.end() + 10)
        window = text[left:right]
        if any(k in window for k in WEAK_FALLBACK_KEYWORDS):
            weak_candidates.append(value)
    if not weak_candidates:
        return ""
    return _format_decimal(max(weak_candidates))


def _heuristic_fallback(z4_text: str, z3_text: str) -> Dict[str, str]:
    z4_amount = _best_amount_from_zone(z4_text, "Z4")
    if z4_amount:
        return {"target_amount": z4_amount, "amount_type": "交付金额", "source_zone": "Z4"}

    z3_amount = _best_amount_from_zone(z3_text, "Z3")
    if z3_amount:
        return {"target_amount": z3_amount, "amount_type": "涉案金额", "source_zone": "Z3"}

    return {"target_amount": "", "amount_type": "unknown", "source_zone": "none"}


def extract_target_amount(z4_text: str, z3_text: str, assigned_key: str | None = None) -> Dict[str, Any]:
    z4_text = z4_text or ""
    z3_text = z3_text or ""
    if not z4_text.strip() and not z3_text.strip():
        return {
            "status": "skipped",
            "data": {"target_amount": "", "amount_type": "unknown", "source_zone": "none"},
            "error_msg": "empty_z3_z4",
        }

    if not API_KEYS:
        fallback = _heuristic_fallback(z4_text, z3_text)
        if fallback.get("target_amount"):
            return {"status": "success", "data": fallback, "error_msg": "empty_api_keys_used_fallback"}
        return {
            "status": "failed",
            "data": fallback,
            "error_msg": "empty_api_keys",
        }

    current_key = assigned_key if assigned_key else random.choice(API_KEYS)
    payload = build_payload(z4_text, z3_text, MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    attempt_count = 0
    rate_limit_retries = 0
    start_time = time.time()
    llm_result = None

    while True:
        attempt_count += 1
        llm_result = call_llm_extraction(payload, current_key)

        if llm_result["status"] == "success":
            break

        error_msg = llm_result.get("error_msg", "")
        if is_rate_limit_error(error_msg):
            if RATE_LIMIT_MAX_RETRIES is None or rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                wait_time = compute_rate_limit_delay(rate_limit_retries)
                rate_limit_retries += 1
                time.sleep(wait_time)
                continue
            break
        break

    latency = round(time.time() - start_time, 2)

    if llm_result and llm_result["status"] == "success":
        normalized = _normalize_response(llm_result.get("data", {}))
        if not normalized.get("target_amount"):
            normalized = _heuristic_fallback(z4_text, z3_text)
        return {
            "status": "success",
            "data": normalized,
            "perf": {
                "latency": latency,
                "key_tail": current_key[-4:],
                "attempts": attempt_count,
                "rate_limit_retries": rate_limit_retries,
            },
        }

    fallback = _heuristic_fallback(z4_text, z3_text)
    if fallback.get("target_amount"):
        return {
            "status": "success",
            "data": fallback,
            "error_msg": llm_result.get("error_msg") if llm_result else "used_fallback",
            "perf": {
                "latency": latency,
                "key_tail": current_key[-4:] if current_key else "",
                "attempts": attempt_count,
                "rate_limit_retries": rate_limit_retries,
            },
        }

    return {
        "status": "failed",
        "data": fallback,
        "error_msg": llm_result.get("error_msg") if llm_result else "unknown_error",
        "raw_response": llm_result.get("raw_response") if llm_result else None,
        "perf": {
            "latency": latency,
            "key_tail": current_key[-4:] if current_key else "",
            "attempts": attempt_count,
            "rate_limit_retries": rate_limit_retries,
        },
    }
