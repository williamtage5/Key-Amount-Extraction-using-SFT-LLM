import random
import time

from api_client import call_llm_extraction
from config import (
    API_KEYS,
    MODEL_NAME,
    RATE_LIMIT_INITIAL_DELAY,
    RATE_LIMIT_BACKOFF_FACTOR,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MAX_RETRIES,
    TEMPERATURE,
    MAX_TOKENS,
)
from prompt_manager import build_payload


def is_rate_limit_error(error_msg: str) -> bool:
    if not error_msg:
        return False
    lowered = str(error_msg).lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def compute_rate_limit_delay(retry_count: int) -> int:
    delay = RATE_LIMIT_INITIAL_DELAY * (RATE_LIMIT_BACKOFF_FACTOR ** retry_count)
    return min(delay, RATE_LIMIT_MAX_DELAY)


def extract_citations(reasoning_text, assigned_key=None):
    if not reasoning_text:
        return {"status": "skipped", "data": {"citations": []}, "error_msg": "empty_text"}

    current_key = assigned_key if assigned_key else random.choice(API_KEYS)
    payload = build_payload(reasoning_text, MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

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
        return {
            "status": "success",
            "data": llm_result.get("data", {}),
            "perf": {
                "latency": latency,
                "key_tail": current_key[-4:],
                "attempts": attempt_count,
                "rate_limit_retries": rate_limit_retries,
            },
        }

    return {
        "status": "failed",
        "data": {"citations": []},
        "error_msg": llm_result.get("error_msg") if llm_result else "unknown_error",
        "raw_response": llm_result.get("raw_response") if llm_result else None,
        "perf": {
            "latency": latency,
            "key_tail": current_key[-4:],
            "attempts": attempt_count,
            "rate_limit_retries": rate_limit_retries,
        },
    }
