# scripts/extraction_service.py

import time
import random
from config import (
    API_KEYS,
    MODEL_NAME,
    RATE_LIMIT_INITIAL_DELAY,
    RATE_LIMIT_BACKOFF_FACTOR,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MAX_RETRIES,
)
from prompt_manager import build_payload
from llm_client import call_llm_extraction

# ================= Helper: Rate Limit Logic =================

def is_rate_limit_error(error_msg: str) -> bool:
    """Check if the error string indicates a 429/rate limit scenario."""
    if not error_msg:
        return False
    lowered = str(error_msg).lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered

def compute_rate_limit_delay(retry_count: int) -> int:
    """Exponential backoff delay bounded by MAX_DELAY."""
    delay = RATE_LIMIT_INITIAL_DELAY * (RATE_LIMIT_BACKOFF_FACTOR ** retry_count)
    return min(delay, RATE_LIMIT_MAX_DELAY)

# ================= Core Service Logic =================

def process_merged_row(merged_data, assigned_key=None):
    """
    Process a single merged record (Fee Text + Role Context).

    Args:
        merged_data (dict): The dictionary yielded by data_merger.
                            Structure: {'meta': {...}, 'input_data': {'plaintiffs':[], ...}}
        assigned_key (str, optional): Specific API Key to use (for load balancing).

    Returns:
        dict: Final structured result including status and performance metrics.
    """
    
    # 1. Unpack Data
    meta = merged_data.get('meta', {})
    input_data = merged_data.get('input_data', {})
    
    plaintiffs = input_data.get('plaintiffs', [])
    defendants = input_data.get('defendants', [])
    cost_texts = input_data.get('cost_texts', [])
    
    # Safety check: If for some reason cost_texts is empty, skip calling LLM
    if not cost_texts:
        return {
            "meta": meta,
            "status": "skipped",
            "error_msg": "No cost text found in input",
            "extraction_result": None
        }

    # 2. Key Selection (Load Balancing)
    current_key = assigned_key if assigned_key else random.choice(API_KEYS)

    # 3. Build Payload
    # The prompt_manager will join the cost_texts with newlines and format the name lists
    payload = build_payload(cost_texts, plaintiffs, defendants, MODEL_NAME)

    # 4. Execute API Call with Retry Logic
    attempt_count = 0
    rate_limit_retries = 0
    start_time = time.time()
    llm_result = None

    while True:
        attempt_count += 1
        
        # Call the LLM Client
        llm_result = call_llm_extraction(payload, current_key)
        
        # Check Success
        if llm_result['status'] == 'success':
            break
        
        # Handle Failures
        error_msg = llm_result.get('error_msg', '')
        
        # Check for 429 Rate Limit
        if is_rate_limit_error(error_msg):
            if RATE_LIMIT_MAX_RETRIES is None or rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                wait_time = compute_rate_limit_delay(rate_limit_retries)
                rate_limit_retries += 1
                # Optional: Log to console if needed, but usually main_runner handles logging
                # print(f"[429] Retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            else:
                # Exceeded max retries
                break
        
        # Other errors (500, 400, etc.) -> Break immediately, do not retry indefinitely
        break

    latency = round(time.time() - start_time, 2)

    # 5. Encapsulate Result
    final_package = {
        "meta": meta,
        "extraction_result": {}, 
        "status": "pending",
        "perf": {
            "latency": latency,
            "key_tail": current_key[-4:],
            "attempts": attempt_count,
            "rate_limit_retries": rate_limit_retries
        }
    }

    if llm_result['status'] == 'success':
        final_package['status'] = 'success'
        final_package['extraction_result'] = llm_result['data']
    else:
        final_package['status'] = 'failed'
        final_package['error_msg'] = llm_result.get('error_msg')
        final_package['raw_response'] = llm_result.get('raw_response')

    return final_package