from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    INPUT_DIR,
    OUTPUT_ROOT,
    MODEL_PATH,
    MAX_NEW_TOKENS,
    PROMPT_TEMPLATE,
    TEMPERATURE,
    TOP_P,
)
from io_utils import iter_records, ensure_output_path, write_json_array
from parser_utils import extract_json_object


def build_prompt(text: str) -> str:
    # Avoid str.format to prevent conflicts with JSON braces in the template.
    return PROMPT_TEMPLATE.replace("{text}", text)


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install a CUDA-enabled PyTorch and NVIDIA driver.")
    # Disable async weight materialization to avoid Windows access violation during load.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        trust_remote_code=True,
        quantization_config=bnb_config,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def generate_one(tokenizer, model, text: str) -> Dict:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=False,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # remove prompt prefix if present
    if generated.startswith(prompt):
        generated = generated[len(prompt) :]
    return extract_json_object(generated)


def main():
    parser = argparse.ArgumentParser(description="Annotate Z4_Reasoning with local SFT model.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    args = parser.parse_args()

    tokenizer, model = load_model()

    current_file = None
    buffer: List[Dict] = []
    count = 0

    for file_path, record in iter_records(INPUT_DIR):
        if current_file is None:
            current_file = file_path

        if file_path != current_file:
            out_path = ensure_output_path(OUTPUT_ROOT, current_file)
            write_json_array(out_path, buffer)
            buffer = []
            current_file = file_path

        z4 = (record.get("Z4_Reasoning") or "").strip()
        if not z4:
            continue

        result = generate_one(tokenizer, model, z4)
        citations = result.get("citations", []) if isinstance(result, dict) else []

        buffer.append(
            {
                "source_file": record.get("source_file"),
                "case_no": record.get("case_no"),
                "case_name": record.get("case_name"),
                "Z4_Reasoning": z4,
                "citations": citations,
            }
        )

        count += 1
        if args.limit is not None and count >= args.limit:
            break

    if current_file is not None and buffer:
        out_path = ensure_output_path(OUTPUT_ROOT, current_file)
        write_json_array(out_path, buffer)

    print(f"Done. Samples: {count}. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
