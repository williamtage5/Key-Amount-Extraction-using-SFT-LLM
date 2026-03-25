from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "SFT_model" / "qwen2.5-7b-law-ie-merged" / "qwen2.5-7b-law-ie-merged"

INPUT_DIR = PROJECT_ROOT / "data" / "Z4_Reasoning_extraction"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "SFT model annotation"

OLLAMA_MODEL = "Law-Qwen-4bit"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 1.0

PROMPT_TEMPLATE = (
    "你是中国法律条文结构化抽取专家。"
    "请从输入的裁判理由/法律适用文本中，抽取所有被引用的法律条文，"
    "输出JSON对象，格式为 {\"citations\":[{\"law_name\":\"...\",\"article\":\"...\"}]}。"
    "仅输出JSON，不要解释。\n\n"
    "【待抽取文本】\n"
    "{text}\n"
)
