from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "yishen_partition"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "loan_amount_extraction"

API_PIPELINE_SCRIPTS = PROJECT_ROOT / "model" / "api_prompt_pipeline" / "scripts"
API_KEYS_FILE = (
    PROJECT_ROOT
    / "model"
    / "Legal-Provisions-Extraction-From-API-to-SFT-Model-Annotation"
    / "create train dataset"
    / "api_keys.json"
)


def _load_api_config():
    api_config_path = API_PIPELINE_SCRIPTS / "config.py"
    if not api_config_path.exists():
        raise FileNotFoundError(f"API config not found: {api_config_path}")

    spec = importlib.util.spec_from_file_location("api_pipeline_config", api_config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_api_cfg = _load_api_config()

if API_KEYS_FILE.exists():
    API_KEYS = json.loads(API_KEYS_FILE.read_text(encoding="utf-8"))
else:
    API_KEYS = _api_cfg.API_KEYS

MODEL_NAME = _api_cfg.MODEL_NAME
API_BASE_URL = _api_cfg.API_BASE_URL
API_TIMEOUT = _api_cfg.API_TIMEOUT
RATE_LIMIT_INITIAL_DELAY = _api_cfg.RATE_LIMIT_INITIAL_DELAY
RATE_LIMIT_BACKOFF_FACTOR = _api_cfg.RATE_LIMIT_BACKOFF_FACTOR
RATE_LIMIT_MAX_DELAY = _api_cfg.RATE_LIMIT_MAX_DELAY
RATE_LIMIT_MAX_RETRIES = _api_cfg.RATE_LIMIT_MAX_RETRIES

TEMPERATURE = 0.0
MAX_TOKENS = 1024
