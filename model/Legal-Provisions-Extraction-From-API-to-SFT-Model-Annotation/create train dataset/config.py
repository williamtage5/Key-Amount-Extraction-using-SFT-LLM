import os
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "Z4_Reasoning_extraction"
OUTPUT_ROOT = PROJECT_ROOT / "data"
API_KEYS_FILE = Path(__file__).resolve().parent / "api_keys.json"

# API settings are reused from the moved pipeline repo
API_PIPELINE_SCRIPTS = PROJECT_ROOT / "model" / "api_prompt_pipeline" / "scripts"

def _load_api_config():
    api_config_path = API_PIPELINE_SCRIPTS / "config.py"
    spec = None
    if api_config_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("api_pipeline_config", api_config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    raise FileNotFoundError(f"API config not found: {api_config_path}")


_api_cfg = _load_api_config()

API_BASE_URL = _api_cfg.API_BASE_URL
if API_KEYS_FILE.exists():
    import json

    API_KEYS = json.loads(API_KEYS_FILE.read_text(encoding="utf-8"))
else:
    API_KEYS = _api_cfg.API_KEYS
MODEL_NAME = _api_cfg.MODEL_NAME
API_TIMEOUT = _api_cfg.API_TIMEOUT
RATE_LIMIT_INITIAL_DELAY = _api_cfg.RATE_LIMIT_INITIAL_DELAY
RATE_LIMIT_BACKOFF_FACTOR = _api_cfg.RATE_LIMIT_BACKOFF_FACTOR
RATE_LIMIT_MAX_DELAY = _api_cfg.RATE_LIMIT_MAX_DELAY
RATE_LIMIT_MAX_RETRIES = _api_cfg.RATE_LIMIT_MAX_RETRIES

# Test settings
DEFAULT_TEST_LIMIT = 50

# Runtime behavior
TEMPERATURE = 0.0
MAX_TOKENS = 4096
