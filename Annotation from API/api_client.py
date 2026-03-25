from __future__ import annotations

import importlib.util
from pathlib import Path

from config import API_PIPELINE_SCRIPTS


def _load_llm_client():
    llm_client_path = Path(API_PIPELINE_SCRIPTS) / "llm_client.py"
    spec = importlib.util.spec_from_file_location("api_pipeline_llm_client", llm_client_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_llm_client = _load_llm_client()
call_llm_extraction = _llm_client.call_llm_extraction

__all__ = ["call_llm_extraction"]
