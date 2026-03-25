from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_vocab(path: Path) -> Dict[str, int]:
    vocab = json.loads(path.read_text(encoding="utf-8"))
    if "<pad>" not in vocab or "<unk>" not in vocab:
        raise ValueError("Vocab must contain <pad> and <unk>")
    return vocab

