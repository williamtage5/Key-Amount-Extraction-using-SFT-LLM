from __future__ import annotations

import re
import unicodedata


def to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_brackets(text: str) -> str:
    return (
        text.replace("〈", "《")
        .replace("〉", "》")
        .replace("＜", "《")
        .replace("＞", "》")
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    text = to_halfwidth(text)
    text = normalize_brackets(text)
    text = normalize_whitespace(text)
    return text
