from __future__ import annotations

from typing import List, Optional, Tuple

from .constants import (
    SENTENCE_SPLIT_REGEX,
    Z1_ANCHOR_CHAR,
    Z1_ANCHOR_MAX_CHARS,
    Z4_ANCHOR_REGEX,
)


def find_z1_anchor(text: str, max_chars: int = Z1_ANCHOR_MAX_CHARS) -> Optional[int]:
    if not text:
        return None
    limit = min(len(text), max_chars)
    idx = text.rfind(Z1_ANCHOR_CHAR, 0, limit)
    if idx == -1:
        return None
    return idx + 1


def find_z4_anchor(text: str) -> Optional[int]:
    if not text:
        return None
    match = Z4_ANCHOR_REGEX.search(text)
    if not match:
        return None
    return match.end()


def sentence_boundaries(text: str) -> List[int]:
    """
    Return a sorted unique list of candidate boundary positions (character offsets),
    including 0 and len(text). Also inject Z1/Z4 anchors as additional candidates.
    """
    if not text:
        return [0]

    boundaries = [0]
    for match in SENTENCE_SPLIT_REGEX.finditer(text):
        end = match.end()
        if end > boundaries[-1]:
            boundaries.append(end)

    z1_end = find_z1_anchor(text)
    z4_end = find_z4_anchor(text)
    for pos in (z1_end, z4_end):
        if pos is not None and 0 < pos <= len(text):
            boundaries.append(pos)

    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    return boundaries


def build_sentence_slices(text: str) -> List[Tuple[int, int]]:
    bounds = sentence_boundaries(text)
    slices: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e > s:
            slices.append((s, e))
    return slices

