from __future__ import annotations

from typing import Dict, List, Tuple

PRONOUNS = {"同法", "本法", "该法", "本条例", "该条例", "本规定", "该规定"}


def resolve_pronouns(citations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Resolve implicit law names like '同法'/'本法'/'该法'.
    Returns (resolved_citations, ambiguous_list)
    """
    resolved: List[Dict] = []
    ambiguous: List[Dict] = []

    # Collect explicit law names
    explicit_laws = []
    for c in citations:
        name = (c.get("law_name") or "").strip()
        if name and name not in PRONOUNS:
            explicit_laws.append(name)

    unique_explicit = list(dict.fromkeys(explicit_laws))
    last_explicit = unique_explicit[-1] if unique_explicit else ""

    for c in citations:
        item = dict(c)
        name = (item.get("law_name") or "").strip()
        if name in PRONOUNS or not name:
            if last_explicit:
                item["law_name"] = last_explicit
            elif len(unique_explicit) == 1:
                item["law_name"] = unique_explicit[0]
            else:
                ambiguous.append(item)
                continue
        else:
            last_explicit = name
            if name not in unique_explicit:
                unique_explicit.append(name)
        resolved.append(item)

    return resolved, ambiguous
