from __future__ import annotations

import re
from typing import Dict, List

from text_normalizer import normalize_for_match


def _build_law_article_patterns(law_name: str, article: str) -> List[str]:
    # Allow optional book quotes around law name
    law = re.escape(law_name)
    article_esc = re.escape(article)
    return [
        f"《{law}》{article_esc}",
        f"〈{law}〉{article_esc}",
        f"{law}{article_esc}",
    ]


def match_citation(z4_text: str, citation: Dict) -> bool:
    if not z4_text:
        return False
    raw = z4_text
    raw_norm = normalize_for_match(raw)

    source_span = (citation.get("source_span") or "").strip()
    law_name = (citation.get("law_name") or "").strip()
    article = (citation.get("article") or "").strip()

    # 1) Try source_span match
    if source_span:
        if source_span in raw:
            return True
        if normalize_for_match(source_span) in raw_norm:
            return True

    # 2) Try law_name + article match
    if law_name and article:
        article_variants = {article}
        if article.startswith("第"):
            article_variants.add(article[1:])
        patterns = []
        for av in article_variants:
            patterns.extend(_build_law_article_patterns(law_name, av))
        for p in patterns:
            if p in raw:
                return True
            if normalize_for_match(p) in raw_norm:
                return True

        # 3) If law_name appears and articles are listed without repeating law_name,
        # match by article within the nearest law segment.
        law_pat = re.compile(rf"[《〈]{re.escape(law_name)}[》〉]")
        matches = list(law_pat.finditer(raw))
        if matches:
            for i, m in enumerate(matches):
                seg_start = m.end()
                seg_end = len(raw)
                if i + 1 < len(matches):
                    seg_end = min(seg_end, matches[i + 1].start())
                # stop at sentence end
                tail = raw[seg_start:seg_end]
                sent_end = re.search(r"[。；;]", tail)
                if sent_end:
                    seg_end = seg_start + sent_end.start()
                seg = raw[seg_start:seg_end]
                if any(av in seg for av in article_variants):
                    return True
                if any(normalize_for_match(av) in normalize_for_match(seg) for av in article_variants):
                    return True

        # 4) Fallback variants for law name (e.g., remove "最高人民法院" prefix).
        law_variants = {law_name}
        if law_name.startswith("最高人民法院"):
            law_variants.add(law_name.replace("最高人民法院", "", 1).lstrip("关于"))
            law_variants.add(law_name.replace("最高人民法院关于", "关于", 1))

        for lv in list(law_variants):
            if lv and (lv in raw or normalize_for_match(lv) in raw_norm):
                if any(av in raw for av in article_variants) or any(
                    normalize_for_match(av) in raw_norm for av in article_variants
                ):
                    return True

        # 5) Fallback: if law name appears anywhere and article appears anywhere,
        # accept to reduce false negatives (e.g., across sentence boundaries).
        if law_name in raw and any(av in raw for av in article_variants):
            return True
        if normalize_for_match(law_name) in raw_norm and any(
            normalize_for_match(av) in raw_norm for av in article_variants
        ):
            return True

    return False


def filter_aligned_citations(z4_text: str, citations: List[Dict]) -> List[Dict]:
    return [c for c in citations if match_citation(z4_text, c)]
