from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

LAW_TITLE_RE = re.compile(r"[《〈]([^》〉]+)[》〉]")
ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两\d]+条")
SENT_END_RE = re.compile(r"[。；;]")


def iter_records(input_dir: Path):
    for file_path in sorted(input_dir.rglob("*.json")):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(data, list):
                continue
        except Exception:
            continue
        for rec in data:
            yield rec


def extract_expected(z4_text: str) -> Dict[str, List[str]]:
    expected: Dict[str, List[str]] = {}
    matches = list(LAW_TITLE_RE.finditer(z4_text))
    if not matches:
        return expected

    for i, m in enumerate(matches):
        law = m.group(1)
        start = m.end()
        end = len(z4_text)

        # stop at next law title
        if i + 1 < len(matches):
            end = min(end, matches[i + 1].start())

        # also stop at sentence end
        tail = z4_text[start:end]
        sent_end = SENT_END_RE.search(tail)
        if sent_end:
            end = start + sent_end.start()

        seg = z4_text[start:end]
        arts = ARTICLE_RE.findall(seg)
        if arts:
            seen = []
            for a in arts:
                if a not in seen:
                    seen.append(a)
            expected[law] = seen

    return expected


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-article extraction completeness.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    issues = []

    for rec in iter_records(input_dir):
        z4 = rec.get("Z4_Reasoning") or ""
        if "《" not in z4 and "〈" not in z4:
            continue
        expected = extract_expected(z4)
        if not expected:
            continue

        citations = rec.get("citations") or []
        if not isinstance(citations, list):
            continue

        for law, arts in expected.items():
            if len(arts) < 2:
                continue
            got = [c for c in citations if isinstance(c, dict) and (c.get("law_name") or "") == law]
            got_articles = [c.get("article") for c in got if c.get("article")]
            if len(set(got_articles)) < len(set(arts)):
                issues.append({
                    "case_no": rec.get("case_no"),
                    "law": law,
                    "articles_in_text": arts,
                    "articles_in_citations": list(dict.fromkeys(got_articles)),
                })
                if len(issues) >= args.max_cases:
                    break
        if len(issues) >= args.max_cases:
            break

    print(f"found {len(issues)} candidate issues")
    for item in issues:
        print(item)


if __name__ == "__main__":
    main()
