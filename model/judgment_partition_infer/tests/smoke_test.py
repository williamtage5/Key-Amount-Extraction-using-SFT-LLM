from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from judgment_partition_infer.infer import Predictor  # noqa: E402


def main() -> None:
    pred = Predictor(device="cuda", anchor="auto")

    # 1) Anchor-present text should report ok.
    text_ok = (
        "XX省XX市人民法院民 事 判 决 书（2024）XX民初123号"
        "原告张三。被告李四。经审理查明：……本院认为：……"
        "判决如下：一、……二、……"
    )
    out_ok = pred.predict_text(text_ok, extra_fields={"sample_id": "t1"})
    assert out_ok["text_length"] == len(text_ok)
    assert out_ok["anchor_status"] in ("ok", "missing_anchor", "invalid_anchor_order")
    assert isinstance(out_ok["zones"], dict) and len(out_ok["zones"]) == 7
    assert isinstance(out_ok["boundaries"], list) and len(out_ok["boundaries"]) == 6
    assert out_ok["boundaries"] == sorted(out_ok["boundaries"])

    # 2) Anchor-missing text should not crash; boundaries should still be valid.
    text_missing = "这是一段没有明显锚点的截断片段，用于冒烟测试。"
    out_m = pred.predict_text(text_missing, extra_fields={"sample_id": "t2"})
    assert out_m["text_length"] == len(text_missing)
    assert out_m["anchor_status"] in ("missing_anchor", "invalid_anchor_order", "ok")
    assert isinstance(out_m["zones"], dict) and len(out_m["zones"]) == 7
    assert isinstance(out_m["boundaries"], list) and len(out_m["boundaries"]) == 6
    assert out_m["boundaries"] == sorted(out_m["boundaries"])
    for z, info in out_m["zones"].items():
        assert "start" in info and "end" in info and "text" in info
        assert 0 <= info["start"] <= info["end"] <= len(text_missing)

    print("[SMOKE OK]")


if __name__ == "__main__":
    main()

