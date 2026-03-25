import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

project_root = Path(__file__).resolve().parents[1]
model_src = project_root / "model" / "judgment_partition_infer" / "src"
if str(model_src) not in sys.path:
    sys.path.insert(0, str(model_src))

from judgment_partition_infer import Predictor

data_dir = project_root / "data" / "yishen"
out_root = project_root / "data" / "Z4_Reasoning_extraction"
out_root.mkdir(parents=True, exist_ok=True)

files = sorted(data_dir.rglob("*.csv"))

predictor = Predictor(device="cpu")

for file_path in files:
    stem = file_path.stem  # e.g., 2013-05
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0].isdigit():
        year = parts[0]
    else:
        year = "unknown"

    year_dir = out_root / year
    year_dir.mkdir(parents=True, exist_ok=True)

    out_path = year_dir / f"{stem}.json"

    with file_path.open("r", encoding="utf-8-sig", newline="") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        first = True
        f_out.write("[\n")
        for idx, row in enumerate(reader):
            text = row.get("全文") or row.get("full_text") or row.get("text") or ""
            if not text:
                continue
            extra = {
                "source_file": str(file_path.relative_to(project_root)),
                "case_no": row.get("案号"),
                "case_name": row.get("案件名称"),
            }
            pred = predictor.predict_text(text, extra_fields=extra)
            zones = pred.get("zones") or {}
            z4 = zones.get("Z4_Reasoning") or {}
            z4_text = z4.get("text") if isinstance(z4, dict) else ""

            out_record = {
                "source_file": extra["source_file"],
                "case_no": extra["case_no"],
                "case_name": extra["case_name"],
                "Z4_Reasoning": z4_text,
            }
            if not first:
                f_out.write(",\n")
            f_out.write(json.dumps(out_record, ensure_ascii=False))
            first = False
        f_out.write("\n]\n")

print(f"Done. Wrote outputs under: {out_root}")
