from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition yishen CSV samples into Z1~Z7 and save JSON files by year/month."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Input root directory (default: <workspace>/data/yishen)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory (default: <workspace>/data/yishen_partition)",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--anchor", type=str, default="auto", choices=["auto", "off"])
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most N csv files (for smoke test).",
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="Process at most N rows per csv file (for smoke test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output json files.",
    )
    return parser.parse_args()


def detect_workspace_root(script_path: Path) -> Path:
    # Script path: <workspace>/Segment/run_partition_yishen.py
    return script_path.resolve().parents[1]


def resolve_model_src(workspace_root: Path) -> Path:
    candidates = [
        workspace_root / "model" / "judgment_partition_infer" / "src",
        workspace_root / "Segment" / "model" / "judgment_partition_infer" / "src",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find judgment_partition_infer src. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def infer_year_month(file_path: Path) -> Tuple[str, str]:
    stem = file_path.stem
    m = re.search(r"(\d{4})-(\d{2})", stem)
    if m:
        return m.group(1), f"{m.group(1)}-{m.group(2)}"

    parent = file_path.parent.name
    year = parent if re.fullmatch(r"\d{4}", parent) else "unknown"
    return year, stem


def normalize_relpath(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(data_dir.rglob("*.csv"))


def normalize_row_keys(row: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    normalized: Dict[str, Optional[str]] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[key.lstrip("\ufeff")] = value
    return normalized


def main() -> int:
    args = parse_args()
    script_path = Path(__file__)
    workspace_root = detect_workspace_root(script_path)

    data_dir = Path(args.data_dir) if args.data_dir else workspace_root / "data" / "yishen"
    output_root = (
        Path(args.output_root) if args.output_root else workspace_root / "data" / "yishen_partition"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {data_dir}")

    model_src = resolve_model_src(workspace_root)
    if str(model_src) not in sys.path:
        sys.path.insert(0, str(model_src))

    from judgment_partition_infer import Predictor  # pylint: disable=import-outside-toplevel

    csv_files = list(iter_csv_files(data_dir))
    if args.max_files is not None:
        csv_files = csv_files[: args.max_files]

    print(f"[INFO] workspace_root={workspace_root}")
    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] model_src={model_src}")
    print(f"[INFO] csv_files={len(csv_files)}")

    predictor = Predictor(device=args.device, anchor=args.anchor)

    run_stats: Dict[str, int] = {
        "files_seen": 0,
        "files_written": 0,
        "files_skipped_existing": 0,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_skipped_empty": 0,
        "rows_failed": 0,
    }

    for csv_path in csv_files:
        run_stats["files_seen"] += 1
        year, year_month = infer_year_month(csv_path)
        out_dir = output_root / year
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year_month}.json"

        if out_path.exists() and not args.overwrite:
            run_stats["files_skipped_existing"] += 1
            print(f"[SKIP] exists: {out_path}")
            continue

        file_rows_seen = 0
        file_rows_written = 0
        file_rows_skipped_empty = 0
        file_rows_failed = 0

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f_in, out_path.open(
            "w", encoding="utf-8"
        ) as f_out:
            reader = csv.DictReader(f_in)
            first = True
            f_out.write("[\n")

            for row_idx, row in enumerate(reader, start=1):
                if args.max_samples_per_file is not None and file_rows_seen >= args.max_samples_per_file:
                    break
                file_rows_seen += 1

                row_n = normalize_row_keys(row)
                text = (row_n.get("全文") or row_n.get("full_text") or row_n.get("text") or "").strip()
                if not text:
                    file_rows_skipped_empty += 1
                    continue

                sample_id = f"{year_month}_{row_idx:06d}"
                extra_fields = {
                    "sample_id": sample_id,
                    "case_no": row_n.get("案号"),
                    "case_name": row_n.get("案件名称"),
                }

                try:
                    pred = predictor.predict_text(text, extra_fields=extra_fields)
                except Exception as exc:  # keep long-run robustness on large datasets
                    file_rows_failed += 1
                    print(f"[WARN] prediction failed: file={csv_path.name} row={row_idx} err={exc}")
                    continue

                out_record = {
                    "sample_id": sample_id,
                    "source_file": normalize_relpath(csv_path, workspace_root),
                    "case_no": row_n.get("案号"),
                    "case_name": row_n.get("案件名称"),
                    "judgment_date": row_n.get("裁判日期"),
                    "url": row_n.get("原始链接"),
                    "text_length": pred.get("text_length"),
                    "anchor_status": pred.get("anchor_status"),
                    "anchor_end": pred.get("anchor_end"),
                    "boundaries": pred.get("boundaries"),
                    "zones": pred.get("zones"),
                }

                if not first:
                    f_out.write(",\n")
                f_out.write(json.dumps(out_record, ensure_ascii=False))
                first = False
                file_rows_written += 1

            f_out.write("\n]\n")

        run_stats["files_written"] += 1
        run_stats["rows_seen"] += file_rows_seen
        run_stats["rows_written"] += file_rows_written
        run_stats["rows_skipped_empty"] += file_rows_skipped_empty
        run_stats["rows_failed"] += file_rows_failed
        print(
            "[DONE] "
            f"{csv_path.name} -> {out_path} "
            f"(seen={file_rows_seen}, written={file_rows_written}, "
            f"empty={file_rows_skipped_empty}, failed={file_rows_failed})"
        )

    summary_path = output_root / "run_summary.json"
    summary = {
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "device": args.device,
        "anchor": args.anchor,
        "max_files": args.max_files,
        "max_samples_per_file": args.max_samples_per_file,
        "overwrite": args.overwrite,
        "stats": run_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
