from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running without installation: `python infer_cli.py ...`
BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_DIR = BUNDLE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from judgment_partition_infer.infer import Predictor, default_run_dir, write_run_meta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="judgment_partition_infer (JSONL -> JSONL)")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output dir. Default: ./output/<timestamp>/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output jsonl path (overrides output-root/timestamp).",
    )
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint (.pt)")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab json")
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu (cuda falls back to cpu)")
    parser.add_argument("--anchor", type=str, default="auto", choices=["auto", "off"])
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N samples")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    output_root = Path(args.output_root) if args.output_root else (BUNDLE_ROOT / "output")
    run_dir = default_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else (run_dir / "predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(
        model_path=Path(args.model) if args.model else None,
        vocab_path=Path(args.vocab) if args.vocab else None,
        device=args.device,
        anchor=args.anchor,
    )

    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "run_dir": str(run_dir),
        "device_requested": args.device,
        "device_used": str(predictor.torch_device),
        "anchor": args.anchor,
        "model_path": str(predictor.model_path),
        "vocab_path": str(predictor.vocab_path),
    }
    write_run_meta(run_dir / "run_meta.json", meta)

    written = 0
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Infer", unit="line"):
            if args.max_samples is not None and written >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            out = predictor.predict_record(record)
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"[DONE] samples={written} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

