#!/usr/bin/env python3
"""Stage 01 (CPU) — build the compact RWR JSONL from the train-data/ tree.

Pure stdlib (no torch/transformers). Run on any CPU node or your laptop.

    python scripts/01_build_dataset.py --config configs/default.yaml
    python scripts/01_build_dataset.py --train-data /path/to/train-data --limit 200 \
        --out artifacts/rwr_sample.jsonl        # small smoke-test dataset
"""

import argparse
import sys
from pathlib import Path

# make `red_rl` importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwr.config import load_config
from rwr.dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--train-data", dest="train_data_dir", default=None, help="override train-data dir")
    ap.add_argument("--out", dest="dataset_jsonl", default=None, help="override output JSONL path")
    ap.add_argument("--output-cap", dest="output_cap", type=int, default=None)
    ap.add_argument("--benchmark", default=None, help="restrict to one benchmark id")
    ap.add_argument("--limit", type=int, default=None, help="emit only the first N records")
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        train_data_dir=args.train_data_dir,
        dataset_jsonl=args.dataset_jsonl,
        output_cap=args.output_cap,
    ).resolve_paths(PROJECT_ROOT)

    print(f"[01] train-data : {cfg.train_data_dir}")
    print(f"[01] output cap : {cfg.output_cap} chars/output")
    stats = build_dataset(cfg.train_data_dir, cfg.dataset_jsonl, cfg.output_cap,
                          benchmark=args.benchmark, limit=args.limit)
    print(f"[01] wrote {stats['records']} records "
          f"({stats['graded_steps']} graded steps, {stats['successful_runs']} successful) "
          f"-> {stats['out']}")


if __name__ == "__main__":
    main()
