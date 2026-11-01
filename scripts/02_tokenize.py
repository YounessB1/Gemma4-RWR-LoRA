#!/usr/bin/env python3
"""Stage 02 (CPU) — render + tokenize + mask + weight, save an arrow dataset to disk.

Needs transformers (fast tokenizer) + datasets, but NOT torch/CUDA. The tokenizer
download is tiny. This is the heavy CPU step; its output (an on-disk arrow dataset)
is the only thing the GPU training stage consumes, so GPU hours aren't spent tokenizing.

    python scripts/02_tokenize.py --config configs/default.yaml
    python scripts/02_tokenize.py --jsonl artifacts/rwr_sample.jsonl --max-seq-len 2048

If you changed the dataset or prompt, delete the old tokenized dir first (its content
is keyed by neither file name nor prompt — see --out) or pass a fresh --out.
"""

import argparse
import os
import sys
from pathlib import Path

# keep the HF fast-tokenizer from forking its own thread pool under datasets' num_proc
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwr.config import load_config
from rwr.tokenize import load_tokenizer, tokenize_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--model-id", dest="model_id", default=None)
    ap.add_argument("--jsonl", dest="dataset_jsonl", default=None)
    ap.add_argument("--out", dest="tokenized_dir", default=None, help="output arrow dataset dir")
    ap.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None)
    ap.add_argument("--num-proc", dest="num_proc", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        model_id=args.model_id,
        dataset_jsonl=args.dataset_jsonl,
        tokenized_dir=args.tokenized_dir,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
    ).resolve_paths(PROJECT_ROOT)

    print(f"[02] model/tokenizer : {cfg.model_id}")
    print(f"[02] input JSONL     : {cfg.dataset_jsonl}")
    print(f"[02] max_seq_len     : {cfg.max_seq_len} | num_proc {cfg.num_proc}")

    tok = load_tokenizer(cfg.model_id)
    proc = tokenize_dataset(cfg.dataset_jsonl, tok, cfg)

    import numpy as np
    seqs = np.array([len(x) for x in proc["input_ids"]])
    loss = np.array(proc["n_loss_tokens"])
    print(f"[02] usable records  : {len(proc)}")
    print(f"[02] seq len  mean {seqs.mean():.0f}  p95 {np.percentile(seqs,95):.0f}  max {seqs.max()}")
    print(f"[02] loss tok mean {loss.mean():.0f}  total {loss.sum()}")

    Path(cfg.tokenized_dir).parent.mkdir(parents=True, exist_ok=True)
    proc.save_to_disk(cfg.tokenized_dir)
    print(f"[02] saved tokenized dataset -> {cfg.tokenized_dir}")


if __name__ == "__main__":
    main()
