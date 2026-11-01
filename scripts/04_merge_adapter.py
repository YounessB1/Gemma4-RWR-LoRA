#!/usr/bin/env python3
"""Stage 04 (optional, GPU/CPU) — merge the LoRA adapter into full bf16 weights.

Only needed if you want a single standalone checkpoint with no PEFT dependency.
Produces ~60GB of weights, so mind the disk. For serving, prefer keeping the small
adapter and loading it with vLLM `--enable-lora` (no merge required).

    python scripts/04_merge_adapter.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwr.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--adapter-dir", dest="adapter_dir", default=None)
    ap.add_argument("--merged-dir", dest="merged_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, adapter_dir=args.adapter_dir, merged_dir=args.merged_dir).resolve_paths(PROJECT_ROOT)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[04] merging {cfg.adapter_dir} into base {cfg.model_id} (bf16) ...")
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16,
                                                    device_map="auto", trust_remote_code=True)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16,
                                                    device_map="auto", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, cfg.adapter_dir).merge_and_unload()
    merged.save_pretrained(cfg.merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg.adapter_dir).save_pretrained(cfg.merged_dir)
    print(f"[04] saved merged model -> {cfg.merged_dir}")


if __name__ == "__main__":
    main()
