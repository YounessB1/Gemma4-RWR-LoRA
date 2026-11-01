"""Single source of truth for every knob, shared by all stages (CPU + GPU).

A plain dataclass with sensible defaults. Load order (later overrides earlier):
    defaults  <  YAML file (--config)  <  explicit CLI flags

Pure Python + pyyaml; importing this module pulls in NO torch/transformers, so it is
safe to use from the CPU-only scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # ---- base model -------------------------------------------------------
    # The weights are Google's gated `google/gemma-4-31b-it`. `polito/gemma-4-31b`
    # is only PoliTO's vLLM serving alias (NOT an HF repo). Log in to HF if a
    # download 401s (accept the Gemma license first).
    model_id: str = "google/gemma-4-31b-it"
    # bf16 LoRA (False) vs 4-bit QLoRA (True). On the H200 (141GB) bf16 LoRA fits
    # comfortably and is higher fidelity. Set True only for <80GB GPUs.
    load_in_4bit: bool = False
    # Attention impl: try this, then fall back sdpa -> eager. "flash_attention_2"
    # only if the flash-attn wheel is installed.
    attn_impl: str = "sdpa"

    # ---- data paths (relative to the project dir, or absolute) ------------
    train_data_dir: str = "train-data"                 # the scored run-dir tree (input to stage 01)
    dataset_jsonl: str = "artifacts/rwr_dataset.jsonl"  # stage 01 output / stage 02 input
    tokenized_dir: str = "artifacts/tokenized"          # stage 02 output / stage 03 input (arrow dir)

    # ---- dataset build + tokenization ------------------------------------
    output_cap: int = 6000        # max chars per tool output (head+tail kept); 0 = no cap
    max_seq_len: int = 24576      # middle-drop longer trajectories, keeping prompt + recent tail (H200: 24576; 40GB: ~2048)
    num_proc: int = 8             # CPU workers for tokenization (set 1 if you hit tokenizer issues)

    # ---- held-out eval -----------------------------------------------------
    # Eval holds out WHOLE targets (bench_ids), never random runs: with ~150 runs
    # per target a random split leaks the target's flag/paths into eval, so eval
    # loss on a seen target just measures memorization. Holding out whole targets
    # makes eval loss a real cross-target / flag-memorization signal.
    n_eval_benchmarks: int = 3    # hold out this many whole targets (chosen by seed) for eval
    eval_benchmarks: tuple = ()   # explicit bench_ids to hold out; overrides n_eval_benchmarks
    eval_steps: int = 100         # run held-out eval every N steps (0 = eval off)
    eval_per_benchmark: int = 50  # cap eval trajectories sampled per held-out target (0 = use all)

    # ---- RWR weighting ----------------------------------------------------
    # weight on an assistant action (command) = min(exp(score / beta), w_max).
    # Per-command loss: CE is averaged over each command's tokens, times its own weight,
    # divided by ONE global constant — so a command's pull depends only on its own score,
    # never on the other commands in its trajectory.
    beta: float = 0.5
    w_max: float = 20.0

    # ---- LoRA -------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # projection names; Gemma-4 wraps each in Gemma4ClippableLinear(.linear=Linear),
    # which modeling.py handles automatically by targeting the inner `.linear`.
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")

    # ---- training ---------------------------------------------------------
    per_device_batch: int = 1
    grad_accum: int = 16          # effective batch = per_device_batch * grad_accum
    lr: float = 1e-4
    warmup_steps: int = 20        # transformers 5.x dropped warmup_ratio; use explicit steps (~3% of ~560)
    epochs: float = 1.0           # start at 1 (see README: why 1 epoch); extend empirically
    max_steps: int = -1           # >0 overrides epochs (use a small value for a smoke test)
    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 3
    # Wall-clock safety: checkpoint + stop cleanly at this many hours, BEFORE SLURM's hard
    # walltime kill (24h) which can truncate a checkpoint. 23.5 leaves ~30 min of margin.
    max_train_hours: float = 23.5
    gradient_checkpointing: bool = True
    seed: int = 0
    resume: bool = True           # resume from the newest checkpoint in output_dir if present

    # ---- outputs ----------------------------------------------------------
    output_dir: str = "artifacts/checkpoints"  # Trainer checkpoints (persistent on HPC)
    adapter_dir: str = "artifacts/adapter"     # final LoRA adapter
    merged_dir: str = "artifacts/merged"       # optional merged full weights (~60GB)

    def resolve_paths(self, root: Path) -> "Config":
        """Make every relative path absolute against `root` (the project dir).
        Absolute paths are left untouched. Returns self for chaining."""
        for name in ("train_data_dir", "dataset_jsonl", "tokenized_dir",
                     "output_dir", "adapter_dir", "merged_dir"):
            p = Path(getattr(self, name))
            if not p.is_absolute():
                setattr(self, name, str((root / p).resolve()))
        return self


def load_config(path: str | None = None, **overrides) -> Config:
    """Build a Config from an optional YAML file plus explicit keyword overrides.
    `overrides` with value None are ignored (so argparse defaults don't clobber YAML)."""
    data: dict = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(**data)
    for name in ("lora_targets", "eval_benchmarks"):   # YAML gives lists; the dataclass wants tuples
        if isinstance(getattr(cfg, name), list):
            setattr(cfg, name, tuple(getattr(cfg, name)))
    return cfg
