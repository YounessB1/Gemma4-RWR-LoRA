#!/usr/bin/env python3
"""Stage 03 (GPU, H200) — RWR LoRA training.

Loads the on-disk tokenized dataset (stage 02), the base model + LoRA adapter, and
runs the weighted-loss Trainer. This is the ONLY stage that needs a GPU.

    python scripts/03_train.py --config configs/default.yaml
    python scripts/03_train.py --max-steps 20 --max-seq-len 2048   # quick GPU smoke

Resumes from the newest checkpoint in output_dir when cfg.resume is set, so a
pre-empted SLURM job can simply be resubmitted. Runs with --max-steps > 0 write to
artifacts/smoke/ (unless --output-dir/--adapter-dir are given) so they can never be
resumed into, or overwrite, the full run.
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

# captured as close to job start as possible, so the wall-clock checkpoint deadline
# (cfg.max_train_hours) is measured against SLURM's walltime, not training start
_T0 = time.time()

# Must be set BEFORE torch initializes CUDA (reduces fragmentation / recovers reserved blocks).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwr.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--model-id", dest="model_id", default=None)
    ap.add_argument("--tokenized", dest="tokenized_dir", default=None)
    ap.add_argument("--output-dir", dest="output_dir", default=None)
    ap.add_argument("--adapter-dir", dest="adapter_dir", default=None)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    ap.add_argument("--max-seq-len", dest="max_seq_len", type=int, default=None)
    ap.add_argument("--epochs", dest="epochs", type=float, default=None)
    # Per-sbatch wall-clock checkpoint+stop budget (hours). Must match the job's
    # SLURM --time: e.g. 23.5 under a 24h partition, ~94 under a 4-day one. Leave
    # unset to use the YAML value.
    ap.add_argument("--max-train-hours", dest="max_train_hours", type=float, default=None)
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=None)
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        model_id=args.model_id,
        tokenized_dir=args.tokenized_dir,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        max_train_hours=args.max_train_hours,
        load_in_4bit=args.load_in_4bit,
    ).resolve_paths(PROJECT_ROOT)

    # A short run (max_steps > 0, e.g. a smoke test) must not share output_dir/adapter_dir
    # with the real run: its final checkpoint would be auto-resumed by the full run.
    # Redirect any dir not given explicitly on the CLI into artifacts/smoke/.
    if cfg.max_steps and cfg.max_steps > 0:
        smoke_root = PROJECT_ROOT / "artifacts" / "smoke"
        if args.output_dir is None:
            cfg.output_dir = str(smoke_root / "checkpoints")
        if args.adapter_dir is None:
            cfg.adapter_dir = str(smoke_root / "adapter")
        print(f"[03] max_steps={cfg.max_steps} -> short run, writing to {cfg.output_dir} / {cfg.adapter_dir}")

    # heavy imports after config (and after the alloc env var) so --help stays fast
    import torch
    from datasets import load_from_disk
    from rwr.tokenize import load_tokenizer
    from rwr.modeling import load_model
    from rwr.trainer import RWRCollator, RWRTrainer, build_training_args, WallClockStopCallback

    print(f"[03] GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print(f"[03] model          : {cfg.model_id} | 4bit={cfg.load_in_4bit}")
    print(f"[03] tokenized data : {cfg.tokenized_dir}")

    proc = load_from_disk(cfg.tokenized_dir)
    if "bench_id" not in proc.column_names:
        raise SystemExit("[03] tokenized data has no 'bench_id' column — re-run stage 02 "
                         "(or patch the dir) so eval can hold out whole targets.")
    if "cmd_ids" not in proc.column_names or "cmd_weight_sum" not in proc.column_names:
        raise SystemExit("[03] tokenized data has no 'cmd_ids'/'cmd_weight_sum' columns — "
                         "re-run stage 02 (the per-command RWR loss needs them).")
    # Hold out WHOLE targets (bench_ids) for eval, never random runs: with ~150 runs
    # per target a random split leaks the target's flag/paths into eval. See README.
    import random as _random
    all_benches = sorted(set(proc["bench_id"]))
    if cfg.eval_benchmarks:
        known = set(all_benches)
        held = [b for b in cfg.eval_benchmarks if b in known]
        missing = [b for b in cfg.eval_benchmarks if b not in known]
        if missing:
            print(f"[03] WARNING: eval_benchmarks not in data, ignored: {missing}")
    else:
        shuffled = all_benches[:]
        _random.Random(cfg.seed).shuffle(shuffled)
        held = shuffled[: cfg.n_eval_benchmarks]
    held_set = set(held)
    if not held_set or len(held_set) >= len(all_benches):
        raise SystemExit(f"[03] bad eval holdout: {sorted(held_set)} of {len(all_benches)} targets")
    eval_ds = proc.filter(lambda r: r["bench_id"] in held_set)
    train_ds = proc.filter(lambda r: r["bench_id"] not in held_set)

    # Subsample eval to a fixed cap PER held-out target (stratified, seeded) — a weighted-mean
    # loss over ~50 trajectories/target is already stable, and keeps each eval fast. 0 = use all.
    if cfg.eval_per_benchmark and cfg.eval_per_benchmark > 0 and len(eval_ds) > 0:
        import collections
        idx_by_bench = collections.defaultdict(list)
        for i, b in enumerate(eval_ds["bench_id"]):
            idx_by_bench[b].append(i)
        rng = _random.Random(cfg.seed)
        keep = []
        for b, idxs in idx_by_bench.items():
            rng.shuffle(idxs)
            keep.extend(idxs[: cfg.eval_per_benchmark])
        eval_ds = eval_ds.select(sorted(keep))

    print(f"[03] held-out eval targets ({len(held_set)}): {sorted(held_set)}")
    print(f"[03] train {len(train_ds)} ({len(all_benches) - len(held_set)} targets) | "
          f"eval {len(eval_ds)} ({len(held_set)} targets, <={cfg.eval_per_benchmark or 'all'}/target)")

    # ONE global loss normalizer from the train set (never per trajectory), so each command's
    # pull on the model depends only on its own weight. Also used for eval -> comparable losses.
    cmd_w = train_ds["cmd_weight_sum"]
    n_cmds = train_ds["n_cmds"]
    loss_norm = sum(cmd_w) / len(cmd_w)
    print(f"[03] per-command RWR: {sum(n_cmds)} commands, mean weight "
          f"{sum(cmd_w) / sum(n_cmds):.3f}, loss_norm (mean cmd-weight sum/row) {loss_norm:.3f}")

    tok = load_tokenizer(cfg.model_id)
    model = load_model(cfg)

    args_tr = build_training_args(cfg)
    trainer = RWRTrainer(
        model=model, args=args_tr,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=RWRCollator(tok.pad_token_id),
        loss_norm=loss_norm,
        callbacks=[WallClockStopCallback(_T0 + cfg.max_train_hours * 3600)],
    )
    print(f"[03] wall-clock checkpoint+stop at {cfg.max_train_hours}h "
          f"(~{cfg.max_train_hours*60:.0f} min from job start)")

    # Resume from the newest COMPLETE checkpoint (pre-emption safe). A job killed
    # mid-save leaves a half-written checkpoint-N: HF writes trainer_state.json last,
    # so its presence marks the dir complete. We parse the step from the dir name
    # defensively (ignore any non-"checkpoint-<int>" match) and, newest-first, skip
    # incomplete dirs instead of crashing the resume on them.
    resume = None
    if cfg.resume:
        def _step(p):
            base = os.path.basename(p.rstrip("/"))
            tail = base.rsplit("-", 1)[-1]
            return int(tail) if tail.isdigit() else -1
        ckpts = sorted((p for p in glob.glob(os.path.join(cfg.output_dir, "checkpoint-*"))
                        if os.path.isdir(p) and _step(p) >= 0),
                       key=_step, reverse=True)
        for p in ckpts:
            if os.path.isfile(os.path.join(p, "trainer_state.json")):
                resume = p
                print(f"[03] resuming from {resume}")
                break
            print(f"[03] skipping incomplete checkpoint (no trainer_state.json): {p}")
        if resume is None and ckpts:
            print("[03] no complete checkpoint found — starting from scratch")

    out = trainer.train(resume_from_checkpoint=resume)
    print("[03] metrics:", out.metrics)

    model.save_pretrained(cfg.adapter_dir)
    tok.save_pretrained(cfg.adapter_dir)
    print(f"[03] saved LoRA adapter + tokenizer -> {cfg.adapter_dir}")


if __name__ == "__main__":
    main()
