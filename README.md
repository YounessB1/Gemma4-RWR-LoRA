# RED-RL — RWR LoRA fine-tuning of Gemma 4 31B (HPC)

Reward-Weighted Regression on CTF-agent trajectories: each past action (the agent's
`<thinking>` + `bash` tool call) is behavior-cloned with its per-token loss scaled by
`exp(reward/β)`, the per-step LLM-judge score. A LoRA adapter is trained on
`google/gemma-4-31b-it`.

This is the modular, HPC version of the notebook — **the pipeline is split by hardware
so only training touches the GPU.**

## Layout

```
RED-RL/
├── red_rl/                 # the package
│   ├── config.py           # one dataclass of knobs (YAML + CLI)        [CPU-safe import]
│   ├── dataset.py          # train-data/ -> rwr_dataset.jsonl           [CPU, stdlib only]
│   ├── render.py           # record -> chat messages + RWR weights      [CPU, no torch]
│   ├── tokenize.py         # render + tokenize + mask -> arrow dataset  [CPU, tokenizer]
│   ├── modeling.py         # load 4bit/bf16 model + attach LoRA         [GPU, torch]
│   └── trainer.py          # weighted chunked-CE Trainer + collator     [GPU, torch]
├── scripts/
│   ├── 01_build_dataset.py # CPU
│   ├── 02_tokenize.py      # CPU
│   ├── 03_train.py         # GPU (H200)
│   └── 04_merge_adapter.py # optional (GPU/CPU)
├── configs/default.yaml    # all defaults
├── requirements/           # conda env + pip requirement files
└── slurm/train_h200.sbatch # example H200 training job
```

## What runs where

| Stage | Script | Hardware | Needs |
|-------|--------|----------|-------|
| 01 build dataset | `scripts/01_build_dataset.py` | **CPU** | stdlib only (reads `train-data/`) |
| 02 tokenize + mask + weight | `scripts/02_tokenize.py` | **CPU** | transformers tokenizer + datasets |
| 03 train | `scripts/03_train.py` | **GPU (H200)** | torch + peft + (bitsandbytes if 4-bit) |
| 04 merge adapter (optional) | `scripts/04_merge_adapter.py` | GPU/CPU | torch + peft |

Stages 01–02 produce `artifacts/tokenized/` (an on-disk arrow dataset). That folder is
the **only** thing stage 03 consumes — so no GPU time is spent on data prep, and you can
re-run training many times against the same cached tokenization.

## Setup (miniforge)

Build a **prefix** env under `$HOME` and call its interpreter directly — no
`conda activate` (unreliable in a non-interactive SLURM shell) and no reliance on
the module's default envs dir (often not user-writable). Do the pip installs in an
interactive session, not on the login node.

```bash
module load miniforge/24.3.0-0
conda env create -f requirements/environment.yml -p ~/envs/red-rl

# GPU/training node — install the CUDA-matched torch FIRST, then the rest:
~/envs/red-rl/bin/pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu124
~/envs/red-rl/bin/pip install -r requirements/requirements-gpu.txt

# (CPU-only node for stages 01/02, if separate):
~/envs/red-rl/bin/pip install -r requirements/requirements-cpu.txt
```

`torch` is deliberately not pinned in the requirement files — install the wheel that
matches the cluster's CUDA (H200 = Hopper / sm_90, use a CUDA 12.x/13.x build).

The base model is gated: run `~/envs/red-rl/bin/huggingface-cli login` once (or export
`HF_TOKEN`) and accept the model license, or stages 02/03 will 401.

## Data

Stage 01 reads the scored `train-data/<BENCH_ID>/<run>/` tree (each run dir must have
`reward.json` + `context_window.json` + `metadata.json`). Point `train_data_dir` at it
(config or `--train-data`), or symlink it into the project dir:

```bash
ln -s /path/to/train-data train-data
```

If `train-data/` is large and lives elsewhere, you can run stage 01 wherever the tree
is, then copy just the small `artifacts/rwr_dataset.jsonl` to the cluster.

## Run

```bash
# 01 (CPU) — build the JSONL
python scripts/01_build_dataset.py --config configs/default.yaml
#   smoke subset:
python scripts/01_build_dataset.py --limit 200 --out artifacts/rwr_sample.jsonl

# 02 (CPU) — tokenize + mask + weight -> artifacts/tokenized
python scripts/02_tokenize.py --config configs/default.yaml
#   smoke:
python scripts/02_tokenize.py --jsonl artifacts/rwr_sample.jsonl --out artifacts/tokenized_smoke --max-seq-len 2048

# 03 (GPU) — train on the H200 (create the log dir first, or SLURM can't open its output file)
mkdir -p logs && sbatch slurm/train_h200.sbatch
#   or directly / for a GPU smoke (inside a GPU session):
~/envs/red-rl/bin/python scripts/03_train.py --config configs/default.yaml
~/envs/red-rl/bin/python scripts/03_train.py --tokenized artifacts/tokenized_smoke --max-steps 20 --max-seq-len 2048 \
    --output-dir artifacts/ckpt_smoke --adapter-dir artifacts/adapter_smoke
```

`03` resumes from the newest checkpoint in `output_dir` automatically, so a pre-empted
SLURM job can just be resubmitted.

## Deploy the adapter

Keep the small LoRA adapter and serve it with vLLM (no merge needed):

```bash
vllm serve google/gemma-4-31b-it --enable-lora --lora-modules red=artifacts/adapter
```

Or merge to standalone weights (~60GB) with `scripts/04_merge_adapter.py`.

## Design notes / caveats

- **RWR, not AWR/PPO**: offline reward-weighted behavior cloning, no value baseline.
  `w = min(exp(score/β), w_max)`, applied **per command**: CE is averaged over each
  command's tokens (length-neutral), multiplied by that command's own `w`, and divided by
  one global constant (mean per-row sum of command weights on the train set). No
  per-trajectory normalization, so a command's pull depends only on its own score.
- **One fixed system prompt** for every trajectory (`render.FIXED_SYSTEM_PROMPT`). It is
  masked during training (conditioning only, never a loss target).
- **Faithful action format**: rendered through the real Gemma-4 chat template with the
  `bash` tool schema; tool-call `arguments` are passed as a dict (the template requires
  it). Masking uses offset mapping over a `tokenize=False` render (the `tokenize=True`
  path is unreliable on transformers 5.x and this template has no `{% generation %}`).
- **Gemma-4 specifics handled in code**: `Gemma4ClippableLinear` LoRA targets (inner
  `.linear`), `dtype` vs `torch_dtype`, `TrainingArguments` kwarg filtering, popping
  `labels` + chunked CE to avoid the 262k-vocab fp32 OOM.
- **Epochs**: start at 1 (large self-generated corpus, flag-memorization risk); extend
  to 2–3 only if held-out CTF capture rate keeps improving.
- **GPU sizing**: H200 (141GB) runs bf16 LoRA at `max_seq_len=16384` comfortably. For a
  40GB card, set `load_in_4bit: true` and `max_seq_len: ~2048`.
