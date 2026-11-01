#!/bin/bash
#SBATCH --job-name=red-rl-train
#SBATCH --partition=gpu_h200         # Legion's H200 partition (24h max walltime)
#SBATCH --gres=gpu:1                 # node advertises Gres=gpu:8 (no type) -> request gpu:N, not gpu:h200:N
#SBATCH --cpus-per-task=16           # 1 GPU's fair share on compute-7-12 (128 cores / 8 GPUs)
#SBATCH --mem=192G                   # well under the ~257G/GPU share; bump if you hit a RAM OOM
#SBATCH --time=24:00:00              # gpu_h200 hard cap is 24h; checkpoint + resume for longer runs
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ---------------------------------------------------------------------------
# Stage 03 (GPU): RWR LoRA training on one H200.
# Assumes stages 01 + 02 already produced artifacts/tokenized (run those on a CPU
# node or the login node first — they don't need a GPU), and the env is built at
# ~/envs/red-rl (see README "Setup").
#
# IMPORTANT: create the log dir BEFORE submitting, or SLURM fails to open its
# output file (it opens logs/train_%j.out before this script runs):
#     mkdir -p logs && sbatch slurm/train_h200.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs   # belt-and-suspenders; the pre-submit mkdir above is the one that matters

# Call the env's interpreter directly — no `conda activate`, which is unreliable in a
# non-interactive SLURM shell. The env is a self-contained prefix under $HOME.
PY="$HOME/envs/red-rl/bin/python"
[ -x "$PY" ] || { echo "FATAL: $PY not found — build the env first (see README 'Setup')" >&2; exit 1; }

# The base model is gated: a HF token must be available (huggingface-cli login once,
# or export HF_TOKEN). Keep the model cache on $HOME so it survives between jobs.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Fragmentation guard (also set inside the script, harmless to export here).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # stream loss/grad_norm to the log live (don't block-buffer stdout/err)

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"; nvidia-smi -L

# Full run uses configs/default.yaml. For a GPU smoke, append e.g. --max-steps 20.
"$PY" scripts/03_train.py --config configs/default.yaml
