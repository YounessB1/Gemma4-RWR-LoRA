#!/bin/bash
#SBATCH --job-name=red-rl-train
#SBATCH --partition=restrict_h200       # same node as gpu_h200 (compute-7-12) but a 10-day walltime cap
#SBATCH --gres=gpu:1                     # node advertises Gres=gpu:8 (no type) -> request gpu:N, not gpu:h200:N
#SBATCH --cpus-per-task=16               # 1 GPU's fair share on compute-7-12 (128 cores / 8 GPUs)
#SBATCH --mem=192G                       # well under the ~257G/GPU share; bump if you hit a RAM OOM
#SBATCH --time=4-00:00:00                # 4 days: one full epoch (~52h left from ckpt-163) fits with wide margin; cap is 10 days
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ---------------------------------------------------------------------------
# Stage 03 (GPU): RWR LoRA training on one H200, on the restrict_h200 partition.
#
# Unlike gpu_h200 (24h cap -> checkpoint+resume chain), restrict_h200 allows up
# to 10 days, so a single job finishes the whole epoch. We still checkpoint
# (save_steps) for crash safety (see the rack 3/4/6 power warning on login), and
# keep a wall-clock safety stop BELOW this job's --time via --max-train-hours.
#
# Resume is automatic: cfg.resume picks the newest complete checkpoint in
# artifacts/checkpoints, so this job continues from wherever the last one stopped.
#
# IMPORTANT: create the log dir BEFORE submitting (SLURM opens logs/train_%j.out
# before this script runs): mkdir -p logs && sbatch slurm/train_restrict_h200.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Call the env's interpreter directly (conda activate is unreliable in a non-interactive SLURM shell).
PY="$HOME/envs/red-rl/bin/python"
[ -x "$PY" ] || { echo "FATAL: $PY not found — build the env first (see README 'Setup')" >&2; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1   # stream loss/grad_norm to the log live (don't block-buffer stdout/err)

echo "job $SLURM_JOB_ID on $SLURM_NODELIST"; nvidia-smi -L

# --max-train-hours 94 keeps the wall-clock safety checkpoint ~2h below the 4-day --time.
# A 1-epoch run finishes (~52h from ckpt-163) long before this fires; it only matters if a
# node slowdown stretches the run toward the cap. For a smoke, append e.g. --max-steps 20.
"$PY" scripts/03_train.py --config configs/default.yaml --max-train-hours 94
