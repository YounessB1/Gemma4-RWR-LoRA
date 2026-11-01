"""GPU: the RWR collator, the weighted-loss Trainer, and a resilient args builder.

Imports torch — training node only.

RWR loss, per COMMAND (assistant action), not per token or per trajectory:
    loss = sum_c  w_c * mean_{t in c} CE_t   /   (loss_norm * batch_size)
  * mean over a command's tokens -> a command counts once, whatever its length.
  * w_c is the command's own RWR weight; loss_norm is ONE global constant (mean per-row
    sum of command weights over the train set, computed in stage 03). Nothing depends on
    the other commands in the same trajectory, so a command's pull on the model is set
    only by its own score — a 1.0 command always outweighs a 0.0 one by exp(1/beta).
  * loss_norm only keeps the loss ~ a mean CE so `lr` keeps its usual meaning.

Two memory-critical details for Gemma-4's 262k-token vocabulary:
  * We pop `labels` before model(**inputs) so the model does NOT compute its own
    (huge, full-vocab fp32) cross-entropy — we override the loss anyway.
  * We compute our CE chunked over tokens so we never materialize a full
    [num_tokens, vocab] fp32 tensor at once.
"""

from __future__ import annotations

import inspect
import time

import torch
from transformers import Trainer, TrainingArguments, TrainerCallback

from .config import Config


class WallClockStopCallback(TrainerCallback):
    """Checkpoint + stop cleanly before SLURM's hard walltime kill.

    SLURM hard-kills at the partition cap (24h); a kill mid-save truncates the checkpoint.
    This fires at a wall-clock `deadline` (seconds since epoch, measured from job start), sets
    should_save + should_training_stop so the Trainer writes a COMPLETE checkpoint and the job
    exits gracefully — a resubmit then resumes from it. Checked at on_step_end, so it triggers
    within one step (~minutes) of the deadline; set the deadline below the cap with margin."""

    def __init__(self, deadline_epoch: float):
        self.deadline = deadline_epoch
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._fired and time.time() >= self.deadline:
            self._fired = True
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                print(f"[trainer] wall-clock deadline hit at step {state.global_step} "
                      f"-> saving checkpoint and stopping (resubmit to resume)", flush=True)
        return control


class RWRCollator:
    """Pad input_ids/labels/weights/cmd_ids to the batch max and build the attention mask."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, lab, wts, cmd, att = [], [], [], [], []
        for f in feats:
            n = len(f["input_ids"])
            pad = maxlen - n
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            lab.append(f["labels"] + [-100] * pad)
            wts.append([float(x) for x in f["weights"]] + [0.0] * pad)
            cmd.append(f["cmd_ids"] + [-1] * pad)
            att.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "weights": torch.tensor(wts, dtype=torch.float),
            "cmd_ids": torch.tensor(cmd, dtype=torch.long),
        }


class RWRTrainer(Trainer):
    # chunk size (in tokens) for the memory-bounded cross-entropy
    LOSS_CHUNK = 512

    def __init__(self, *args, loss_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # global normalizer (mean per-row sum of command weights on the train set)
        self.loss_norm = float(loss_norm)
        # Our compute_loss is already normalized per micro-batch (by loss_norm * batch size),
        # which does NOT depend on num_items_in_batch. Modern HF Trainer, when model_accepts_loss_kwargs is
        # True (Gemma's forward has **kwargs), assumes the loss was token-normalized and SKIPS the
        # /grad_accum division — which would make our effective LR grad_accum (16x) too high.
        # Force it off so the Trainer divides by grad_accum, i.e. correct mean accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("weights")
        labels = inputs.pop("labels")          # pop -> model skips its own full-vocab loss
        cmd_ids = inputs.pop("cmd_ids")
        outputs = model(**inputs)
        logits = outputs.logits
        bsz = labels.size(0)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))  # view, no copy (dim0==1)
        shift_labels = labels[:, 1:].reshape(-1)
        shift_w = weights[:, 1:].reshape(-1)
        shift_cmd = cmd_ids[:, 1:]

        # one slot per (row, command): key = row * n_cmd + cmd_id
        n_cmd = int(shift_cmd.max().clamp_min(-1).item()) + 1
        rows = torch.arange(bsz, device=shift_cmd.device).unsqueeze(1)
        key = (rows * n_cmd + shift_cmd).reshape(-1)
        valid = (shift_cmd.reshape(-1) >= 0) & (shift_labels != -100)

        # Select the labeled positions BEFORE the fp32 cross-entropy. Most tokens in a row
        # are masked tool output (label -100); computing CE over all of them would retain a
        # full [n_tokens, vocab] fp32 tensor per chunk for the backward pass — tens of GB on
        # Gemma-4's 262k vocab. Gathering first makes the fp32 CE (and what autograd saves for
        # it) scale with the count of labeled tokens, not the sequence length.
        idx = valid.nonzero(as_tuple=True)[0]
        key, w = key[idx], shift_w[idx]
        sel_logits, sel_labels = shift_logits[idx], shift_labels[idx]
        if idx.numel():
            ce = torch.cat([
                torch.nn.functional.cross_entropy(
                    sel_logits[i:i + self.LOSS_CHUNK].float(),
                    sel_labels[i:i + self.LOSS_CHUNK],
                    reduction="none")
                for i in range(0, sel_logits.size(0), self.LOSS_CHUNK)])
        else:  # no labeled tokens in this micro-batch (dataset filtering makes this unreachable)
            ce = shift_logits.new_zeros(0, dtype=torch.float32)
        n_slots = bsz * n_cmd
        ce_sum = ce.new_zeros(n_slots).index_add(0, key, ce)
        tok_cnt = ce.new_zeros(n_slots).index_add(0, key, torch.ones_like(ce))
        w_sum = ce.new_zeros(n_slots).index_add(0, key, w.float())
        present = tok_cnt > 0
        cnt = tok_cnt[present]
        # w is constant within a command, so w_sum / cnt is exactly that command's weight
        cmd_loss = (w_sum[present] / cnt) * (ce_sum[present] / cnt)
        loss = cmd_loss.sum() / (self.loss_norm * bsz)
        return (loss, outputs) if return_outputs else loss


def build_training_args(cfg: Config) -> TrainingArguments:
    """Construct TrainingArguments, passing only the kwargs THIS transformers build
    accepts (5.x refactored/renamed some, e.g. dropped warmup_ratio). Reports drops."""
    desired = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="paged_adamw_8bit" if cfg.load_in_4bit else "adamw_torch",
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=("steps" if cfg.eval_steps and cfg.eval_steps > 0 else "no"),
        eval_steps=(cfg.eval_steps if cfg.eval_steps and cfg.eval_steps > 0 else None),
        per_device_eval_batch_size=cfg.per_device_batch,
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,   # keep our custom "weights" column
    )
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = [k for k in desired if k not in valid]
    if dropped:
        print(f"[trainer] this transformers build doesn't accept {dropped} -> dropped")
    return TrainingArguments(**{k: v for k, v in desired.items() if k in valid})
