"""Stage 02 (CPU): render + tokenize + offset-based label masking + RWR weights.

Why offset-based masking (not apply_chat_template(tokenize=True)):
  * On transformers 5.x, apply_chat_template(..., tokenize=True) is unreliable
    (observed returning ~2 tokens for a full conversation).
  * The Gemma-4 template has no {% generation %} markers, so return_assistant_tokens_mask
    is empty.
So we render to a STRING (tokenize=False works), tokenize that ourselves with
return_offsets_mapping=True, and map each assistant turn's character span -> tokens.
The template is append-only, so a prefix render's length is an exact char boundary.

Needs `transformers` (fast tokenizer) + `datasets`. Does NOT need torch/CUDA.
"""

from __future__ import annotations

from transformers import AutoTokenizer
from datasets import Dataset

from .render import build_messages, TOOLS
from .config import Config


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token   # a distinct pad id for the collator (masked out of loss anyway)
    return tok


def _system_supported(tok) -> bool:
    """Does this template accept a system role? (Gemma-4 does; some older ones fold it in.)"""
    try:
        tok.apply_chat_template(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True)
        return True
    except Exception:
        return False


MARKER = "\n...[output trimmed to fit context]...\n"


def _shrink(s: str, target: int) -> str:
    """Head+tail trim of a (masked) tool output to ~target chars, keeping both ends
    where the facts cluster (command echo/headers up top, discovered path/flag/token
    near the bottom). target below the marker length collapses to just the marker."""
    if len(s) <= target:
        return s
    if target <= len(MARKER):
        return MARKER.strip()
    budget = target - len(MARKER)
    head = budget * 2 // 3
    tail = budget - head
    return s[:head] + MARKER + (s[-tail:] if tail > 0 else "")


def _prune_outputs_to_fit(msgs, render, tok, max_len):
    """Option 1 — fit by shrinking the OLDEST tool outputs first, never dropping a step.

    Tool outputs are masked context (never a loss target), so trimming them costs no
    supervision while keeping every assistant action in its true position, each still
    conditioned on the full ordered chain of prior actions. This preserves causality:
    the model never sees a fabricated state (which middle-drop would create by splicing
    the task prefix straight onto a recent tail). Successively smaller floors trim the
    oldest outputs harder until the row fits. A trajectory whose ACTION chain alone
    exceeds max_len is left over-length for the caller's last-resort middle-drop."""
    msgs = [dict(m) for m in msgs]
    tool_idx = [k for k, m in enumerate(msgs) if m.get("role") == "tool" and m.get("content")]
    if not tool_idx:
        return msgs
    for floor in (2000, 1000, 400, 120, 0):
        full = render(msgs)
        ids = tok(full, add_special_tokens=False)["input_ids"]
        over = len(ids) - max_len
        if over <= 0:
            return msgs
        cpt = max(1.0, len(full) / len(ids))      # chars per token on this render
        need = int(over * cpt * 1.1) + 1           # chars to shed (10% margin)
        for k in tool_idx:                         # oldest first
            if need <= 0:
                break
            c = msgs[k]["content"]
            if len(c) <= floor:
                continue
            msgs[k]["content"] = _shrink(c, floor)
            need -= len(c) - len(msgs[k]["content"])
    return msgs


def make_encoder(tok, cfg: Config, system_supported: bool):
    def _render(messages, add_gen=False):
        return tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=add_gen)

    def encode(record):
        msgs, weights = build_messages(record, cfg.beta, cfg.w_max, system_supported)
        full_text = _render(msgs)
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)  # fast tokenizer
        if len(enc["input_ids"]) > cfg.max_seq_len:
            # Overflow: keep every action, shrink the oldest (masked) tool outputs to fit.
            # Labeling below runs on these trimmed msgs, so char spans stay self-consistent.
            msgs = _prune_outputs_to_fit(msgs, _render, tok, cfg.max_seq_len)
            full_text = _render(msgs)
            enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]

        labels = [-100] * len(ids)
        tweights = [0.0] * len(ids)
        # cmd_ids[t] = index of the assistant action (command) token t belongs to, -1 if
        # none. The trainer averages CE within each command, so every command counts by
        # its own weight regardless of its token length.
        cmd_ids = [-1] * len(ids)
        cmd_w = []
        first_action = next((k for k, w in enumerate(weights) if w is not None), len(msgs))
        # Each assistant turn's char span = [len(render(msgs[:k])), len(render(msgs[:k+1]))).
        # Label the tokens whose offsets fall fully inside it (header + content + <end>;
        # structural tokens are a negligible fraction and harmless to include).
        for k, w in enumerate(weights):
            if w is None:
                continue
            c = len(cmd_w)
            cmd_w.append(w)
            cstart = len(_render(msgs[:k]))
            cend = len(_render(msgs[:k + 1]))
            for t, (s, e) in enumerate(offs):
                if e > s and s >= cstart and e <= cend:
                    labels[t] = ids[t]
                    tweights[t] = w
                    cmd_ids[t] = c

        if len(ids) > cfg.max_seq_len:
            # Last resort: the ACTION chain alone exceeds the budget (output-pruning
            # above could not fit it), so some action tokens must go. Middle-drop, NOT
            # left-truncate: always keep the leading context prefix (<bos> + system
            # prompt + task) so every kept action still sees what the target/objective
            # was, then keep the most-recent tail. Only the middle (older steps) is
            # dropped. Left-truncation instead drops the prefix on every long row,
            # blinding the model to the task.
            prefix_end_char = len(_render(msgs[:first_action]))
            n_prefix = sum(1 for (s, e) in offs if e > s and e <= prefix_end_char)
            n_prefix = min(n_prefix, cfg.max_seq_len // 2)  # guard: never let the prefix eat the row
            tail = cfg.max_seq_len - n_prefix
            keep = list(range(n_prefix)) + list(range(len(ids) - tail, len(ids)))
            ids = [ids[i] for i in keep]
            labels = [labels[i] for i in keep]
            tweights = [tweights[i] for i in keep]
            cmd_ids = [cmd_ids[i] for i in keep]
        # commands still (at least partly) labeled after truncation; position 0 is never a
        # loss target after the next-token shift, so it doesn't count
        present = {c for l, c in zip(labels[1:], cmd_ids[1:]) if l != -100 and c >= 0}
        return {"input_ids": ids, "labels": labels, "weights": tweights, "cmd_ids": cmd_ids,
                "n_cmds": len(present),
                # sum of command weights in this row; stage 03 averages it into the loss normalizer
                "cmd_weight_sum": float(sum(cmd_w[c] for c in present)),
                "n_loss_tokens": sum(l != -100 for l in labels),
                # carried through so stage 03 can hold out whole targets for eval
                "bench_id": record.get("bench_id", ""), "run": record.get("run", "")}

    return encode


def tokenize_dataset(jsonl_path: str, tok, cfg: Config) -> Dataset:
    """Load the JSONL, tokenize + mask + weight every record, drop any with no labels."""
    ds = Dataset.from_json(str(jsonl_path))
    system_supported = _system_supported(tok)

    # Fail loud on ONE record before the full map, so a bad mask doesn't silently empty
    # the dataset (and never cache an empty result).
    probe = make_encoder(tok, cfg, system_supported)(ds[0])
    if probe["n_loss_tokens"] == 0:
        raise RuntimeError("masking produced no labels on record 0 — inspect render/tokenize")

    encode = make_encoder(tok, cfg, system_supported)
    proc = ds.map(encode, remove_columns=ds.column_names, num_proc=cfg.num_proc, desc="tokenize+weight")
    proc = proc.filter(lambda r: r["n_cmds"] > 0)
    if len(proc) == 0:
        raise RuntimeError("no records produced labels — not writing an empty tokenized dataset")
    return proc
