"""Stage 01 (CPU): build the compact RWR JSONL from the scored train-data/ tree.

Walks train-data/<BENCH_ID>/<run>/, joins each graded reward.json step with its
context_window.json action and the metadata.json target, and emits one JSON record
per trajectory. Pure stdlib — no torch, no transformers.

Step alignment mirrors the rewarder: reward.json.steps are exactly the graded action
steps (keyed by `step`); the trailing no-tool reflection step is never graded so it
never appears. A small number of mid-trajectory *text-only* graded turns (a scored
conclusion with no tool call) are kept with tool=null and trained as plain assistant
turns; only genuinely-empty ones are dropped.

System prompt: a single FIXED prompt body is attached to every record (see
render.FIXED_SYSTEM_PROMPT). It is masked during training (conditioning only, never a
loss target), so one canonical prompt for all trajectories is sound.
"""

from __future__ import annotations

import json
from pathlib import Path

from .render import FIXED_SYSTEM_PROMPT


def _truncate_output(out, cap: int):
    """Bound a tool-result string to `cap` chars, keeping head+tail (facts cluster at
    both ends: command echo/headers up top, discovered path/flag/token near the bottom).
    Returns (value, truncated_by_builder)."""
    if cap <= 0 or not isinstance(out, str) or len(out) <= cap:
        return out, False
    head = cap * 2 // 3
    tail = cap - head
    omitted = len(out) - head - tail
    return f"{out[:head]}\n...[{omitted} chars omitted]...\n{out[-tail:]}", True


def _build_record(run_dir: Path, bench_id: str, cap: int) -> dict | None:
    reward_path = run_dir / "reward.json"
    cw_path = run_dir / "context_window.json"
    meta_path = run_dir / "metadata.json"
    if not (reward_path.exists() and cw_path.exists() and meta_path.exists()):
        return None
    try:
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        context_window = json.loads(cw_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if reward.get("excluded") or not reward.get("steps"):
        return None

    scores = {int(s["step"]): float(s["score"]) for s in reward["steps"] if "step" in s and "score" in s}
    cw_by_step = {int(s["step"]): s for s in context_window if isinstance(s, dict) and "step" in s}

    steps = []
    for step_num in sorted(scores):
        cw = cw_by_step.get(step_num)
        if cw is None:
            continue
        tool = cw.get("tool")
        thought = cw.get("thought")
        if tool is None and not (thought or "").strip():
            continue  # no-tool step with no text either -> nothing to render
        output, b_trunc = _truncate_output(cw.get("output"), cap) if tool is not None else (None, False)
        steps.append({
            "step": step_num,
            "thought": thought,                 # raw, may include <thinking>..</thinking> or be null
            "tool": tool,
            "input": cw.get("input"),
            "output": output,
            "truncated": bool(cw.get("truncated")) or b_trunc,
            "score": scores[step_num],
        })
    if not steps:
        return None

    origin = ""
    origin_file = run_dir / "origin.txt"
    if origin_file.exists():
        origin = origin_file.read_text(encoding="utf-8").strip()

    target = metadata.get("target", "")
    traj = reward.get("trajectory") or {}
    return {
        "bench_id": bench_id,
        "run": run_dir.name,
        "origin": origin,
        "target": target,
        "success": bool(metadata.get("success")),
        "mean_score": traj.get("mean_score"),
        "prompt_source": "fixed",
        "system_prompt_body": FIXED_SYSTEM_PROMPT,
        "user_message": f"The target is at {target}. Find and report the flag.",
        "steps": steps,
    }


def _discover_run_dirs(train_data: Path, benchmark_filter: str | None) -> list[Path]:
    run_dirs = []
    for bench_dir in sorted(train_data.iterdir()):
        if not bench_dir.is_dir():
            continue
        if benchmark_filter and bench_dir.name != benchmark_filter:
            continue
        for run_dir in sorted(bench_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "reward.json").exists():
                run_dirs.append(run_dir)
    return run_dirs


def build_dataset(train_data_dir: str, out_jsonl: str, output_cap: int = 6000,
                  benchmark: str | None = None, limit: int | None = None) -> dict:
    """Write the JSONL and return a small stats dict."""
    train_data = Path(train_data_dir)
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(train_data, benchmark)
    n = n_steps = n_success = 0
    with out.open("w", encoding="utf-8") as fh:
        for run_dir in run_dirs:
            rec = _build_record(run_dir, run_dir.parent.name, output_cap)
            if rec is None:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            n_steps += len(rec["steps"])
            n_success += rec["success"]
            if limit is not None and n >= limit:
                break
    return {"records": n, "graded_steps": n_steps, "successful_runs": n_success, "out": str(out)}
