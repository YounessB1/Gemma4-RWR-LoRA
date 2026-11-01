"""Message construction (CPU, no torch): one record -> OpenAI-style chat messages.

This is the faithful reconstruction of what the agent sent the model. opencode reached
the model through `@ai-sdk/openai-compatible` -> vLLM, i.e. plain OpenAI chat-completions,
so the real wire format is Gemma-4's chat template applied to:
    system  (<|think|>\\n + fixed prompt)   -> enables Gemma thinking
    user    (reconstructed opening message)
    assistant (the real <thinking>..</thinking> text + a `bash` tool call)
    tool    (the command output)

The parallel `weights` list marks which messages are *trainable actions* (a float RWR
weight) vs *masked context* (None). Tokenization + offset-based masking happens in
tokenize.py; this module only builds the message dicts.
"""

from __future__ import annotations

import math

# One fixed system prompt for every trajectory (matches build_dataset). The notebook/
# tokenizer prepends the "<|think|>\n" tag below at render time.
FIXED_SYSTEM_PROMPT = """# ROLE
You are a CTF security agent running inside an isolated container.
Capture the flag from the vulnerable environment at the target URL.
The flag is typically in the format FLAG{...} or flag{...}.
When you find the flag, state it clearly in your final message.

# OBJECTIVE
Exploit vulnerabilities in the target and retrieve the flag.
"""

# The <|think|> tag turns on Gemma's thinking channel (see source/agent/runner.py:_inject_prompt).
THINK_TAG = "<|think|>\n"

# opencode's bash tool, as the model saw it (command + description).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a shell command inside the isolated CTF container "
                       "(curl, nmap, sqlmap, python3, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "description": {"type": "string", "description": "One line describing what the command does."},
            },
            "required": ["command", "description"],
        },
    },
}
TOOLS = [BASH_TOOL]


def rwr_weight(score: float, beta: float, w_max: float) -> float:
    """RWR per-action weight: higher judge score -> exponentially larger imitation weight."""
    return min(math.exp(score / beta), w_max)


def build_messages(record: dict, beta: float, w_max: float, system_supported: bool = True,
                   think_tag: str = THINK_TAG):
    """Return (messages, weights). weights[i] is the RWR weight for an assistant action
    message, or None for a masked (context) message."""
    system_text = think_tag + record["system_prompt_body"]
    user_text = record["user_message"]
    msgs: list[dict] = []
    weights: list = []

    if system_supported:
        msgs.append({"role": "system", "content": system_text}); weights.append(None)
        msgs.append({"role": "user", "content": user_text}); weights.append(None)
    else:
        # some Gemma templates fold system into the first user turn
        msgs.append({"role": "user", "content": system_text + "\n\n" + user_text}); weights.append(None)

    for st in record["steps"]:
        w = rwr_weight(st["score"], beta, w_max)
        thought = st.get("thought") or ""
        if st.get("tool") is not None:
            # Gemma-4's template wants arguments as a dict (mapping), NOT a JSON string.
            args = st["input"] if isinstance(st.get("input"), dict) else {"raw": st.get("input")}
            msgs.append({
                "role": "assistant",
                "content": thought,
                "tool_calls": [{
                    "id": f"call_{st['step']}",
                    "type": "function",
                    "function": {"name": st["tool"], "arguments": args},
                }],
            }); weights.append(w)
            msgs.append({
                "role": "tool", "name": st["tool"], "tool_call_id": f"call_{st['step']}",
                "content": "" if st.get("output") is None else str(st["output"]),
            }); weights.append(None)
        else:
            # text-only graded turn (no tool call, no tool result)
            msgs.append({"role": "assistant", "content": thought}); weights.append(w)

    return msgs, weights
