"""GPU: load the base model (4-bit QLoRA or bf16) and attach the LoRA adapter.

Imports torch — only used on the training node (stage 03). Encodes two Gemma-4-specific
fixes we had to discover:
  * dtype vs torch_dtype: transformers >=5 renamed `torch_dtype` -> `dtype` in
    from_pretrained; we try the new name and fall back.
  * Gemma4ClippableLinear: Gemma-4 wraps each projection in a custom module with the
    real Linear as `.linear`. PEFT can't attach to the wrapper, so we target the inner
    `.linear` (decided per projection, so a model that wraps only some still gets every
    target). The wrapper's forward runs self.linear(x) then clips, so LoRA (base+delta on
    .linear) composes cleanly, with clipping applied after.
"""

from __future__ import annotations


import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import Config


def _from_pretrained(cfg: Config, quant, attn: str):
    kw = dict(quantization_config=quant, device_map="auto",
              attn_implementation=attn, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16, **kw)
    except TypeError:  # transformers <5 uses torch_dtype
        return AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16, **kw)


def _resolve_lora_targets(model, cfg: Config):
    """Return the EXACT module paths to adapt — the leaf `nn.Linear` projections in the TEXT
    decoder. Gemma-4 is multimodal: the vision/audio towers use the SAME projection names
    (q_proj, ...) but wrap them in Gemma4ClippableLinear(.linear=Linear), while the text
    decoder uses plain Linear. A text-only training run never executes the vision/audio towers,
    so any adapter placed there gets ZERO gradient (lora_B stays 0 -> no learning). The old
    name-level heuristic saw a `*.q_proj.linear` existed (in the vision tower) and redirected
    EVERY target to `<proj>.linear`, landing all adapters on the unused towers and none on the
    text decoder. We instead enumerate real Linear modules and keep only those under the text
    decoder (identified by a `...language_model` submodule; if there is none, the whole model
    is the decoder). Returning exact paths also matches a wrapped text projection's inner
    `.linear` correctly, should any exist."""
    import torch.nn as nn
    want = set(cfg.lora_targets)
    # text-decoder root: the `language_model` submodule on a multimodal model, else "" (all)
    text_root = next((n for n, _ in model.named_modules() if n.endswith("language_model")), "")
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in want:   # last path component is the projection name
            continue
        if text_root and not name.startswith(text_root + "."):
            continue                               # skip vision/audio tower projections
        targets.append(name)
    if not targets:
        raise RuntimeError(f"_resolve_lora_targets found no Linear modules for {sorted(want)} "
                           f"under text root '{text_root or '<model>'}' — check the architecture")
    return targets


def load_model(cfg: Config):
    """Load the base model (quantized or bf16), prep for training, and attach LoRA.
    Returns the PEFT model (already on GPU via device_map)."""
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = None
    for attn in dict.fromkeys([cfg.attn_impl, "sdpa", "eager"]):  # try configured, then degrade
        try:
            model = _from_pretrained(cfg, quant, attn)
            print(f"[modeling] loaded with attn_implementation={attn}")
            break
        except Exception as e:
            print(f"[modeling] attn '{attn}' failed ({str(e)[:120]}); trying next")
    if model is None:
        raise RuntimeError("could not load model with any attention implementation")

    model.config.use_cache = False  # required with gradient checkpointing
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # With a frozen base + gradient checkpointing, the checkpointed activations carry no
        # grad unless the input embeddings produce one -> backward would fail with "no input
        # requires grad". prepare_model_for_kbit_training does this for the 4-bit path; do it
        # explicitly for bf16. Idempotent, so harmless if a peft version already handles it.
        model.enable_input_require_grads()

    targets = _resolve_lora_targets(model, cfg)
    print(f"[modeling] LoRA target_modules = {targets}")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    model.print_trainable_parameters()
    return model
