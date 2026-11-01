"""RED-RL: Reward-Weighted Regression (RWR) LoRA fine-tuning of Gemma 4 31B.

Modular, HPC-friendly pipeline split by hardware:

  CPU-only (any node / login node):
    red_rl.dataset    - build the compact JSONL from the train-data/ tree
    red_rl.render     - one record -> chat messages + per-token RWR weights
    red_rl.tokenize   - render + tokenize + offset-based label masking -> arrow dataset

  GPU-only (the H200 training node):
    red_rl.modeling   - load the 4-bit/bf16 base model + attach the LoRA adapter
    red_rl.trainer    - weighted chunked-CE Trainer + collator + TrainingArguments builder

  Shared:
    red_rl.config     - one dataclass of knobs, loadable from YAML + CLI overrides

Entry points live in ../scripts (01_build_dataset, 02_tokenize = CPU; 03_train = GPU).
"""

__all__ = ["config", "dataset", "render", "tokenize", "modeling", "trainer"]
