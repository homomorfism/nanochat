# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack LLM implementation (like ChatGPT) in a minimal, hackable codebase. It runs the entire pipeline on a single 8xH100 node: tokenization → pretraining → midtraining → SFT → RL → inference → web UI.

## Common Commands

### Environment Setup
```bash
# Create venv and install dependencies (uses uv)
uv venv && uv sync --extra gpu
source .venv/bin/activate

# Build the Rust BPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Full Training Pipeline
```bash
bash speedrun.sh          # $100 tier (~4h on 8xH100)
bash run1000.sh           # $800 tier (~33h on 8xH100)
```

### Individual Training Stages
```bash
# Tokenizer
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# Base model pretraining (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Midtraining (conversation tokens, tool use)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# SFT
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# RL (optional, GSM8K only)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

### Inference
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"  # CLI chat
python -m scripts.chat_web                              # Web UI
```

### Testing
```bash
python -m pytest tests/test_rustbpe.py -v -s
python -m pytest tests/test_engine.py -v -s
```

### CPU/MPS Development (smaller models)
```bash
# Example for local development without GPUs:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
```

## Architecture

### Configuration System
Scripts use `nanochat/configurator.py` - a simple alternative to argparse. Define variables at script top, then:
```bash
python -m scripts.base_train -- --depth=26 --device_batch_size=16
```

### Model (`nanochat/gpt.py`)
GPT architecture with modern features:
- Rotary embeddings (no positional embeddings)
- QK normalization
- Untied embedding/lm_head weights
- ReLU² activation in MLP
- RMSNorm (no learnable params)
- Group-Query Attention (GQA) support
- Logit softcapping

Model size controlled by `--depth`: `n_embd = depth * 64`, `n_heads ≈ n_embd / 128`

### Optimizers
Two optimizers used together:
- **Muon** (`nanochat/muon.py`): For transformer block linear layers
- **AdamW** (`nanochat/adamw.py`): For embeddings and lm_head

### Training Pipeline
1. **Tokenizer** (`scripts/tok_train.py`): Custom Rust BPE + tiktoken for inference
2. **Base training** (`scripts/base_train.py`): Pretraining on FineWeb data
3. **Midtraining** (`scripts/mid_train.py`): Teaches conversation format, tool use, multiple choice
4. **SFT** (`scripts/chat_sft.py`): Per-sequence domain adaptation
5. **RL** (`scripts/chat_rl.py`): Optional reinforcement learning on GSM8K

### Special Tokens
```python
<|bos|>                    # Document start
<|user_start|> / <|user_end|>
<|assistant_start|> / <|assistant_end|>
<|python_start|> / <|python_end|>      # Tool invocation
<|output_start|> / <|output_end|>      # Tool output
```

### Inference Engine (`nanochat/engine.py`)
- KV cache management for efficient generation
- Built-in calculator tool via `<|python_start|>...<|python_end|>` blocks
- Batch generation with forced token injection for tool outputs

### Task System (`tasks/`)
Tasks implement conversation datasets for training/eval:
- `TaskMixture`: Combines multiple tasks
- Built-in tasks: `SmolTalk`, `MMLU`, `GSM8K`, `ARC`, `HumanEval`, `SpellingBee`
- `CustomJSON`: Load custom JSONL conversation files

### Data Storage
Artifacts stored in `$NANOCHAT_BASE_DIR` (default: `~/.cache/nanochat`):
- `tokenized_data/`: Pretraining data shards
- `tokenizer/`: Trained tokenizer files
- `base_checkpoints/`, `mid_checkpoints/`, `sft_checkpoints/`, `rl_checkpoints/`

## Key Implementation Details

- **Gradient accumulation**: Automatically computed from `total_batch_size` and `device_batch_size`
- **OOM handling**: Reduce `--device_batch_size` (default 32 → 16, 8, 4, 2, 1)
- **Distributed training**: Uses `torchrun` with NCCL backend; single GPU works by omitting `torchrun`
- **Wandb logging**: Set `WANDB_RUN=name` to enable; "dummy" disables logging
- **Reports**: `python -m nanochat.report generate` creates `report.md` with training metrics
