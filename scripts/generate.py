"""
Generate text from a trained base model with a seed phrase.

Usage:
    python -m scripts.generate --prompt "To be or not to be"
    python -m scripts.generate --prompt "ROMEO:" --temperature 0.8 --max_tokens 200
    python -m scripts.generate --prompt "First Citizen:" --temperature 1.0 --top_k 50
"""

import os
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# Parse arguments

parser = argparse.ArgumentParser(description='Generate text from a trained model')
parser.add_argument('-p', '--prompt', type=str, required=True, help='Seed phrase to start generation')
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Sampling temperature (0=greedy, higher=more random)')
parser.add_argument('-k', '--top_k', type=int, default=50, help='Top-k sampling (None=disabled)')
parser.add_argument('-n', '--max_tokens', type=int, default=256, help='Maximum tokens to generate')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
parser.add_argument('--phase', type=str, default='base', choices=['base', 'mid', 'sft', 'rl'], help='Model phase to load')
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Load model and tokenizer
print(f"Loading {args.phase} model...")
model, tokenizer, meta = load_model(args.phase, device, phase="eval")
model.eval()

# -----------------------------------------------------------------------------
# Generate

print(f"\n{'='*60}")
print(f"Prompt: {args.prompt}")
print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
print(f"{'='*60}\n")

# Tokenize the prompt
bos_token_id = tokenizer.get_bos_token_id()
prompt_tokens = tokenizer.encode(args.prompt, prepend=bos_token_id)

# Generate using Engine for efficiency
engine = Engine(model, tokenizer)

for sample_idx in range(args.num_samples):
    if args.num_samples > 1:
        print(f"\n--- Sample {sample_idx + 1} ---")

    # Print the prompt
    print(args.prompt, end="", flush=True)

    # Generate tokens
    with autocast_ctx:
        for token_column, token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + sample_idx,  # different seed per sample
        ):
            token = token_column[0]
            # Stop on BOS token (document boundary)
            if token == bos_token_id:
                break
            # Decode and print incrementally
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)

    print()  # newline after generation

print(f"\n{'='*60}")
print("Generation complete!")
