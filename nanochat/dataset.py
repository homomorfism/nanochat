"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

Shakespeare dataset: Downloads and preprocesses Shakespeare's works into parquet format.
"""

import os
import argparse
import requests
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Shakespeare dataset configuration

# URL for the complete works of Shakespeare (from Karpathy's char-rnn repo)
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Configuration for how to split Shakespeare into documents
CHUNK_SIZE = 2000  # characters per document chunk
OVERLAP = 200  # overlap between chunks to preserve context
VAL_RATIO = 0.1  # 10% for validation

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
# Shakespeare-specific functions

def download_shakespeare():
    """Download the Shakespeare text file."""
    raw_path = os.path.join(DATA_DIR, "shakespeare_raw.txt")

    if os.path.exists(raw_path):
        print(f"Shakespeare text already exists at {raw_path}")
        with open(raw_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"Downloading Shakespeare from {SHAKESPEARE_URL}...")
    response = requests.get(SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    text = response.text

    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Downloaded {len(text):,} characters to {raw_path}")

    return text

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks, trying to break at paragraph boundaries."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk - take everything remaining
            chunks.append(text[start:])
            break

        # Try to find a good breaking point (paragraph or sentence)
        # Look for double newline (paragraph break) first
        search_start = max(start + chunk_size - 200, start)
        search_end = min(start + chunk_size + 200, len(text))
        search_region = text[search_start:search_end]

        # Prefer paragraph breaks
        para_break = search_region.rfind('\n\n')
        if para_break != -1:
            end = search_start + para_break + 2
        else:
            # Fall back to sentence break
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                sent_break = search_region.rfind(punct)
                if sent_break != -1:
                    end = search_start + sent_break + len(punct)
                    break

        chunks.append(text[start:end])
        start = end - overlap  # overlap for context continuity

    return chunks

def create_parquet_shards(chunks, val_ratio=VAL_RATIO):
    """Create train and val parquet files from text chunks."""
    # Shuffle chunks for better distribution
    import random
    random.seed(42)
    shuffled = chunks.copy()
    random.shuffle(shuffled)

    # Split into train/val
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_chunks = shuffled[:val_size]
    train_chunks = shuffled[val_size:]

    print(f"Train chunks: {len(train_chunks)}, Val chunks: {val_size}")

    # Create train parquet (shard_00000.parquet)
    train_path = os.path.join(DATA_DIR, "shard_00000.parquet")
    train_table = pa.table({'text': train_chunks})
    pq.write_table(train_table, train_path, row_group_size=64)
    print(f"Wrote train shard to {train_path}")

    # Create val parquet (shard_00001.parquet) - last shard is always val
    val_path = os.path.join(DATA_DIR, "shard_00001.parquet")
    val_table = pa.table({'text': val_chunks})
    pq.write_table(val_table, val_path, row_group_size=64)
    print(f"Wrote val shard to {val_path}")

    return train_path, val_path

def prepare_shakespeare_dataset(num_epochs=1):
    """
    Main function to download and prepare the Shakespeare dataset.
    num_epochs: repeat the data this many times to create more training data.
    """
    # Check if already prepared
    parquet_files = list_parquet_files()
    if len(parquet_files) >= 2:
        print(f"Dataset already prepared with {len(parquet_files)} shards")
        return parquet_files

    # Download Shakespeare
    text = download_shakespeare()
    print(f"Total characters: {len(text):,}")

    # Split into chunks
    chunks = split_into_chunks(text)
    print(f"Split into {len(chunks)} chunks")

    # Repeat for more epochs if requested
    if num_epochs > 1:
        chunks = chunks * num_epochs
        print(f"After {num_epochs}x epochs: {len(chunks)} total chunks")

    # Create parquet files
    train_path, val_path = create_parquet_shards(chunks)

    print(f"\nShakespeare dataset ready!")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    return [train_path, val_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Shakespeare dataset")
    parser.add_argument("-n", "--num-epochs", type=int, default=1,
                        help="Number of times to repeat the data (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Characters per chunk (default: {CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"Overlap between chunks (default: {OVERLAP})")
    args = parser.parse_args()

    # Update globals if overridden
    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap

    print(f"Preparing Shakespeare dataset...")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Overlap: {OVERLAP}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Target directory: {DATA_DIR}")
    print()

    prepare_shakespeare_dataset(num_epochs=args.num_epochs)
