"""
Download roneneldan/TinyStories from HuggingFace and convert to RWKV .bin format.

TinyStories is a dataset of ~2.1M short children's stories (~475M tokens when tokenized
with the RWKV World tokenizer). It has clear difficulty gradients (vocabulary richness,
syntactic complexity) making it ideal for curriculum learning experiments.

Usage:
    cd RWKV-MLX
    python scripts/download_tinystories.py --output data/

    # Test with a small sample first
    python scripts/download_tinystories.py --output data/ --max_stories 5000

Output files:
    data/tinystories_train.bin   — training split (~2.1M stories)
    data/tinystories_valid.bin   — validation split (~21k stories)

Each file is a flat uint32 binary of RWKV World token IDs, with stories separated
by <|eos|> tokens (ID 65536), matching the format expected by train.py.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running from RWKV-MLX/
sys.path.insert(0, str(Path(__file__).parent.parent))
from rwkv_mlx.tokenizer import RWKVTokenizer, EOS_TOKEN_ID, PADDED_VOCAB_SIZE

VOCAB_FILE = Path(__file__).parent.parent / "rwkv_vocab_v20230424.txt"
DATASET_ID = "roneneldan/TinyStories"


def get_tokenizer() -> RWKVTokenizer:
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"Vocab file not found: {VOCAB_FILE}\n"
            "Expected at RWKV-MLX/rwkv_vocab_v20230424.txt"
        )
    return RWKVTokenizer(str(VOCAB_FILE))


def tokenize_split(stories: list, tokenizer: RWKVTokenizer, split_name: str) -> np.ndarray:
    """Tokenize a list of story strings into a flat uint32 array."""
    all_tokens = []
    skipped = 0

    for story in tqdm(stories, desc=f"Tokenizing {split_name}", unit="story"):
        text = story.strip()
        if not text:
            skipped += 1
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(EOS_TOKEN_ID)  # story boundary

    if skipped:
        print(f"  Skipped {skipped} empty stories")

    arr = np.array(all_tokens, dtype=np.uint32)
    print(f"  {len(stories) - skipped:,} stories → {len(arr):,} tokens "
          f"({len(arr) / 1e6:.1f}M)")
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Download TinyStories and convert to RWKV .bin format"
    )
    parser.add_argument("--output", type=str, default="data/",
                        help="Output directory (default: data/)")
    parser.add_argument("--max_stories", type=int, default=None,
                        help="Limit stories per split (for quick testing)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Import datasets here so the error is clear if not installed
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with:  pip install datasets")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading RWKV World tokenizer from {VOCAB_FILE}...")
    tokenizer = get_tokenizer()

    # Download dataset
    print(f"\nDownloading {DATASET_ID}...")
    print("(This may take a few minutes on first run; cached afterwards)")
    ds = load_dataset(DATASET_ID, cache_dir=args.cache_dir)

    for split_name, output_name in [("train", "tinystories_train"), ("validation", "tinystories_valid")]:
        if split_name not in ds:
            print(f"Split '{split_name}' not found, skipping.")
            continue

        split = ds[split_name]
        stories = [row["text"] for row in split]

        if args.max_stories:
            stories = stories[: args.max_stories]
            print(f"\n{split_name}: using first {len(stories):,} stories (--max_stories)")
        else:
            print(f"\n{split_name}: {len(stories):,} stories")

        tokens = tokenize_split(stories, tokenizer, split_name)

        out_path = os.path.join(args.output, f"{output_name}.bin")
        tokens.tofile(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved → {out_path} ({size_mb:.1f} MB)")

    print("\nDone! Files written to:", args.output)
    print("\nNext steps:")
    print("  # Score for curriculum learning")
    print("  python scripts/rank_dataset.py \\")
    print("      --model models/out-try-4 \\")
    print("      --checkpoint models/out-try-4/rwkv-71000.safetensors \\")
    print("      --data data/tinystories_train.bin \\")
    print("      --output data/tinystories_asc.bin \\")
    print("      --save_scores data/tinystories_scores.npy \\")
    print("      --direction asc")


if __name__ == "__main__":
    main()
