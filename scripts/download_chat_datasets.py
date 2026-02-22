"""
Download chat datasets from HuggingFace and convert to RWKV SFT JSONL format.

Supported datasets:
  - openbmb/UltraChat      (~949k multi-turn conversations)
  - anaonymous-aad/GenQA_dialog  (~819k multi-turn conversations)

Output format (one JSON object per line):
  {"input": "<|im_start|>user\\n...\\n<|im_end|>\\n", "output": "<|im_start|>assistant\\n...\\n<|im_end|>"}

One example is generated per assistant turn. The input contains the full conversation
history up to (but not including) that assistant turn. This allows the model to learn
multi-turn context while computing loss only on the output tokens.

Usage:
    cd RWKV-MLX
    python scripts/download_chat_datasets.py --output data/

    # Download only one dataset
    python scripts/download_chat_datasets.py --output data/ --datasets ultrachat

    # Test with a small sample
    python scripts/download_chat_datasets.py --output data/ --max_examples 5000

    # Custom cache directory (default: .cache/ in project dir)
    python scripts/download_chat_datasets.py --output data/ --cache_dir /tmp/hf_cache
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm


# ── SFT token markers ────────────────────────────────────────────────────────

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def fmt_turn(role: str, content: str) -> str:
    """Format a single conversation turn."""
    return f"{IM_START}{role}\n{content}\n{IM_END}\n"


def build_sft_examples(turns: list[tuple[str, str]]) -> list[dict]:
    """
    Build SFT examples from a list of (role, content) tuples.

    One example per assistant turn:
      input  = all turns up to (not including) this assistant turn
      output = this assistant turn

    Skips examples where the assistant message is empty.
    """
    examples = []
    history = ""

    for role, content in turns:
        content = content.strip()
        if role == "user":
            history += fmt_turn("user", content)
        elif role == "assistant":
            if not content:
                # skip empty assistant turns
                history += fmt_turn("assistant", content)
                continue
            output = fmt_turn("assistant", content)
            if history:  # need at least one user turn in input
                examples.append({"input": history, "output": output})
            history += output
        else:
            # unknown role — treat as user
            history += fmt_turn(role, content)

    return examples


# ── Dataset-specific parsers ─────────────────────────────────────────────────

def parse_ultrachat(row: dict) -> list[tuple[str, str]]:
    """
    UltraChat schema:
      {"id": "...", "data": ["user1", "asst1", "user2", "asst2", ...], "split": "train"}

    data is a flat alternating list: even indices = user, odd = assistant.
    """
    data = row.get("data") or []
    turns = []
    for i, msg in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append((role, str(msg)))
    return turns


def parse_genqa(row: dict) -> list[tuple[str, str]]:
    """
    GenQA_dialog schema:
      {"split": "...", "prompt": "...",
       "messages": [{"role": "user"/"assistant", "content": "..."}, ...]}
    """
    messages = row.get("messages") or []
    turns = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        turns.append((role, str(content)))
    return turns


# ── Per-dataset processing ───────────────────────────────────────────────────

DATASET_CONFIGS = {
    "ultrachat": {
        "hf_id": "openbmb/UltraChat",
        "splits": ["train"],          # only train split exists
        "parser": parse_ultrachat,
        "output": "ultrachat_sft.jsonl",
    },
    "genqa": {
        "hf_id": "anaonymous-aad/GenQA_dialog",
        "splits": ["train", "test"],
        "parser": parse_genqa,
        "output": "genqa_sft.jsonl",
    },
}


def process_dataset(
    name: str,
    config: dict,
    output_dir: str,
    cache_dir: str,
    max_examples: int | None,
    load_dataset,
) -> None:
    hf_id = config["hf_id"]
    parser = config["parser"]
    out_path = os.path.join(output_dir, config["output"])

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({hf_id})")
    print(f"Output:  {out_path}")
    print(f"{'='*60}")

    print(f"Downloading {hf_id}...")
    print("(First run downloads and caches; subsequent runs are instant)")
    ds = load_dataset(hf_id, cache_dir=cache_dir)

    total_examples = 0
    total_convs = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for split_name in config["splits"]:
            if split_name not in ds:
                print(f"  Split '{split_name}' not found, skipping.")
                continue

            split = ds[split_name]
            rows = list(split)

            print(f"\n  Split '{split_name}': {len(rows):,} conversations")

            for row in tqdm(rows, desc=f"  Converting {split_name}", unit="conv"):
                turns = parser(row)
                examples = build_sft_examples(turns)

                for ex in examples:
                    fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_examples += 1

                    if max_examples and total_examples >= max_examples:
                        break

                total_convs += 1

                if max_examples and total_examples >= max_examples:
                    print(f"\n  Reached --max_examples {max_examples}, stopping.")
                    break

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  {total_convs:,} conversations → {total_examples:,} SFT examples")
    print(f"  Saved → {out_path} ({size_mb:.1f} MB)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    # Default cache dir: .cache/ inside the RWKV-MLX project directory
    project_dir = Path(__file__).parent.parent
    default_cache = str(project_dir / ".cache")

    parser = argparse.ArgumentParser(
        description="Download chat datasets and convert to RWKV SFT JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", type=str, default="data/",
        help="Output directory for JSONL files (default: data/)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=default_cache,
        help=f"HuggingFace cache directory (default: {default_cache})",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Max SFT examples per dataset (for quick testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with:  pip install datasets")
        sys.exit(1)

    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output}")
    if args.max_examples:
        print(f"Max examples per dataset: {args.max_examples:,}")

    for name in args.datasets:
        config = DATASET_CONFIGS[name]
        process_dataset(
            name=name,
            config=config,
            output_dir=args.output,
            cache_dir=args.cache_dir,
            max_examples=args.max_examples,
            load_dataset=load_dataset,
        )

    print("\n\nDone! Files written to:", args.output)
    print("\nNext steps:")
    print("  # Fine-tune on chat data")
    print("  python -m rwkv_mlx.train \\")
    print("      --data data/ultrachat_sft.jsonl \\")
    print("      --sft_data data/ultrachat_sft.jsonl \\")
    print("      --resume models/out-try-5/rwkv-116000.safetensors \\")
    print("      --output models/chat-try-1 \\")
    print("      --lr_init 1e-4 --lr_final 1e-5 --total_steps 10000")


if __name__ == "__main__":
    main()
