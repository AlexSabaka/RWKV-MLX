"""
Preprocess OpenWebText2 (or similar JSONL/JSONL.ZST datasets) for RWKV-MLX training.

Reads .jsonl or .jsonl.zst files, tokenizes text using the RWKV World tokenizer,
and writes a single concatenated binary file (.bin, uint32) ready for training.

Usage:
    # Process entire directory of .jsonl.zst files
    python scripts/data_preprocess.py \
        /Volumes/2TB/idump/openwebtext2.jsonl.zst \
        --vocab /path/to/rwkv_vocab_v20230424.txt \
        --output data/openwebtext2.bin

    # Process a single file
    python scripts/data_preprocess.py \
        data/my_corpus.jsonl \
        --vocab /path/to/rwkv_vocab_v20230424.txt \
        --output data/my_corpus.bin

    # Process only first N documents (for testing)
    python scripts/data_preprocess.py \
        /Volumes/2TB/idump/openwebtext2.jsonl.zst \
        --vocab /path/to/rwkv_vocab_v20230424.txt \
        --output data/test_small.bin \
        --max-docs 10000

    # Filter by minimum word count
    python scripts/data_preprocess.py \
        /Volumes/2TB/idump/openwebtext2.jsonl.zst \
        --vocab /path/to/rwkv_vocab_v20230424.txt \
        --output data/openwebtext2.bin \
        --min-words 50

Output format:
    - .bin file: concatenated uint32 token IDs (documents separated by <|eos|> token)
    - Compatible with TextDataset in rwkv_mlx/train.py
    - Metadata printed to stdout (total tokens, documents, etc.)
"""

import argparse
import io
import json
import os
import struct
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np


def find_vocab_file() -> str | None:
    """Try to find the RWKV vocab file in common locations."""
    candidates = [
        Path(__file__).parent.parent / "rwkv_vocab_v20230424.txt",
        Path(__file__).parent.parent.parent / "RWKV-v7" / "rwkv_vocab_v20230424.txt",
        Path(__file__).parent.parent.parent / "RWKV-v5" / "tokenizer" / "rwkv_vocab_v20230424.txt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def iter_jsonl_zst(path: str):
    """Iterate over lines in a .jsonl.zst file, yielding parsed JSON objects."""
    import zstandard
    with open(path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text_stream:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_jsonl(path: str):
    """Iterate over lines in a plain .jsonl file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_documents(input_path: str):
    """Iterate over documents from a file or directory.

    Supports:
        - Single .jsonl file
        - Single .jsonl.zst file
        - Directory of .jsonl.zst files (sorted by name)
        - Directory of .jsonl files
    """ 
    input_path = Path(input_path)

    if input_path.is_file():
        if str(input_path).endswith(".jsonl.zst"):
            yield from iter_jsonl_zst(str(input_path))
        elif str(input_path).endswith(".jsonl"):
            yield from iter_jsonl(str(input_path))
        else:
            raise ValueError(f"Unsupported file type: {input_path}")
    elif input_path.is_dir():
        files = sorted(input_path.iterdir())
        zst_files = [f for f in files if f.name.endswith(".jsonl.zst")]
        jsonl_files = [f for f in files if f.name.endswith(".jsonl") and not f.name.endswith(".jsonl.zst")]
        all_files = zst_files or jsonl_files
        if not all_files:
            raise ValueError(f"No .jsonl or .jsonl.zst files found in {input_path}")
        # print(f"Found {len(all_files)} files to process")
        for i, fpath in enumerate(all_files):
            # print(f"  [{i+1}/{len(all_files)}] {fpath.name}")
            if fpath.name.endswith(".jsonl.zst"):
                yield from iter_jsonl_zst(str(fpath))
            else:
                yield from iter_jsonl(str(fpath))
    else:
        raise ValueError(f"Path does not exist: {input_path}")


def extract_text(doc: dict, text_field: str) -> str | None:
    """Extract text content from a document dict."""
    if text_field in doc:
        return doc[text_field]
    # Common alternatives
    for key in ("text", "content", "body", "passage"):
        if key in doc:
            return doc[key]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL/JSONL.ZST datasets for RWKV-MLX training"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input .jsonl/.jsonl.zst file or directory of .jsonl.zst files",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to rwkv_vocab_v20230424.txt (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train.bin",
        help="Output .bin file path (default: data/train.bin)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="JSON field containing the text (default: 'text')",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum number of documents to process (0 = all)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Minimum word count to include a document (0 = no filter)",
    )
    parser.add_argument(
        "--eos-token",
        type=int,
        default=None,
        help="EOS token ID to insert between documents (default: <|eos|> = 65536)",
    )
    parser.add_argument(
        "--eos-count",
        type=int,
        default=2,
        help="Number of EOS tokens between documents (default: 2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of documents to buffer before flushing to disk (default: 100000)",
    )

    args = parser.parse_args()

    # Find tokenizer vocab
    vocab_path = args.vocab or find_vocab_file()
    if vocab_path is None:
        print("ERROR: Could not find rwkv_vocab_v20230424.txt")
        print("  Please specify --vocab /path/to/rwkv_vocab_v20230424.txt")
        sys.exit(1)

    print(f"Loading tokenizer from {vocab_path}...")

    # Add the package to path so we can import the tokenizer
    pkg_root = str(Path(__file__).parent.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from rwkv_mlx.tokenizer import RWKVTokenizer, EOS_TOKEN_ID

    tokenizer = RWKVTokenizer(vocab_path)
    print(f"  Vocab size: {tokenizer.vocab_size} ({len(tokenizer.idx2token)} active tokens)")

    # EOS token (default: <|eos|> special token)
    eos_id = args.eos_token if args.eos_token is not None else EOS_TOKEN_ID
    eos_tokens = np.array([eos_id] * args.eos_count, dtype=np.uint32)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process documents and write tokens
    print(f"\nProcessing {args.input}...")
    print(f"Output: {args.output}")
    if args.max_docs > 0:
        print(f"Max documents: {args.max_docs}")
    if args.min_words > 0:
        print(f"Min words per document: {args.min_words}")
    print()

    total_tokens = 0
    total_docs = 0
    skipped_empty = 0
    skipped_short = 0
    token_buffer = []
    buffer_tokens = 0
    t0 = time.time()
    last_report = t0

    with open(str(output_path), "wb") as out_f:
        with tqdm(iter_documents(args.input), desc="Processing documents", unit="doc") as doc_iter:
            for doc in doc_iter:
                text = extract_text(doc, args.text_field)
                if not text or not text.strip():
                    skipped_empty += 1
                    continue

                # Word count filter
                if args.min_words > 0:
                    word_count = len(text.split())
                    if word_count < args.min_words:
                        skipped_short += 1
                        continue

                # Tokenize
                tokens = tokenizer.encode(text)
                if not tokens:
                    skipped_empty += 1
                    continue

                # Append tokens + EOS separator
                token_arr = np.array(tokens, dtype=np.uint32)
                token_buffer.append(token_arr)
                token_buffer.append(eos_tokens)
                buffer_tokens += len(tokens) + args.eos_count
                total_tokens += len(tokens) + args.eos_count
                total_docs += 1

                # Flush buffer periodically
                if total_docs % args.chunk_size == 0:
                    combined = np.concatenate(token_buffer)
                    out_f.write(combined.tobytes())
                    token_buffer = []
                    buffer_tokens = 0

                # Progress report every 10 seconds
                now = time.time()
                if now - last_report >= 10.0:
                    elapsed = now - t0
                    tokens_per_sec = total_tokens / elapsed
                    doc_iter.set_postfix({
                        "docs": f"{total_docs:,}",
                        "tokens": f"{total_tokens:,}",
                        "speed": f"{tokens_per_sec / 1e3:,.0f} Ktok/s",
                        "empty/short": f"{skipped_empty}/{skipped_short}"
                    })
                    last_report = now

                if args.max_docs > 0 and total_docs >= args.max_docs:
                    break

        # Flush remaining buffer
        if token_buffer:
            combined = np.concatenate(token_buffer)
            out_f.write(combined.tobytes())

    elapsed = time.time() - t0
    file_size = output_path.stat().st_size

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Documents processed: {total_docs:,}")
    print(f"  Documents skipped:   {skipped_empty:,} (empty) + {skipped_short:,} (too short)")
    print(f"  Total tokens:        {total_tokens:,}")
    print(f"  Avg tokens/doc:      {total_tokens / max(total_docs, 1):.0f}")
    print(f"  Output file:         {output_path} ({file_size / 1e9:.2f} GB)")
    print(f"  Throughput:          {total_docs / elapsed:,.0f} docs/s, {total_tokens / elapsed:,.0f} tok/s")
    print(f"\nTo train:")
    print(f"  python -m rwkv_mlx.train --data {output_path}")


if __name__ == "__main__":
    main()
