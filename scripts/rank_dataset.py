"""
Rank dataset documents by perplexity for influence-driven curriculum learning.

Scores each document in a .bin file using a trained RWKV model's forward pass
(per-document NLL), then writes a new .bin with documents in sorted order.

Based on: "Influence-driven Curriculum Learning for Pre-training on Limited Data"
(Schoenegger et al., 2025, arXiv:2508.15475).

Adaptation: We use perplexity as a practical proxy for training data influence.
Full TracinCP requires per-checkpoint gradients for all document pairs — O(N²·T)
which is intractable for large datasets. Perplexity correlates with self-influence
(||∇ℓ(w,z)||²) and is computable in a single forward pass.

  Low NLL  → easy/prototypical → high influence → schedule early (C_up / ascending)
  High NLL → hard/outlier      → low influence  → schedule late  (C_down / descending)

Also computes MATTR (Moving Average Type-Token Ratio) as a free model-agnostic
difficulty heuristic that correlates with the influence ordering (per the paper).

Usage:
    cd RWKV-MLX

    # Score + sort (ascending = easy-first)
    python scripts/rank_dataset.py \\
        --model models/out-try-4 \\
        --checkpoint models/out-try-4/rwkv-71000.safetensors \\
        --data data/tinystories_train.bin \\
        --output data/tinystories_asc.bin \\
        --save_scores data/tinystories_scores.npy \\
        --direction asc

    # Load pre-computed scores (skip forward pass)
    python scripts/rank_dataset.py \\
        --data data/tinystories_train.bin \\
        --load_scores data/tinystories_scores.npy \\
        --output data/tinystories_desc.bin \\
        --direction desc

    # Score only (no output .bin), save scores for later
    python scripts/rank_dataset.py \\
        --model models/out-try-4 \\
        --checkpoint models/out-try-4/rwkv-71000.safetensors \\
        --data data/tinystories_train.bin \\
        --save_scores data/tinystories_scores.npy

    # MATTR-only ranking (no model needed)
    python scripts/rank_dataset.py \\
        --data data/tinystories_train.bin \\
        --mattr_only \\
        --output data/tinystories_mattr_asc.bin \\
        --direction asc
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

# Allow running as a script from RWKV-MLX/
sys.path.insert(0, str(Path(__file__).parent.parent))
from rwkv_mlx.model import RWKV, RWKVConfig

EOS_TOKEN = 65536  # <|eos|>


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str, checkpoint: str) -> tuple:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    config = RWKVConfig(
        n_layer=cfg["n_layer"],
        n_embd=cfg["n_embd"],
        vocab_size=cfg.get("vocab_size", 65568),
        head_size=cfg.get("head_size", 64),
        ctx_len=cfg.get("ctx_len", 512),
    )
    model = RWKV(config)
    print(f"Loading checkpoint: {checkpoint}")
    model.load_weights(checkpoint)
    mx.eval(model.parameters())
    return model, config


# ---------------------------------------------------------------------------
# Data loading and document splitting
# ---------------------------------------------------------------------------

def load_bin(path: str) -> np.ndarray:
    file_size = os.path.getsize(path)
    if file_size % 4 == 0:
        data = np.memmap(path, dtype=np.uint32, mode="r")
        if data[:min(1000, len(data))].max() > 65568:
            data = np.memmap(path, dtype=np.uint16, mode="r")
    else:
        data = np.memmap(path, dtype=np.uint16, mode="r")
    return data


def split_documents(tokens: np.ndarray, eos_id: int = EOS_TOKEN) -> list:
    """Split a flat token array into documents at EOS boundaries."""
    eos_positions = np.where(tokens == eos_id)[0]
    docs = []
    prev = 0
    for pos in eos_positions:
        if pos > prev:
            docs.append(tokens[prev : pos + 1].astype(np.int32))
        prev = pos + 1
    if prev < len(tokens):
        docs.append(tokens[prev:].astype(np.int32))
    return docs


# ---------------------------------------------------------------------------
# Perplexity scoring
# ---------------------------------------------------------------------------

def score_document_nll(model: RWKV, tokens: np.ndarray, chunk_len: int) -> float:
    """Mean per-token NLL for a document (forward-pass only, no gradients)."""
    if len(tokens) < 2:
        return float("nan")

    total_loss = 0.0
    n_chunks = 0
    for i in range(0, len(tokens) - 1, chunk_len):
        chunk = tokens[i : i + chunk_len + 1]
        if len(chunk) < 2:
            continue
        # Pad to chunk_len if needed
        if len(chunk) < chunk_len + 1:
            pad = np.zeros(chunk_len + 1 - len(chunk), dtype=np.int32)
            chunk = np.concatenate([chunk, pad])

        x = mx.array(chunk[:-1][None])  # (1, chunk_len)
        y = mx.array(chunk[1:][None])   # (1, chunk_len)
        logits, _ = model(x)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="mean",
        )
        mx.eval(loss)
        total_loss += loss.item()
        n_chunks += 1

    return total_loss / n_chunks if n_chunks > 0 else float("nan")


def score_all(model: RWKV, docs: list, chunk_len: int, batch_size: int) -> np.ndarray:
    """Score all documents. Returns float32 array of NLL values."""
    scores = []
    # Batch forward passes for efficiency
    for i in tqdm(range(0, len(docs), batch_size), desc="Scoring docs", unit="batch"):
        batch_docs = docs[i : i + batch_size]
        batch_scores = []
        for doc in batch_docs:
            s = score_document_nll(model, doc, chunk_len)
            batch_scores.append(s)
        scores.extend(batch_scores)

    scores = np.array(scores, dtype=np.float32)
    # Replace NaN with median of valid scores
    valid = ~np.isnan(scores)
    if valid.sum() < len(scores):
        median = float(np.median(scores[valid])) if valid.any() else 5.0
        scores[~valid] = median
        print(f"  Replaced {(~valid).sum()} NaN scores with median {median:.3f}")
    return scores


# ---------------------------------------------------------------------------
# MATTR scoring (model-agnostic difficulty heuristic)
# ---------------------------------------------------------------------------

def mattr_score(tokens: np.ndarray, window: int = 25) -> float:
    """Moving Average Type-Token Ratio over a sliding window.

    High MATTR = high vocabulary diversity = harder/more complex text.
    Correlates with influence ordering per Schoenegger et al. 2025.
    """
    tokens = tokens[tokens != EOS_TOKEN]  # ignore EOS
    if len(tokens) < window:
        return float(len(set(tokens.tolist()))) / max(len(tokens), 1)
    ttrs = []
    for i in range(len(tokens) - window + 1):
        window_toks = tokens[i : i + window]
        ttrs.append(len(set(window_toks.tolist())) / window)
    return float(np.mean(ttrs))


def score_all_mattr(docs: list) -> np.ndarray:
    scores = np.array(
        [mattr_score(d) for d in tqdm(docs, desc="MATTR scoring", unit="doc")],
        dtype=np.float32,
    )
    return scores


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sorted_bin(docs: list, scores: np.ndarray, output_path: str, direction: str):
    if direction == "asc":
        order = np.argsort(scores)
        label = "easy-first (low NLL first)"
    else:
        order = np.argsort(-scores)
        label = "hard-first (high NLL first)"

    sorted_tokens = np.concatenate([docs[i] for i in order]).astype(np.uint32)
    sorted_tokens.tofile(output_path)

    print(f"\nWrote {len(sorted_tokens):,} tokens ({len(docs):,} docs) → {output_path}")
    print(f"  Direction: {direction} ({label})")
    print(f"  Score range: {scores.min():.3f} – {scores.max():.3f} (mean={scores.mean():.3f})")
    q25, q75 = np.percentile(scores, [25, 75])
    print(f"  Q25={q25:.3f}, Q75={q75:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank dataset documents by perplexity for curriculum learning"
    )
    # Model (required unless --load_scores or --mattr_only)
    parser.add_argument("--model", type=str, default=None,
                        help="Model directory (contains config.json)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (.safetensors)")
    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Input .bin dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output sorted .bin (omit to score-only)")
    # Score I/O
    parser.add_argument("--save_scores", type=str, default=None,
                        help="Save NLL scores to .npy")
    parser.add_argument("--load_scores", type=str, default=None,
                        help="Load pre-computed scores (skip forward pass)")
    parser.add_argument("--mattr_only", action="store_true",
                        help="Use MATTR instead of perplexity (no model needed)")
    # Ranking
    parser.add_argument("--direction", choices=["asc", "desc"], default="asc",
                        help="asc=easy-first (low NLL), desc=hard-first (high NLL)")
    # Scoring params
    parser.add_argument("--chunk_len", type=int, default=512,
                        help="Token chunk length for forward pass")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Documents per batch (higher = slightly faster)")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit number of documents (for testing)")

    args = parser.parse_args()

    # Validate
    if not args.mattr_only and args.load_scores is None:
        if args.model is None or args.checkpoint is None:
            parser.error("--model and --checkpoint required unless --load_scores or --mattr_only")

    # Load data
    print(f"Loading {args.data}...")
    data = load_bin(args.data)
    print(f"  {len(data):,} tokens")
    docs = split_documents(data)
    print(f"  {len(docs):,} documents")

    if args.max_docs:
        docs = docs[: args.max_docs]
        print(f"  Limited to {len(docs):,} documents")

    # Score
    if args.load_scores:
        print(f"Loading scores from {args.load_scores}...")
        scores = np.load(args.load_scores)
        if len(scores) != len(docs):
            raise ValueError(
                f"Score count {len(scores)} != document count {len(docs)}"
            )
    elif args.mattr_only:
        scores = score_all_mattr(docs)
    else:
        model, _ = load_model(args.model, args.checkpoint)
        scores = score_all(model, docs, args.chunk_len, args.batch_size)

    # Report
    valid = ~np.isnan(scores)
    print(f"\nScore statistics ({('NLL' if not args.mattr_only else 'MATTR')}):")
    print(f"  Mean: {scores[valid].mean():.4f}")
    print(f"  Std:  {scores[valid].std():.4f}")
    print(f"  Min:  {scores[valid].min():.4f}")
    print(f"  Max:  {scores[valid].max():.4f}")
    if not args.mattr_only:
        mean_ppl = math.exp(min(float(scores[valid].mean()), 20))
        print(f"  Mean ppl: {mean_ppl:.1f}")

    # Save scores
    if args.save_scores:
        np.save(args.save_scores, scores)
        print(f"\nSaved scores → {args.save_scores}")

    # Write sorted output
    if args.output:
        write_sorted_bin(docs, scores, args.output, args.direction)


if __name__ == "__main__":
    main()
