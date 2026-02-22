"""
RWKV-v7 Text Generation (Inference) for MLX.

Supports both GPT-mode (parallel prefill) and RNN-mode (sequential generation).
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model import RWKV, RWKVConfig
from .tokenizer import RWKVTokenizer


def sample_logits(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
) -> int:
    """Sample from logits with temperature, top-p, and top-k."""
    if temperature == 0:
        return logits.argmax().item()

    probs = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)

    if top_k > 0:
        top_k_indices = mx.argpartition(-probs, top_k)[:top_k]
        mask = mx.zeros_like(probs)
        mask[top_k_indices] = 1.0
        probs = probs * mask

    if top_p < 1.0:
        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs)
        # Find how many tokens to keep: first index where cumsum >= top_p
        cutoff_mask = (cumulative_probs >= top_p)
        # Use argmax on bool array to find first True; if none, keep all
        first_above = mx.argmax(cutoff_mask).item()
        cutoff_idx = first_above + 1 if cutoff_mask[first_above].item() else len(sorted_probs)
        # Zero out everything beyond cutoff
        mask = mx.zeros_like(probs)
        mask[sorted_indices[:cutoff_idx]] = 1.0
        probs = probs * mask

    # Renormalize
    probs = probs / (probs.sum() + 1e-12)

    # Sample
    return mx.random.categorical(mx.log(probs + 1e-12)).item()


def generate(
    model: RWKV,
    tokenizer: RWKVTokenizer,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0,
    use_rnn_mode: bool = True,
) -> str:
    """Generate text from a prompt.

    Args:
        model: RWKV model
        tokenizer: tokenizer
        prompt: input text
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        top_k: top-k sampling (0 to disable)
        use_rnn_mode: True for RNN mode (faster for generation),
                      False for GPT mode (faster for prefill)

    Returns:
        generated text (including prompt)
    """
    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    print(prompt, end="", flush=True)

    if use_rnn_mode:
        # RNN mode: process tokens one at a time
        state = model.init_state()
        t0 = time.time()

        # Prefill
        for token in tokens:
            logits, state = model.forward_rnn(token, state)
            mx.eval(logits, *state)

        # Generate
        for i in range(max_tokens):
            token = sample_logits(logits, temperature, top_p, top_k)
            generated.append(token)

            # Print incremental output
            try:
                text = tokenizer.decode(generated[len(tokens):])
                new_text = text[len(tokenizer.decode(generated[len(tokens):-1])):]
                print(new_text, end="", flush=True)
            except (UnicodeDecodeError, IndexError):
                pass

            t1 = time.time()
            logits, state = model.forward_rnn(token, state)
            mx.eval(logits, *state)

        elapsed = time.time() - t0
        gen_tokens = max_tokens
        print(f"\n\n[{gen_tokens / elapsed:.1f} tokens/s, {elapsed:.2f}s total]")

    else:
        # GPT mode: parallel prefill, then extend
        t0 = time.time()

        # Prefill
        idx = mx.array([tokens])
        out = model(idx)
        mx.eval(out)

        logits = out[0, -1]

        # Generate one token at a time by extending the context
        for i in range(max_tokens):
            token = sample_logits(logits, temperature, top_p, top_k)
            generated.append(token)

            try:
                text = tokenizer.decode(generated[len(tokens):])
                new_text = text[len(tokenizer.decode(generated[len(tokens):-1])):]
                print(new_text, end="", flush=True)
            except (UnicodeDecodeError, IndexError):
                pass

            # Extend context (inefficient but correct)
            idx = mx.array([generated])
            out = model(idx)
            mx.eval(out)
            logits = out[0, -1]

        elapsed = time.time() - t0
        print(f"\n\n[{max_tokens / elapsed:.1f} tokens/s, {elapsed:.2f}s total]")

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text with RWKV-v7 on MLX")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights directory")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer vocab file (rwkv_vocab_v20230424.txt)")
    parser.add_argument("--prompt", type=str, default="The Eiffel tower is in the city of",
                        help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 to disable)")
    parser.add_argument("--mode", choices=["rnn", "gpt"], default="rnn",
                        help="Inference mode: rnn (sequential) or gpt (parallel)")

    args = parser.parse_args()

    # Load config
    model_dir = Path(args.model)
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        config = RWKVConfig(**cfg)
    else:
        print("No config.json found, using defaults")
        config = RWKVConfig()

    # Load model
    print(f"Loading model from {model_dir}...")
    model = RWKV(config)
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists():
        model.load_weights(str(weights_path))
    else:
        # Try loading from .safetensors files
        safetensor_files = list(model_dir.glob("*.safetensors"))
        # Sort by modification time, newest first
        safetensor_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if safetensor_files:
            model.load_weights(str(safetensor_files[0]))
        else:
            raise FileNotFoundError(f"No weights found in {model_dir}")
    mx.eval(model.parameters())

    # Load tokenizer
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        # Try common locations
        for p in [
            model_dir / "rwkv_vocab_v20230424.txt",
            Path("rwkv_vocab_v20230424.txt"),
            Path(__file__).parent.parent / "rwkv_vocab_v20230424.txt",
            Path(__file__).parent.parent.parent / "RWKV-v7" / "rwkv_vocab_v20230424.txt",
        ]:
            if p.exists():
                tokenizer_path = str(p)
                break

    if tokenizer_path is None:
        raise FileNotFoundError(
            "Tokenizer not found. Please provide --tokenizer path to rwkv_vocab_v20230424.txt"
        )

    tokenizer = RWKVTokenizer(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path}")
    
    # Load prompt
    # if Path(args.prompt).exists():
    #     with open(args.prompt, "r") as f:
    #         args.prompt = f.read()
    #     print(f"Loaded prompt from {args.prompt}")

    # Generate
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model: L{config.n_layer} D{config.n_embd}, {n_params:,} params")
    print(f"Checkpoint: {weights_path if weights_path.exists() else safetensor_files[0]}")
    print(f"Mode: {'RNN' if args.mode == 'rnn' else 'GPT'}")
    print(f"Temperature: {args.temperature}, top_p: {args.top_p}")
    print("---")

    generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_rnn_mode=(args.mode == "rnn"),
    )


if __name__ == "__main__":
    main()
