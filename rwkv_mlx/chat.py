"""
RWKV-v7 Interactive Chat REPL for MLX.

Uses RNN mode for efficient multi-turn generation. The RNN state persists
across turns, so the model never re-processes old conversation history.

Usage:
    python -m rwkv_mlx.chat --model models/chat-try-1
    python -m rwkv_mlx.chat --model models/chat-try-1 --system "You are a helpful assistant."

Commands during chat:
    /reset          — Clear conversation history (keeps system prompt)
    /system <text>  — Change system prompt and reset
    /quit           — Exit
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .model import RWKV, RWKVConfig
from .tokenizer import RWKVTokenizer, SPECIAL_TOKENS
from .generate import sample_logits


# Token IDs that end an assistant turn
_IM_END = SPECIAL_TOKENS["<|im_end|>"]   # 65539
_EOS    = SPECIAL_TOKENS["<|eos|>"]      # 65536
_STOP_IDS = {_IM_END, _EOS}


class ChatSession:
    """Multi-turn chat session backed by persistent RWKV RNN state.

    The RNN state naturally encodes full conversation history, so each
    new user turn is processed incrementally without re-reading past turns.
    """

    def __init__(
        self,
        model: RWKV,
        tokenizer: RWKVTokenizer,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.state = model.init_state()

        if system_prompt:
            self._prefill(f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n")

        # Snapshot state after system prompt — used to reset conversation
        # A shallow list copy is sufficient: forward_rnn replaces list slots
        # with new arrays rather than mutating existing array objects.
        self._base_state: list = list(self.state)

    # ── internals ─────────────────────────────────────────────────────────

    def _prefill(self, text: str) -> mx.array | None:
        """Encode text and feed it token-by-token into the RNN state.

        Returns the final logit vector (needed to kick off generation),
        or None if the text is empty.
        """
        tokens = self.tokenizer.encode(text)
        if not tokens:
            return None
        logits = None
        for tok in tokens:
            logits, self.state = self.model.forward_rnn(tok, self.state)
            mx.eval(logits, *self.state)
        return logits

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Revert to the state after the system prompt (clears conversation)."""
        self.state = list(self._base_state)

    def respond(
        self,
        user_message: str,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ):
        """Process a user message and yield assistant response text incrementally.

        Feeds:  <|im_start|>user\\n{user_message}\\n<|im_end|>\\n<|im_start|>assistant\\n
        Yields: text chunks as they are decoded
        Stops:  on <|im_end|>, <|eos|>, or max_tokens
        After:  feeds <|im_end|>\\n to close the assistant turn in state
        """
        prompt = (
            f"<|im_start|>user\n{user_message}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        logits = self._prefill(prompt)
        if logits is None:
            return

        # Generate response tokens, accumulating raw bytes for correct UTF-8
        byte_buf = bytearray()
        for _ in range(max_tokens):
            token_id = sample_logits(logits, temperature, top_p, top_k)
            if token_id in _STOP_IDS:
                break

            # Accumulate raw bytes; yield only once we have a valid UTF-8 boundary
            byte_buf += self.tokenizer.idx2token.get(token_id, b"")
            try:
                chunk = byte_buf.decode("utf-8")
                yield chunk
                byte_buf = bytearray()
            except UnicodeDecodeError:
                pass  # incomplete multi-byte sequence — wait for next token

            logits, self.state = self.model.forward_rnn(token_id, self.state)
            mx.eval(logits, *self.state)

        # Flush any remaining bytes (shouldn't happen with valid UTF-8, but be safe)
        if byte_buf:
            yield byte_buf.decode("utf-8", errors="replace")

        # Close the assistant turn in state so the next user turn is formatted correctly
        self._prefill("<|im_end|>\n")


# ── model / tokenizer loading ─────────────────────────────────────────────────

def _load_model_and_tokenizer(
    model_dir: str,
    tokenizer_path: str | None = None,
    checkpoint: str | None = None,
) -> tuple[RWKV, RWKVTokenizer, RWKVConfig]:
    model_dir = Path(model_dir)

    # Config
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        config = RWKVConfig(**cfg)
    else:
        print("Warning: no config.json found, using defaults", file=sys.stderr)
        config = RWKVConfig()

    # Weights
    model = RWKV(config)
    if checkpoint:
        weights_path = Path(checkpoint)
    else:
        explicit = model_dir / "model.safetensors"
        if explicit.exists():
            weights_path = explicit
        else:
            candidates = sorted(
                model_dir.glob("*.safetensors"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(f"No .safetensors weights found in {model_dir}")
            weights_path = candidates[0]

    print(f"Loading weights from {weights_path}...")
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())

    # Tokenizer
    if tokenizer_path is None:
        for p in [
            model_dir / "rwkv_vocab_v20230424.txt",
            Path("rwkv_vocab_v20230424.txt"),
            Path(__file__).parent.parent / "rwkv_vocab_v20230424.txt",
        ]:
            if p.exists():
                tokenizer_path = str(p)
                break
    if tokenizer_path is None:
        raise FileNotFoundError(
            "Tokenizer not found. Pass --tokenizer path to rwkv_vocab_v20230424.txt"
        )

    tokenizer = RWKVTokenizer(tokenizer_path)
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model: L{config.n_layer} D{config.n_embd}, {n_params:,} params")

    return model, tokenizer, config


# ── REPL ──────────────────────────────────────────────────────────────────────

def _repl(
    session: ChatSession,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    print("\nRWKV Chat — /reset to clear history, /system <text> to change prompt, /quit to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                print("Bye!")
                break
            elif cmd == "reset":
                session.reset()
                print("[Conversation cleared]")
            elif cmd == "system":
                new_system = parts[1] if len(parts) > 1 else ""
                # Re-initialise session — rebuilds state from scratch
                session.__init__(session.model, session.tokenizer, new_system or None)
                print(f"[System prompt updated. Conversation cleared.]")
            else:
                print(f"Unknown command: {user_input}")
            continue

        print("Assistant: ", end="", flush=True)
        for chunk in session.respond(
            user_input,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            print(chunk, end="", flush=True)
        print()  # newline after response


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive RWKV chat REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model directory (must contain config.json)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific .safetensors checkpoint (default: newest in --model dir)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to rwkv_vocab_v20230424.txt (auto-detected if omitted)")
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt (optional)")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Max tokens per assistant response (default: 500)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling, 0 to disable (default: 0)")
    args = parser.parse_args()

    model, tokenizer, _ = _load_model_and_tokenizer(
        args.model, args.tokenizer, args.checkpoint
    )

    session = ChatSession(model, tokenizer, args.system)
    if args.system:
        print(f"System: {args.system}")

    _repl(session, args.max_tokens, args.temperature, args.top_p, args.top_k)


if __name__ == "__main__":
    main()
