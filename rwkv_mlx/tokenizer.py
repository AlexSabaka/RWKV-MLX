"""
RWKV World Tokenizer with special token support.

Port of the RWKV tokenizer from RWKV-v7/rwkv_v7_demo.py.
Uses the rwkv_vocab_v20230424.txt vocabulary file (65536 base tokens)
plus additional special tokens for chat/reasoning (IDs 65536+).
"""

import re


# Special tokens appended after the base RWKV World vocab (65536 tokens).
# IDs 65536..65545. Vocab size padded to 65568 for alignment.
SPECIAL_TOKENS = {
    "<|eos|>": 65536,
    "<|pad|>": 65537,
    "<|im_start|>": 65538,
    "<|im_end|>": 65539,
    "<|think|>": 65540,
    "<|/think|>": 65541,
    "<|tool_call|>": 65542,
    "<|/tool_call|>": 65543,
    "<|tool_response|>": 65544,
    "<|/tool_response|>": 65545,
}

# Padded vocab size (multiple of 32 for GPU alignment)
PADDED_VOCAB_SIZE = 65568

# Convenience aliases
EOS_TOKEN_ID = SPECIAL_TOKENS["<|eos|>"]
PAD_TOKEN_ID = SPECIAL_TOKENS["<|pad|>"]

# Pre-compiled regex for splitting text on special tokens
_SPECIAL_PATTERN = re.compile(
    "(" + "|".join(re.escape(tok) for tok in SPECIAL_TOKENS) + ")"
)


class RWKVTokenizer:
    """RWKV World vocabulary tokenizer with special token support.

    Base vocab: 65536 byte-level tokens from rwkv_vocab_v20230424.txt
    Special tokens: 10 tokens at IDs 65536-65545
    Total vocab size: 65568 (padded for alignment)
    """

    def __init__(self, vocab_file: str):
        self.idx2token: dict[int, bytes] = {}
        sorted_tokens: list[bytes] = []

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[:line.index(' ')])
                x = eval(line[line.index(' '):line.rindex(' ')])
                x = x.encode("utf-8") if isinstance(x, str) else x
                assert isinstance(x, bytes)
                assert len(x) == int(line[line.rindex(' '):])
                sorted_tokens.append(x)
                self.idx2token[idx] = x

        # Register special tokens
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.special_tokens_reverse: dict[int, str] = {}
        for name, idx in self.special_tokens.items():
            self.idx2token[idx] = name.encode("utf-8")
            self.special_tokens_reverse[idx] = name

        self.token2idx: dict[bytes, int] = {v: k for k, v in self.idx2token.items()}

        # Precompute lookup tables for fast byte-level matching
        self.table: list[list[list[bytes]]] = [[[] for _ in range(256)] for _ in range(256)]
        self.good: list[set[int]] = [set() for _ in range(256)]
        self.wlen: list[int] = [0] * 256

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = s[0]
                s1 = s[1]
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

        self.vocab_size = PADDED_VOCAB_SIZE

    def encode_bytes(self, src: bytes) -> list[int]:
        """Encode bytes to token indices (base vocab only, no special tokens)."""
        src_len = len(src)
        tokens: list[int] = []
        i = 0
        while i < src_len:
            s = src[i:i + 1]
            if i < src_len - 1:
                s0 = src[i]
                s1 = src[i + 1]
                if s1 in self.good[s0]:
                    sss = src[i:i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decode token indices to bytes."""
        return b''.join(self.idx2token.get(t, b'\xef\xbf\xbd') for t in tokens)

    def encode(self, text: str) -> list[int]:
        """Encode a string to token indices.

        Special tokens (like <|im_start|>) are recognized as atomic units
        and mapped to their IDs directly. Regular text between them is
        encoded using the byte-level tokenizer.
        """
        parts = _SPECIAL_PATTERN.split(text)
        tokens: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self.encode_bytes(part.encode("utf-8")))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode token indices to a string.

        Special tokens are decoded to their string representation
        (e.g. <|im_start|>). Unknown token IDs produce the Unicode
        replacement character.
        """
        parts: list[str] = []
        byte_buf: list[int] = []

        def flush_bytes():
            if byte_buf:
                raw = self.decode_bytes(byte_buf)
                parts.append(raw.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for t in tokens:
            if t in self.special_tokens_reverse:
                flush_bytes()
                parts.append(self.special_tokens_reverse[t])
            else:
                byte_buf.append(t)

        flush_bytes()
        return "".join(parts)

    @property
    def eos_token_id(self) -> int:
        return EOS_TOKEN_ID

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID
