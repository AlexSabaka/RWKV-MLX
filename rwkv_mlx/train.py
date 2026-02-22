"""
RWKV-v7 Training Pipeline for MLX on Apple Silicon.

Optimized for Mac mini M4 with 16GB unified memory.

Key optimizations:
- mx.compile: traces the training step once, caches the computation graph
- mx.checkpoint: gradient checkpointing per block to reduce activation memory
- MLX-native LR schedules (cosine decay with linear warmup)
- Proper gradient accumulation support
"""

import argparse
import json
import math
import os
import time
from functools import partial
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from .model import RWKV, RWKVConfig


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

class TextDataset:
    """Text dataset loaded from a binary token file.

    Supports two sampling modes:
    - Random (default): picks random starting positions each call — good for
      pre-training where data order doesn't matter.
    - Sequential: advances a cursor through the file in order — use with
      pre-sorted .bin files (from rank_dataset.py) for curriculum learning.
    """

    def __init__(self, data_path: str, ctx_len: int, tokenizer=None):
        self.ctx_len = ctx_len
        self._pos = 0  # cursor for sequential mode

        if data_path.endswith(".npy"):
            self.data = np.load(data_path).astype(np.int32)
        elif data_path.endswith(".bin"):
            file_size = os.path.getsize(data_path)
            if file_size % 4 == 0:
                # Memory-map as uint32 (zero-copy, OS pages in on demand)
                self.data = np.memmap(data_path, dtype=np.uint32, mode='r')
                # Sanity check: sample a few values to detect format
                check = self.data[:min(1000, len(self.data))]
                if check.max() > 65568:
                    # Doesn't look like uint32 token IDs, try uint16
                    self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            else:
                self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        elif data_path.endswith(".txt"):
            if tokenizer is None:
                raise ValueError("Tokenizer required for .txt files")
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.data = np.array(tokenizer.encode(text), dtype=np.int32)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        self.n_tokens = len(self.data)
        print(f"Loaded {self.n_tokens:,} tokens from {data_path}")

    def get_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Get a random batch of (input, target) pairs."""
        indices = np.random.randint(0, self.n_tokens - self.ctx_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.ctx_len] for i in indices])
        y = np.stack([self.data[i + 1:i + self.ctx_len + 1] for i in indices])
        return mx.array(x), mx.array(y)

    def get_batch_sequential(self, batch_size: int) -> tuple[mx.array, mx.array]:
        """Get a sequential (ordered) batch — reads through the file in order.

        Wraps around at end of file. Use with pre-sorted .bin files to
        implement curriculum learning (easy-first or hard-first ordering).
        Each call advances the cursor by batch_size * ctx_len tokens so
        windows are non-overlapping within an epoch.
        """
        x_rows, y_rows = [], []
        for _ in range(batch_size):
            if self._pos + self.ctx_len + 1 > self.n_tokens:
                self._pos = 0  # wrap around (new epoch)
            x_rows.append(self.data[self._pos : self._pos + self.ctx_len])
            y_rows.append(self.data[self._pos + 1 : self._pos + self.ctx_len + 1])
            self._pos += self.ctx_len
        x = np.stack(x_rows)
        y = np.stack(y_rows)
        return mx.array(x), mx.array(y)


class SFTDataset:
    """Supervised fine-tuning dataset from JSONL with input/output pairs.

    Tokenizes {"input": "...", "output": "..."} records on-the-fly. Loss is
    computed only on output tokens (mask=1) — input tokens are masked out
    (mask=0). Sequences longer than ctx_len are truncated; shorter ones are
    right-padded with <|pad|> tokens.

    Args:
        data_path: path to .jsonl file with "input"/"output" fields
        ctx_len: max sequence length
        vocab_file: path to RWKV World vocab file (default: auto-detect)
    """

    PAD_ID = 65537   # <|pad|>
    EOS_ID = 65536   # <|eos|>

    def __init__(self, data_path: str, ctx_len: int, vocab_file: str = None,
                 max_records: int = None):
        from pathlib import Path
        from .tokenizer import RWKVTokenizer

        if vocab_file is None:
            # Auto-detect: look next to this file and in the project root
            candidates = [
                Path(__file__).parent.parent / "rwkv_vocab_v20230424.txt",
                Path(__file__).parent / "rwkv_vocab_v20230424.txt",
                Path("rwkv_vocab_v20230424.txt"),
            ]
            for p in candidates:
                if p.exists():
                    vocab_file = str(p)
                    break
            if vocab_file is None:
                raise FileNotFoundError(
                    "Could not find rwkv_vocab_v20230424.txt — pass --vocab_file"
                )

        self.tokenizer = RWKVTokenizer(vocab_file)
        self.ctx_len = ctx_len

        # Raw text records — tokenization is lazy (done on first access per record)
        self._raw: list[tuple[str, str]] = []   # (input_str, output_str)
        self._cache: list[tuple | None] = []    # (input_ids, output_ids) or None

        import json
        from tqdm import tqdm
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading SFT records", unit="rec"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._raw.append((obj["input"], obj["output"]))
                if max_records and len(self._raw) >= max_records:
                    break

        self._cache = [None] * len(self._raw)
        print(f"Loaded {len(self._raw):,} SFT records from {data_path}")
        print(f"  Tokenization is lazy — records are encoded on first access")

    def _get_tokens(self, i: int) -> tuple[list, list]:
        """Return (input_ids, output_ids) for record i, tokenizing on first call."""
        if self._cache[i] is None:
            inp_str, out_str = self._raw[i]
            inp = self.tokenizer.encode(inp_str)
            out = self.tokenizer.encode(out_str)
            out.append(self.EOS_ID)
            self._cache[i] = (inp, out)
        return self._cache[i]

    def _pack(self, inp: list, out: list) -> tuple:
        """Pack input+output into (tokens, mask) of length ctx_len."""
        tokens = inp + out
        mask = [0] * len(inp) + [1] * len(out)

        # Truncate to ctx_len+1 (need one extra for target shift)
        tokens = tokens[: self.ctx_len + 1]
        mask   = mask  [: self.ctx_len + 1]

        # Right-pad with PAD_ID / mask=0
        pad_len = self.ctx_len + 1 - len(tokens)
        tokens += [self.PAD_ID] * pad_len
        mask   += [0]           * pad_len

        return tokens, mask

    def get_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        """Get a random batch of (x, y, mask) triples."""
        indices = np.random.randint(0, len(self._raw), size=batch_size)
        x_rows, y_rows, m_rows = [], [], []
        for i in indices:
            inp, out = self._get_tokens(int(i))
            tokens, mask = self._pack(inp, out)
            x_rows.append(tokens[:-1])
            y_rows.append(tokens[1:])
            m_rows.append(mask[1:])  # mask aligned to target positions
        return (
            mx.array(np.array(x_rows, dtype=np.int32)),
            mx.array(np.array(y_rows, dtype=np.int32)),
            mx.array(np.array(m_rows, dtype=np.float32)),
        )


# ---------------------------------------------------------------------------
# Weight Initialization
# ---------------------------------------------------------------------------

def init_weights(model: RWKV, config: RWKVConfig):
    """Initialize model weights following RWKV-v7 conventions.

    This matches the initialization in RWKV-v7/train_temp/src/model.py
    """
    C = config.n_embd
    H = config.n_head
    N = config.head_size
    n_layer = config.n_layer

    def ortho_init(shape, scale=1.0):
        """Orthogonal initialization."""
        if len(shape) == 2:
            rows, cols = shape
            flat = np.random.randn(rows, cols).astype(np.float32)
            q, _ = np.linalg.qr(flat if rows >= cols else flat.T)
            q = q.T if rows < cols else q
            q = q[:rows, :cols]
            gain = math.sqrt(rows / cols) if rows > cols else 1.0
            return mx.array(q * gain * scale)
        return mx.array(np.random.randn(*shape).astype(np.float32) * scale)

    # Embedding
    scale = 1e-4
    emb_w = mx.random.uniform(-scale, scale, model.emb.weight.shape)
    model.emb.weight = emb_w

    # Head (output projection)
    head_shape = model.head.weight.shape
    if config.vocab_size > config.n_embd:
        h_scale = 0.5 * math.sqrt(config.vocab_size / config.n_embd)
    else:
        h_scale = 0.5
    model.head.weight = ortho_init(head_shape, h_scale)

    for layer_id in range(n_layer):
        block = model.blocks[layer_id]
        att = block.att
        ffn = block.ffn

        ratio_0_to_1 = layer_id / (n_layer - 1) if n_layer > 1 else 0
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

        # Token shift mixing parameters
        ddd = np.arange(C, dtype=np.float32) / C
        att.x_r = mx.array((1.0 - ddd ** (0.2 * ratio_1_to_almost0)).reshape(1, 1, C))
        att.x_w = mx.array((1.0 - ddd ** (0.9 * ratio_1_to_almost0)).reshape(1, 1, C))
        att.x_k = mx.array((1.0 - ddd ** (0.7 * ratio_1_to_almost0)).reshape(1, 1, C))
        att.x_v = mx.array((1.0 - ddd ** (0.7 * ratio_1_to_almost0)).reshape(1, 1, C))
        att.x_a = mx.array((1.0 - ddd ** (0.9 * ratio_1_to_almost0)).reshape(1, 1, C))
        att.x_g = mx.array((1.0 - ddd ** (0.2 * ratio_1_to_almost0)).reshape(1, 1, C))

        # Decay (w) parameters
        linear = np.arange(C, dtype=np.float32) / (C - 1) - 0.5
        zigzag = np.zeros(C, dtype=np.float32)
        for n in range(C):
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])

        www = -6 + 6 * (np.arange(C, dtype=np.float32) / (C - 1)) ** (1 + ratio_0_to_1 ** 0.3)
        att.w0 = mx.array((www + 0.5 + zigzag * 2.5).reshape(1, 1, C))
        att.w1 = mx.zeros((C, att.w1.shape[1]))
        att.w2 = ortho_init(att.w2.shape, 0.1)

        # In-context learning rate (a) parameters
        att.a0 = mx.array((-0.19 + zigzag * 0.3 + linear * 0.4).reshape(1, 1, C))
        att.a1 = mx.zeros((C, att.a1.shape[1]))
        att.a2 = ortho_init(att.a2.shape, 0.1)

        # Value residual (v) parameters
        att.v0 = mx.array((0.73 - linear * 0.4).reshape(1, 1, C))
        att.v1 = mx.zeros((C, att.v1.shape[1]))
        att.v2 = ortho_init(att.v2.shape, 0.1)

        # Gate parameters
        att.g1 = mx.zeros((C, att.g1.shape[1]))
        att.g2 = ortho_init(att.g2.shape, 0.1)

        # Key normalization parameters
        att.k_k = mx.array((0.71 - linear * 0.1).reshape(1, 1, C))
        att.k_a = mx.array(np.full((1, 1, C), 1.02, dtype=np.float32))
        att.r_k = mx.array(np.full((H, N), -0.04, dtype=np.float32))

        # Linear projections (orthogonal init, matching reference)
        att.receptance.weight = ortho_init(att.receptance.weight.shape, 1.0)
        att.key.weight = ortho_init(att.key.weight.shape, 0.1)
        att.value.weight = ortho_init(att.value.weight.shape, 1.0)
        att.output.weight = mx.zeros(att.output.weight.shape)

        # GroupNorm: ln_x weight scales by layer depth
        layer_scale = ((1 + layer_id) / n_layer) ** 0.7
        att.ln_x.weight = mx.full(att.ln_x.weight.shape, layer_scale)

        # FFN
        ffn_ratio = 1.0 - (layer_id / n_layer)
        ddd_ffn = np.arange(C, dtype=np.float32) / C
        ffn.x_k = mx.array((1.0 - ddd_ffn ** (ffn_ratio ** 4)).reshape(1, 1, C))
        ffn.key.weight = ortho_init(ffn.key.weight.shape, 1.0)
        ffn.value.weight = mx.zeros(ffn.value.weight.shape)

    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def l2_wrap_loss(logits: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss with L2 regularization on logit magnitudes.

    Matches the reference RWKV L2Wrap: gradient of 1e-4/(B*T) * max_logit
    applied to the argmax position. Using 0.5*max^2 gives gradient of max,
    matching the reference backward pass.
    """
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    # L2 regularization on max logits (matching RWKV L2Wrap)
    # Reference gradient: max_val * 1e-4 / (B*T) at the argmax position
    # Using 0.5*max^2 so d/d(max) = max, then scale by 1e-4/(B*T)
    factor = 1e-4 / (logits.shape[0] * logits.shape[1])
    l2_reg = factor * 0.5 * (logits.max(axis=-1) ** 2).sum()
    return loss + l2_reg


def l2_wrap_loss_masked(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Cross-entropy loss with mask — only compute loss on mask=1 positions.

    Used for supervised fine-tuning (SFT): mask=0 on input tokens, mask=1 on
    output/response tokens. L2 regularization is applied across all positions
    as usual (it acts on activations, not the learning signal).

    Args:
        logits:  (B, T, V)
        targets: (B, T) integer token ids
        mask:    (B, T) float, 1.0 where loss should be computed
    """
    flat_logits  = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask    = mask.reshape(-1)

    # Per-token CE loss without reduction, then masked mean
    per_token_loss = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")
    n_active = flat_mask.sum() + 1e-9
    loss = (per_token_loss * flat_mask).sum() / n_active

    # L2 reg on max logits across all positions (same as non-SFT)
    factor = 1e-4 / (logits.shape[0] * logits.shape[1])
    l2_reg = factor * 0.5 * (logits.max(axis=-1) ** 2).sum()
    return loss + l2_reg


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    config: RWKVConfig,
    data_path: str,
    output_dir: str = "out",
    lr_init: float = 6e-4,
    lr_final: float = 6e-5,
    warmup_steps: int = 10,
    micro_bsz: int = 2,
    grad_accum: int = 1,
    total_steps: int = 10000,
    save_every: int = 1000,
    log_every: int = 10,
    weight_decay: float = 0.001,
    grad_clip: float = 1.0,
    resume: str = None,
    sequential: bool = False,
    sft_data: str = None,
    vocab_file: str = None,
    max_sft_examples: int = None,
):
    """Main training function for RWKV-v7 on MLX."""
    os.makedirs(output_dir, exist_ok=True)

    effective_bsz = micro_bsz * grad_accum
    print(f"RWKV-v7 MLX Training")
    print(f"  Model: L{config.n_layer} D{config.n_embd} (head_size={config.head_size})")
    print(f"  Batch: {micro_bsz} x {grad_accum} = {effective_bsz} effective")
    print(f"  Context: {config.ctx_len}")
    print(f"  LR: {lr_init} -> {lr_final}, warmup {warmup_steps} steps")
    print(f"  Steps: {total_steps}")
    print(f"  Metal WKV kernels: True (custom forward+backward)")
    print(f"  Compiled: True (mx.compile)")
    if sft_data:
        print(f"  Mode: SFT (loss mask on input tokens)")
        print(f"  SFT data: {sft_data}")
    elif sequential:
        print(f"  Mode: Sequential (curriculum — batches read in order)")

    # Create model
    model = RWKV(config)
    start_step = 0
    if resume:
        print(f"Loading checkpoint from {resume}...")
        model.load_weights(resume)
        # Look for saved training state (step number) alongside the checkpoint
        state_path = os.path.join(os.path.dirname(resume), "training_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                saved_state = json.load(f)
            start_step = saved_state.get("step", 0)
            print(f"  Resuming from step {start_step}")
        else:
            print("  No training_state.json found — LR schedule will restart from step 0")
    else:
        print("Initializing weights...")
        model = init_weights(model, config)

    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"  Parameters: {n_params:,}")

    # Estimate memory
    bytes_per_param = 2  # float16
    model_mem_gb = n_params * bytes_per_param / 1e9
    optim_mem_gb = n_params * 4 * 2 / 1e9  # Adam m/v in float32
    act_mem_gb = micro_bsz * config.ctx_len * config.n_embd * config.n_layer * 4 / 1e9
    total_mem_gb = model_mem_gb + optim_mem_gb + act_mem_gb
    print(f"  Est. memory: model={model_mem_gb:.2f}GB, optim={optim_mem_gb:.2f}GB, act~{act_mem_gb:.2f}GB")
    print(f"  Total est.: {total_mem_gb:.2f}GB (available: ~11GB)")

    # Load data — SFT mode takes precedence over plain text dataset
    if sft_data:
        sft_dataset = SFTDataset(sft_data, config.ctx_len, vocab_file=vocab_file,
                                 max_records=max_sft_examples)
        dataset = None  # not needed; all batches come from sft_dataset
    else:
        sft_dataset = None
        dataset = TextDataset(data_path, config.ctx_len)

    # LR schedule: linear warmup then cosine decay
    warmup_schedule = optim.linear_schedule(lr_init * 0.01, lr_init, warmup_steps)
    decay_schedule = optim.cosine_decay(lr_init, total_steps - warmup_steps, lr_final)
    lr_schedule = optim.join_schedules(
        [warmup_schedule, decay_schedule], [warmup_steps]
    )

    # Use Adam without weight decay; apply WD manually to 2D+ params only
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        betas=[0.9, 0.99],
        eps=1e-18,
        weight_decay=0.0,  # handled manually below
    )
    # Restore step counter so LR schedule resumes at the right position
    if start_step > 0:
        optimizer.state["step"] = start_step

    # Build gradient scale tree: 2x for w0 params (matching reference)
    grad_scale = tree_map(lambda _: mx.array(1.0), model.parameters())
    for name, _ in tree_flatten(model.parameters()):
        if name.endswith(".w0"):
            # Navigate the nested dict to set scale
            parts = name.split(".")
            d = grad_scale
            for p in parts[:-1]:
                d = d[int(p)] if p.isdigit() else d[p]
            d[parts[-1]] = mx.array(2.0)

    # Build weight decay mask: 1.0 for 2D+ weight matrices, 0.0 for rest
    wd_mask = tree_map(
        lambda p: mx.ones_like(p) if p.ndim >= 2 else mx.zeros_like(p),
        model.parameters()
    )

    # Loss function — SFT mode uses masked loss; CLM mode uses full loss
    if sft_dataset is not None:
        def loss_fn(model, x, y, mask):
            logits, aux_loss = model(x)
            return l2_wrap_loss_masked(logits, y, mask) + aux_loss
    else:
        def loss_fn(model, x, y):
            logits, aux_loss = model(x)
            return l2_wrap_loss(logits, y) + aux_loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Batch fetching helper — centralises sequential/random/SFT selection
    def get_batch():
        if sft_dataset is not None:
            return sft_dataset.get_batch(micro_bsz)  # returns (x, y, mask)
        elif sequential:
            x, y = dataset.get_batch_sequential(micro_bsz)
            return x, y
        else:
            x, y = dataset.get_batch(micro_bsz)
            return x, y

    # Compiled training step (for grad_accum == 1, CLM only)
    # SFT is excluded from mx.compile because the mask adds a dynamic argument
    # that breaks the static-graph assumption.
    state = [model.state, optimizer.state]
    use_compile = (grad_accum == 1 and sft_dataset is None)

    if use_compile:
        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_train_step(x, y):
            loss, grads = loss_and_grad_fn(model, x, y)
            # Scale w0 grads by 2x (effective 2x learning rate)
            grads = tree_map(lambda g, s: g * s, grads, grad_scale)
            # Gradient clipping by norm
            grad_norm_sq = sum((g * g).sum() for _, g in tree_flatten(grads))
            grad_norm = mx.sqrt(grad_norm_sq)
            clip_scale = mx.minimum(mx.array(1.0), mx.array(grad_clip) / (grad_norm + 1e-6))
            grads = tree_map(lambda g: g * clip_scale, grads)
            optimizer.update(model, grads)
            # Manual weight decay on 2D+ params only (reference: no WD on 1D params)
            new_params = tree_map(
                lambda p, m: p - p * m * (weight_decay * lr_schedule(optimizer.state["step"])),
                model.parameters(), wd_mask
            )
            model.update(new_params)
            return loss, grad_norm

    # Training loop
    print("\nStarting training...")
    if use_compile:
        print("  Using compiled single-step mode")
    elif grad_accum > 1:
        print(f"  Using gradient accumulation ({grad_accum} micro-batches, uncompiled)")
    else:
        mode = "SFT" if sft_dataset is not None else "sequential"
        print(f"  Using uncompiled mode ({mode})")
    print("  First step will be slow (graph tracing)...\n")

    start_time = time.time()
    running_loss = 0.0
    best_loss = float("inf")
    tokens_processed = 0
    last_log_time = start_time

    with tqdm(range(start_step, total_steps + 1), total=total_steps, desc="Training", unit="step", initial=start_step) as pbar:
        for step in pbar:
            step_tokens = effective_bsz * config.ctx_len

            if use_compile:
                # Fast path: fully compiled step (CLM, grad_accum==1)
                x, y = get_batch()
                loss, grad_norm = compiled_train_step(x, y)
                mx.eval(state)
            else:
                # Uncompiled path: SFT, sequential, or grad_accum > 1
                accumulated_grads = None
                accumulated_loss = mx.array(0.0)

                for _ in range(grad_accum):
                    batch = get_batch()
                    if sft_dataset is not None:
                        x, y, mask = batch
                        loss_micro, grads = loss_and_grad_fn(model, x, y, mask)
                    else:
                        x, y = batch
                        loss_micro, grads = loss_and_grad_fn(model, x, y)
                    # Evaluate each micro-batch to keep graph bounded
                    mx.eval(loss_micro, grads)

                    grads = tree_map(lambda g: g / grad_accum, grads)
                    if accumulated_grads is None:
                        accumulated_grads = grads
                    else:
                        accumulated_grads = tree_map(mx.add, accumulated_grads, grads)
                    accumulated_loss = accumulated_loss + loss_micro

                # Gradient clipping
                grad_norm_sq = sum((g * g).sum() for _, g in tree_flatten(accumulated_grads))
                grad_norm = mx.sqrt(grad_norm_sq)
                clip_scale = mx.minimum(mx.array(1.0), mx.array(grad_clip) / (grad_norm + 1e-6))
                accumulated_grads = tree_map(lambda g: g * clip_scale, accumulated_grads)

                # Scale w0 grads by 2x
                accumulated_grads = tree_map(lambda g, s: g * s, accumulated_grads, grad_scale)
                optimizer.update(model, accumulated_grads)
                # Manual weight decay on 2D+ params only
                cur_lr = lr_schedule(optimizer.state["step"])
                new_params = tree_map(
                    lambda p, m: p - p * m * (weight_decay * cur_lr),
                    model.parameters(), wd_mask
                )
                model.update(new_params)
                mx.eval(model.state, optimizer.state, grad_norm)
                loss = accumulated_loss / grad_accum

            tokens_processed += step_tokens
            loss_val = loss.item()
            running_loss += loss_val

            if step % log_every == 0:
                avg_loss = running_loss / log_every
                now = time.time()
                elapsed_since_log = now - last_log_time
                tokens_since_log = log_every * step_tokens
                kt_per_sec = tokens_since_log / elapsed_since_log / 1000
                ppl = math.exp(min(avg_loss, 20))
                # Get current LR from optimizer
                cur_lr = optimizer.learning_rate(optimizer.state["step"]).item() if callable(optimizer.learning_rate) else optimizer.learning_rate
                pbar.set_postfix({
                    "loss": f"{avg_loss:.2f}",
                    "ppl": f"{ppl:.2f}",
                    "lr": f"{cur_lr:.2e}",
                    "ts": f"{kt_per_sec:.1f} Kt/s",
                    "gn": f"{grad_norm.item():.2f}"
                })
                running_loss = 0.0
                last_log_time = now

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # Save checkpoint
            if step > start_step and (step % save_every == 0 or step == total_steps):
                ckpt_path = os.path.join(output_dir, f"rwkv-{step}.safetensors")
                flat_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(ckpt_path, flat_params)
                pbar.write(f"Saved checkpoint: {ckpt_path}")

                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    json.dump({
                        "n_layer": config.n_layer,
                        "n_embd": config.n_embd,
                        "vocab_size": config.vocab_size,
                        "head_size": config.head_size,
                        "ctx_len": config.ctx_len,
                    }, f, indent=2)

                with open(os.path.join(output_dir, "training_state.json"), "w") as f:
                    json.dump({"step": step}, f)

    total_time = time.time() - start_time
    total_kt = tokens_processed / 1000
    print(f"\nTraining complete in {total_time:.1f}s ({total_kt / total_time:.1f} Kt/s avg)")
    print(f"Best loss: {best_loss:.4f} (ppl {math.exp(min(best_loss, 20)):.2f})")


def main():
    parser = argparse.ArgumentParser(description="Train RWKV-v7 on Apple Silicon with MLX")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data (.npy, .bin, or .txt). Required unless --sft_data is given.")
    parser.add_argument("--output", type=str, default="out", help="Output directory")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size")
    parser.add_argument("--ctx_len", type=int, default=512, help="Context length")
    parser.add_argument("--micro_bsz", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_init", type=float, default=6e-4, help="Initial learning rate")
    parser.add_argument("--lr_final", type=float, default=6e-5, help="Final learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--total_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--sequential", action="store_true",
                        help="Read batches in order (for curriculum-ranked .bin files)")
    parser.add_argument("--sft_data", type=str, default=None,
                        help="JSONL file for supervised fine-tuning (input/output pairs)")
    parser.add_argument("--max_sft_examples", type=int, default=None,
                        help="Max SFT records to load (optional cap for large datasets)")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to rwkv_vocab_v20230424.txt (auto-detected if omitted)")

    args = parser.parse_args()

    if args.data is None and args.sft_data is None:
        parser.error("--data is required unless --sft_data is provided")

    config = RWKVConfig(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
        ctx_len=args.ctx_len,
    )

    train(
        config=config,
        data_path=args.data,
        output_dir=args.output,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        micro_bsz=args.micro_bsz,
        grad_accum=args.grad_accum,
        total_steps=args.total_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        resume=args.resume,
        sequential=args.sequential,
        sft_data=args.sft_data,
        vocab_file=args.vocab_file,
        max_sft_examples=args.max_sft_examples,
    )


if __name__ == "__main__":
    main()
